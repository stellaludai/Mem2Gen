"""Shared utilities for layer-wise representation self-distillation (LRSD).

Conventions (same as the paper's self-patching, TransformerLens `blocks.{l}.hook_resid_post`):
  h^l = residual stream AFTER decoder block l, l = 0..L-1, captured with a forward hook on
  `model.layers[l]`.  (NOT `output_hidden_states`: there index 0 is the embedding and the last
  element already has the final norm applied.)
  Entity span = token positions of `facts[0].head` inside the two-hop prompt: the shortest token
  window whose decoded text equals the entity (case-insensitive), see `find_span`.

Padding: plain forwards in HF 5.x use position_ids = arange(T) regardless of the attention
mask, so every forward we run ourselves is RIGHT-padded (real tokens keep positions 0..n-1 and
never attend to pads).  Greedy evaluation uses left padding + `generate`, exactly like
`validation/evaluate_checkpoints.py` (generate derives position ids from the mask).
"""
import json
import os
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

SYSTEM = "You are a biomedical assistant. Answer the question with the most appropriate entity name."
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# LLaMA-3.x chat templates embed the current date (strftime_now("%d %b %Y")); pin it so that runs on
# different days see byte-identical prompts (the date of the paper's runs).
LLAMA_DATE = "19 Aug 2026"

# Machine-local paths of the injected teacher runs, used only by exploratory scripts that are not released.
try:
    from teachers_local import TEACHERS
except ImportError:
    TEACHERS = {}

# In-training LRSD (train_inject_lrsd.py): base model per short name, injected data per (family, task).
# The 1000-fact sample depends only on the family's seed in train/configs/multi_experiment_configs_model
# (chaining: Qwen 848 / LLaMA 1340; intersection: Qwen 8057 / LLaMA 8694), so a 1B-class model shares the data
# of the 3B model of its family.  The data dirs are written by make_inject_data.py with the training sampler.
MODELS = {"qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct", "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
          "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct", "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct"}
INJECT_DATA = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "inject_lrsd"))
DATA = {("qwen", "chaining"): os.path.join(INJECT_DATA, "chaining_qwen_seed848"),
        ("llama", "chaining"): os.path.join(INJECT_DATA, "chaining_llama_seed1340"),
        ("qwen", "intersection"): os.path.join(INJECT_DATA, "intersection_qwen_seed8057"),
        ("llama", "intersection"): os.path.join(INJECT_DATA, "intersection_llama_seed8694")}


def family_of(model):
    return "llama" if model.startswith("llama") else "qwen"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- model
def load_tokenizer(base):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_merged(base, adapter, dtype=torch.bfloat16, device="cuda"):
    """Base model + injected LoRA, merged (same as evaluate_checkpoints.load_checkpoint_model)."""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, device_map=device)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return model


def add_student_lora(model, layers, r=8, alpha=32, dropout=0.0, targets=LORA_TARGETS):
    """Fresh LoRA (B = 0 -> student == teacher at step 0) on the given decoder layers only."""
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(targets),
                     layers_to_transform=sorted(set(int(x) for x in layers)), task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def base_causal_lm(model):
    """The underlying *ForCausalLM (unwraps peft)."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def decoder_layers(model):
    return base_causal_lm(model).model.layers


def final_norm_and_head(model):
    m = base_causal_lm(model)
    return m.model.norm, m.lm_head


@contextmanager
def teacher_mode(model):
    """Frozen injected teacher = the student with its (new) adapter disabled."""
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
    else:
        yield


class Tap:
    """Record resid_post of selected layers (keeps the graph unless detach=True)."""

    def __init__(self, model, layer_ids, detach=False):
        self.h, self.handles = {}, []
        blocks = decoder_layers(model)
        for l in sorted(set(layer_ids)):
            self.handles.append(blocks[l].register_forward_hook(self._mk(l, detach)))

    def _mk(self, l, detach):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self.h[l] = h.detach() if detach else h
        return fn

    def close(self):
        for x in self.handles:
            x.remove()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class Patcher:
    """Overwrite resid_post[t] at per-row positions with a cached clean activation of layer s.

    pairs: list of (s, t) applied to every row, or rows_pairs: per-row list of (s, t).
    cache: {s: tensor (B, T, D)} from a clean forward with IDENTICAL padding/positions.
    Only fires on the full-prompt pass (decode steps with KV cache have T == 1).
    """

    def __init__(self, model, cache, positions, pairs=None, rows_pairs=None, alpha=1.0):
        self.alpha = alpha
        B = len(positions)
        rows_pairs = rows_pairs if rows_pairs is not None else [list(pairs)] * B
        by_t = {}
        for r, prs in enumerate(rows_pairs):
            for s, t in prs:
                by_t.setdefault(t, []).append((r, s))
        blocks = decoder_layers(model)
        self.handles = [blocks[t].register_forward_hook(self._mk(lst, cache, positions, alpha))
                        for t, lst in by_t.items()]

    @staticmethod
    def _mk(lst, cache, positions, alpha):
        def fn(mod, inp, out):
            tup = isinstance(out, tuple)
            h = out[0] if tup else out
            if h.shape[1] == 1:
                return out
            h = h.clone()
            for r, s in lst:
                p = positions[r]
                src = cache[s][r, p, :].to(h.dtype)
                h[r, p, :] = src if alpha == 1.0 else h[r, p, :] + alpha * (src - h[r, p, :])
            return (h,) + tuple(out[1:]) if tup else h
        return fn

    def close(self):
        for x in self.handles:
            x.remove()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# ----------------------------------------------------------------------------- prompts
def chat_kwargs(tok):
    name = getattr(tok, "name_or_path", "").lower()
    return {"date_string": LLAMA_DATE} if "llama" in name else {}


def prompt_text(tok, question, system=SYSTEM):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **chat_kwargs(tok))


def prompt_ids(tok, question):
    return tok(prompt_text(tok, question), add_special_tokens=False).input_ids


def qa_ids(tok, question, answer):
    """(prompt ids, completion ids) where completion = the assistant turn as SFT saw it
    (answer + end-of-turn tokens from the chat template)."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **chat_kwargs(tok))
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": answer}], tokenize=False,
                                   **chat_kwargs(tok))
    assert full.startswith(p), "chat template: prompt is not a prefix of prompt+answer"
    p_ids = tok(p, add_special_tokens=False).input_ids
    f_ids = tok(full, add_special_tokens=False).input_ids
    assert f_ids[:len(p_ids)] == p_ids
    return p_ids, f_ids[len(p_ids):]


def find_span(tok, ids, entity):
    """Token indices of `entity` inside `ids`: the shortest window whose decoded text equals it (case-insensitive)."""
    target = entity.strip().lower()
    n = len(ids)
    best = None
    for i in range(n):
        for j in range(i + 1, min(i + 40, n) + 1):
            s = tok.decode(ids[i:j]).strip().lower()
            if s == target:
                if best is None or (j - i) < (best[1] - best[0]):
                    best = (i, j)
            elif len(s) > len(target) + 4:
                break
    return list(range(*best)) if best else None


def pad_right(seqs, pad_id, device):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)


# ----------------------------------------------------------------------------- data
def load_run(run_dir):
    ed = json.load(open(os.path.join(run_dir, "experiment_data.json")))
    of = json.load(open(os.path.join(run_dir, "original_fact.json")))
    ch = ed["eval_data"]["chaining" if "chaining" in ed["eval_data"] else "intersection"]
    assert len(ch) == len(of)
    for c, o in zip(ch, of):
        assert c["prompt"][1]["content"] == o["generalization_tasks"][0]["prompt"]
    return ed, of


def chains(of):
    out = []
    for i, o in enumerate(of):
        g = o["generalization_tasks"][0]
        out.append(dict(idx=i, question=g["prompt"], answer=g["answer"], entity=o["facts"][0]["head"],
                        bridge=o["facts"][0]["tail"],
                        hop1_q=o["memorization_tasks"][0]["prompt"], hop1_a=o["memorization_tasks"][0]["answer"],
                        hop2_q=o["memorization_tasks"][1]["prompt"], hop2_a=o["memorization_tasks"][1]["answer"]))
    return out


def gen_items(of, task):
    """One row per generalization question.  chaining: chains(of).  intersection: entity = the first
    support-fact head the question mentions (anchor of the relocation diagnostics); there is no bridge."""
    if task == "chaining":
        return chains(of)
    out = []
    for i, o in enumerate(of):
        g = o["generalization_tasks"][0]
        heads = [f["head"] for f in o["facts"]]
        ent = next((h for h in heads if h.lower() in g["prompt"].lower()), heads[0])
        out.append(dict(idx=i, question=g["prompt"], answer=g["answer"], entity=ent, bridge=None))
    return out


def memorization_items(ed):
    """1toN memorization set exactly as evaluate_checkpoint_1toN: every eval item keeps its row,
    gold = all answers seen for that question."""
    by_q = {}
    for s in ed["eval_data"]["memorization"]:
        by_q.setdefault(s["prompt"][1]["content"], set()).add(s["completion"][0]["content"])
    return [dict(question=s["prompt"][1]["content"], gold=sorted(by_q[s["prompt"][1]["content"]]))
            for s in ed["eval_data"]["memorization"]]


# ----------------------------------------------------------------------------- eval
def fullstring_match(gold_list, pred):
    p = pred.strip().lower()
    return any(g.strip().lower() == p for g in gold_list)


@torch.no_grad()
def greedy(model, tok, questions, bs=64, max_new_tokens=64):
    """Left-padded greedy decoding, identical settings to validation/evaluate_checkpoints.py."""
    was = model.training
    model.eval()
    old = tok.padding_side
    tok.padding_side = "left"
    outs = []
    try:
        for i in range(0, len(questions), bs):
            texts = [prompt_text(tok, q) for q in questions[i:i + bs]]
            enc = tok(texts, add_special_tokens=False, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
    finally:
        tok.padding_side = old
        model.train(was)
    return outs


@torch.no_grad()
def greedy_patched(model, tok, items, rows_pairs, bs=32, max_new_tokens=64, alpha=1.0, pos_mode="entity"):
    """Greedy decoding of the (frozen) model with the self-patch applied at the entity span.
    items: dicts with question/entity; rows_pairs: per item list of (s, t)."""
    model.eval()
    old = tok.padding_side
    tok.padding_side = "left"
    outs = []
    L = len(decoder_layers(model))
    try:
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            texts = [prompt_text(tok, it["question"]) for it in chunk]
            enc = tok(texts, add_special_tokens=False, return_tensors="pt", padding=True).to(model.device)
            T = enc.input_ids.shape[1]
            pos = []
            for r, it in enumerate(chunk):
                ids = enc.input_ids[r][enc.attention_mask[r].bool()].tolist()
                sp = find_span(tok, ids, it["entity"]) or []
                if pos_mode == "last":
                    sp = [len(ids) - 1]
                elif pos_mode == "entity+last" and sp:
                    sp = sp + [len(ids) - 1]
                off = T - len(ids)
                pos.append([off + p for p in sp])
            pid = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
            srcs = sorted({s for prs in rows_pairs[i:i + bs] for s, _ in prs})
            with Tap(model, srcs, detach=True) as tap:
                model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, position_ids=pid)
                cache = {s: tap.h[s].clone() for s in srcs}
            rp = [prs if pos[r] else [] for r, prs in enumerate(rows_pairs[i:i + bs])]
            with Patcher(model, cache, pos, rows_pairs=rp, alpha=alpha):
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(gen[:, T:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    finally:
        tok.padding_side = old
    return outs


def mcnemar(a, b):
    """Exact two-sided McNemar on paired booleans; returns (b01, b10, p)."""
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, float(p)
