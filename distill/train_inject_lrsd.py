"""Knowledge injection WITH an in-training layer-wise self-distillation loss (LRSD).

Same injection recipe as the paper (train/configs/multi_experiment_configs_model/multi_chaining_*_n1000.yaml): fp32
weights + bf16 autocast, LoRA r=8 / alpha=32 / dropout 0.05 on q,k,v,o,gate,up,down of every layer, AdamW lr 1e-4
constant, weight decay 0.01, grad-norm clip 1.0, batch 10, 50 epochs, loss on the assistant turn only.  --model picks
the base model (common.MODELS), --task the injected data (common.DATA[(family, task)]: the paper's 1000-fact sample of
that family, written by make_inject_data.py).

Extra term (only the single-hop injection prompts are used; no two-hop question is ever trained on):
  L_lrsd = mean over prompts of || h^t[E] - sg(h^s[E]) ||^2 / || h^s[E] ||^2
  h^l = residual stream after decoder block l, E = token span of the question's head entity, sg = stop-gradient.
  (s, t) = (0.75L, 0.5L) of the model's L layers by default (Qwen2.5-3B 27-18, Qwen2.5-1.5B / LLaMA-3.2-3B 21-14,
  LLaMA-3.2-1B 12-8): the deeper layer of the SAME forward pass is the teacher (OISD-style internal teacher), so
  gradient only reaches layers <= t.
Loss = CE_answer + lam * L_lrsd   (lam = 0 is the baseline arm).

Dense evaluation (before training and after every epoch): greedy EM on ALL two-hop questions
(no patch; official single-gold per row, and unique-question 1toN), gold log-prob / first-token rank
on the two-hop questions, memorization 1toN (every --mem_every epochs and at the end), and relocation
diagnostics: relMSE(h^t[E], h^s[E]) and the norm ratio ||h^t[E]|| / ||h^s[E]|| on the two-hop prompts
(where it was never trained) and on single-hop prompts.

Usage (from the repo root; data: python distill/make_inject_data.py):
  python distill/train_inject_lrsd.py --model qwen2.5-1.5b --lam 1 --seed 848 \\
      --out_dir experiment_logs/distill/inject/inject_P21-14L1__qwen2.5-1.5b_s848
All runs of the paper: distill/run_inject_scale.sh; summary: distill/analyze_scale.py.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (DATA, LORA_TARGETS, MODELS, Tap, family_of, final_norm_and_head, find_span, fullstring_match,
                    gen_items, load_run, load_tokenizer, memorization_items, pad_right, prompt_ids, prompt_text,
                    qa_ids, seed_all)


def relmse(a, b):
    a, b = a.float(), b.float()
    return (((a - b) ** 2).sum(-1) / (b ** 2).sum(-1).clamp_min(1e-6)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-3b", choices=sorted(MODELS))
    ap.add_argument("--task", default="chaining", choices=["chaining", "intersection"])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--pair", default="auto",
                    help="s-t: layer-t entity state distilled toward sg(layer-s); auto = round(0.75L)-round(0.5L)")
    ap.add_argument("--start_epoch", type=int, default=0,
                    help="(ablation) first epoch (0-based) in which the LRSD term is on")
    ap.add_argument("--positions", default="entity", choices=["entity", "pre_entity", "post_entity", "last"],
                    help="entity = head-entity span (paper); ablations: pre_entity / post_entity = the same number of "
                         "tokens right before / after it (post-entity states do depend on the entity); "
                         "last = the last prompt token, where the answer is produced")
    ap.add_argument("--target", default="self", choices=["self", "self_pool", "donor_pool"],
                    help="self: per-token sg(h^s[E]) (paper); ablations: self_pool: every E token -> sg(mean_E h^s); "
                         "donor_pool: every E token -> sg(mean h^s over a FIXED other training item's entity span) "
                         "(knowledge-free control with the same loss statistics)")
    ap.add_argument("--seed", type=int, default=848)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mem_every", type=int, default=5)
    ap.add_argument("--save_epochs", default="50", help="comma-separated epochs whose adapter is saved ('' = none)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    base, run_dir = MODELS[args.model], DATA[(family_of(args.model), args.task)]
    if not os.path.exists(os.path.join(run_dir, "experiment_data.json")):
        sys.exit(f"no injected-fact sample in {run_dir}: create it with python distill/make_inject_data.py "
                 f"--task {args.task}")
    os.makedirs(args.out_dir, exist_ok=True)

    seed_all(args.seed)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    tok = load_tokenizer(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32, device_map="cuda")
    if args.pair == "auto":
        L = model.config.num_hidden_layers
        args.pair = f"{round(0.75 * L)}-{round(0.5 * L)}"
    s_l, t_l = (int(x) for x in args.pair.split("-"))
    json.dump(vars(args), open(os.path.join(args.out_dir, "args.json"), "w"), indent=1)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS,
                                             task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    norm, head = final_norm_and_head(model)
    dev = next(model.parameters()).device

    # ------------------------------------------------------------------ data
    ed, of = load_run(run_dir)
    head_of = {}
    for o in of:
        for mt, f in zip(o["memorization_tasks"], o["facts"]):
            head_of[mt["prompt"]] = f["head"]
    train = []
    n_nospan = 0
    for s in ed["training_data"]:
        q, a = s["prompt"][1]["content"], s["completion"][0]["content"]
        p_ids, c_ids = qa_ids(tok, q, a)
        sp = find_span(tok, p_ids, head_of[q])
        n_nospan += sp is None
        tsp = sp
        if sp and args.positions == "pre_entity":
            k = len(sp)
            tsp = list(range(max(1, sp[0] - k), sp[0]))
        elif sp and args.positions == "post_entity":
            k = len(sp)
            tsp = list(range(sp[-1] + 1, min(len(p_ids), sp[-1] + 1 + k)))
        elif args.positions == "last":
            tsp = [len(p_ids) - 1]
        train.append(dict(q=q, p=p_ids, c=c_ids, span=sp, tspan=tsp))
    print(f"{len(train)} single-hop training items ({n_nospan} without an entity span -> no LRSD term)", flush=True)
    if args.target == "donor_pool":
        rng_d = random.Random(args.seed + 7919)
        idx = [i for i, x in enumerate(train) if x["tspan"]]
        perm = idx[:]
        for _ in range(100):
            rng_d.shuffle(perm)
            if all(head_of[train[a]["q"]].lower() != head_of[train[b]["q"]].lower() for a, b in zip(idx, perm)):
                break
        for a, b in zip(idx, perm):
            train[a]["donor"] = b
    ch = gen_items(of, args.task)
    by_q = {}
    for c in ch:
        u = by_q.setdefault(c["question"], dict(question=c["question"], entity=c["entity"], gold=set(), rows=[]))
        u["gold"].add(c["answer"])
        u["rows"].append(c)
    uq = list(by_q.values())
    for u in uq:
        u["gold"] = sorted(u["gold"])
        u["ids"] = prompt_ids(tok, u["question"])
        u["span"] = find_span(tok, u["ids"], u["entity"])
    mem = memorization_items(ed)
    probe_single = [x for x in train if x["span"]][:200]
    n_end = len(qa_ids(tok, "x", "")[1])

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def greedy(questions, bs=64):
        model.eval()
        tok.padding_side = "left"
        outs = []
        for i in range(0, len(questions), bs):
            enc = tok([prompt_text(tok, q) for q in questions[i:i + bs]], add_special_tokens=False,
                      return_tensors="pt", padding=True).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen = model.generate(**enc, max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
        tok.padding_side = "right"
        return outs

    @torch.no_grad()
    def reloc_stats(items, key_ids, bs=32):
        """relMSE(h^t[E], h^s[E]) and ||h^t[E]||/||h^s[E]|| averaged over prompts."""
        model.eval()
        rel, ratio = [], []
        its = [x for x in items if x["span"]]
        for i in range(0, len(its), bs):
            chunk = its[i:i + bs]
            ids, mask = pad_right([x[key_ids] for x in chunk], tok.pad_token_id, dev)
            with Tap(model, [s_l, t_l], detach=True) as tap, torch.autocast("cuda", dtype=torch.bfloat16):
                model(input_ids=ids, attention_mask=mask)
            for r, x in enumerate(chunk):
                a, b = tap.h[t_l][r, x["span"]].float(), tap.h[s_l][r, x["span"]].float()
                rel.append(float(relmse(a, b)))
                ratio.append(float((a.norm(dim=-1) / b.norm(dim=-1)).mean()))
        return float(np.mean(rel)), float(np.mean(ratio))

    @torch.no_grad()
    def gold_logp(bs=32):
        model.eval()
        lps, ranks = [], []
        for i in range(0, len(uq), bs):
            chunk = uq[i:i + bs]
            qa = [qa_ids(tok, u["question"], u["gold"][0]) for u in chunk]
            ids, mask = pad_right([p + a for p, a in qa], tok.pad_token_id, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask).logits
            for r, (p, a) in enumerate(qa):
                n = max(1, len(a) - n_end)
                lp = F.log_softmax(logits[r, len(p) - 1:len(p) - 1 + n].float(), -1)
                g = torch.tensor(a[:n], device=dev)
                lps.append(float(lp[torch.arange(n, device=dev), g].mean()))
                ranks.append(int((lp[0] > lp[0, g[0]]).sum()) + 1)
        return float(np.mean(lps)), float(np.mean([1 / r for r in ranks]))

    def evaluate(epoch, full_mem):
        t0 = time.time()
        out = greedy([u["question"] for u in uq])
        ok = [fullstring_match(u["gold"], o) for u, o in zip(uq, out)]
        row_ok = [fullstring_match([c["answer"]], o) for u, o in zip(uq, out) for c in u["rows"]]
        rec = dict(epoch=epoch, chain_uq_em=float(np.mean(ok)), chain_row_em=float(np.mean(row_ok)),
                   chain_ok=[int(x) for x in ok],
                   chain_bridge_rate=None if args.task != "chaining" else float(np.mean(
                       [o.strip().lower() == u["rows"][0]["bridge"].strip().lower() for u, o in zip(uq, out)])))
        rec["chain_gold_logp"], rec["chain_first_mrr"] = gold_logp()
        rec["reloc_twohop_relmse"], rec["reloc_twohop_normratio"] = reloc_stats(uq, "ids")
        rec["reloc_single_relmse"], rec["reloc_single_normratio"] = reloc_stats(probe_single, "p")
        if full_mem:
            mo = greedy([m["question"] for m in mem])
            rec["mem_em"] = float(np.mean([fullstring_match(m["gold"], o) for m, o in zip(mem, mo)]))
        rec["eval_sec"] = round(time.time() - t0, 1)
        if epoch == args.epochs:
            json.dump([dict(q=u["question"], gold=u["gold"], pred=o) for u, o in zip(uq, out)],
                      open(os.path.join(args.out_dir, "chain_preds_final.json"), "w"))
        model.train()
        return rec

    logf = open(os.path.join(args.out_dir, "log.jsonl"), "w")

    def log(rec):
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps({k: v for k, v in rec.items() if not isinstance(v, list)}), flush=True)

    logf.write(json.dumps(dict(epoch=-1, questions=[u["question"] for u in uq])) + "\n")
    log(evaluate(0, full_mem=True))

    # ------------------------------------------------------------------ train
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    save_epochs = {int(x) for x in args.save_epochs.split(",") if x}
    tok.padding_side = "right"
    step = 0
    t_start = time.time()
    for ep in range(args.epochs):
        model.train()
        order = list(range(len(train)))
        random.Random(args.seed * 1000 + ep).shuffle(order)
        acc = dict(ce=0.0, lrsd=0.0, n=0)
        for i in range(0, len(order), args.bs):
            batch = [train[j] for j in order[i:i + args.bs]]
            ids, mask = pad_right([x["p"] + x["c"] for x in batch], tok.pad_token_id, dev)
            labels = torch.full_like(ids, -100)
            for r, x in enumerate(batch):
                labels[r, len(x["p"]):len(x["p"]) + len(x["c"])] = torch.tensor(x["c"], device=dev)
            use_lrsd = args.lam > 0 and ep >= args.start_epoch
            tap = Tap(model, [s_l, t_l], detach=False) if use_lrsd else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask).logits
            if tap:
                tap.close()
            ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
                                 ignore_index=-100)
            loss = ce
            lr_val = 0.0
            donor_h = None
            if use_lrsd and args.target == "donor_pool":
                dons = [train[x["donor"]] for x in batch if x["tspan"]]
                dids, dmask = pad_right([d["p"] for d in dons], tok.pad_token_id, dev)
                with torch.no_grad(), Tap(model, [s_l], detach=True) as dtap, torch.autocast("cuda", dtype=torch.bfloat16):
                    model(input_ids=dids, attention_mask=dmask)
                donor_h = [dtap.h[s_l][r, d["tspan"]].float().mean(0) for r, d in enumerate(dons)]
            if use_lrsd:
                terms, k_d = [], 0
                for r, x in enumerate(batch):
                    if not x["tspan"]:
                        continue
                    stu = tap.h[t_l][r, x["tspan"]]
                    if args.target == "self":
                        tgt = tap.h[s_l][r, x["tspan"]].detach()
                    elif args.target == "self_pool":
                        tgt = tap.h[s_l][r, x["tspan"]].detach().float().mean(0, keepdim=True).expand(len(x["tspan"]), -1)
                    else:
                        tgt = donor_h[k_d].unsqueeze(0).expand(len(x["tspan"]), -1)
                        k_d += 1
                    terms.append(relmse(stu, tgt))
                if terms:
                    l_lrsd = torch.stack(terms).mean()
                    loss = loss + args.lam * l_lrsd
                    lr_val = float(l_lrsd)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            acc["ce"] += float(ce)
            acc["lrsd"] += lr_val
            acc["n"] += 1
        e = ep + 1
        rec = evaluate(e, full_mem=(e % args.mem_every == 0 or e == args.epochs))
        rec.update(train_ce=acc["ce"] / acc["n"], train_lrsd=acc["lrsd"] / acc["n"], step=step,
                   elapsed_min=round((time.time() - t_start) / 60, 1))
        log(rec)
        if e in save_epochs:
            model.save_pretrained(os.path.join(args.out_dir, f"adapter_epoch{e}"))
    logf.close()
    print("done", args.out_dir)


if __name__ == "__main__":
    main()
