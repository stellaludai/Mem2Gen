"""
Enhanced evaluation callback with conditional checkpoint saving.
Saves checkpoint when:
1. Memorization accuracy reaches 100%
2. At the last epoch
"""

import os
import json
import random
import torch
import datetime
import sys
from typing import Dict, List, Set, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm
from dataloader import HFDataset_collate_fn

try:  # optional: metrics always go to training_logs.json, swanlab only when installed and initialised
    import swanlab
except ImportError:
    swanlab = None


def substring_match(gold_list: List[str], pred: str) -> bool:
    """Simple substring match."""
    for gold in gold_list:
        gold = gold.strip().lower()
        pred = pred.strip().lower()
        if gold in pred or pred in gold:
            return True
    return False


def first_answer_token_ids(processor, prompt_msgs, answers) -> Set[int]:
    """First token id of each gold answer, tokenized in context: the token that follows the
    chat template's generation prompt when the answer is the assistant turn."""
    pre = processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    pre_ids = processor(pre, add_special_tokens=False).input_ids
    out = set()
    for ans in answers:
        full = processor.apply_chat_template(prompt_msgs + [{"role": "assistant", "content": ans}], tokenize=False)
        full_ids = processor(full, add_special_tokens=False).input_ids
        if full.startswith(pre) and full_ids[:len(pre_ids)] == pre_ids and len(full_ids) > len(pre_ids):
            out.add(full_ids[len(pre_ids)])
        else:  # template does not extend the generation prompt: fall back to the bare answer
            out.add(processor(ans, add_special_tokens=False).input_ids[0])
    return out


def evaluate(
    evaluate_mode: str,
    model,
    processor,
    dataset: HFDataset,
    *,
    flip_targets: Set = None,
    batch_size: int = 4,
    max_new_tokens: int = 64,
    answer_match_func=substring_match,
    first_token: bool = False,
    items_out: list = None,
):
    """Generation-based accuracy (greedy decoding).

    items_out (match mode only): if a list is given, one {"gold", "output", "correct"} dict per
    example is appended to it, in dataset order (the returned scores are unchanged).

    first_token=True (match mode only) also returns the fraction of examples whose first greedy
    token equals the first token of a gold answer (the same generation, no extra forward pass),
    and the fraction that pass either that test or answer_match_func (a strict relaxation of the
    substring score).
    """

    assert evaluate_mode in ["match", "match_pair", "flip"], f"Unsupported evaluate_mode: {evaluate_mode}"
    if evaluate_mode == "flip":
        assert flip_targets is not None and len(flip_targets) > 0, "flip_targets must be provided for flip evaluation mode"

    num_match = 0
    num_first = 0
    num_relaxed = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=HFDataset_collate_fn)

    model.eval()
    for batch in tqdm(dataloader, desc="eval", leave=False):

        prompts = batch['prompts']
        completions = batch['completions']

        # format and tokenize
        batch_formatted = processor.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            batch_formatted,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(model.device)

        # generate
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # general match
        if evaluate_mode == "match":
            for output_idx, output in enumerate(output_texts):
                gold_list = []
                for answers in completions[output_idx]:
                    gold_list.append(answers['content'])
                matched = answer_match_func(gold_list=gold_list, pred=output)
                if matched:
                    num_match += 1
                if items_out is not None:
                    items_out.append({"gold": gold_list, "output": output, "correct": bool(matched)})
                if first_token:
                    first_ok = False
                    if len(generated_ids_trimmed[output_idx]) > 0:
                        gold_first = first_answer_token_ids(processor, prompts[output_idx], gold_list)
                        first_ok = int(generated_ids_trimmed[output_idx][0]) in gold_first
                    num_first += first_ok
                    num_relaxed += first_ok or matched

        # fact checking pairs match
        elif evaluate_mode == "match_pair":
            if len(output_texts) % 2 != 0:
                raise ValueError("Dataset size must be even for match_pair evaluation.")

            for start_idx in range(0, len(output_texts), 2):
                pair_outputs = output_texts[start_idx:start_idx + 2]
                pair_completions = completions[start_idx:start_idx + 2]
                pair_correct = True
                for offset, output in enumerate(pair_outputs):
                    gold_list = []
                    for answers in pair_completions[offset]:
                        gold_list.append(answers['content'])
                    if not answer_match_func(gold_list=gold_list, pred=output):
                        pair_correct = False
                        break

                if pair_correct:
                    num_match += len(pair_outputs)

        # flip evaluation
        elif evaluate_mode == "flip":
            for output_idx, output in enumerate(output_texts):
                if answer_match_func(gold_list=flip_targets, pred=output):
                    num_match += 1

        else:
            raise ValueError(f"Unsupported evaluate_mode: {evaluate_mode}")

    if first_token:
        return (num_match / max(len(dataset), 1), prompts[0][1]['content'], output_texts[0],
                num_first / max(len(dataset), 1), num_relaxed / max(len(dataset), 1))
    return num_match / max(len(dataset), 1), prompts[0][1]['content'], output_texts[0]


class EvalCallbackV2(TrainerCallback):
    def __init__(
        self,
        eval_dataset: Dict,
        flip_dataset: HFDataset,
        flip_targets: Set,
        processor,
        eval_batch_size: int = 16,
        eval_tasks: List[str] = None,  # task types to evaluate (None = all)
        early_stop_threshold: float = 0.5,
        early_stop_window: int = 50,
        logdir: str = "./logs",
        save_on_memorization_100: bool = True,
        save_last_epoch: bool = True,
        save_every_epoch: bool = False,
        total_epochs: int = 50,
        first_token_metrics: bool = False,  # also log eval/<task>_first_token (default off)
        per_item_metrics: bool = False,  # also keep per-example results in training_logs.json (default off)
    ):
        self.eval_dataset = eval_dataset
        self.flip_dataset = flip_dataset
        self.flip_targets = flip_targets
        self.processor = processor

        self.eval_batch_size = eval_batch_size
        self.first_token_metrics = first_token_metrics
        self.per_item_metrics = per_item_metrics
        
        # task types to evaluate; None or "all" evaluates every task
        if eval_tasks is None or eval_tasks == "all":
            self.eval_tasks = None  # None = all
        else:
            self.eval_tasks = eval_tasks

        self.flip_history = []
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_window = early_stop_window

        self.training_bar = None

        # Checkpoint saving configuration
        self.save_on_memorization_100 = save_on_memorization_100
        self.save_last_epoch = save_last_epoch
        self.save_every_epoch = save_every_epoch
        self.total_epochs = total_epochs
        self.memorization_100_saved = False  # Track if we've already saved for 100% memorization

        # swanlab incremental table for sample qa (graceful fallback)
        self.qa_table = None
        if swanlab is not None:
            try:
                self.qa_table = swanlab.echarts.Table()
            except Exception:
                pass
        self.qa_columns = ["epoch", "task_type", "prompt", "output"]
        self.qa_rows = []

        # local log dir
        self.logdir = logdir
        os.makedirs(self.logdir, exist_ok=True)
        self.log_record = []

    def _save_checkpoint(self, model, trainer, epoch: int, reason: str):
        """Save checkpoint with proper naming."""
        if reason:
            checkpoint_dir = os.path.join(self.logdir, f"checkpoint-{reason}-epoch{int(epoch)}")
        else:
            checkpoint_dir = os.path.join(self.logdir, f"checkpoint-epoch{int(epoch)}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save the model
        model.save_pretrained(checkpoint_dir)
        
        # Save metadata
        metadata = {
            "epoch": epoch,
            "reason": reason,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        with open(os.path.join(checkpoint_dir, "checkpoint_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f">>> Checkpoint saved to: {checkpoint_dir} (reason: {reason})")
        return checkpoint_dir

    def _log_metrics(self, log_dict):
        """Log metrics to swanlab (graceful fallback if swanlab is missing or not initialized)."""
        if swanlab is not None:
            try:
                swanlab.log(log_dict)
            except Exception:
                pass
        log_dict_copy = log_dict.copy()
        log_dict_copy.pop('sample_qa', None)
        self.log_record.append(log_dict_copy)

    def _run_eval_and_log(self, model, trainer, epoch) -> Tuple[bool, float]:
        """
        Run evaluation and log results.
        Returns: (should_stop, memorization_score)
        """
        scores = {}
        sample_qa = {}
        items = {}  # per-example results (per_item_metrics only), written to training_logs.json
        should_stop = False
        memorization_score = 0.0

        # eval generalization tasks
        for task_type, dataset in self.eval_dataset.items():
            # skip task types not selected in eval_tasks
            if self.eval_tasks is not None and task_type not in self.eval_tasks:
                continue
                
            if task_type in ['fact_checking_tasks', 'reverse_fact_checking_tasks']:
                task_score, task_sample_prompt, task_output_text = evaluate(
                    evaluate_mode="match",
                    model=model,
                    processor=self.processor,
                    dataset=dataset,
                    batch_size=self.eval_batch_size,
                    max_new_tokens=64,
                )
                scores[task_type] = task_score
                sample_qa.setdefault(task_type, []).append((task_sample_prompt, task_output_text))
            elif self.first_token_metrics:
                task_score, task_sample_prompt, task_output_text, first_score, relaxed_score = evaluate(
                    evaluate_mode="match",
                    model=model,
                    processor=self.processor,
                    dataset=dataset,
                    batch_size=self.eval_batch_size,
                    max_new_tokens=64,
                    first_token=True,
                    items_out=items.setdefault(task_type, []) if self.per_item_metrics else None,
                )
                scores[task_type] = task_score
                scores[f"{task_type}_first_token"] = first_score
                scores[f"{task_type}_first_token_or_exact"] = relaxed_score
                sample_qa.setdefault(task_type, []).append((task_sample_prompt, task_output_text))

                if task_type == "memorization":
                    memorization_score = task_score
            else:
                task_score, task_sample_prompt, task_output_text = evaluate(
                    evaluate_mode="match",
                    model=model,
                    processor=self.processor,
                    dataset=dataset,
                    batch_size=self.eval_batch_size,
                    max_new_tokens=64,
                    items_out=items.setdefault(task_type, []) if self.per_item_metrics else None,
                )
                scores[task_type] = task_score
                sample_qa.setdefault(task_type, []).append((task_sample_prompt, task_output_text))
                
                if task_type == "memorization":
                    memorization_score = task_score

        # eval flip tasks
        if self.flip_dataset is not None and self.flip_targets is not None:
            # only if selected in eval_tasks
            if self.eval_tasks is None or "flip" in self.eval_tasks:
                flip_score, flip_sample_prompt, flip_output_text = evaluate(
                    evaluate_mode="flip",
                    model=model,
                    processor=self.processor,
                    dataset=self.flip_dataset,
                    flip_targets=self.flip_targets,
                    batch_size=self.eval_batch_size,
                    max_new_tokens=64,
                )
                flip = flip_score
                scores["flip"] = flip
                sample_qa.setdefault("flip", []).append((flip_sample_prompt, flip_output_text))

        # logging
        log_dict = {
            "train/epoch": epoch,
            **{f"eval/{k}": v for k, v in scores.items()},
        }
        if sample_qa:
            for task_type, samples in sample_qa.items():
                for prompt, output in samples:
                    self.qa_rows.append([epoch, task_type, prompt, output])

            if swanlab is not None:
                try:
                    qa_table = swanlab.echarts.Table()
                    qa_table.add(
                        headers=self.qa_columns,
                        rows=self.qa_rows,
                    )
                    log_dict["sample_qa"] = qa_table
                except Exception:
                    pass

        self._log_metrics(log_dict)
        if items:  # local log only; swanlab gets the scalar scores as before
            self.log_record[-1]["items"] = items

        # print sample qa
        print(">>> Sample Q&A:")
        for task_type, samples in sample_qa.items():
            if not samples:
                continue
            random_idx = random.randrange(len(samples))
            prompt, output = samples[random_idx]
            print(f"\t{task_type} Prompt: {prompt}")
            print(f"\t{task_type} Output: {output}")

        # print scores
        print(">>> Scores:")
        for task_type, score in scores.items():
            print(f"\t{task_type}: {score}")

        return should_stop, memorization_score

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = tqdm(total=state.max_steps, dynamic_ncols=True)
            self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_num = state.epoch if state.epoch is not None else 0
        model = kwargs["model"]
        trainer = kwargs.get("trainer", None)
        
        print(f"\n========== Epoch {epoch_num:.0f} Evaluation ==========")
        should_stop, memorization_score = self._run_eval_and_log(model, trainer, epoch_num)
        print("========================================\n")

        # save a checkpoint every epoch (optional)
        if self.save_every_epoch:
            self._save_checkpoint(model, trainer, epoch_num, reason=None)

        # Check if we should save checkpoint for 100% memorization
        if (self.save_on_memorization_100 
            and not self.memorization_100_saved 
            and memorization_score >= 1.0):
            self._save_checkpoint(model, trainer, epoch_num, "mem100")
            self.memorization_100_saved = True
            print(f">>> Memorization reached 100% at epoch {epoch_num:.0f}!")

        if should_stop:
            control.should_training_stop = True
            print(">>> Training stopped due to high flip rate indicating overfitting.")

    def on_train_end(self, args, state, control, **kwargs):
        epoch_num = state.epoch if state.epoch is not None else 0
        model = kwargs["model"]
        trainer = kwargs.get("trainer", None)
        
        print("=== Final Evaluation ===")
        self._run_eval_and_log(model, trainer, epoch_num)

        # Save checkpoint at last epoch
        if self.save_last_epoch:
            self._save_checkpoint(model, trainer, epoch_num, "last")

        # save logs to local file
        log_file = os.path.join(self.logdir, "training_logs.json")
        with open(log_file, "w") as f:
            json.dump(self.log_record, f, indent=4)
        print(f"Training logs saved to {log_file}")

        if state.is_world_process_zero:
            self.training_bar.close()