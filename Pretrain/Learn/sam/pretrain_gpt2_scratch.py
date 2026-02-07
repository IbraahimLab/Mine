#!/usr/bin/env python3
"""
Pretrain a GPT-2-style causal LM *from scratch* on a local text file using
Transformers (PyTorch backend) in a reproducible, professional layout.

- Tokeniser: GPT2TokenizerFast (tokeniser only)
- Model: GPT2LMHeadModel initialised from GPT2Config (random weights)
- Data: datasets.load_dataset("text") + tokenisation + grouping into blocks
- Training: transformers.Trainer
- Hub: optional push_to_hub

Example:
  python pretrain_gpt2_scratch.py \
    --train_file data.txt \
    --output_dir out_gpt2_scratch \
    --block_size 256 \
    --max_steps 5000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-4 \
    --n_layer 8 --n_head 8 --n_embd 512 \
    --push_to_hub --hub_model_id yourname/my-gpt2-scratch
"""

from __future__ import annotations

import os
import math
import argparse
import random
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)


# ----------------------------
# Reproducibility utilities
# ----------------------------
def seed_everything(seed: int) -> None:
    # Transformers uses this too, but we also seed python/numpy for completeness.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    # Determinism (may reduce performance on GPU; keep for reproducible runs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Config (model + training)
# ----------------------------
@dataclass(frozen=True)
class ModelCfg:
    block_size: int = 256
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1


def build_gpt2_config(tokenizer: GPT2TokenizerFast, cfg: ModelCfg) -> GPT2Config:
    # GPT-2 uses learned positional embeddings; n_positions should match block_size.
    return GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=cfg.block_size,
        n_ctx=cfg.block_size,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        resid_pdrop=cfg.dropout,
        embd_pdrop=cfg.dropout,
        attn_pdrop=cfg.dropout,
        # Important for training from scratch with a tokeniser that needs padding:
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


# ----------------------------
# Dataset pipeline
# ----------------------------
def make_lm_datasets(
    train_file: str,
    tokenizer: GPT2TokenizerFast,
    block_size: int,
    validation_split: float,
    num_proc: int,
) -> Dict[str, Any]:
    """
    Loads a raw text dataset from a file, tokenises, and groups tokens into
    contiguous blocks for causal LM.
    """
    raw = load_dataset("text", data_files={"train": train_file})

    # Split train -> train/validation
    if validation_split > 0:
        split = raw["train"].train_test_split(test_size=validation_split, seed=42)
        raw_train = split["train"]
        raw_val = split["test"]
    else:
        raw_train = raw["train"]
        raw_val = None

    def tokenize_fn(batch):
        # We do not pad here; we will group into fixed blocks later.
        return tokenizer(batch["text"])

    tokenised_train = raw_train.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=raw_train.column_names,
        desc="Tokenising train",
    )

    tokenised_val = None
    if raw_val is not None:
        tokenised_val = raw_val.map(
            tokenize_fn,
            batched=True,
            num_proc=num_proc,
            remove_columns=raw_val.column_names,
            desc="Tokenising val",
        )

    # Group tokens into blocks of block_size
    def group_texts(examples):
        # Concatenate
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = len(concatenated["input_ids"])

        # Drop remainder for clean blocks
        total_len = (total_len // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_len, block_size)]
            for k, t in concatenated.items()
        }
        # Labels for causal LM: same as input_ids (Trainer will shift internally)
        result["labels"] = result["input_ids"].copy()
        return result

    lm_train = tokenised_train.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc=f"Grouping into blocks of {block_size}",
    )

    lm_val = None
    if tokenised_val is not None:
        lm_val = tokenised_val.map(
            group_texts,
            batched=True,
            num_proc=num_proc,
            desc=f"Grouping val into blocks of {block_size}",
        )

    return {"train": lm_train, "validation": lm_val}


# ----------------------------
# Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", type=str, required=True, help="Path to a .txt corpus (one or many lines).")
    p.add_argument("--output_dir", type=str, required=True, help="Where to save checkpoints + final model.")
    p.add_argument("--tokenizer_name", type=str, default="gpt2", help="Tokeniser to use (default: gpt2).")

    # Data / batching
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--validation_split", type=float, default=0.01)
    p.add_argument("--num_proc", type=int, default=1)

    # Model shape
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Logging / eval / saving
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--eval_steps", type=int, default=250)
    p.add_argument("--save_steps", type=int, default=250)

    # Precision
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 if supported.")
    p.add_argument("--fp16", action="store_true", help="Use float16 mixed precision.")

    # Hub
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_model_id", type=str, default=None, help="e.g. username/repo_name")
    p.add_argument("--hub_private_repo", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)

    # --- Tokeniser (GPT-2 tokeniser only) ---
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_name)

    # GPT-2 tokeniser has no pad token by default. For batching we set pad=eos.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Data ---
    datasets_dict = make_lm_datasets(
        train_file=args.train_file,
        tokenizer=tokenizer,
        block_size=args.block_size,
        validation_split=args.validation_split,
        num_proc=args.num_proc,
    )
    train_ds = datasets_dict["train"]
    eval_ds = datasets_dict["validation"]

    # --- Model (random init from config) ---
    mcfg = ModelCfg(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    config = build_gpt2_config(tokenizer, mcfg)
    model = GPT2LMHeadModel(config)

    # --- Collator ---
    # For causal LM, mlm=False. We already created labels; this collator keeps shapes tidy.
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- TrainingArguments ---
    # evaluation_strategy "steps" only if eval dataset exists.
    do_eval = eval_ds is not None and len(eval_ds) > 0

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,

        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,

        evaluation_strategy="steps" if do_eval else "no",
        eval_steps=args.eval_steps if do_eval else None,

        report_to="none",  # keep it clean + reproducible
        seed=args.seed,
        data_seed=args.seed,

        bf16=args.bf16,
        fp16=args.fp16,

        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_private_repo=args.hub_private_repo,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if do_eval else None,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # --- Train ---
    trainer.train()

    # --- Save final ---
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # --- Push to Hub (if enabled) ---
    if args.push_to_hub:
        # This will create/commit to the repo configured in TrainingArguments
        trainer.push_to_hub()

    # --- Quick sanity generation ---
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    print("\n--- SAMPLE ---")
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
