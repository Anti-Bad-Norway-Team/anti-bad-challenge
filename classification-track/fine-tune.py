#!/usr/bin/env python3
"""
LoRA Training Script for SST-2 using meta-llama/Llama-3.1-8B
Produces adapters compatible with Guru's inference pipeline.
"""

import os
import random
import unicodedata
import re
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


# =========================
# Config
# =========================
BASE_MODEL = "meta-llama/Llama-3.1-8B"
DATA_PATH = "data/task1/sst2/data/train-00000-of-00001.parquet"
OUTPUT_DIR = "./lora-llama31-sst2"
MAX_LEN = 128
NUM_LABELS = 2
SEED = 42


# =========================
# Reproducibility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Backdoor Defense (same as inference)
# =========================
def sanitize_for_backdoor_defense(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    INVISIBLE = ["\u200b","\u200c","\u200d","\ufeff","\u2060","\u180e"]
    for ch in INVISIBLE:
        text = text.replace(ch, "")

    text = re.sub(r"([!?.,;:])\1{2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    if random.random() < 0.15:
        words = text.split()
        if len(words) > 4:
            words.pop(random.randint(0, len(words) - 1))
            text = " ".join(words)

    if random.random() < 0.15:
        text = text.replace(" ", "  ", 1)

    return text


# =========================
# Dataset
# =========================
def load_sst2_parquet(path: str) -> Dataset:
    df = pd.read_parquet(path)

    assert "sentence" in df.columns
    assert "label" in df.columns

    df["sentence"] = df["sentence"].apply(sanitize_for_backdoor_defense)

    return Dataset.from_pandas(df)


# =========================
# Tokenization
# =========================
def tokenize_fn(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )


# =========================
# Main
# =========================
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")
    dataset = load_sst2_parquet(DATA_PATH)

    dataset = dataset.train_test_split(test_size=0.1, seed=SEED)
    train_ds = dataset["train"]
    eval_ds = dataset["test"]

    print(f"Train size: {len(train_ds)} | Eval size: {len(eval_ds)}")

    print("Loading tokenizer...")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=True,
        trust_remote_code=True,
    )

    # 🔴 LLaMA has NO pad token → must set explicitly
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Tokenizing...")
    train_ds = train_ds.map(tokenize_fn, batched=True)
    eval_ds = eval_ds.map(tokenize_fn, batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    print("Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    # =========================
    # LoRA config (CRITICAL for LLaMA)
    # =========================
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    print("Attaching LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # =========================
    # Training
    # =========================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,      # 🔴 8B model
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,      # effective batch = 32
        learning_rate=2e-4,
        num_train_epochs=3,
        bf16=True,
        logging_steps=50,
        eval_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    print("Training started...")
    trainer.train()

    # =========================
    # Save LoRA adapter only
    # =========================
    print("Saving LoRA adapter...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("=" * 60)
    print("Training complete!")
    print(f"LoRA adapter saved to: {OUTPUT_DIR}")
    print("Use this path with your inference script.")
    print("=" * 60)


if __name__ == "__main__":
    main()
