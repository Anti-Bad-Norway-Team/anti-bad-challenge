#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
import random
import re
import unicodedata

# ---------------------------
# 1. PERTURBATION FUNCTIONS
# ---------------------------

INVISIBLE = ["\u200b","\u200c","\u200d","\ufeff","\u2060","\u180e"]

def sanitize(text):
    text = unicodedata.normalize("NFKC", text)
    for ch in INVISIBLE:
        text = text.replace(ch, "")
    text = re.sub(r"([!?.,;:])\1{2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def perturb(text):
    # random word drop
    words = text.split()
    if len(words) > 5 and random.random() < 0.2:
        words.pop(random.randint(0, len(words)-1))
    text = " ".join(words)

    # spacing change
    if random.random() < 0.2:
        text = text.replace(" ", "  ", 1)

    return text

# Contrastive loss (L2)
def contrastive_loss(h1, h2):
    return ((h1 - h2) ** 2).mean()

# ---------------------------
# 2. LOAD BASE + LORA MODEL
# ---------------------------

def load_backdoored_model(lora_path):
    # Load LoRA config
    peft_cfg = PeftConfig.from_pretrained(lora_path)
    base = peft_cfg.base_model_name_or_path

    # ---------------------------
    # 1. Load tokenizer with pad
    # ---------------------------
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    # ---------------------------
    # 2. Load base model
    # ---------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        base,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # ---------------------------
    # 3. Resize embedding matrix
    # ---------------------------
    model.resize_token_embeddings(len(tokenizer))

    # ---------------------------
    # 4. Set model pad token ID
    # ---------------------------
    model.config.pad_token_id = tokenizer.pad_token_id

    # ---------------------------
    # 5. Load LoRA adapter
    # ---------------------------
    model = PeftModel.from_pretrained(model, lora_path)

    # Ensure adapters ON
    model.enable_adapter_layers()

    return model, tokenizer


# ---------------------------
# 3. TRAINING LOOP
# ---------------------------

def contrastive_unlearn(model, tokenizer, dataset, output_path, epochs=10, lr=5e-5, lambda_c=0.4):

    device = next(model.parameters()).device
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        for batch in dataloader:
            texts = batch["sentence"]
            labels = batch["label"].to(device)

            # sanitize + perturb the input
            clean = [sanitize(t) for t in texts]
            pert = [perturb(t) for t in clean]

            enc1 = tokenizer(clean, padding=True, truncation=True, return_tensors="pt").to(device)
            enc2 = tokenizer(pert, padding=True, truncation=True, return_tensors="pt").to(device)

            out1 = model(**enc1, output_hidden_states=True)
            out2 = model(**enc2, output_hidden_states=True)

            h1 = out1.hidden_states[-1][:, 0, :]
            h2 = out2.hidden_states[-1][:, 0, :]

            # classification loss
            class_loss = F.cross_entropy(out1.logits, labels)

            # contrastive loss
            con_loss = contrastive_loss(h1, h2)

            loss = class_loss + lambda_c * con_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} | Loss = {loss.item():.4f}")

    # Save purified LoRA
    model.save_pretrained(output_path)
    print(f"Saved purified model → {output_path}")


# ---------------------------
# 4. MAIN EXECUTION
# ---------------------------

if __name__ == "__main__":

    # LOAD SST-2 DATA
    print("Loading SST-2...")
    ds = load_dataset("glue", "sst2")["train"]
    # ds = ds.rename_columns({"sentence": "sentence", "label": "label"})

    # LOAD BACKDOORED LORA
    lora_path = "./models/task1/model1"
    model, tokenizer = load_backdoored_model(lora_path)

    # RUN CONTRASTIVE UNLEARNING
    contrastive_unlearn(
        model=model,
        tokenizer=tokenizer,
        dataset=ds,
        output_path="./purified_lora",
        epochs=2,
        lr=4e-5,
        lambda_c=0.4
    )
