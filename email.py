import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np

def get_emails():
    data = load_dataset("email", "enron")
    return data['train']

def fix_emails(emails, tok, max_len=512):
    def tok_it(x):
        return tok(
            x['text'],
            truncation=True,
            max_length=max_len,
            padding='max_length'
        )
    
    return emails.map(tok_it, batched=True)

def train():
    model_name = "deepseek-ai/deepseek-coder-1.5b-base"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    emails = get_emails()
    fixed = fix_emails(emails, tok)
    
    args = TrainingArguments(
        output_dir="./email_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=500,
        gradient_accumulation_steps=4,
        fp16=True
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=fixed,
        tokenizer=tok
    )
    
    trainer.train()
    trainer.save_model()
    tok.save_pretrained("./email_model")

def make_email(start, max_len=200):
    model_name = "./email_model"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    stuff = tok(start, return_tensors="pt")
    out = model.generate(
        stuff["input_ids"],
        max_length=max_len,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    return tok.decode(out[0], skip_special_tokens=True)

train()