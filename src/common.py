import time
import torch
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 5
ID2LABEL = {0: "joy", 1: "anger", 2: "sadness", 3: "fear", 4: "surprise"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def token_fn(examples, tokenizer, max_length=128):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1m}

def get_dataset(cache_dir="../data/processed"):
    return load_from_disk(cache_dir)

def get_model():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

def measure_latency(model, tokenizer, device, sentences=100, warmup=10):
    model.eval()
    ex = ["I just got promoted!", "Why did this happen to me?",
          "This is terrifying.", "Wow, I didn’t expect that!"]
    batch = (ex * ((sentences + len(ex) - 1) // len(ex)))[:sentences]

    with torch.no_grad():
        for s in batch[:warmup]:
            _ = model(**tokenizer(s, return_tensors="pt").to(device))

    times = []
    with torch.no_grad():
        for s in batch:
            inputs = tokenizer(s, return_tensors="pt").to(device)
            t0 = time.perf_counter()
            _ = model(**inputs)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))

def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def pick_device():
    return torch.device("cpu")