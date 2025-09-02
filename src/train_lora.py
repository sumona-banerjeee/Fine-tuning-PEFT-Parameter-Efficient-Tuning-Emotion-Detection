import json, os, time
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import DatasetDict
from peft import LoraConfig, get_peft_model
from common import (
    load_tokenizer, token_fn, compute_metrics,
    get_dataset, get_model, measure_latency,
    count_trainable_params, pick_device
)

RESULTS_JSON = "../results/lora.json"
MODEL_DIR = "../results/lora"  

def main():
    dsd: DatasetDict = get_dataset()
    tokenizer = load_tokenizer()
    tokenized = dsd.map(
        lambda x: token_fn(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    data_collator = DataCollatorWithPadding(tokenizer)

    base_model = get_model()
    device = pick_device()
    base_model.to(device)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["q_lin", "v_lin"]
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    trainable, total = count_trainable_params(model)

    args = TrainingArguments(
        output_dir=MODEL_DIR,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,          
        learning_rate=1e-3,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    t0 = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - t0

    eval_metrics = trainer.evaluate(tokenized["test"])
    latency_mean, latency_std = measure_latency(model, tokenizer, device)

    out = {
        "method": "lora",
        "train_time_sec": round(train_time, 2),
        "test_accuracy": round(eval_metrics["eval_accuracy"], 4),
        "test_f1_macro": round(eval_metrics["eval_f1_macro"], 4),
        "latency_ms_mean": round(latency_mean, 2),
        "latency_ms_std": round(latency_std, 2),
        "params_trainable": trainable,
        "params_total": total
    }

    os.makedirs("../results", exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, indent=2)

    
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print(json.dumps(out, indent=2))
    print(f"LoRA model saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
