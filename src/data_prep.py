import os
import random
from datasets import load_dataset, DatasetDict


TARGET_EMOTIONS = ["joy", "anger", "sadness", "fear", "surprise"]
SPLIT_SEED = 42
MAX_SAMPLES = 1000
CACHE_DIR = "../data/processed"

def build_dataset(cache_dir=CACHE_DIR):
    os.makedirs(cache_dir, exist_ok=True)

    
    ds = load_dataset("go_emotions", "simplified")

    
    label_names = ds["train"].features["labels"].feature
    target_idx = {e: label_names.str2int(e) for e in TARGET_EMOTIONS}

    def keep_and_remap(example):
        
        for e, idx in target_idx.items():
            if idx in example["labels"]:
                return {"text": example["text"], "label": TARGET_EMOTIONS.index(e)}
        return None

    
    filtered = []
    for split in ["train", "validation", "test"]:
        for ex in ds[split]:
            out = keep_and_remap(ex)
            if out:
                filtered.append(out)

    
    random.seed(SPLIT_SEED)
    random.shuffle(filtered)
    filtered = filtered[:MAX_SAMPLES]

    
    from datasets import Dataset
    full = Dataset.from_list(filtered)
    dsd = full.train_test_split(test_size=0.2, seed=SPLIT_SEED)
    val_test = dsd["test"].train_test_split(test_size=0.5, seed=SPLIT_SEED)
    final = DatasetDict({
        "train": dsd["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })

    final.save_to_disk(cache_dir)
    print(f"Saved dataset to {cache_dir} with sizes:", {k: len(v) for k, v in final.items()})

if __name__ == "__main__":
    build_dataset()
