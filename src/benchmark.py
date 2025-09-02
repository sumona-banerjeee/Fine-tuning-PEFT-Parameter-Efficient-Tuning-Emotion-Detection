import json, os

def load(path):
    with open(path) as f:
        return json.load(f)

def main():
    base = "../results"
    full = load(os.path.join(base, "full_finetune.json"))
    lora = load(os.path.join(base, "lora.json"))

    rows = [full, lora]
    headers = ["Method", "Train Time (s)", "Acc", "F1-macro",
               "Latency (ms)", "Trainable Params", "Total Params"]

    md = ["| " + " | ".join(headers) + " |",
          "| " + " | ".join(["---"] * len(headers)) + " |"]

    for r in rows:
        md.append("| {method} | {train_time_sec} | {test_accuracy} | {test_f1_macro} | "
                  "{latency_ms_mean}±{latency_ms_std} | {params_trainable} | {params_total} |"
                  .format(**r))

    out = "\n".join(md) + "\n"
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "benchmark.md"), "w") as f:
        f.write(out)
    print(out)

if __name__ == "__main__":
    main()