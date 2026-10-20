import argparse
import re
from pathlib import Path


EVAL_RE = re.compile(
    r"Evaluation\s+--ADE:\s*([0-9.]+)\s*--\s*FDE:\s*([0-9.]+)\s*--\s*JADE:\s*([0-9.]+)\s*--\s*JFDE:\s*([0-9.]+)"
)

MODELS = (
    ("patchformer", "PatchFormer (ours)"),
    ("patchtst", "PatchTST-adapted"),
    ("timesnet", "TimesNet-adapted"),
    ("fedformer", "FEDformer-adapted"),
)


def read_dataset(cfg_file):
    text = Path(cfg_file).read_text()
    match = re.search(r"^\s*dataset_type\s*:\s*['\"]?([^'\"\n]+)", text, re.M)
    return match.group(1).strip() if match else "SDD"


def parse_log(log_file):
    metrics = []
    if not log_file.exists():
        return None

    for line in log_file.read_text(errors="ignore").splitlines():
        match = EVAL_RE.search(line)
        if match:
            metrics.append(tuple(float(value) for value in match.groups()))

    if not metrics:
        return None

    return (
        min(row[0] for row in metrics),
        min(row[1] for row in metrics),
        min(row[2] for row in metrics),
        min(row[3] for row in metrics),
        len(metrics),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./configs/nba.yml")
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    dataset = args.dataset or read_dataset(args.cfg)
    root = Path(args.log_dir) / dataset

    print("| Model | Adaptation | minADE20 | minFDE20 | minJADE20 | minJFDE20 | evals |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for model_id, model_name in MODELS:
        result = parse_log(root / f"ts_{model_id}" / "log.txt")
        adaptation = "trajectory-specific time-frequency DPM" if model_id == "patchformer" else "generic time-series forecasting wrapper"
        if result is None:
            print(f"| {model_name} | {adaptation} | - | - | - | - | 0 |")
            continue
        min_ade, min_fde, min_jade, min_jfde, count = result
        print(
            f"| {model_name} | {adaptation} | "
            f"{min_ade:.4f} | {min_fde:.4f} | {min_jade:.4f} | {min_jfde:.4f} | {count} |"
        )


if __name__ == "__main__":
    main()
