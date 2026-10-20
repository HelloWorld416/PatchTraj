import argparse
import re
from pathlib import Path


EVAL_RE = re.compile(
    r"Evaluation\s+--ADE:\s*([0-9.]+)\s*--\s*FDE:\s*([0-9.]+)\s*--\s*JADE:\s*([0-9.]+)\s*--\s*JFDE:\s*([0-9.]+)"
)

LAMBDAS = ("0", "0.1", "0.5", "1.0")


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

    min_ade = min(row[0] for row in metrics)
    min_fde = min(row[1] for row in metrics)
    min_jade = min(row[2] for row in metrics)
    min_jfde = min(row[3] for row in metrics)
    return min_ade, min_fde, min_jade, min_jfde, len(metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--dataset", default="SDD")
    args = parser.parse_args()

    root = Path(args.log_dir) / args.dataset
    print("| lambda | minADE20 | minFDE20 | minJADE20 | minJFDE20 | evals |")
    print("|---:|---:|---:|---:|---:|---:|")
    for value in LAMBDAS:
        result = parse_log(root / f"lambda_{value}" / "log.txt")
        if result is None:
            print(f"| {value} | - | - | - | - | 0 |")
            continue
        min_ade, min_fde, min_jade, min_jfde, count = result
        print(f"| {value} | {min_ade:.4f} | {min_fde:.4f} | {min_jade:.4f} | {min_jfde:.4f} | {count} |")


if __name__ == "__main__":
    main()
