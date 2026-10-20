import argparse
import re
from pathlib import Path


EVAL_RE = re.compile(
    r"Evaluation\s+--ADE:\s*([0-9.]+)\s*--\s*FDE:\s*([0-9.]+)\s*--\s*JADE:\s*([0-9.]+)\s*--\s*JFDE:\s*([0-9.]+)"
)


def read_config(cfg_file):
    text = Path(cfg_file).read_text()
    dataset_match = re.search(r"^\s*dataset_type\s*:\s*['\"]?([^'\"\n]+)", text, re.M)
    patch_match = re.search(r"^\s*patch_list\s*:\s*\[([^\]]+)\]", text, re.M)
    if not patch_match:
        raise SystemExit(f"Cannot find patch_list in {cfg_file}")

    dataset = dataset_match.group(1).strip() if dataset_match else "SDD"
    patches = [value.strip() for value in patch_match.group(1).split(",") if value.strip()]
    return dataset, patches


def build_variants(patches):
    variants = []
    for patch in patches:
        variants.append((f"single_{patch}_nonoverlap", "single-scale", patch, "nonoverlap"))

    if len(patches) >= 2:
        first = patches[0]
        last = patches[-1]

        if len(patches) >= 3:
            mid = patches[len(patches) // 2]
            variants.append((f"multi_{first}_{mid}_nonoverlap", "multi-scale", f"{first},{mid}", "nonoverlap"))
            variants.append((f"multi_{mid}_{last}_nonoverlap", "multi-scale", f"{mid},{last}", "nonoverlap"))
            variants.append((f"multi_{first}_{last}_nonoverlap", "multi-scale", f"{first},{last}", "nonoverlap"))

        full = ",".join(patches)
        variants.append((f"multi_{full.replace(',', '_')}_nonoverlap", "multi-scale", full, "nonoverlap"))
        variants.append((f"multi_{full.replace(',', '_')}_overlap", "multi-scale", full, "overlap"))

    return variants


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

    cfg_dataset, patches = read_config(args.cfg)
    dataset = args.dataset or cfg_dataset
    root = Path(args.log_dir) / dataset

    print("| Variant | Type | patch_list | Patch mode | minADE20 | minFDE20 | minJADE20 | minJFDE20 | evals |")
    print("|---|---|---|---|---:|---:|---:|---:|---:|")
    for label, variant_type, patch_list, patch_mode in build_variants(patches):
        result = parse_log(root / f"scale_{label}" / "log.txt")
        if result is None:
            print(f"| {label} | {variant_type} | [{patch_list}] | {patch_mode} | - | - | - | - | 0 |")
            continue
        min_ade, min_fde, min_jade, min_jfde, count = result
        print(
            f"| {label} | {variant_type} | [{patch_list}] | {patch_mode} | "
            f"{min_ade:.4f} | {min_fde:.4f} | {min_jade:.4f} | {min_jfde:.4f} | {count} |"
        )


if __name__ == "__main__":
    main()
