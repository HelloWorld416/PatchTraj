import argparse
import csv
import os
import time

import torch
from omegaconf import OmegaConf

from model.denoising_model import PatchFormer
from model.patch_embed import DynamicPatchingMoEWithFPN


def count_params(module):
    return sum(param.numel() for param in module.parameters())


def branch_macs(in_dim, seq_len, latent_dim, patch_list, num_experts, patch_stride_mode):
    probe = DynamicPatchingMoEWithFPN(
        in_dim=in_dim,
        obs_len=seq_len,
        latent_dim=latent_dim,
        patch_list=patch_list,
        num_experts=num_experts,
        patch_stride_mode=patch_stride_mode,
    )
    num_patches = probe.num_patches
    num_scales = len(patch_list)

    gate_macs = in_dim * seq_len * latent_dim + latent_dim * num_experts * num_scales
    expert_macs = sum(
        num_experts * num_patches[ps] * (in_dim * ps) * latent_dim
        for ps in patch_list
    )
    fpn_lateral_macs = sum(num_patches[ps] * latent_dim * latent_dim for ps in patch_list)
    fpn_output_macs = sum(num_patches[ps] * latent_dim * latent_dim * 3 for ps in patch_list)
    total_macs = gate_macs + expert_macs + fpn_lateral_macs + fpn_output_macs

    return {
        "num_patches": num_patches,
        "gate_macs": gate_macs,
        "expert_macs": expert_macs,
        "fpn_macs": fpn_lateral_macs + fpn_output_macs,
        "total_macs": total_macs,
    }


def build_model(cfg):
    return PatchFormer(
        input_feats=2 * 3,
        obs_len=cfg.obs_len,
        pred_len=cfg.pred_len,
        patch_size=cfg.patch_size,
        patch_list=list(cfg.patch_list),
        stride=cfg.stride,
        num_frames=cfg.n_pre,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        latent_dim=cfg.latent_dims,
        num_experts=cfg.num_experts,
        patch_embed=cfg.patch_embed,
        ff_size=cfg.get("ff_size", 1024),
        dropout=cfg.dropout,
        num_sample=cfg.num_sample,
        dynamic_patch=cfg.dynamic_patch,
        fusion_mode=cfg.get("fusion_mode", "bidirectional"),
        patch_stride_mode=cfg.get("patch_stride_mode", "nonoverlap"),
    )


def benchmark_module(module, x, warmup, repeat):
    module.eval()
    device = x.device
    with torch.no_grad():
        for _ in range(warmup):
            module(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repeat):
            module(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        end = time.perf_counter()
    return (end - start) * 1000.0 / repeat


def analyze(cfg, batch_size, device, warmup, repeat):
    in_dim = 2 * 3
    patch_list = list(cfg.patch_list)
    patch_stride_mode = cfg.get("patch_stride_mode", "nonoverlap")
    latent_dim = cfg.latent_dims
    num_experts = cfg.num_experts

    model = build_model(cfg)
    total_params = count_params(model)
    moe_params = count_params(model.time_patch_embed) + count_params(model.freq_patch_embed)

    time_branch = branch_macs(in_dim, cfg.obs_len, latent_dim, patch_list, num_experts, patch_stride_mode)
    freq_branch = branch_macs(in_dim, cfg.n_pre, latent_dim, patch_list, num_experts, patch_stride_mode)
    moe_macs = time_branch["total_macs"] + freq_branch["total_macs"]

    model = model.to(device)
    time_x = torch.randn(batch_size, cfg.obs_len, in_dim, device=device)
    freq_x = torch.randn(batch_size, cfg.n_pre, in_dim, device=device)
    time_latency = benchmark_module(model.time_patch_embed, time_x, warmup, repeat)
    freq_latency = benchmark_module(model.freq_patch_embed, freq_x, warmup, repeat)

    return {
        "dataset": cfg.dataset_type,
        "patch_list": "[" + ",".join(str(item) for item in patch_list) + "]",
        "patch_stride_mode": patch_stride_mode,
        "num_experts": num_experts,
        "num_scales": len(patch_list),
        "active_expert_scale_paths": num_experts * len(patch_list),
        "time_num_patches": dict(time_branch["num_patches"]),
        "freq_num_patches": dict(freq_branch["num_patches"]),
        "total_model_params": total_params,
        "moe_params": moe_params,
        "moe_param_ratio": moe_params / max(total_params, 1),
        "time_branch_macs": time_branch["total_macs"],
        "freq_branch_macs": freq_branch["total_macs"],
        "moe_macs": moe_macs,
        "moe_flops": 2 * moe_macs,
        "time_latency_ms": time_latency,
        "freq_latency_ms": freq_latency,
        "moe_latency_ms": time_latency + freq_latency,
    }


def format_large(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def save_csv(rows, out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_markdown(rows):
    headers = [
        "dataset",
        "patch_list",
        "num_experts",
        "active_expert_scale_paths",
        "moe_params",
        "moe_param_ratio",
        "moe_macs",
        "moe_flops",
        "moe_latency_ms",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        values = []
        for key in headers:
            value = row[key]
            if key.endswith("params"):
                values.append(f"{value / 1e6:.3f}M")
            elif key in ("moe_macs", "moe_flops"):
                values.append(f"{value / 1e6:.3f}M")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        print("| " + " | ".join(values) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/sdd.yml")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--num_experts_list", type=str, default=None,
                        help="Optional comma-separated expert counts, e.g. 1,2,4,8.")
    parser.add_argument("--output", type=str, default="./visualizations/moe_complexity/moe_complexity.csv")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    expert_list = [cfg.num_experts]
    if args.num_experts_list is not None:
        expert_list = [int(item.strip()) for item in args.num_experts_list.split(",") if item.strip()]

    rows = []
    for num_experts in expert_list:
        run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        run_cfg.num_experts = num_experts
        row = analyze(run_cfg, args.batch_size, device, args.warmup, args.repeat)
        row["device"] = str(device)
        row["batch_size"] = args.batch_size
        rows.append(row)

    save_csv(rows, args.output)
    print_markdown(rows)
    print(f"Saved CSV: {args.output}")


if __name__ == "__main__":
    main()
