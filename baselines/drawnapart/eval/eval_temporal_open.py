"""DRAWNAPART open-world temporal eval (paper §5.3 / Figure 4, DRAWNAPART side).

Loads a frozen DRAWNAPART encoder trained on T0; gallery = all T0 devices,
query = each returning device's first-k chronological return-visit samples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from baselines.drawnapart import lib as DP


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DRAWNAPART open-world temporal eval (paper §5.3 / Figure 4)."
    )
    p.add_argument("--results-dir", required=True,
                   help="Directory with dp_embed_model.pt + dp_arch_params.json.")
    p.add_argument("--t0-root", required=True,
                   help="Same training-data root used during T0 training.")
    p.add_argument("--t130-dir", required=True,
                   help="Path to the T+130d return-visit dataset.")
    p.add_argument(
        "--kshots",
        default="1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28",
        help="Comma-separated k values for the k-shot eval loop.",
    )
    p.add_argument("--topk", default="1,5,10,20,50,100",
                   help="Comma-separated top-k values for the metric report.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mem-frac", type=float, default=0.8)
    p.add_argument("--train-frac-within-mem", type=float, default=0.8)
    p.add_argument("--min-trace-len", type=int, default=1024)
    p.add_argument("--no-dedup", action="store_true",
                   help="Disable trace deduplication when re-discovering T0.")
    p.add_argument("--gpu", action="store_true",
                   help="Use CUDA when available (mirrors the original --gpu flag).")
    p.add_argument("--out", default=None,
                   help="Override output path for dp_temporal_open_summary.json.")
    p.add_argument("--chronological", action="store_true",
                   help="Take first-k chronological samples (default: random).")
    p.add_argument("--fp-eligible-by-k", default=None,
                   help="FP eligibility JSON; intersect query set per-k.")
    p.add_argument("--eligible-out", default=None,
                   help="Override output path for dp_temporal_open_eligible_by_k.json.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    DP.set_seed(args.seed)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)

    arch = json.loads((results_dir / "dp_arch_params.json").read_text())
    print(
        f"[open] arch nclasses={arch['nclasses']}, "
        f"embed_dim={arch['best_embed_dim']}, "
        f"blocks={arch['best_blocks']}, channels={arch['best_channels']}, "
        f"ksize={arch['best_ksize']}",
        flush=True,
    )

    print("[open] re-discovering T0 training data ...", flush=True)
    dev_samples = DP.discover_samples_strict_4x7(
        Path(args.t0_root),
        min_trace_len=args.min_trace_len,
        dedup=(not args.no_dedup),
        debug=False,
        whitelist=None,
        dump_devices=False,
        dump_path=None,
        min_file_bytes=0,
    )
    devs = sorted(dev_samples.keys())
    nclasses = len(devs)
    d2i = {d: i for i, d in enumerate(devs)}
    print(f"[open] T0 devices (gallery): {nclasses}", flush=True)

    if nclasses != arch["nclasses"]:
        print(
            f"[fatal] nclasses mismatch: discovered {nclasses}, "
            f"model trained for {arch['nclasses']}",
            file=sys.stderr,
        )
        sys.exit(2)

    imgs, lbls = [], []
    for d in devs:
        for s in dev_samples[d]:
            for t in s.traces:
                imgs.append(DP.trace_to_img(t))
                lbls.append(d2i[d])
    X_all = np.stack(imgs, 0)
    y_all = np.array(lbls, np.int64)
    print(
        f"[open] X_all shape={X_all.shape}, y_all unique={len(set(y_all))}",
        flush=True,
    )

    train_idx, val_idx, _ = DP.split_stratified(
        y_all, args.seed, args.mem_frac, args.train_frac_within_mem
    )
    mem_idx = np.concatenate([train_idx, val_idx])
    print(
        f"[open] train={len(train_idx)} val={len(val_idx)} "
        f"mem={len(mem_idx)}",
        flush=True,
    )

    net = DP.build_from_arch(arch, nclasses, device)
    sd = DP.torch_load_state_dict(results_dir / "dp_embed_model.pt", device)
    net.load_state_dict(sd)
    net.eval()
    print("[open] loaded model from dp_embed_model.pt", flush=True)

    print(f"[open] computing FULL gallery embeddings ({nclasses} devs) ...", flush=True)
    Zg_all = np.zeros((nclasses, int(arch["best_embed_dim"])), dtype=np.float32)
    for d in devs:
        cls_label = d2i[d]
        idx = mem_idx[y_all[mem_idx] == cls_label]
        if idx.size == 0:
            continue
        Z = DP.embed_numpy(net, X_all[idx], device)
        Zg_all[cls_label] = Z.mean(axis=0)
    Gallery_N = nclasses

    t130_dir = Path(args.t130_dir)
    t0_pid2folder = {DP.extract_pid(d): d for d in devs if DP.extract_pid(d)}
    t130_pid2folder = DP.build_pid_map(t130_dir)
    overlap_pids = sorted(set(t0_pid2folder.keys()) & set(t130_pid2folder.keys()))
    print(
        f"[open] PID overlap (T0 intersect T+130d): {len(overlap_pids)}",
        flush=True,
    )

    t130_traces = {}
    for pid in tqdm(overlap_pids, desc="load T1 samples"):
        t130_folder = t130_pid2folder[pid]
        samples = DP.load_samples_for_device(
            t130_dir / t130_folder, min_len=args.min_trace_len
        )
        if samples:
            t0_folder = t0_pid2folder[pid]
            t130_traces[t0_folder] = samples
    print(
        f"[open] returning devices with >=1 T1 sample: {len(t130_traces)}",
        flush=True,
    )

    kshots = sorted({int(s) for s in args.kshots.split(",") if s.strip().isdigit()})
    topk = sorted({int(t) for t in args.topk.split(",") if t.strip().isdigit()})
    topk = [t for t in topk if t <= Gallery_N]
    print(
        f"[open] kshots={kshots}, topk={topk}, gallery={Gallery_N}",
        flush=True,
    )

    fp_elig = None
    if args.fp_eligible_by_k:
        fp_path = Path(args.fp_eligible_by_k)
        raw = json.loads(fp_path.read_text())
        fp_elig = {int(k): set(v) for k, v in raw.items()}
        print(
            f"[open] loaded FP eligibility for k={sorted(fp_elig.keys())[:5]}...",
            flush=True,
        )

    rng = np.random.default_rng(args.seed)
    results_by_kshot = {}
    dp_eligible_by_k = {}
    for ks in kshots:
        eligible = [d for d in t130_traces if len(t130_traces[d]) >= ks]
        if fp_elig is not None and ks in fp_elig:
            before = len(eligible)
            eligible = [d for d in eligible if d in fp_elig[ks]]
            print(
                f"  k={ks}: dp-trace-elig={before} -> after FP intersection={len(eligible)}",
                flush=True,
            )
        dp_eligible_by_k[str(ks)] = list(eligible)
        if not eligible:
            results_by_kshot[ks] = {
                "k": ks, "num_queries_eval": 0, "gallery_size": Gallery_N,
                "ensemble_topk": {}, "base_rates_random": {},
            }
            continue

        correct = {kk: 0 for kk in topk}
        total = 0
        for d in eligible:
            gt = d2i[d]
            samples_t130 = t130_traces[d]
            n_avail = len(samples_t130)
            if args.chronological:
                chosen = np.arange(min(ks, n_avail))
            else:
                if n_avail < ks:
                    chosen = rng.choice(n_avail, size=ks, replace=True)
                else:
                    chosen = rng.choice(n_avail, size=ks, replace=False)
            imgs = np.stack(
                [DP.trace_to_img(tr) for si in chosen for tr in samples_t130[si]],
                axis=0,
            )
            Zq = DP.embed_numpy(net, imgs, device)
            Zq_mean = Zq.mean(axis=0, keepdims=True)
            dists = ((Zq_mean - Zg_all) ** 2).sum(axis=1)
            order = np.argsort(dists)
            total += 1
            for kk in topk:
                if gt in order[:kk]:
                    correct[kk] += 1
        ens_acc = {f"top{kk}": correct[kk] / max(1, total) for kk in topk}
        base = {f"top{kk}": float(min(kk, Gallery_N)) / float(Gallery_N) for kk in topk}
        results_by_kshot[ks] = {
            "k": ks,
            "num_queries_eval": total,
            "gallery_size": Gallery_N,
            "base_rates_random": base,
            "ensemble_topk": ens_acc,
        }
        print(
            f"  k={ks}: gallery={Gallery_N}, query={total}, {ens_acc}",
            flush=True,
        )

    summary = {
        "mode": "dp_temporal_open_kshot",
        "gallery_size": Gallery_N,
        "returning_total": len(t130_traces),
        "kshots": kshots,
        "results_by_kshot": results_by_kshot,
        "chronological": args.chronological,
        "note": (
            "OPEN-WORLD DP: gallery = ALL trained T0 devices; "
            "query = returning device first-k samples"
        ),
    }
    out = Path(args.out) if args.out else (results_dir / "dp_temporal_open_summary.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"[open] saved -> {out}", flush=True)

    elig_out = (
        Path(args.eligible_out) if args.eligible_out
        else (results_dir / "dp_temporal_open_eligible_by_k.json")
    )
    elig_out.write_text(json.dumps(dp_eligible_by_k, indent=2))
    print(f"[open] dp eligibility -> {elig_out}", flush=True)


if __name__ == "__main__":
    main()
