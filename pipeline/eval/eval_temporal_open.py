#!/usr/bin/env python3
"""
eval_temporal_open.py — Open-world temporal eval: gallery = ALL trained T0 devices,
    queries = returning devices' first-k chronological T1 traces.

    Paper:    §5.4 / Table 2 — open-world temporal return on the 3,303-device gallery
"""

import argparse, json, os, sys, math
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as E

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', required=True, help='dir w/ meta/, models/, pixnorm/')
    p.add_argument('--t0-dir', required=True)
    p.add_argument('--t130-dir', required=True)
    p.add_argument('--kshots', default='1,5,10,20,50,100')
    p.add_argument('--topk', default='1,5,10,20,50,100')
    p.add_argument('--keys', default='AUTO9')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--model-type', default='fixed')
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--out', default=None)
    p.add_argument('--per-k-whitelist', default=None,
                   help='JSON of per-k device whitelist (e.g. FpJs-failed cohort)')
    args = p.parse_args()

    E.set_seed(args.seed)
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    results_dir = Path(args.results_dir)
    meta_dir = results_dir / 'meta'
    sig = json.loads((meta_dir / 'plan_signature.json').read_text())
    idx2dev_train = json.loads((meta_dir / 'devices_kept.json').read_text())
    cal_plan_t0 = json.loads((meta_dir / 'cal_plan.json').read_text())

    keys = E.resolve_keys(args.keys, sig.get('keys', []))
    print(f'[open] keys={keys}', flush=True)
    print(f'[open] gallery (T0 trained devices): {len(idx2dev_train)}', flush=True)

    kshots = sorted({int(s) for s in args.kshots.split(',') if s.strip().isdigit()})
    topk_list = sorted({int(t) for t in args.topk.split(',') if t.strip().isdigit()})
    min_k = min(kshots)
    max_topk = max(topk_list)
    if max_topk > len(idx2dev_train):
        topk_list = [t for t in topk_list if t <= len(idx2dev_train)]
    print(f'[open] kshots={kshots}, topk={topk_list}', flush=True)

    # Load encoder models
    C_train = len(idx2dev_train)
    models, norms = E.load_models_auto(args, keys, results_dir, C_train, device)

    # Build full T0 gallery: prototypes for all 3,303 devices
    pid2idx = {}
    for i, dev in enumerate(idx2dev_train):
        pid = E.extract_pid(dev)
        if pid: pid2idx[pid] = i
    print(f'[open] built pid2idx: {len(pid2idx)} unique PIDs in gallery', flush=True)

    print('[open] computing T0 prototypes for full gallery ...', flush=True)
    protos_all = {}
    for k in keys:
        rows = []
        for dev in idx2dev_train:
            paths = [Path(p) for p in cal_plan_t0.get(k, {}).get(dev, [])]
            Fd = E.feats_for_paths(models[k], paths, norms[k][0], norms[k][1],
                                   device=device, bs=args.batch_size)
            rows.append(Fd.mean(axis=0) if Fd.size else np.zeros((256,), dtype=np.float32))
        P = np.stack(rows, axis=0)
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        protos_all[k] = P
        print(f'  prototype[{k}]: {P.shape}', flush=True)

    # Load T1 returning devices
    print(f'[open] loading T+130d traces from {args.t130_dir} ...', flush=True)
    dkl_t130 = E.collect_device_key_lists_all_vis(Path(args.t130_dir), keys, min_traces=min_k)

    # Find returning devices: PID present in BOTH T0 gallery AND T1
    t130_pid2folder = {}
    for p in E.list_device_folders(Path(args.t130_dir)):
        pid = E.extract_pid(p.name)
        if pid: t130_pid2folder[pid] = p.name

    overlap = []
    for pid, t0_idx in pid2idx.items():
        if pid not in t130_pid2folder: continue
        t130_f = t130_pid2folder[pid]
        if t130_f not in dkl_t130: continue
        overlap.append((pid, t0_idx, t130_f))
    print(f'[open] returning devices (PID overlap with sufficient T1 traces): {len(overlap)}', flush=True)

    # Compute query features for all returning devices
    print('[open] computing T1 query features ...', flush=True)
    t130_feats = {}
    for (pid, t0_idx, t130_f) in overlap:
        for k in keys:
            paths = [Path(p) for p in dkl_t130[t130_f][k]]
            F = E.feats_for_paths(models[k], paths, norms[k][0], norms[k][1],
                                  device=device, bs=args.batch_size)
            if F.size:
                F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
            t130_feats[(t130_f, k)] = F

    # Optional per-k whitelist
    per_k_wl = None
    if args.per_k_whitelist:
        wl_raw = json.loads(Path(args.per_k_whitelist).read_text())
        per_k_wl = {int(k): set(v) for k, v in wl_raw.items()}
        print(f'[open] loaded per-k whitelist for k={sorted(per_k_wl.keys())[:5]}...', flush=True)

    # Map t0_folder name to overlap entry for whitelist lookup
    t0folder_to_overlap = {idx2dev_train[t0_idx]: (pid, t0_idx, t130_f)
                            for (pid, t0_idx, t130_f) in overlap}

    # For each k: gallery=full 3,303; query=first-k T1 traces (chronological)
    GALLERY_N = len(idx2dev_train)
    results_by_kshot = {}
    for ks in kshots:
        ens_correct = {kk: 0 for kk in topk_list}
        per_key_correct = {k: {kk: 0 for kk in topk_list} for k in keys}
        total = 0
        eligible_n = 0
        # Apply whitelist (per-k)
        if per_k_wl is not None and ks in per_k_wl:
            wl_set = per_k_wl[ks]
            iter_overlap = [t0folder_to_overlap[f] for f in wl_set if f in t0folder_to_overlap]
            print(f'  k={ks}: whitelist={len(wl_set)} → {len(iter_overlap)} eval queries')
        else:
            iter_overlap = overlap
        for (pid, t0_idx, t130_f) in iter_overlap:
            # device must have >= ks traces in every key
            ok = True
            for k in keys:
                F = t130_feats.get((t130_f, k))
                if F is None or F.shape[0] < ks: ok = False; break
            if not ok: continue
            eligible_n += 1
            gt = t0_idx
            perkey_scores = {}
            for k in keys:
                F = t130_feats[(t130_f, k)]
                idx = np.arange(ks)
                perkey_scores[k] = (F[idx] @ protos_all[k].T).mean(axis=0)
            total += 1
            for k, Sv in perkey_scores.items():
                order_k = np.argsort(-Sv)
                for kk in topk_list:
                    if gt in order_k[:kk]: per_key_correct[k][kk] += 1
            S_avg = np.mean(np.stack(list(perkey_scores.values()), axis=0), axis=0)
            order = np.argsort(-S_avg)
            for kk in topk_list:
                if gt in order[:kk]: ens_correct[kk] += 1

        ens_acc = {f'top{kk}': ens_correct[kk]/max(1,total) for kk in topk_list}
        per_key_acc = {k: {f'top{kk}': per_key_correct[k][kk]/max(1,total) for kk in topk_list} for k in keys}
        base_ks = {f'top{kk}': float(min(kk, GALLERY_N))/float(GALLERY_N) for kk in topk_list}
        results_by_kshot[ks] = {
            'k': ks, 'num_queries_eval': total, 'num_queries_eligible': eligible_n,
            'gallery_size': GALLERY_N,
            'base_rates_random': base_ks,
            'ensemble_topk': ens_acc, 'per_key_topk': per_key_acc,
        }
        print(f'  k={ks}: gallery={GALLERY_N}, query={total}, ensemble {ens_acc}', flush=True)

    summary = {
        'mode': 'vote_temporal_open_kshot',
        't0_dir': str(args.t0_dir), 't130_dir': str(args.t130_dir),
        'keys': keys,
        'gallery_size': GALLERY_N,
        'num_returning_total': len(overlap),
        'kshots': kshots,
        'results_by_kshot': results_by_kshot,
        'note': 'OPEN-WORLD: gallery = ALL T0 trained devices (3,303); query = returning T1 first-k chronological',
    }
    out_path = Path(args.out) if args.out else (results_dir / 'vote_temporal_open_summary.json')
    out_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'[open] saved -> {out_path}', flush=True)

if __name__ == '__main__':
    main()
