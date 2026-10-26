#!/usr/bin/env python3
"""
eval_temporal_dp_match.py — Open-world temporal eval at the 1,022-device matched scale used
    for the head-to-head LearnedFP vs DRAWNAPART comparison.

    Paper:    §5.3 / Figure 4 — DRAWNAPART-matched temporal re-identification (1,022-device gallery).
              Figure 5 (held-out) is this evaluation with encoders trained on D_train only, D_open
              returning devices as queries (--per-k-whitelist), and 175 DB_T1 devices placed under
              --t0-dir as distractors (1,197 gallery).
"""

import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', required=True)
    p.add_argument('--t0-dir', required=True)
    p.add_argument('--t130-dir', required=True)
    p.add_argument('--kshots', default='1,5,10,20,50,100')
    p.add_argument('--topk', default='1,5,10,20,50,100')
    p.add_argument('--keys', default='AUTO9')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--model-type', default='bayes')
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--per-k-whitelist', default=None)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--out', default=None)
    p.add_argument('--eligible-out', default=None,
                   help='where to dump fp eligibility json (run1 only)')
    args = p.parse_args()

    F.set_seed(args.seed)
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    results_dir = Path(args.results_dir)
    meta_dir = results_dir / 'meta'
    sig = json.loads((meta_dir / 'plan_signature.json').read_text())
    idx2dev_train = json.loads((meta_dir / 'devices_kept.json').read_text())
    cal_plan_t0 = json.loads((meta_dir / 'cal_plan.json').read_text())

    keys = F.resolve_keys(args.keys, sig.get('keys', []))
    print(f'[open] keys={keys}', flush=True)
    print(f'[open] T0 trained gallery: {len(idx2dev_train)}', flush=True)

    kshots = sorted({int(s) for s in args.kshots.split(',') if s.strip().isdigit()})
    topk_list = sorted({int(t) for t in args.topk.split(',') if t.strip().isdigit()})
    topk_list = [t for t in topk_list if t <= len(idx2dev_train)]
    min_k = min(kshots)
    print(f'[open] kshots={kshots}, topk={topk_list}', flush=True)

    # per-k whitelist
    per_k_wl = None
    if args.per_k_whitelist:
        wl_path = Path(args.per_k_whitelist)
        raw = json.loads(wl_path.read_text())
        per_k_wl = {int(k): set(v) for k, v in raw.items()}
        print(f'[open] loaded per-k whitelist for k={sorted(per_k_wl.keys())[:5]}...', flush=True)

    C_train = len(idx2dev_train)
    models, norms = F.load_models_auto(args, keys, results_dir, C_train, device)

    # Build full gallery: prototypes for all 1,022 trained devs
    pid2idx = {}
    for i, dev in enumerate(idx2dev_train):
        pid = F.extract_pid(dev)
        if pid: pid2idx[pid] = i
    print(f'[open] pid2idx: {len(pid2idx)} unique PIDs', flush=True)

    print('[open] computing T0 prototypes for full gallery ...', flush=True)
    protos_all = {}
    for k in keys:
        rows = []
        for dev in idx2dev_train:
            paths = [Path(p) for p in cal_plan_t0.get(k, {}).get(dev, [])]
            Fd = F.feats_for_paths(models[k], paths, norms[k][0], norms[k][1],
                                   device=device, bs=args.batch_size)
            rows.append(Fd.mean(axis=0) if Fd.size else np.zeros((256,), dtype=np.float32))
        P = np.stack(rows, axis=0)
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        protos_all[k] = P
        print(f'  proto[{k}]: {P.shape}', flush=True)

    # Load T1 returning
    print(f'[open] loading T1 from {args.t130_dir} ...', flush=True)
    dkl_t130 = F.collect_device_key_lists_all_vis(Path(args.t130_dir), keys, min_traces=min_k)
    t130_pid2folder = {}
    for pp in F.list_device_folders(Path(args.t130_dir)):
        pid = F.extract_pid(pp.name)
        if pid: t130_pid2folder[pid] = pp.name

    overlap = []
    for pid, t0_idx in pid2idx.items():
        if pid not in t130_pid2folder: continue
        t130_f = t130_pid2folder[pid]
        if t130_f not in dkl_t130: continue
        # find t0 folder name (for whitelist matching)
        t0_f = idx2dev_train[t0_idx]
        overlap.append((pid, t0_idx, t0_f, t130_f))
    print(f'[open] returning devices (PID overlap with T1 traces): {len(overlap)}', flush=True)

    # Compute T1 query features for all returning devices
    print('[open] computing T1 query features ...', flush=True)
    t130_feats = {}
    for (pid, t0_idx, t0_f, t130_f) in overlap:
        for k in keys:
            paths = [Path(p) for p in dkl_t130[t130_f][k]]
            Ff = F.feats_for_paths(models[k], paths, norms[k][0], norms[k][1],
                                   device=device, bs=args.batch_size)
            if Ff.size:
                Ff = Ff / (np.linalg.norm(Ff, axis=1, keepdims=True) + 1e-9)
            t130_feats[(t130_f, k)] = Ff

    # Eligibility per k (FP trace-side: each key has >= ks T1 traces)
    fp_eligible_by_k = {}
    for ks in kshots:
        elig_t0_folders = []
        for (pid, t0_idx, t0_f, t130_f) in overlap:
            ok = True
            for k in keys:
                Ff = t130_feats.get((t130_f, k))
                if Ff is None or Ff.shape[0] < ks: ok = False; break
            if ok: elig_t0_folders.append(t0_f)
        fp_eligible_by_k[str(ks)] = elig_t0_folders

    elig_out = Path(args.eligible_out) if args.eligible_out else (results_dir / 'fp_temporal_open_eligible_by_k.json')
    elig_out.write_text(json.dumps(fp_eligible_by_k, indent=2))
    print(f'[open] dumped fp eligible -> {elig_out}', flush=True)
    for ks in kshots:
        print(f'  k={ks}: {len(fp_eligible_by_k[str(ks)])} FP-eligible devices')

    GALLERY_N = len(idx2dev_train)
    results_by_kshot = {}
    for ks in kshots:
        # FP trace-eligible
        elig = [(pid,t0_idx,t0_f,t130_f) for (pid,t0_idx,t0_f,t130_f) in overlap
                if all((t130_feats.get((t130_f,k)) is not None and t130_feats[(t130_f,k)].shape[0] >= ks) for k in keys)]
        # apply per-k whitelist (DP intersection)
        if per_k_wl is not None and ks in per_k_wl:
            wl = per_k_wl[ks]
            before = len(elig)
            elig = [e for e in elig if e[2] in wl]
            print(f'  k={ks}: trace-elig={before} → after DP wl={len(elig)}')

        if not elig:
            results_by_kshot[ks] = {'k':ks,'num_queries_eval':0,'gallery_size':GALLERY_N,
                                    'ensemble_topk':{},'per_key_topk':{},'base_rates_random':{}}
            continue

        ens_correct = {kk:0 for kk in topk_list}
        per_key_correct = {k:{kk:0 for kk in topk_list} for k in keys}
        total = 0
        for (pid, t0_idx, t0_f, t130_f) in elig:
            gt = t0_idx
            perkey_scores = {}
            for k in keys:
                Ff = t130_feats[(t130_f, k)]
                idx = np.arange(ks)
                perkey_scores[k] = (Ff[idx] @ protos_all[k].T).mean(axis=0)
            total += 1
            for k, Sv in perkey_scores.items():
                order_k = np.argsort(-Sv)
                for kk in topk_list:
                    if gt in order_k[:kk]: per_key_correct[k][kk] += 1
            S_avg = np.mean(np.stack(list(perkey_scores.values()),axis=0),axis=0)
            order = np.argsort(-S_avg)
            for kk in topk_list:
                if gt in order[:kk]: ens_correct[kk] += 1
        ens_acc = {f'top{kk}': ens_correct[kk]/max(1,total) for kk in topk_list}
        per_key_acc = {k:{f'top{kk}': per_key_correct[k][kk]/max(1,total) for kk in topk_list} for k in keys}
        base = {f'top{kk}': float(min(kk, GALLERY_N))/float(GALLERY_N) for kk in topk_list}
        results_by_kshot[ks] = {'k':ks,'num_queries_eval':total,'gallery_size':GALLERY_N,
                                'base_rates_random':base,'ensemble_topk':ens_acc,'per_key_topk':per_key_acc}
        print(f'  k={ks}: gallery={GALLERY_N}, query={total}, {ens_acc}')

    summary = {
        'mode':'vote_temporal_open_kshot_combined',
        't0_dir':str(args.t0_dir),'t130_dir':str(args.t130_dir),
        'keys':keys,'gallery_size':GALLERY_N,'num_returning_total':len(overlap),
        'per_k_whitelist':str(args.per_k_whitelist) if args.per_k_whitelist else None,
        'kshots':kshots,'results_by_kshot':results_by_kshot,
        'note':'OPEN-WORLD FP @ DP-set scale: gallery=1022; query=returning T1 first-k chronological',
    }
    out = Path(args.out) if args.out else (results_dir / 'vote_temporal_open_summary.json')
    out.write_text(json.dumps(summary, indent=2))
    print(f'[open] saved -> {out}')


if __name__ == '__main__':
    main()
