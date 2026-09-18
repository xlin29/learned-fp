#!/usr/bin/env python3
"""
eval_coldstart_open.py — Open-world cold-start eval: gallery = trained T0 + N self-enrollments;
    query = each new device's last-k chronological traces.

    Paper:    §5.4 / Table 2 — cold-start tracking against the 3,896-device gallery
"""
import argparse, json, os, sys, math
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as E


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', required=True)
    p.add_argument('--unseen-dir', required=True)
    p.add_argument('--kshots', default='1,2,3,4,5,10,20')
    p.add_argument('--topk', default='1,5,10,20,50,100')
    p.add_argument('--keys', default='AUTO9')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--model-type', default='fixed')
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--min-traces', type=int, default=2)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    E.set_seed(args.seed)
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    results_dir = Path(args.results_dir)
    meta_dir = results_dir / 'meta'
    sig = json.loads((meta_dir / 'plan_signature.json').read_text())
    idx2dev_train = json.loads((meta_dir / 'devices_kept.json').read_text())
    cal_plan_t0 = json.loads((meta_dir / 'cal_plan.json').read_text())

    keys = E.resolve_keys(args.keys, sig.get('keys', []))
    print(f'[hybrid] keys={keys}', flush=True)
    print(f'[hybrid] T0 trained gallery: {len(idx2dev_train)}', flush=True)

    kshots = sorted({int(s) for s in args.kshots.split(',') if s.strip().isdigit()})
    topk_list = sorted({int(t) for t in args.topk.split(',') if t.strip().isdigit()})
    min_k = min(kshots)
    min_traces = max(args.min_traces, min_k + 1)

    C_train = len(idx2dev_train)
    models, norms = E.load_models_auto(args, keys, results_dir, C_train, device)

    # ----- Build T0 gallery prototypes for all 3,303 trained -----
    print(f'[hybrid] computing T0 prototypes ({C_train} devices) ...', flush=True)
    t0_protos = {}
    for k in keys:
        rows = []
        for dev in idx2dev_train:
            paths = [Path(p) for p in cal_plan_t0.get(k, {}).get(dev, [])]
            Fd = E.feats_for_paths(models[k], paths, norms[k][0], norms[k][1],
                                   device=device, bs=args.batch_size)
            rows.append(Fd.mean(axis=0) if Fd.size else np.zeros((256,), dtype=np.float32))
        T0 = np.stack(rows, axis=0)
        T0 = T0 / (np.linalg.norm(T0, axis=1, keepdims=True) + 1e-9)
        t0_protos[k] = T0
        print(f'  T0 proto[{k}]: {T0.shape}', flush=True)

    # ----- Load unseen devices and chrono-split -----
    base_dir = Path(args.unseen_dir)
    print(f'[hybrid] discovering unseen devices in {base_dir} ...', flush=True)
    dkl = E.collect_device_key_lists_all_vis(base_dir, keys, min_traces=min_traces)
    cal_plan, test_plan = E.build_chronological_split_all_vis(dkl, keys, max_query_pool=20)
    all_unseen = sorted(set.intersection(*[set(cal_plan[k].keys()) for k in keys]))
    print(f'[hybrid] unseen devices after intersection: {len(all_unseen)}', flush=True)

    # ----- Build unseen self-prototypes (from cal split = first 80%) -----
    print('[hybrid] computing unseen self-prototypes ...', flush=True)
    unseen_protos = {}
    test_feats = {}
    for k in keys:
        rows = []
        for d in all_unseen:
            paths = [Path(p) for p in cal_plan[k].get(d, [])]
            Fd = E.feats_for_paths(models[k], paths, norms[k][0], norms[k][1],
                                   device=device, bs=args.batch_size)
            rows.append(Fd.mean(axis=0) if Fd.size else np.zeros((256,), dtype=np.float32))
        Pu = np.stack(rows, axis=0)
        Pu = Pu / (np.linalg.norm(Pu, axis=1, keepdims=True) + 1e-9)
        unseen_protos[k] = Pu
    print(f'  unseen proto: {len(all_unseen)} devs', flush=True)

    for d in all_unseen:
        for k in keys:
            paths = [Path(p) for p in test_plan[k].get(d, [])]
            Ff = E.feats_for_paths(models[k], paths, norms[k][0], norms[k][1],
                                   device=device, bs=args.batch_size)
            if Ff.size:
                Ff = Ff / (np.linalg.norm(Ff, axis=1, keepdims=True) + 1e-9)
            test_feats[(d, k)] = Ff

    # ----- Combined gallery -----
    combined = {}
    for k in keys:
        combined[k] = np.concatenate([t0_protos[k], unseen_protos[k]], axis=0)
    GALLERY_N = combined[keys[0]].shape[0]
    print(f'[hybrid] COMBINED gallery: {GALLERY_N} = {C_train} T0 + {len(all_unseen)} unseen', flush=True)

    # eligibility per k
    dev_min_test = {d: min(len(test_plan[k].get(d, [])) for k in keys) for d in all_unseen}
    fp_eligible_by_k = {str(ks): [d for d in all_unseen if dev_min_test[d] >= ks] for ks in kshots}

    topk_list = [t for t in topk_list if t <= GALLERY_N]
    rng = np.random.default_rng(args.seed)
    results_by_kshot = {}
    dev2unseen_idx = {d: i for i, d in enumerate(all_unseen)}

    for ks in kshots:
        elig = [d for d in all_unseen if dev_min_test[d] >= ks]
        if not elig:
            results_by_kshot[ks] = {'k':ks,'num_queries_eval':0,'gallery_size':GALLERY_N,
                                    'ensemble_topk':{},'per_key_topk':{},'base_rates_random':{}}
            continue
        ens_correct = {kk:0 for kk in topk_list}
        per_key_correct = {k:{kk:0 for kk in topk_list} for k in keys}
        total = 0
        for d in elig:
            gt = C_train + dev2unseen_idx[d]  # offset by T0 size
            perkey_scores = {}
            for k in keys:
                Ff = test_feats.get((d, k))
                if Ff is None or Ff.shape[0] < ks: continue
                idx = np.arange(ks)
                perkey_scores[k] = (Ff[idx] @ combined[k].T).mean(axis=0)
            if not perkey_scores: continue
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
        per_key_acc = {k:{f'top{kk}': per_key_correct[k][kk]/max(1,total) for kk in topk_list} for k in keys}
        base = {f'top{kk}': float(min(kk, GALLERY_N))/float(GALLERY_N) for kk in topk_list}
        results_by_kshot[ks] = {'k':ks,'num_queries_eval':total,'gallery_size':GALLERY_N,
                                'num_t0_distractors':C_train,'num_unseen_in_gallery':len(all_unseen),
                                'base_rates_random':base,'ensemble_topk':ens_acc,'per_key_topk':per_key_acc}
        print(f'  k={ks}: gallery={GALLERY_N} (T0={C_train} + unseen={len(all_unseen)}), query={total}, ensemble {ens_acc}', flush=True)

    summary = {
        'mode':'vote_unseen_hybrid_open_kshot',
        'unseen_dir':str(args.unseen_dir),'keys':keys,
        'gallery_size':GALLERY_N,'num_t0_distractors':C_train,
        'num_unseen_total':len(all_unseen),
        'kshots':kshots,'results_by_kshot':results_by_kshot,
        'note':'HYBRID OPEN-WORLD COLD-START: gallery = T0 enrolled + unseen self; query = unseen device last-k chronological',
    }
    out = Path(args.out) if args.out else (results_dir / 'vote_unseen_open_summary.json')
    out.write_text(json.dumps(summary, indent=2))
    print(f'[hybrid] saved -> {out}', flush=True)


if __name__ == '__main__':
    main()
