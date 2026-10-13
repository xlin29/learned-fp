#!/usr/bin/env python3
"""LearnedFP pipeline CLI: reads a YAML config from pipeline/configs/, turns
its keys into flags and dispatches to the train_*/eval_* script.

    python pipeline/cli.py train --config <yaml> --data <BASE_DIR> --out <RESULTS_DIR>
    python pipeline/cli.py eval  --config <yaml> --data <BASE_DIR> --models <RESULTS_DIR> \\
                                 --mode {vote,vote_temporal,vote_unseen} [--t130-dir <RETURN_VISIT_DIR>]

Configs: bayes_dp.yaml (paper §5.2, Bayesian search), fixed_3k.yaml (§5.4
closed-set / Figure 6 with --mode vote), temporal_chrono.yaml (§5.4 Table 2
temporal row, --mode vote_temporal), unseen_chrono.yaml (§5.4 Table 2
cold-start row, --mode vote_unseen), demo_lab.yaml (pipeline/demo.sh).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Missing dependency: PyYAML. Install with: pip install -r pipeline/requirements.txt",
          file=sys.stderr)
    sys.exit(2)

PIPELINE_ROOT = Path(__file__).resolve().parent

SCRIPTS = {
    "train_bayes":         PIPELINE_ROOT / "train" / "train_bayes.py",
    "train_fixed":         PIPELINE_ROOT / "train" / "train_fixed.py",
    "evaluate":            PIPELINE_ROOT / "eval"  / "evaluate.py",
    "eval_temporal_open":  PIPELINE_ROOT / "eval"  / "eval_temporal_open.py",
    "eval_coldstart_open": PIPELINE_ROOT / "eval"  / "eval_coldstart_open.py",
}

# Number of rendering keys when keys=AUTO9 (paper §4.1: 9 canvas keys).
DEFAULT_NUM_KEYS = 9

# Which YAML keys feed which subparser. The underlying train_*.py and the four
# eval scripts each take their own argparse vocabulary; the CLI must filter or
# argparse rejects "unrecognized arguments". Keep these in sync with the
# argparse blocks in:
#   train/train_{bayes,fixed}.py   ↔ _TRAIN_KEYS / _VOTE_KEYS
#   eval/evaluate.py vote          ↔ _VOTE_KEYS
#   eval/eval_temporal_open.py     ↔ _TEMPORAL_OPEN_KEYS
#   eval/eval_coldstart_open.py    ↔ _UNSEEN_OPEN_KEYS
_TRAIN_KEYS = {
    "seed", "gpu", "keys", "key_index",
    # fixed trainer
    "epochs", "batch_size", "lr", "wd", "label_smoothing", "ema",
    "warmup_epochs", "dropout", "test_frac",
    # bayes trainer
    "search_trials", "search_epochs", "val_frac_in_cal",
    # paths
    "base_dir", "results_dir",
}
_VOTE_KEYS = {
    "seed", "gpu", "keys", "batch_size", "dropout", "topk",
    "n_test_per_key", "per_key_trace_agg", "fuse_mode",
    "base_dir", "results_dir",
    # only meaningful for evaluate.py (ignored by train_*/vote)
    "model_type",
}
# Open-world temporal: gallery = ALL T0 trained, query = returning T1 first-k chrono.
# Maps to eval_temporal_open.py (paper §5.4 / Table 2, temporal row).
_TEMPORAL_OPEN_KEYS = {
    "seed", "gpu", "keys", "batch_size", "dropout", "topk", "kshots",
    "results_dir", "t0_dir", "t130_dir", "model_type", "per_k_whitelist",
}
# Open-world cold-start: gallery = T0 trained + unseen self-enrol, query = unseen last-k chrono.
# Maps to eval_coldstart_open.py (paper §5.4 / Table 2, cold-start row).
_UNSEEN_OPEN_KEYS = {
    "seed", "gpu", "keys", "batch_size", "dropout", "topk", "kshots", "min_traces",
    "results_dir", "unseen_dir", "model_type",
}


def _select(cfg: dict, allowed: set[str]) -> dict:
    return {k: v for k, v in cfg.items() if k in allowed}


def kebab(key: str) -> str:
    return "--" + key.replace("_", "-")


def cfg_to_args(cfg: dict) -> list[str]:
    """Flatten a dict into argparse-style CLI args. Booleans → presence flags."""
    out: list[str] = []
    for k, v in cfg.items():
        flag = kebab(k)
        if isinstance(v, bool):
            if v:
                out.append(flag)
        elif v is None:
            continue
        else:
            out.append(flag)
            out.append(str(v))
    return out


def _run(cmd: list[str]) -> None:
    print("[cli] $", " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.check_call(cmd)


def cmd_train(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(args.config.read_text())
    variant = cfg.pop("variant", None)
    if variant not in ("bayes", "fixed"):
        sys.exit(f"config 'variant' must be 'bayes' or 'fixed' (got {variant!r})")
    script = SCRIPTS[f"train_{variant}"]

    keys_spec = cfg.get("keys", "AUTO9")
    if isinstance(keys_spec, str) and keys_spec.upper().startswith("AUTO"):
        n = keys_spec[4:].strip() or str(DEFAULT_NUM_KEYS)
        n_keys = int(n)
    elif isinstance(keys_spec, str):
        n_keys = len([k for k in keys_spec.split(",") if k.strip()])
    else:
        sys.exit(f"config 'keys' must be a string (got {type(keys_spec).__name__})")

    base = {"base_dir": str(args.data), "results_dir": str(args.out)}

    # Train each key sequentially.
    for idx in range(n_keys):
        train_cfg = _select({**cfg, **base, "key_index": idx}, _TRAIN_KEYS)
        _run([sys.executable, str(script), "train", *cfg_to_args(train_cfg)])

    # Auto-vote on the trained models, mirroring vote_*.sh on the production cluster.
    vote_cfg = _select({**cfg, **base}, _VOTE_KEYS)
    _run([sys.executable, str(script), "vote", *cfg_to_args(vote_cfg)])


def cmd_eval(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(args.config.read_text())
    variant = cfg.pop("variant", None)
    if variant not in ("bayes", "fixed"):
        sys.exit(f"config 'variant' must be 'bayes' or 'fixed' (got {variant!r})")

    base = {"results_dir": str(args.models), "model_type": variant}
    if args.mode == "vote_temporal":
        # Open-world temporal: gallery = full 3,303 T0 trained, query = returning T1.
        # Paper §5.4 / Table 2, temporal row.
        if not args.t130_dir:
            sys.exit("--t130-dir is required for mode=vote_temporal")
        base["t0_dir"] = str(args.data)
        base["t130_dir"] = str(args.t130_dir)
        cfg = _select({**cfg, **base}, _TEMPORAL_OPEN_KEYS)
        _run([sys.executable, str(SCRIPTS["eval_temporal_open"]), *cfg_to_args(cfg)])
        return
    if args.mode == "vote_unseen":
        # Open-world cold-start: gallery = T0 trained + unseen self-enrol; query = unseen last-k.
        # Paper §5.4 / Table 2, cold-start row.
        base["unseen_dir"] = str(args.data)
        cfg = _select({**cfg, **base}, _UNSEEN_OPEN_KEYS)
        _run([sys.executable, str(SCRIPTS["eval_coldstart_open"]), *cfg_to_args(cfg)])
        return
    # Closed-set vote within DB_T0 (paper §5.4 per-key setting / Figure 6).
    base["base_dir"] = str(args.data)
    cfg = _select({**cfg, **base}, _VOTE_KEYS)
    _run([sys.executable, str(SCRIPTS["evaluate"]), args.mode, *cfg_to_args(cfg)])


def main() -> None:
    p = argparse.ArgumentParser(description="LearnedFP pipeline CLI.")
    sp = p.add_subparsers(dest="cmd", required=True)

    pt = sp.add_parser("train", help="train per-key encoders + final vote")
    pt.add_argument("--config", type=Path, required=True,
                    help="YAML config under pipeline/configs/ (bayes_dp.yaml, fixed_3k.yaml, demo_lab.yaml)")
    pt.add_argument("--data", type=Path, required=True,
                    help="dataset root (parent of dev_*/visit*/ dirs)")
    pt.add_argument("--out", type=Path, required=True,
                    help="results dir — encoders + meta + per-key normalizers land here")
    pt.set_defaults(func=cmd_train)

    pe = sp.add_parser("eval", help="evaluate trained encoders (Top-k tables)")
    pe.add_argument("--config", type=Path, required=True)
    pe.add_argument("--data", type=Path, required=True,
                    help="dataset root: T0 for vote, T0 for vote_temporal, "
                         "unseen-cohort root for vote_unseen")
    pe.add_argument("--models", type=Path, required=True,
                    help="results dir produced by `train`")
    pe.add_argument("--mode",
                    choices=["vote", "vote_temporal", "vote_unseen"],
                    default="vote",
                    help="vote = closed-set within DB_T0 (paper Figure 6 setting); "
                         "vote_temporal = open-world return-visit (Table 2, temporal); "
                         "vote_unseen = open-world cold-start (Table 2, cold-start)")
    pe.add_argument("--t130-dir", type=Path,
                    help="return-visit dataset root (only used when --mode=vote_temporal)")
    pe.set_defaults(func=cmd_eval)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
