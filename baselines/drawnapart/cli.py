#!/usr/bin/env python3
"""DRAWNAPART baseline CLI: reads a YAML recipe from configs/, turns its keys
into flags and launches the evaluator through run_released.py, which applies
the stored scaler. One recipe, eval_temporal_open (paper §5.3, Figure 4);
Figure 5 goes through dp_coldstart_crosscampaign.py directly.

    python -m baselines.drawnapart.cli eval \\
        --config baselines/drawnapart/configs/eval_temporal_open.yaml \\
        --t0-root /path/to/matched_set_T0 \\
        --t130-dir /path/to/returning_visits \\
        --results-dir ./out_dp_results

--results-dir is what dp_keras_to_torch.py wrote; the summary is written
beside the checkpoint unless --out names another file.
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
    print(
        "Missing dependency: PyYAML. Install with: "
        "pip install -r baselines/drawnapart/requirements.txt",
        file=sys.stderr,
    )
    sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Map config "recipe" string -> python module path.
RECIPES = {
    "eval_temporal_open": "baselines.drawnapart.eval.eval_temporal_open",
}

EVAL_RECIPES = set(RECIPES)


def kebab(key: str) -> str:
    return "--" + key.replace("_", "-")


def cfg_to_args(cfg: dict) -> list[str]:
    """Flatten a dict into argparse flags. Booleans -> presence flags."""
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


def _run(module: str, cfg: dict) -> None:
    # run_released imports the evaluator, swaps trace_to_img for the fitted
    # scaler stored beside the encoder, and calls its main() in-process.
    cmd = [sys.executable, "-m", "baselines.drawnapart.run_released",
           "--pipeline", module, "--", *cfg_to_args(cfg)]
    print("[cli] $", " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


def _load_recipe(config_path: Path) -> tuple[str, dict]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    recipe = cfg.pop("recipe", None)
    if recipe not in RECIPES:
        sys.exit(
            f"config 'recipe' must be one of {sorted(RECIPES)} "
            f"(got {recipe!r} from {config_path})"
        )
    return recipe, cfg


def cmd_eval(args: argparse.Namespace) -> None:
    recipe, cfg = _load_recipe(args.config)
    if recipe not in EVAL_RECIPES:
        sys.exit(f"recipe {recipe!r} is a train recipe; use `cli.py train` instead.")

    if args.results_dir is not None:
        cfg["results_dir"] = str(args.results_dir)
    if args.out is not None:
        cfg["out"] = str(args.out)
    if args.t0_root is not None:
        cfg["t0_root"] = str(args.t0_root)
    if args.t130_dir is not None:
        cfg["t130_dir"] = str(args.t130_dir)
    if args.fp_eligible_by_k is not None:
        cfg["fp_eligible_by_k"] = str(args.fp_eligible_by_k)

    _run(RECIPES[recipe], cfg)


def main() -> None:
    p = argparse.ArgumentParser(description="DRAWNAPART baseline pipeline CLI.")
    sp = p.add_subparsers(dest="cmd", required=True)

    pe = sp.add_parser("eval", help="evaluate a trained DRAWNAPART encoder")
    pe.add_argument("--config", type=Path, required=True,
                    help="YAML recipe under baselines/drawnapart/configs/")
    pe.add_argument("--results-dir", type=Path,
                    help="directory holding dp_embed_model.pt, dp_arch_params.json and "
                         "dp_position_scaler.pkl (written by dp_keras_to_torch.py); the "
                         "summary is written there unless --out is given")
    pe.add_argument("--out", type=Path,
                    help="write the summary JSON to this path instead")
    pe.add_argument("--t0-root", type=Path,
                    help="T0 dataset root (rebuilds the enrollment gallery)")
    pe.add_argument("--t130-dir", type=Path,
                    help="return-visit dataset root")
    pe.add_argument("--fp-eligible-by-k", type=Path,
                    help="LearnedFP per-k eligibility JSON")
    pe.set_defaults(func=cmd_eval)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
