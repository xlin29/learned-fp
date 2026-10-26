"""Launch a DRAWNAPART evaluation with the preprocessing its encoder was trained with.

data.trace_to_img is a per-trace z-score (the DRAWNAPART paper's wording); the
encoder trained by dp_keras_baseline.py uses a per-position StandardScaler
fitted on its training split (their released notebook's behaviour). This loads
the scaler stored beside the encoder, replaces trace_to_img in every module of
this package with a version that applies it, and calls the evaluator's main().
fit_position_scaler, scaled_trace_to_img and the patch loop are reproduced from
the launcher that produced the paper's DRAWNAPART numbers.

    python -m baselines.drawnapart.run_released \\
        --pipeline baselines.drawnapart.eval.eval_temporal_open -- <evaluator args>
"""
import argparse
import importlib
import sys

from . import data as _data

SCALER_NAME = "dp_position_scaler.pkl"


def get_position_scaler(mod, args_list):
    """Load the scaler stored beside the encoder (--model-dir for the Figure 5
    shim, --results-dir otherwise); stop if it is missing rather than refit."""
    import pickle
    from pathlib import Path

    def opt(name, default=None):
        return args_list[args_list.index(name) + 1] if name in args_list else default

    # directories that hold an encoder, most specific first
    enc = [opt(n) for n in ("--model-dir", "--t0-results-dir", "--t0-results", "--results-dir")]
    enc = [e for e in enc if e]
    for e in enc:
        path = Path(e) / SCALER_NAME
        if path.exists():
            print(f"[scaler] loaded {path}", flush=True)
            return pickle.load(open(path, "rb"))
    raise SystemExit(
        f"no {SCALER_NAME} beside the encoder ({', '.join(enc) or 'no encoder directory given'}). "
        "This run evaluates a trained encoder, so the scaler must come from its "
        "training (dp_keras_to_torch.py copies it next to dp_embed_model.pt); "
        "refitting here would use the wrong traces.")


def fit_position_scaler(mod, args_list):
    """Fit a StandardScaler over the 1024 trace positions on the training split,
    walking devices in the order the pipeline does and using its split_stratified.
    DRAWNAPART's released notebook uses such a scaler (its pickle is not published);
    --scaler per-trace keeps the paper's per-trace normalisation instead."""
    import numpy as np
    from pathlib import Path
    from sklearn.preprocessing import StandardScaler

    def opt(name, default, cast=str):
        if name in args_list:
            return cast(args_list[args_list.index(name) + 1])
        return default

    base = Path(opt("--base-dir", None) or opt("--t0-root", None)
                or opt("--mprest-dir", None) or opt("--unseen-dir", ""))
    seed = opt("--seed", 42, int)
    mem_frac = opt("--mem-frac", 0.80, float)
    tr_frac = opt("--train-frac-within-mem", 0.80, float)
    wl_path = opt("--device-whitelist", None)
    wl = mod.load_whitelist(Path(wl_path)) if wl_path else None

    dev_samples = mod.discover_samples_strict_4x7(base, whitelist=wl)
    devs = sorted(dev_samples)
    raw, lbl = [], []
    for i, d in enumerate(devs):                      # same order as main()
        for s in dev_samples[d]:
            for t in s.traces:
                raw.append(np.asarray(t, dtype=np.float32).reshape(-1)[:1024])
                lbl.append(i)
    X = np.stack(raw, 0)
    y = np.asarray(lbl, np.int64)
    assert X.shape[0] == len(devs) * 4 * 7, (X.shape, len(devs))

    tr, _, _ = mod.split_stratified(y, seed, mem_frac, tr_frac)
    sc = StandardScaler().fit(X[tr])
    print(f"[scaler] fitted on {tr.size}/{X.shape[0]} training traces "
          f"({len(devs)} devices)", flush=True)
    return sc


def apply_fitted_scaler(sc):
    """Replace trace_to_img in every loaded module of this package with one that
    applies the fitted scaler `sc`."""
    import numpy as np

    def scaled_trace_to_img(x):
        v = np.asarray(x, dtype=np.float32).reshape(-1)[:1024]
        v = sc.transform(v.reshape(1, -1))[0]
        return v.reshape(1, 32, 32).astype(np.float32)

    # Patch every loaded module that defines it (data, lib's re-export, or an
    # evaluator's own import), so no path keeps the per-trace z-score.
    patched = []
    for name, m in list(sys.modules.items()):
        if m is not None and getattr(m, "__file__", "") and \
                "/drawnapart/" in str(m.__file__) and hasattr(m, "trace_to_img"):
            m.trace_to_img = scaled_trace_to_img
            patched.append(name)
    print(f"[patch] trace_to_img -> fitted StandardScaler in "
          f"{len(patched)} module(s): {', '.join(sorted(patched))}", flush=True)
    if not patched:
        raise SystemExit("no module exposed trace_to_img; refusing to run "
                         "with mixed preprocessing")


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pipeline", required=True,
                    help="evaluator module, e.g. baselines.drawnapart.eval.eval_temporal_open")
    ap.add_argument("--scaler", choices=["per-trace", "fitted"], default="fitted",
                    help="per-trace follows the paper ('we normalized each "
                         "trace'); fitted follows the released notebook, which "
                         "standardises with a scaler fitted on training data")
    known, rest = ap.parse_known_args()

    mod = importlib.import_module(known.pipeline)

    # argparse leaves the `--` separator in the remainder, and a leading `--`
    # makes the pipeline's parser read every following token as a positional.
    while rest and rest[0] == "--":
        rest = rest[1:]

    if known.scaler == "fitted":
        apply_fitted_scaler(get_position_scaler(_data, rest))
    else:
        print("[patch] trace_to_img unchanged: per-trace z-score (the paper's "
              "wording)", flush=True)
    sys.argv = [known.pipeline + ".py"] + rest
    print(f"[patch] forwarding {len(rest)} args to {known.pipeline}.main()",
          flush=True)
    mod.main()


if __name__ == "__main__":
    main()
