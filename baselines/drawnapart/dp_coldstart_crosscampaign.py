#!/usr/bin/env python3
"""Held-out re-identification with an encoder that never saw the query devices: paper Figure 5
(the script that produced that curve; "cold-start" in the file name refers to this setting).

eval_temporal_open.py sizes the classifier head and the gallery from one number, so an encoder
trained on D_train alone cannot be pointed at the full gallery directly. This copies the
architecture file with nclasses rewritten to the gallery size, builds the network with its head
at the checkpoint's width so the state dict loads strictly, applies the stored scaler, and
forwards everything after `--` to eval_temporal_open.main(). Identification is 1-NN in embedding
space, so the head's width does not enter the result. Distractor devices join the gallery by
placing their directories under --t0-root; --gallery-size auto counts them.

    python -m baselines.drawnapart.dp_coldstart_crosscampaign \\
        --encoder-dir <D_TRAIN_RESULTS> --gallery-size auto --shim-dir <SHIM> -- \\
        --results-dir <SHIM> --t0-root <GALLERY_ROOT> --t130-dir <RETURN_ROOT> \\
        --fp-eligible-by-k <D_OPEN_QUERIES.json> --chronological --kshots 1,2,3,4
"""
import argparse, json, shutil, sys
from pathlib import Path

from . import data as DP
from . import lib
from .released_model import ReleasedDPCNN
from .run_released import get_position_scaler, apply_fitted_scaler


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--encoder-dir", required=True,
                    help="the MP65 encoder: trained on 664, never saw MPrest")
    ap.add_argument("--gallery-size", default="auto",
                    help="devices the gallery will hold; 'auto' runs the evaluation's own "
                         "discovery over --t0-root and counts them, which is the only way to be "
                         "right when the root mixes campaigns and some devices fail the 4x7 test")
    ap.add_argument("--shim-dir", required=True, help="where to write the patched arch file")
    ap.add_argument("--scaler", choices=["per-trace", "fitted"], default="fitted")
    own = sys.argv[1:sys.argv.index("--")] if "--" in sys.argv else sys.argv[1:]
    if "-h" in own or "--help" in own:
        ap.print_help()
        print("\nEverything after a lone `--` is passed to eval_temporal_open.main() unchanged "
              "(--results-dir must name --shim-dir; see `python -m "
              "baselines.drawnapart.eval.eval_temporal_open --help`).")
        return 0
    known, rest = ap.parse_known_args()
    while rest and rest[0] == "--":
        rest = rest[1:]

    enc = Path(known.encoder_dir)
    shim = Path(known.shim_dir)
    shim.mkdir(parents=True, exist_ok=True)

    arch = json.loads((enc / "dp_arch_params.json").read_text())
    head_width = int(arch["nclasses"])

    if str(known.gallery_size) == "auto":
        def opt(name, default=None):
            return rest[rest.index(name) + 1] if name in rest else default
        root = opt("--t0-root")
        if not root:
            raise SystemExit("--gallery-size auto needs --t0-root among the forwarded arguments")
        print("[shim] counting gallery devices under %s ..." % root, flush=True)
        found = DP.discover_samples_strict_4x7(
            Path(root), min_trace_len=int(opt("--min-trace-len", 1024)),
            dedup=("--no-dedup" not in rest), debug=False, whitelist=None,
            dump_devices=False, dump_path=None)
        gallery_size = len(found)
        print("[shim] the evaluation's own discovery finds %d" % gallery_size, flush=True)
    else:
        gallery_size = int(known.gallery_size)

    print("[shim] encoder trained on %d devices; gallery will hold %d"
          % (head_width, gallery_size), flush=True)
    if head_width >= gallery_size:
        raise SystemExit("this only makes sense when the encoder saw fewer devices than the "
                         "gallery holds; got %d vs %d" % (head_width, gallery_size))
    arch["nclasses"] = gallery_size
    (shim / "dp_arch_params.json").write_text(json.dumps(arch, indent=2))
    for name in ("dp_embed_model.pt", "dp_position_scaler.pkl", "dp_devices_kept.json"):
        src = enc / name
        if src.exists() and not (shim / name).exists():
            shutil.copy2(src, shim / name)
    print("[shim] wrote %s (nclasses %d -> %d, weights copied unchanged)"
          % (shim, head_width, gallery_size), flush=True)

    def fixed_head(arch_blob, nclasses, device):
        """Build at the checkpoint's classifier width, whatever the caller asked for."""
        return ReleasedDPCNN(nclasses=head_width, embed_dim=int(arch_blob.get("embed_dim", 256))).to(device)

    lib.build_from_arch = fixed_head
    print("[patch] classifier head pinned to %d so the checkpoint loads strictly" % head_width,
          flush=True)

    from .eval import eval_temporal_open as EV
    if getattr(EV, "DP", None) is not lib:
        raise SystemExit("eval_temporal_open.DP is not the module we patched")

    if known.scaler == "fitted":
        # it looks for the encoder under --model-dir / --t0-results-dir, not --results-dir, and
        # refuses to refit -- the shim carries the D_train scaler, which is the one this encoder
        # was trained with
        apply_fitted_scaler(get_position_scaler(DP, ["--model-dir", str(shim)] + rest))

    sys.argv = [sys.argv[0]] + rest
    EV.main()


if __name__ == "__main__":
    main()
