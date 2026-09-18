#!/usr/bin/env python3
# DRAWNAPART code, redistributed with the authors' permission.
#
# build_keras() below reproduces the layer list of DRAWNAPART's get_clf_model()
# (without its softmax head, as their get_triplet_model() does), taken as
# written from their released code:
#
#     https://github.com/drawnapart/drawnapart
#     fpstalker_drawnapart.ipynb, commit bddd4f68fed265078c63edcf8f72e602f43084f9
#
# and included here with the written permission of its authors (September
# 2026), on the condition that this link and the following attribution
# accompany it:
#
#     Tomer Laor, Naif Mehanna, Vitaly Dyadyuk, Antonin Durey, Pierre
#     Laperdrix, Clémentine Maurice, Yossi Oren, Romain Rouvoy, Walter
#     Rudametkin, and Yuval Yarom. "DRAWN APART: A Device Identification
#     Technique based on Remote GPU Fingerprinting." Network and Distributed
#     System Security Symposium (NDSS), 2022.
#
# The weight mapping and the equivalence check are part of the LearnedFP
# artifact and are MIT-licensed with it.
"""Move the trained Keras weights into the PyTorch `ReleasedDPCNN` checkpoint the evaluators load
(the converter that produced the paper's checkpoints).

Keras stores convolution kernels as (H, W, in, out) and dense kernels as (in, out); PyTorch as
(out, in, H, W) and (out, in). Flattening is layout-safe because the unpadded stack reaches 1x1.
Before writing, both networks are run on the same inputs and must agree within --tol.

Writes dp_embed_model.pt, dp_arch_params.json and a copy of dp_position_scaler.pkl to --out-dir.
"""
import argparse, json, shutil
from pathlib import Path

import numpy as np

from .released_model import ReleasedDPCNN


def build_keras(nclasses):
    """DRAWNAPART's get_triplet_model(): their get_clf_model() layer list without the softmax
    head. Their code, reproduced layer for layer (see the file header)."""
    import tensorflow as tf
    DROPOUT = 0.119510
    layers = [tf.keras.layers.InputLayer((32, 32, 1))]
    for _ in range(3):
        layers += [tf.keras.layers.Conv2D(128, (4, 4), activation='relu'),
                   tf.keras.layers.Dropout(DROPOUT),
                   tf.keras.layers.AveragePooling2D()]
    layers += [tf.keras.layers.Flatten(),
               tf.keras.layers.Dense(256, activation='relu'),
               tf.keras.layers.Dense(256, activation=None),
               tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))]
    return tf.keras.Sequential(layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keras-dir", required=True, help="holds triplet.weights.h5 and keras_meta.json")
    ap.add_argument("--out-dir", required=True, help="results directory the evaluations will load")
    ap.add_argument("--scaler-from", default=None,
                    help="directory holding the dp_position_scaler.pkl the encoder was trained "
                         "with; defaults to --keras-dir, where dp_keras_baseline.py writes it")
    ap.add_argument("--tol", type=float, default=2e-4)
    args = ap.parse_args()

    import torch

    meta = json.loads((Path(args.keras_dir) / "keras_meta.json").read_text())
    C = int(meta["classes"])
    print("[meta] %d classes, epoch %s selected, val 1-NN %.4f"
          % (C, meta.get("epoch_selected"), meta.get("val_1nn", -1)), flush=True)

    kmodel = build_keras(C)
    kmodel.load_weights(str(Path(args.keras_dir) / "triplet.weights.h5"))
    kw = kmodel.get_weights()
    print("[keras] %d weight tensors: %s" % (len(kw), [w.shape for w in kw]), flush=True)
    if len(kw) != 10:
        raise SystemExit("expected 10 tensors (3 convs + 2 dense, weight and bias each), got %d"
                         % len(kw))

    net = ReleasedDPCNN(nclasses=C)
    sd = net.state_dict()
    conv_keys = [k for k in sd if k.startswith("backbone") and k.endswith("weight")
                 and sd[k].dim() == 4]
    lin_keys = [k for k in sd if k.startswith("proj") and k.endswith("weight")
                and sd[k].dim() == 2]
    conv_keys.sort(key=lambda k: int(k.split(".")[1]))
    lin_keys.sort(key=lambda k: int(k.split(".")[1]))
    print("[torch] conv %s, linear %s" % (conv_keys, lin_keys), flush=True)
    if len(conv_keys) != 3 or len(lin_keys) != 2:
        raise SystemExit("ReleasedDPCNN does not have 3 convolutions and 2 projection layers")

    new = {}
    for i, k in enumerate(conv_keys):                      # (H,W,in,out) -> (out,in,H,W)
        w, b = kw[2 * i], kw[2 * i + 1]
        new[k] = torch.from_numpy(np.transpose(w, (3, 2, 0, 1)).copy())
        new[k.replace("weight", "bias")] = torch.from_numpy(b.copy())
    for j, k in enumerate(lin_keys):                       # (in,out) -> (out,in)
        w, b = kw[6 + 2 * j], kw[7 + 2 * j]
        new[k] = torch.from_numpy(w.T.copy())
        new[k.replace("weight", "bias")] = torch.from_numpy(b.copy())

    for k, v in new.items():
        if sd[k].shape != v.shape:
            raise SystemExit("shape mismatch at %s: torch %s vs keras-derived %s"
                             % (k, tuple(sd[k].shape), tuple(v.shape)))
    sd.update(new)
    net.load_state_dict(sd)
    net.eval()

    # ---- the check: same input, same output, or nothing gets written
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 32, 32, 1)).astype(np.float32)
    zk = kmodel.predict(X, verbose=0)
    with torch.no_grad():
        zt = net.embedding(torch.from_numpy(X.transpose(0, 3, 1, 2).copy())).numpy()
    err = float(np.abs(zk - zt).max())
    cos = float((zk * zt).sum(1).mean())
    print("[check] max |keras - torch| = %.3e, mean cosine = %.6f" % (err, cos), flush=True)
    if err > args.tol:
        raise SystemExit("the two networks disagree by %.3e; the weight mapping is wrong and the "
                         "checkpoint would be silently useless" % err)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / "dp_embed_model.pt")
    json.dump({"model_class": "ReleasedDPCNN (weights from DRAWNAPART's Keras network)",
               "nclasses": C, "best_embed_dim": 256, "best_blocks": 3, "best_channels": 128,
               "best_ksize": 4, "best_dropout": 0.119510, "best_activation": "relu",
               "preprocessing": "fitted per-position StandardScaler",
               "keras_epoch_selected": meta.get("epoch_selected"),
               "keras_val_1nn": meta.get("val_1nn"),
               "keras_classifier_test_acc": meta.get("classifier_test_acc"),
               "max_abs_diff_vs_keras": err},
              open(out / "dp_arch_params.json", "w"), indent=1)
    scaler_dir = Path(args.scaler_from or args.keras_dir)
    src = scaler_dir / "dp_position_scaler.pkl"
    if not src.exists():
        raise SystemExit("no scaler at %s" % src)
    shutil.copy(src, out / "dp_position_scaler.pkl")
    print("[write] %s  (state dict, arch json, scaler from %s)"
          % (out, scaler_dir.name), flush=True)


if __name__ == "__main__":
    main()
