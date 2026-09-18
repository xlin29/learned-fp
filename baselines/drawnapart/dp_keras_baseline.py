#!/usr/bin/env python3
# DRAWNAPART code, redistributed with the authors' permission.
#
# The model definition in this file (the layer list of get_clf_model and the
# construction of get_triplet_model, marked inline) is DRAWNAPART's, taken as
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
# The rest of this file (data handling, scaler, training driver, best-epoch
# selection, output) is part of the LearnedFP artifact and is MIT-licensed
# with it.
"""Train DRAWNAPART's Keras network on a device corpus (the script that produced the paper's
DRAWNAPART numbers).

Network and recipe are theirs, from `fpstalker_drawnapart.ipynb`: classifier pre-training, then
the softmax layer dropped and semi-hard triplet fine-tuning, keeping the epoch with the best
validation 1-NN accuracy. `tensorflow_addons` supplies the triplet loss where it is installable;
otherwise a local semi-hard implementation is used and keras_meta.json records which one ran.
Traces are standardised with a per-position StandardScaler fitted on the training split, as in
their released notebook.

Writes triplet.weights.h5, keras_meta.json and dp_position_scaler.pkl to --save-weights;
dp_keras_to_torch.py turns them into the checkpoint the evaluators load.
"""
import argparse, json, os, pickle, time
from pathlib import Path

import numpy as np

from . import data as DP
from .run_released import fit_position_scaler, SCALER_NAME


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="device root holding dev_*/drawnapart/*.ndjson")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mem-frac", type=float, default=0.80)
    ap.add_argument("--train-frac-within-mem", type=float, default=0.80)
    ap.add_argument("--cls-epochs", type=int, default=30)
    ap.add_argument("--cls-batch", type=int, default=32)
    ap.add_argument("--trip-epochs", type=int, default=30)
    ap.add_argument("--trip-batch", type=int, default=1024)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--train-devices-under", default=None,
                    help="train only on devices whose folder also appears under this root "
                         "(paper Figure 5 needs the 664-device D_train encoder, never exposed to D_open)")
    ap.add_argument("--save-weights", required=True,
                    help="directory for the trained Keras weights, so the same network can be "
                         "converted to the PyTorch checkpoint every k-dependent evaluation loads")
    args = ap.parse_args()

    t = time.time()
    dev_samples = DP.discover_samples_strict_4x7(Path(args.data))
    devs = sorted(dev_samples)
    if args.train_devices_under:
        keep = {p.name for p in Path(args.train_devices_under).iterdir() if p.name.startswith("dev_")}
        before = len(devs)
        devs = [d for d in devs if d in keep]
        if not devs:
            raise SystemExit("no training device survives the restriction to %s"
                             % args.train_devices_under)
        dev_samples = {d: dev_samples[d] for d in devs}
        print("[data] training restricted to %s: %d of %d devices"
              % (Path(args.train_devices_under).name, len(devs), before), flush=True)
    print("[data] %d T0 devices, %.0fs" % (len(devs), time.time() - t), flush=True)

    # Scaler fitted on the training rows only, and only on the training devices when
    # --train-devices-under restricts them.
    scaler_args = ["--base-dir", args.data, "--seed", str(args.seed),
                   "--mem-frac", str(args.mem_frac),
                   "--train-frac-within-mem", str(args.train_frac_within_mem)]
    if args.train_devices_under:
        scaler_args = ["--base-dir", args.train_devices_under] + scaler_args[2:]
        print("[scaler] fitted on %s only, not the full T0 root"
              % Path(args.train_devices_under).name, flush=True)
    scaler = fit_position_scaler(DP, scaler_args)

    raw, lbl = [], []
    for i, d in enumerate(devs):                       # same order as the pipeline's main()
        for s in dev_samples[d]:
            for tr_ in s.traces:
                raw.append(np.asarray(tr_, np.float32).reshape(-1)[:1024])
                lbl.append(i)
    X = scaler.transform(np.stack(raw, 0)).reshape(-1, 32, 32, 1).astype(np.float32)
    y = np.asarray(lbl, np.int64)
    C = len(devs)
    print("[data] %d traces, %d classes" % (X.shape[0], C), flush=True)

    tr, va, te = DP.split_stratified(y, args.seed, args.mem_frac, args.train_frac_within_mem)
    print("[split] train=%d val=%d test=%d" % (tr.size, va.size, te.size), flush=True)

    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)
    print("[env] tensorflow %s, GPUs=%s"
          % (tf.__version__, len(tf.config.list_physical_devices("GPU"))), flush=True)

    # ---- DRAWNAPART's code: their get_clf_model() from fpstalker_drawnapart.ipynb, layer for
    # ---- layer, with the class count taken from the data instead of the notebook's 714.
    DROPOUT = 0.119510
    layers = [tf.keras.layers.Input((32, 32, 1))]
    for _ in range(3):
        layers += [tf.keras.layers.Conv2D(128, (4, 4), activation='relu'),
                   tf.keras.layers.Dropout(DROPOUT),
                   tf.keras.layers.AveragePooling2D()]
    layers += [tf.keras.layers.Flatten(),
               tf.keras.layers.Dense(256, activation='relu'),
               tf.keras.layers.Dense(256, activation=None),
               tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)),
               tf.keras.layers.Dense(C, activation='softmax')]
    clf = tf.keras.Sequential(layers)
    clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    # ---- end of DRAWNAPART's get_clf_model()

    clf.fit(X[tr], tf.keras.utils.to_categorical(y[tr], C),
            epochs=args.cls_epochs, batch_size=args.cls_batch, verbose=2)
    cls_acc = float((clf.predict(X[te], verbose=0).argmax(1) == y[te]).mean())
    print("[keras] classifier test acc = %.4f (base %.4f)" % (cls_acc, 1.0 / C), flush=True)

    # ---- DRAWNAPART's code: their get_triplet_model(), the classifier without its softmax
    trip = tf.keras.models.Sequential([tf.keras.layers.InputLayer((32, 32, 1))])
    for layer in clf.layers[:-1]:
        trip.add(layer)
    # ---- end of DRAWNAPART's get_triplet_model(); the loss below is theirs too
    # ---- (tfa.losses.TripletSemiHardLoss) where tensorflow_addons is installed

    loss_impl = "tfa.losses.TripletSemiHardLoss"
    try:
        import tensorflow_addons as tfa
        trip_loss = tfa.losses.TripletSemiHardLoss()
    except Exception as e:
        print("[keras] tensorflow_addons unavailable (%s); local semi-hard loss" % e, flush=True)
        loss_impl = "local semi-hard (tensorflow_addons unavailable)"
        margin = args.margin

        def trip_loss(y_true, y_pred):
            y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
            d = tf.reduce_sum(tf.square(y_pred), 1, keepdims=True)
            D = tf.maximum(d + tf.transpose(d) - 2.0 * tf.matmul(y_pred, y_pred, transpose_b=True), 0.0)
            same = tf.equal(tf.expand_dims(y_true, 0), tf.expand_dims(y_true, 1))
            eye = tf.eye(tf.shape(D)[0], dtype=tf.bool)
            pos = tf.logical_and(same, tf.logical_not(eye))
            neg = tf.logical_not(same)
            big = tf.reduce_max(D) + 1.0
            dp_ = tf.reduce_max(tf.where(pos, D, -big), axis=1)
            sh = tf.logical_and(neg, tf.logical_and(D > tf.expand_dims(dp_, 1),
                                                    D < tf.expand_dims(dp_ + margin, 1)))
            dn = tf.where(tf.reduce_any(sh, axis=1),
                          tf.reduce_min(tf.where(sh, D, big), axis=1),
                          tf.reduce_min(tf.where(neg, D, big), axis=1))
            valid = tf.logical_and(tf.reduce_any(pos, axis=1), tf.reduce_any(neg, axis=1))
            l = tf.nn.relu(dp_ - dn + margin)
            return tf.reduce_sum(tf.where(valid, l, 0.0)) / tf.maximum(
                tf.reduce_sum(tf.cast(valid, tf.float32)), 1.0)

    trip.compile(optimizer='adam', loss=trip_loss)

    # Their procedure: "We took the weights of the epoch that yielded a model with the best
    # accuracy using a 1-Nearest Neighbor classifier." One epoch at a time; 1-NN with the training
    # split as gallery and the validation split as query.
    def knn_top1(Zg, yg, Zq, yq):
        Zg = Zg / np.maximum(np.linalg.norm(Zg, axis=1, keepdims=True), 1e-9)
        Zq = Zq / np.maximum(np.linalg.norm(Zq, axis=1, keepdims=True), 1e-9)
        hit = 0
        for i in range(0, Zq.shape[0], 512):
            nn = (Zq[i:i + 512] @ Zg.T).argmax(1)
            hit += int((yg[nn] == yq[i:i + 512]).sum())
        return hit / float(Zq.shape[0])

    best = {"acc": -1.0, "epoch": -1, "w": None}
    for ep in range(1, args.trip_epochs + 1):
        trip.fit(X[tr], y[tr].astype(np.float32), epochs=1,
                 batch_size=args.trip_batch, verbose=0)
        acc = knn_top1(trip.predict(X[tr], verbose=0), y[tr],
                       trip.predict(X[va], verbose=0), y[va])
        if acc > best["acc"]:
            best.update(acc=acc, epoch=ep, w=[w.copy() for w in trip.get_weights()])
        print("[keras] triplet epoch %2d/%d  val 1-NN top-1 = %.4f%s"
              % (ep, args.trip_epochs, acc, "  <- best" if best["epoch"] == ep else ""), flush=True)
    if best["w"] is None:
        raise SystemExit("no epoch was scored; the selection loop did not run")
    trip.set_weights(best["w"])
    print("[keras] restored epoch %d (val 1-NN top-1 %.4f), as their procedure selects"
          % (best["epoch"], best["acc"]), flush=True)

    Path(args.save_weights).mkdir(parents=True, exist_ok=True)
    trip.save_weights(os.path.join(args.save_weights, "triplet.weights.h5"))
    json.dump({"classes": C, "devices": devs, "epoch_selected": best["epoch"],
               "val_1nn": best["acc"], "classifier_test_acc": cls_acc,
               "triplet_loss_impl": loss_impl,
               "preprocessing": "fitted per-position StandardScaler"},
              open(os.path.join(args.save_weights, "keras_meta.json"), "w"), indent=1)
    pickle.dump(scaler, open(os.path.join(args.save_weights, SCALER_NAME), "wb"))
    print("[keras] weights, meta and scaler -> %s" % args.save_weights, flush=True)

    def embed(Xb):
        return trip.predict(Xb, verbose=0)

    probe = embed(X[:8])
    if probe.shape[1] != 256:
        raise SystemExit("embedding width is %d, expected 256" % probe.shape[1])
    print("[keras] embedding width %d" % probe.shape[1], flush=True)


if __name__ == "__main__":
    main()
