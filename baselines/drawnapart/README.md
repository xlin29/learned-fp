# `baselines/drawnapart/` — DRAWNAPART (NDSS'22) baseline

The code used to run and evaluate DRAWNAPART (Laor et al., NDSS'22), the
head-to-head ML comparison baseline; LearnedFP itself lives under `pipeline/`.

This directory holds a training driver that runs DRAWNAPART's recipe, a
converter that moves the trained weights into the torch checkpoint the
evaluations load, the k-shot temporal evaluation behind the paper's
DRAWNAPART curves, and the cohort intersection that scores both methods on
identical device sets. It does not hold DRAWNAPART traces (the crowdsourced
corpus is withheld) or their collector; reproducing the paper's DRAWNAPART
numbers needs your own traces in the layout under "Data".

## Upstream and attribution

- Paper: Tomer Laor, Naif Mehanna, Vitaly Dyadyuk, Antonin Durey, Pierre
  Laperdrix, Clémentine Maurice, Yossi Oren, Romain Rouvoy, Walter
  Rudametkin, and Yuval Yarom. "DRAWN APART: A Device Identification
  Technique based on Remote GPU Fingerprinting." Network and Distributed
  System Security Symposium (NDSS), 2022.
- Repository: <https://github.com/drawnapart/drawnapart>, commit
  `bddd4f68fed265078c63edcf8f72e602f43084f9` (the only version published)
- Architecture and training recipe: `fpstalker_drawnapart.ipynb`
- Collector: `drawn_apart_extension/`; the paper's traces follow its
  offscreen path at the parameters of their `gen10_offscreen_sinh_gpu_timer`
  dataset

The model definition in `dp_keras_baseline.py` (their `get_clf_model` and
`get_triplet_model`, marked inline) and the matching layer list in
`dp_keras_to_torch.py` are copied from that notebook as written, with the
class count read from the data, and are redistributed with the authors'
written permission; each file's header carries the attribution they asked
for, and `NOTICE` at the repo root records the terms.

## Environment

Training is Keras, evaluation is torch, and the converter needs both in one
process. Install the two requirements files in this order:

```bash
pip install -r baselines/drawnapart/requirements-tf.txt
pip install -r baselines/drawnapart/requirements.txt
```

The second command prints an `ERROR: pip's dependency resolver ...` line
about TensorFlow's `typing-extensions` bound and exits 0; that is expected
(`requirements.txt` explains it). On Linux without a GPU add
`--extra-index-url https://download.pytorch.org/whl/cpu` to the second
command, otherwise PyPI's CUDA build of torch is pulled. On `aarch64`,
where `tensorflow_addons` has no wheel, training falls back to a local
semi-hard triplet loss and records that in `keras_meta.json`.

## Train → convert → evaluate

The two training-side scripts are the ones that produced the paper's
DRAWNAPART numbers.

```bash
# 1. their recipe: classifier pre-training, then semi-hard triplet fine-tuning,
#    keeping the epoch with the best validation 1-NN accuracy
python -m baselines.drawnapart.dp_keras_baseline --data <DEVICE_ROOT> --save-weights out_dp_keras

# 2. Keras weights -> torch checkpoint; refuses to write unless both networks
#    agree on the same inputs
python -m baselines.drawnapart.dp_keras_to_torch --keras-dir out_dp_keras --out-dir out_dp_results

# 3. temporal re-identification (paper Figure 4)
python -m baselines.drawnapart.cli eval \
    --config baselines/drawnapart/configs/eval_temporal_open.yaml \
    --t0-root /path/to/matched_set_T0 \
    --t130-dir /path/to/returning_visits \
    --results-dir out_dp_results \
    [--fp-eligible-by-k <fp_temporal_open_eligible_by_k.json>]
```

Both methods are scored on the same devices at each k. Each evaluation
writes the per-k set of devices it can score (this one
`dp_temporal_open_eligible_by_k.json`, the LearnedFP side
`fp_temporal_open_eligible_by_k.json` from
`pipeline/eval/eval_temporal_dp_match.py`) and accepts the other's file
(`--fp-eligible-by-k` here, `--per-k-whitelist` there), so each side can be
restricted to the intersection.

Traces are standardised with a per-position `StandardScaler` fitted on the
training split, as their released notebook does; step 1 stores it next to
the weights, step 2 copies it into the checkpoint directory, and step 3
applies it to every trace through `run_released.py` and stops if it is
missing. The summary lands beside the checkpoint as
`dp_temporal_open_summary.json`.

**Figure 5** (held-out re-identification) scores an encoder trained on
D_train only against the full gallery; `dp_coldstart_crosscampaign.py`
resizes the checkpoint's classifier head to the gallery and forwards
everything after `--` to the same evaluator:

```bash
python -m baselines.drawnapart.dp_keras_baseline --data <T0_ROOT> --train-devices-under <D_TRAIN_ROOT> --save-weights out_dp_keras_dtrain
python -m baselines.drawnapart.dp_keras_to_torch --keras-dir out_dp_keras_dtrain --out-dir out_dp_dtrain
python -m baselines.drawnapart.dp_coldstart_crosscampaign \
    --encoder-dir out_dp_dtrain --gallery-size auto --shim-dir out_dp_fig5_shim -- \
    --results-dir out_dp_fig5_shim --t0-root <GALLERY_ROOT> --t130-dir <RETURN_ROOT> \
    --fp-eligible-by-k <D_OPEN_QUERIES.json> --chronological --kshots 1,2,3,4 --out out_dp_fig5.json
```

Distractor devices enter the gallery by placing their directories under
`<GALLERY_ROOT>` alongside the enrolled ones; here the eligibility file is
the one from the LearnedFP Figure 5 run, so both methods are again scored on
the same devices, D_open's returning ones.

## Data

One directory per device, traces under `dev_<id>/drawnapart/`, one
`.ndjson` (or `.ndjson.gz`) file per sample, one JSON object per line, of
which the loader reads only `traces`:

```json
{"traces": [{"times_ms": [/* 1024 floats */]}, {"times_ms": [/* ... */]}]}
```

A device is kept when it has exactly 4 sample files and each yields at
least 7 traces of length 1024 (the 4x7 protocol the paper trained under);
duplicate traces within a file are dropped. Directory names must carry
`__pid-<hex>__`, the token that pairs a device's enrollment and return-visit
directories. Return-visit files are taken in file-name order, so
`--chronological` needs names that sort by time (the paper's data used
`samples_<YYYY-MM-DD-HH-MM-SS>.ndjson`). `--t0-root` points at the T0 visits of the DRAWNAPART-matched
set (or one of its partitions, D_train and D_open of paper §5.2, whose
internal names MP65 and MPrest appear in the code comments) and
`--t130-dir` at the same devices' return visits (§5.3).
