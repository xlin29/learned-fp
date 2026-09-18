# Evaluation pipeline

Per-key CNN training, profile construction, similarity ranking, and Top-k
evaluation — the back half of LearnedFP. Ships code, fixed configurations
and a CPU demo against the bundled lab gallery; what is withheld is listed
in the top-level README.

## Quick demo (≈2-3 min on CPU)

```bash
pip install -r pipeline/requirements.txt
bash pipeline/demo.sh
```

What the demo does:

1. Picks 5 lab devices from `lab_data/devices/` spanning the three
   paper Table 4 categories (browser-default / browser-defense /
   extension), extracts their tarballs into a temp dir.
2. Trains `train_fixed.py` on all 9 paper rendering keys
   (paper §4.1, Figure 2) for 2 epochs with `batch_size=64`. Each key
   gets its own per-device CNN; the cross-key vote fusion path
   (paper §5.4, "Per-key contribution") is exercised at eval time.
3. Runs `evaluate.py vote` against the same split and prints Top-k.
4. Reports `PASS` if the train + eval pipeline returns valid Top-k
   numbers for the 5-class smoke gallery.

The demo is not a paper-scale reproduction. At 5 devices × 2 epochs
the Top-k numbers stay at the 5-class base rate (top1=0.2, top5+=1.0).
The paper's runs (§5.3 matched comparison, §5.4 full population) trained for
30 epochs or with the Bayesian search of Appendix C.

Every config pins `seed: 42`, and the Bayesian trainer passes the same seed
to scikit-optimize as `random_state`, so the same command on the same
hardware yields bit-for-bit the same output.

Every training and eval script opens with a docstring that names the paper
section it implements (`Paper: §X.Y`); each YAML under `configs/` names the
table row of the paper it reproduces.

## CLI

```bash
python3 pipeline/cli.py train --config <yaml> --data <BASE_DIR> --out <RESULTS_DIR>
python3 pipeline/cli.py eval  --config <yaml> --data <BASE_DIR> --models <RESULTS_DIR> \
                              --mode {vote, vote_temporal, vote_unseen} \
                              [--t130-dir <RETURN_VISIT_DIR>]
```

`train` runs the per-key trainer 9× (one per rendering key), then
auto-runs `vote`. `eval` routes by `--mode` to one of three eval paths:

- `--mode vote` → `evaluate.py vote` — closed-set within DB_T0, training and test samples from the same campaign (the setting of paper §5.4 "Per-key contribution" / Figure 6)
- `--mode vote_temporal` → `eval_temporal_open.py` — open-world chrono, gallery = full T0 (paper §5.4 / Table 2, temporal re-identification)
- `--mode vote_unseen` → `eval_coldstart_open.py` — open-world chrono, gallery = T0 + unseen self (paper §5.4 / Table 2, cold-start)

Pair each `--mode` with the matching YAML config (the `recipe` /
description in the YAML names the table row of the paper it reproduces). For the
per-browser breakdown and the defense eval, invoke directly — they
reuse `evaluate.py` internals:

```bash
# per-browser split of the closed-set vote; the browser is read from each
# device's stored user agent. `vote_temporal` (--t0-dir/--t130-dir) and
# `vote_unseen` split the two open-world evals the same way.
python pipeline/eval/evaluate_per_browser.py vote \
    --base-dir <DATA> --results-dir <RESULTS_DIR>

python pipeline/eval/eval_defense.py \
    --base-dir <DATA> --results-dir <RESULTS_DIR>
```

## Two trainers

| | `train_bayes.py` | `train_fixed.py` |
|---|---|---|
| Paper | §5.2, DRAWNAPART-matched comparison | §5.4, full population |
| Hyperparameters | 79-trial Gaussian-process search, 5 epochs per trial | fixed defaults (Appendix C) |
| Model file | `models/fp_rawcnn_<key>.pt` | `models/rawcnn_allvis_<key>.pt` |
| Normalizer | `pixnorm/fp_rawcnn_pixnorm_<key>.json` | `pixnorm/rawcnn_pixnorm_allvis_<key>.json` |
| Hparam record | `hparams/fp_rawcnn_best_<key>.json` | none; defaults live in the config |

`evaluate.py` reads either via `--model-type bayes|fixed`.

## Reproducing the paper

The full reproduction requires the crowdsourced corpus referenced
in paper §5.1 (DB_T0, the returning and new devices of DB_T1, and the
DRAWNAPART-matched subset used in §5.3). We do
not redistribute it (Open Science, Appendix A). With access to a comparable
dataset, the configurations are:

**Bayesian (paper §5.2)** — 9 keys × 79 trials × 5 epochs each.

**Fixed (paper §5.4)** — 9 keys × 30 epochs on the full population.

The CLI plus the YAML configs in `configs/` carry the paper's
configuration on a single host:

```bash
# Bayesian (paper §5.2 matched DP comparison)
python3 pipeline/cli.py train --config pipeline/configs/bayes_dp.yaml \
                              --data /path/to/D_train --out ./out_bayes
python3 pipeline/cli.py eval --config pipeline/configs/bayes_dp.yaml \
                             --data /path/to/D_train --models ./out_bayes --mode vote

# Fixed (paper §5.4 3k-scale)
python3 pipeline/cli.py train --config pipeline/configs/fixed_3k.yaml \
                              --data /path/to/DB_T0 --out ./out_3k
python3 pipeline/cli.py eval --config pipeline/configs/fixed_3k.yaml \
                             --data /path/to/DB_T0 --models ./out_3k --mode vote

# Open-world chrono temporal return (paper §5.4 / Table 2, temporal)
python3 pipeline/cli.py eval --config pipeline/configs/temporal_chrono.yaml \
                             --data /path/to/DB_T0 --t130-dir /path/to/returning_visits \
                             --models ./out_3k --mode vote_temporal

# Open-world chrono cold-start (paper §5.4 / Table 2, cold-start)
python3 pipeline/cli.py eval --config pipeline/configs/unseen_chrono.yaml \
                             --data /path/to/new_devices --models ./out_3k \
                             --mode vote_unseen

# DRAWNAPART-matched temporal re-identification (paper §5.3 / Figure 4, LearnedFP side):
# the matched gallery, queried by its returning devices. Invoked directly.
python3 pipeline/eval/eval_temporal_dp_match.py \
    --results-dir ./out_bayes --t0-dir /path/to/matched_set_T0 --t130-dir /path/to/returning_visits \
    [--per-k-whitelist <dp_temporal_open_eligible_by_k.json>]
```

Both methods are scored on the same devices at each k. Each evaluation
writes the per-k set of devices it can score (this one
`fp_temporal_open_eligible_by_k.json`, the DRAWNAPART side
`dp_temporal_open_eligible_by_k.json`) and accepts the other's file
(`--per-k-whitelist` here, `--fp-eligible-by-k` there), so each side can be
restricted to the intersection.

Figure 5 (held-out) is the same evaluation with encoders trained on D_train
only (`bayes_dp.yaml` on the 65% partition), queries restricted to D_open's
returning devices via `--per-k-whitelist`, and the DB_T1 devices that serve
as distractors placed under `--t0-dir` alongside the enrolled ones. The DRAWNAPART side of
that figure needs a shim to load a D_train-width encoder against the larger
gallery; see `baselines/drawnapart/README.md`.

For the per-browser breakdown and defense eval scripts, invoke
[`evaluate_per_browser.py`](eval/evaluate_per_browser.py) and
[`eval_defense.py`](eval/eval_defense.py) directly — see the
[CLI](#cli) section above.

## Docker

```bash
cd pipeline
docker build -t learnedfp-pipeline .

# 2-3 min demo. Use `bash` as entrypoint so the host file's executable bit
# doesn't matter (zip distributions of the artifact strip +x). Must be
# `bash`, not `sh` — demo.sh uses set -o pipefail which dash rejects.
docker run --rm -v "$(pwd)/..:/workspace" \
  --entrypoint bash learnedfp-pipeline /workspace/pipeline/demo.sh

# Or run the CLI directly (default entrypoint).
docker run --rm -v "$(pwd)/..:/workspace" learnedfp-pipeline --help
```

The image is CPU-only and sized for the demo. For paper-scale GPU
training, replace torch with the matching CUDA wheel:

```dockerfile
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch
```
