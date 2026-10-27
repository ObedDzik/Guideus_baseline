# GUIDE-US in the lab's core-level benchmark

This folder defines how GUIDE-US must be trained and scored so that it can be added as a row to the lab's core-level
micro-ultrasound benchmark, next to MedSAM-UNETR, MedSAM-FT, MicroSegNet, ProstNFound, ProstNFound+ and Prost-RL.
It is written for the person or coding agent who runs GUIDE-US. Follow it literally; when something here is
impossible on your machine, stop and report it instead of working around it.

The benchmark has three cohorts:

| Cohort | Use | Cores / patients the benchmark reports on |
|---|---|---|
| NCT2013 (5 sites, transrectal, systematic) | training and five-fold cross-validation | 6,588 / 693 |
| OPTIMUM OL + UA (2 sites, systematic + targeted) | external, zero-shot | 1,460 / 111 |
| OPTIMUM PU (1 site, transperineal) | external, zero-shot | 558 / 42 |

Every other row was trained on NCT2013 only, with the same five patient-level folds, selected on the NCT validation
fold, and then scored without any change on OPTIMUM OL+UA and PU through one fixed preprocessing path. GUIDE-US has
to do the same.

## Summary

1. Train five fold models with `guideus_cfg_benchmark_nct.yaml` (`fold=0` ... `fold=4`) using this repository's
   `guideus_pnf_train.py`. It refuses a config that breaks the protocol or data that are not the benchmark fold, and it
   stores the training config and a data manifest inside every checkpoint. Checkpoints trained before this update
   carry no such record and cannot be exported: retrain.
2. Export each fold with `benchmark/export.py`. It scores the fold's NCT validation cores, every OPTIMUM OL+UA core and
   every PU core, and checks that your cohorts and model inputs are identical to the benchmark's.
3. Run `benchmark/check_export.py` on the export folder. It must print `PASS`. It recomputes the identifiers,
   checksums and input digests from the files themselves.
4. Hand the export folder to the benchmark maintainers (cluster shared folder). Do not commit predictions,
   checkpoints or any patient-level file to this repository: it is public.

The benchmark maintainers compute every table metric from your files. You do not need to compute anything on
OPTIMUM or PU, and you must not use OPTIMUM or PU results to choose anything.

## Why the earlier runs in this repository cannot be used

| Setting | Earlier GUIDE-US runs here | Benchmark (this folder) |
|---|---|---|
| Training data | NCT2013, then fine-tuning on OPTIMUM UA+OL (`guideus_cfg0_optimum*.yaml`) | NCT2013 only. OPTIMUM and PU are never used for training, validation, selection or tuning |
| OPTIMUM/PU orientation | not flipped (`_optimum_adapter`); PU optionally flipped once (`pu_extra_flip`) | every OL, UA and PU frame flipped vertically (`probe_top`), as for every other row; NCT frames are already probe-top |
| UA prostate masks | only masks inside each cine folder, otherwise the whole image | the cine folder or the separate UA mask export; cores without a non-empty prostate and needle mask are dropped |
| Missing PSA / age / PSA density | filled with 0.0 | left as NaN and imputed by the transform with the NCT training means (PSA density is missing for every OPTIMUM and PU core) |
| Clinical prompts | `[age, psa]` or none | `[age, psa, psad, pos]` (PSA density and core location included), as for ProstNFound+ and Prost-RL |
| Image-level head | binary GG≥3, or three-class without class weights | three classes (benign / GG1–2 / GG≥3), cross-entropy with per-batch balanced class weights |
| Endpoints | csPCa = GG≥2 in `eval_paper_metrics.py` | csPCa = GG≥3; all-cancer = GG≥1 |
| Input size | 512 px, 128-px masks | 256 px, 64-px masks |
| Mask-decoder upsampler | re-initialised `interpolate` upsampler (this repo's default) | MedSAM's pretrained transposed-convolution upsampler (`output_upscaling_version: default`) |
| Optimisation | AdamW 1e-5 (encoder 1e-6), 50 epochs, batch 20 | AdamW 2e-5 (encoder 1e-5, warm-up 5e-5 for 1 epoch), weight decay 1e-3, 35 epochs, batch 8, seed 30 |
| Checkpoint selection | `val/auc` (heatmap AUROC, all cores) | `val/auc_high_involvement` (heatmap AUROC on benign and >40%-involvement cores of the NCT validation fold) |
| Score in the table | mean needle heatmap | the three-class classifier: P(GG≥1) = 1 − P(benign) and P(GG≥3); the heatmap mean is exported as a secondary score |

The recipe rows (input size, prompts, upsampler, optimisation, selection) are the ones shared by the benchmark's
retrained MedSAM baselines, so GUIDE-US differs from the retrained ProstNFound+ only by what GUIDE-US adds.

What stays exactly as implemented here: the GUIDE-US alignment branch. Encoder tokens under the needle mask are pooled
by attention-based MIL (`src/abmil_ce.py`), and a triplet loss (margin 1, weight 1, L2-normalised embeddings) pulls the
pooled embedding toward a PANDA slide embedding (GigaPath, Radboud provider) of the same ISUP grade and similar
involvement and away from an adjacent-grade one (`src/nct_optimum_dataset.py`, `src/loss_new.py`). The PANDA branch is
used for training only; inference and export use the ultrasound model alone.

## Protocol

These rules define the benchmark row. `benchmark/protocol.py` enforces the checkable ones; the rest is on you.

**Data and split.** NCT2013 only, `cohort_selection_mode: kfold`, five folds: patients are split by
`StratifiedKFold(5, shuffle=True, random_state=0)` stratified by center (`medAI/datasets/nct2013/cohort_selection.py`).
Benign cores of patients who have cancer are removed from training only. First frame, no vertical flip for NCT.
The NCT metadata must be `nct2013/metadata_with_approx_psa_density.csv` with the SHA-256 in `fingerprints.json`.

**Labels.** Three classes from the core's Grade Group: 0 = benign, 1 = GG1–2, 2 = GG≥3. NCT2013 has no GG1 cores, so its
middle class is GG2. Involvement (percentage of cancer in the core) is used by the heatmap MIL loss and by GUIDE-US's
positive sampling, as before.

**Model.** `prostnfound_adapter_medsam_legacy` with a three-class class decoder, prompts `[age, psa, psad, pos]`, MedSAM's
pretrained upsampler, started from the public MedSAM ViT-B weights (`model_checkpoint: null`), plus the GUIDE-US ABMIL
and triplet branch with the settings of the config (`experiment_type: guideus`, `train_mode: us_histo`, ABMIL 256 → 768
with attention width 512, no ACMIL, Radboud PANDA embeddings). These GUIDE-US settings are frozen in every variant.

**Training and selection.** The schedule in `guideus_cfg_benchmark_nct.yaml`. One run per fold, seed 30. The checkpoint
is the epoch with the highest `val/auc_high_involvement` on that fold's NCT validation cores. No early-stopping or
hyperparameter choice may use OPTIMUM or PU, and no other checkpoint may replace `best.pth` afterwards.

**Not allowed.** Training, fine-tuning, calibration, temperature scaling, threshold choice, test-time adaptation or
checkpoint choice on OPTIMUM or PU; changing preprocessing for the external cohorts; dropping or adding cores by hand;
re-running with other seeds or settings and keeping the best.

If you think a rule is wrong for GUIDE-US (for example, you want 512-px inputs), train it additionally as a
`sensitivity` variant (see below). It will be reported as a secondary row; the `primary` row uses the shared recipe.

## Setup

This repository is an extract of the full medAI code base and lacks two modules the trainer imports:
`external_libs` (Segment Anything / MedSAM model code) and `medAI.metrics` (used by the training evaluator). Use one
of two layouts and run every command from its root:

- **Full medAI tree (recommended):** copy the files of this update into your full medAI checkout at the same relative
  paths (`projects/baseline_methods/guideus/...`, including `benchmark/`), and run from that checkout's root, as the
  existing SLURM scripts do.
- **This extract:** add `medAI/metrics` from the full checkout (uncommitted) and put the directory that contains
  `external_libs` on `PYTHONPATH` (the SLURM scripts read it from `EXTRA_PYTHONPATH`). The exporter itself needs
  `external_libs` but not `medAI.metrics`.

Either way the exporter checks that the cohorts and model inputs equal the benchmark's, so a different `medAI`
version cannot go unnoticed in the data path.

```bash
export PYTHONPATH=$PWD                                                # plus external_libs' parent for the extract
export EXACTVU_PCA_DATA_ROOT=/datasets/exactvu_pca                    # contains nct2013/
export BENCHMARK_OPTIMUM_ROOT=/datasets/exactvu_pca/OPTIMUM/UA_OL_PU_annotated_needles_multiframe
export BENCHMARK_OPTIMUM_PROSTATE_MASK_ROOT=/datasets/exactvu_pca/OPTIMUM/processed/UA_annotated_needles
export MEDSAM_CHECKPOINT_DIR=/datasets/exactvu_pca/checkpoint_store   # contains medsam_vit_b_cpu.pth
export JOB_ID=manual_fold0          # training only: part of checkpoint_dir, unique per run; SLURM sets the job id
export WANDB_MODE=disabled          # training only; or export WANDB_API_KEY for online logging
```

`BENCHMARK_OPTIMUM_ROOT` is the OPTIMUM export with one folder per case (`OL-*`, `UA-*`, `PU-*`), each cine holding
`frames/0000.png`, `needle_mask.png`, `info.json` and, when available, `micro_seg_net_prostate_mask.png`.
`BENCHMARK_OPTIMUM_PROSTATE_MASK_ROOT` holds the UA prostate masks that are not inside the cine folders, at
`<case>/<cine>/micro_seg_net_prostate_mask.png`. The paths above are the Killarney paths these configs already used;
the exporter's cohort check tells you whether they are the right exports.

Check the code before using GPUs:

```bash
python -m unittest projects.baseline_methods.guideus.benchmark.test_benchmark -v   # CPU tests, no data needed
```

## Step 1: train the five folds

```bash
python -m projects.baseline_methods.guideus.guideus_pnf_train \
    -c projects/baseline_methods/guideus/guideus_cfg_benchmark_nct.yaml fold=0      # then fold=1 ... fold=4
# or, on Killarney: sbatch --array=0-4 projects/baseline_methods/guideus/train_benchmark_nct.sh
```

In the log of each run, check before letting it continue:

- `Benchmark protocol variant: primary` is printed (otherwise the protocol check did not run), and
  `Benchmark manifest: ...` after the loaders are built. The trainer compares the cores it actually loads with
  `fingerprints.json` and stops if they differ;
- the split sizes printed by cohort selection (for example fold 0: 4,334 training cores after filtering and 1,336
  validation cores; all folds are listed in `fingerprints.json` as NCT validation counts 1,336 / 1,302 / 1,317 /
  1,328 / 1,324);
- `train: N val: M` from the loader shows only NCT cores (no OPTIMUM sub-dataset);
- after epoch 0, `val/auc_high_involvement` is logged and `best.pth` is saved. If the run stops with
  `tracked_metric ... not among validation metrics`, your `medAI.metrics` names the AUROC differently; report it
  rather than switching the selection metric.

Each run stores its resolved config and the data manifest inside every saved checkpoint, writes `config.yaml` and
`benchmark_manifest.json` next to `best.pth` for reading, and writes `training_complete.json` when all epochs have run.
Do not move `best.pth` away from `training_complete.json`; the exporter needs both.

## Step 2: smoke-test the export

Set the five checkpoint paths once (one `best.pth` per fold, from the training logs), then:

```bash
CKPTS=(
  /scratch/obed/guideus_baseline/guideus_benchmark_nct_0/1111111/best.pth   # fold 0: replace with the real job ids
  /scratch/obed/guideus_baseline/guideus_benchmark_nct_1/1111112/best.pth   # fold 1
  /scratch/obed/guideus_baseline/guideus_benchmark_nct_2/1111113/best.pth   # fold 2
  /scratch/obed/guideus_baseline/guideus_benchmark_nct_3/1111114/best.pth   # fold 3
  /scratch/obed/guideus_baseline/guideus_benchmark_nct_4/1111115/best.pth   # fold 4
)
python -m projects.baseline_methods.guideus.benchmark.export \
    --config projects/baseline_methods/guideus/guideus_cfg_benchmark_nct.yaml --fold 0 \
    --checkpoint "${CKPTS[0]}" --out /scratch/obed/guideus_benchmark_export_smoke --smoke 16
```

This scores 16 cores per cohort and is marked as a smoke run; it is never accepted as a result. Delete the smoke folder
afterwards; exports are write-once and refuse to overwrite.

## Step 3: export the five folds

```bash
# CKPTS as set in step 2
for f in 0 1 2 3 4; do
  python -m projects.baseline_methods.guideus.benchmark.export \
      --config projects/baseline_methods/guideus/guideus_cfg_benchmark_nct.yaml --fold $f \
      --checkpoint "${CKPTS[$f]}" --out /scratch/obed/guideus_benchmark_export
done
# or one SLURM job per fold:
# for f in 0 1 2 3 4; do sbatch projects/baseline_methods/guideus/export_benchmark.sh $f "${CKPTS[$f]}" /scratch/obed/guideus_benchmark_export; done
```

Each fold takes about two minutes on one GPU (about 3,500 cores). The exporter:

- refuses to run without `fingerprints.json`;
- refuses a checkpoint that was not trained under the protocol: the config stored in the checkpoint must pass the
  protocol and equal the config you pass (except paths and logging), the stored data manifest must equal the benchmark
  fold, the checkpoint must contain the GUIDE-US ABMIL head, and `training_complete.json` must report the full schedule
  with `best.pth` from the selected epoch;
- loads the ProstNFound part of the checkpoint strictly (the ABMIL head is training-only and not used for scoring);
- builds the cohorts with `benchmark/data.py`, whose loaders and transform under `benchmark/vendored/` are verbatim
  copies of the code that produced the other rows;
- runs the model in full precision (no autocast, TF32 off), batch 8;
- compares the cohort identifiers, the NCT split and cohort-level averages of the model inputs with `fingerprints.json`
  (see "Cohort and input checks"); if anything differs it writes `provenance.json` with the list of problems and exits
  without writing `COMPLETE`;
- writes the files below with their SHA-256 in `provenance.json`, and `COMPLETE`. A fold that fails is moved to
  `fold<k>.failed-<time>` (kept for inspection), so a corrected run can write `fold<k>` again.

## Step 4: check and hand over

```bash
python -m projects.baseline_methods.guideus.benchmark.check_export /scratch/obed/guideus_benchmark_export
```

It must print `PASS`. It also prints NCT validation AUROC per fold as a sanity check (on all validation cores, so it
differs slightly from the table, which uses the 6,588 benchmark cores). Then copy the whole export folder (five `fold*`
folders) to the shared project folder and tell the benchmark maintainers where it is, with the training job IDs and
W&B run names. Keep the checkpoints: if a later check finds a problem, the folds must be re-exported, not retrained.

## Output files

`<out>/fold<k>/` contains:

| File | Rows | Columns |
|---|---|---|
| `val_predictions.csv` | every NCT validation core of fold k | `core_id, patient_id, center, grade_group, involvement, pca, gg3, p_benign, p_gg12, p_gg3, heatmap` |
| `optimum_predictions.csv` | every OPTIMUM OL+UA core (1,587; the same in every fold) | same as above |
| `pu_predictions.csv` | every PU core (566; the same in every fold) | `core_id, patient_id, center, pca, gg3, p_benign, p_gg12, p_gg3, heatmap` (no labels) |
| `provenance.json` | | config and checkpoint SHA-256, checkpoint epoch and score, git commit, versions, data roots, cohort fingerprints, input digests, problems |
| `COMPLETE` | | `success` (or `smoke`) |

Scores, all from one forward pass of the fold model:

- `p_benign, p_gg12, p_gg3`: softmax of the three-class image-level classifier;
- `pca = p_gg12 + p_gg3` (all-cancer score) and `gg3 = p_gg3` (csPCa score). These two are the GUIDE-US row;
- `heatmap`: mean sigmoid heatmap probability inside the needle ∩ prostate mask at mask resolution. It is NaN when that
  intersection is empty, which is expected for 19 NCT cores (over the five folds), 102 OPTIMUM cores and 3 PU cores.
  The classifier scores are finite for every core.

PU files carry no labels on purpose, so no PU result is visible during export. The label columns of the NCT and
OPTIMUM files are informational only: the benchmark joins its own reference labels (Grade Group, involvement) to every
cohort by `core_id` and never uses exported labels for a metric.

## Cohort and input checks

`fingerprints.json` was produced with the benchmark's own loaders on the benchmark's data. For each cohort it holds the
number of cores and patients and the SHA-256 of the sorted, newline-joined identifiers (NCT: per fold, validation
cores, validation patients and training patients; OPTIMUM OL+UA; PU), the SHA-256 of the NCT metadata and needle-mask
files, and an input digest: cohort-level averages of the preprocessed image, of the needle and prostate mask areas and
centroids, and of each normalised clinical prompt. A vertical flip, a different mask source, a crop or a different
prompt imputation changes the digest. Only cohort averages are stored, nothing per core.

The cohorts the exporter scores are slightly larger than the reported ones: all 6,607 NCT validation cores, 1,587
OPTIMUM OL+UA cores and 566 PU cores. The benchmark then restricts every row to the same reported cores: 6,588 NCT cores
(leaving out the 19 whose needle and prostate masks do not overlap at 64 px), 1,460 OPTIMUM cores (leaving out the 102
without overlap and the 25 cores of two patients outside the official OPTIMUM folds) and 558 PU cores (leaving out the 3
without overlap and 5 cores with ambiguous grade labels). Do not filter yourself.

If a check fails, do not use `--allow-mismatch` to get past it: that flag only writes the diagnostics and marks the
export as not accepted. Send `provenance.json` (its `problems` list names the failing cohort and quantity) to the
benchmark maintainers. Typical causes are a different OPTIMUM export, a missing UA prostate-mask root, or a different NCT metadata file.

## What the benchmark computes from your files

The same quantities as for every other row, computed by the benchmark maintainers: core-level AUROC and sensitivity at
60% and 80% specificity for all-cancer (scored by `pca`) and csPCa (scored by `gg3`); AUROC on benign and
>40%-involvement cores and across involvement thresholds; patient-level csPCa AUROC from the mean core score; results
per external site; comparison with the clinicians' PRI-MUS scores on OPTIMUM; and selective prediction from the
three-class probabilities. Values are means ± SD over the five fold models (each on its own NCT validation fold, and
on the whole external cohorts), with paired patient-cluster bootstrap comparisons against the other rows.

## Optional variants

- `image_only`: `guideus_cfg_benchmark_nct_image_only.yaml`, the same run with no clinical prompts. Export it to a
  separate folder. Do this only after the primary row is exported.
- `sensitivity`: copy `guideus_cfg_benchmark_nct.yaml`, set `benchmark_protocol: sensitivity`, change only recipe
  keys (for example `data.image_size: 512` with `data.mask_size: 128`) and describe the change in the hand-over. The
  split rules are still enforced. Reported only as a secondary row.

## PU orientation (checked): no extra flip

Comments elsewhere in this repository state that PU frames on disk are oriented opposite to UA/OL and need one more
flip (`pu_extra_flip`). We checked this on all 566 PU cores, and it is not the case: PU frames are stored the same way
up as OL and UA, and the single `probe_top` flip used for every benchmark row is correct for PU.

- In PU only the needle is transperineal; the imaging probe is still transrectal. After the `probe_top` flip the
  gland lies against the probe edge, as in NCT2013, OL and UA (on average its first row is at 5% of the image height;
  NCT2013 15%, OL 14%, UA 8%).
- The brightness-by-depth profile of PU after the flip correlates 0.96 with that of the NCT2013 training frames (OL+UA:
  0.94); with the extra flip it correlates −0.96, i.e. the far field is where the probe should be.
- The ExactVu display overlay (black box) sits in the same image corner for OL, UA and PU after the flip; the extra flip
  moves it for PU only.
- A diagnostic check with an existing NCT-trained baseline confirmed it: with the extra flip its PU discrimination
  fell below chance.

The needle explains the other conclusion: a transperineal needle runs roughly horizontally from the side of the image
instead of diagonally from the probe, so judging orientation from the needle mask makes PU look upside down. Models
fine-tuned on PU can adapt to either orientation, which is why it did not matter in the earlier runs; for a model
trained on NCT2013 only it does. Do not use `pu_extra_flip` (or any other PU-specific flip) for benchmark runs; the
exporter applies the benchmark orientation and its input digest would reject a flipped PU cohort.

## Files

| File | Purpose |
|---|---|
| `guideus_cfg_benchmark_nct.yaml` | primary benchmark training config (`benchmark_protocol: primary`) |
| `guideus_cfg_benchmark_nct_image_only.yaml` | optional image-only variant |
| `train_benchmark_nct.sh`, `export_benchmark.sh` | Killarney SLURM scripts for steps 1 and 3 |
| `benchmark/protocol.py` | the rules, the training manifest and the checkpoint verification |
| `benchmark/data.py`, `benchmark/vendored/` | benchmark cohorts and preprocessing (vendored files are verbatim; do not edit) |
| `benchmark/fingerprint.py` | identifier fingerprints, input digest, batching, loading `fingerprints.json` |
| `benchmark/export.py`, `benchmark/check_export.py` | export and final check |
| `benchmark/fingerprints.json` | expected cohorts and input digests |
| `benchmark/test_benchmark.py` | CPU unit tests |

Training-code changes made for the benchmark (existing configs behave as before unless noted):

- `guideus_pnf_train.py`: when `benchmark_protocol` is set, runs the protocol check, checks the cores it loads against
  `fingerprints.json`, stores the resolved config and data manifest in every checkpoint, and writes
  `training_complete.json` at the end; it always writes `config.yaml` next to the checkpoint and stops if
  `tracked_metric` is missing from the validation metrics (before, such a run silently never saved a checkpoint).
- `src/loss_new.py`: optional `image_clf_class_weight: balanced` and the `three_class` alias for `gg3`; the proportion
  BCE of the heatmap loss is computed in FP32 log space. Its value is unchanged, but under mixed precision the old form
  became inf/NaN once all probabilities in a bag saturated, so runs of existing configs can differ slightly where that
  happened.
- `src/bmode_dataset.py` (`BModeDatasetV2`): also returns the core-location encodings needed by the `pos` prompt.
