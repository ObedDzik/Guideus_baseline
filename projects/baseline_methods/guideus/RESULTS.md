# guideus baseline results

Two-stage recipe (matches `projects/prostnfound`'s convention): NCT2013
pretrain, then OPTIMUM finetune from the fold-0 NCT checkpoint. Model:
`prostnfound_adapter_medsam_legacy` (same encoder/decoder as prostnfound),
`experiment_type: guideus` (decoder cancer-logits loss + US-histo
alignment loss via needle-pooled ABMIL). No prompts (`model_kw.prompts: []`
— the model factory's default is `[age, psa, approx_psa_density]`, which
was silently active in the first attempt before being caught and fixed).
Configs: `guideus_cfg0_nct.yaml`, `guideus_cfg0_optimum.yaml`.

**Tracked metric**: `val/auc` — the same core-level `ProstNFoundEvaluator`
projects/prostnfound/train.py uses, computed directly from the decoder's
`cancer_logits`. The original config tracked `linear_probe/macro_auc`
instead (a fresh `LogisticRegression` refit each epoch on pooled
needle-region features) — found and fixed after that metric plateaued
within the first ~3 epochs on every fold while `train_loss` kept moving,
a sign the tracked metric wasn't reflecting the model actually training.
`ProstNFoundEvaluator` was already imported but unused; wired into both
`run_train_epoch`/`run_eval_epoch` in `guideus_pnf_train.py`.

## NCT2013, stage 1

| Fold | Job ID | Best val/auc | Checkpoint |
|---|---|---:|---|
| 0 | 5562710 | 0.6627 | `/scratch/obed/guideus_baseline/guideus_nct_0/5562710/best.pth` |
| 1 | 5562711 | 0.6682 | `/scratch/obed/guideus_baseline/guideus_nct_1/5562711/best.pth` |
| 2 | 5562712 | 0.6306 | `/scratch/obed/guideus_baseline/guideus_nct_2/5562712/best.pth` |
| 3 | 5562713 | 0.6292 | `/scratch/obed/guideus_baseline/guideus_nct_3/5562713/best.pth` |
| 4 | 5562714 | 0.5548 | `/scratch/obed/guideus_baseline/guideus_nct_4/5562714/best.pth` |

**Mean = 0.6291, std ≈ 0.040** (n=5 folds)

## OPTIMUM, stage 2

Loads NCT fold-0 checkpoint (job 5562710), finetunes on OPTIMUM
(UA+OL, `data.include_optimum_c3: true`, `data.optimum_centers: [UA, OL]`,
`data.include_nct: false`, `data.include_optimum_c23: false` — the legacy
duplicate single-frame UA export is excluded).

| Fold | Job ID | Best val/auc | Checkpoint |
|---|---|---:|---|
| 0 | 5567062 | 0.6544 | `/scratch/obed/guideus_baseline/guideus_optimum_0/5567062/best.pth` |
| 1 | 5567063 | 0.5345 | `/scratch/obed/guideus_baseline/guideus_optimum_1/5567063/best.pth` |
| 2 | 5567064 | 0.5847 | `/scratch/obed/guideus_baseline/guideus_optimum_2/5567064/best.pth` |
| 3 | 5567065 | 0.6341 | `/scratch/obed/guideus_baseline/guideus_optimum_3/5567065/best.pth` |
| 4 | 5567066 | 0.6586 | `/scratch/obed/guideus_baseline/guideus_optimum_4/5567066/best.pth` |

**Mean = 0.6133, std ≈ 0.047** (n=5 folds)

Fold 1 plateaued after epoch 0 and never improved through the rest of
the 50-epoch run (the other 4 folds all show healthy, continued
improvement across many epochs) — a real, fold-specific weak result, not
a repeat of the earlier tracked-metric bug (confirmed via the same
`val/auc` evaluator as every other fold).

## OPTIMUM, stage 2 — PU (transperineal) added to training only

Same as the row above but `data.include_pu_in_train: true`,
`data.pu_extra_flip: true` — val stays UA/OL-only, unchanged. This
loader has no precomputed `splits.json` (unlike prostnfound's), so PU
gets its own from-scratch `KFold` split and only that fold's "train" role
is added (~80% of PU cases, a different subset per fold) — see
`src/nct_optimum_loader_pnf.py`'s comments for the exact mechanism, and
note this bypasses `ProstNFoundTransform`'s own `pu_extra_flip` entirely
(this loader's `_optimum_adapter` renames `"image"` to `"bmode"` before
the shared transform ever runs, so its own PU-flip logic lives directly
in `_optimum_adapter` instead). Config: `guideus_cfg0_optimum_pu.yaml`.
Verified directly before running: `include_pu_in_train=True` grows fold-0
train from 1349 to 1794 cases (445 PU cases added) while val stays
exactly 341; the PU flip changes only PU items' pixels, confirmed via a
direct `_optimum_adapter` unit check (same verification pattern used for
prostnfound's own PU fix).

| Fold | Job ID | Best val/auc | Checkpoint |
|---|---|---:|---|
| 0 | 5575388 | 0.6519 | `/scratch/obed/guideus_baseline/guideus_optimum_pu_0/5575388/best.pth` |
| 1 | 5575389 | 0.5339 | `/scratch/obed/guideus_baseline/guideus_optimum_pu_1/5575389/best.pth` |
| 2 | 5575390 | 0.5816 | `/scratch/obed/guideus_baseline/guideus_optimum_pu_2/5575390/best.pth` |
| 3 | 5575391 | 0.6168 | `/scratch/obed/guideus_baseline/guideus_optimum_pu_3/5575391/best.pth` |
| 4 | 5575392 | 0.6339 | `/scratch/obed/guideus_baseline/guideus_optimum_pu_4/5575392/best.pth` |

**Mean = 0.6036, std ≈ 0.042** (n=5 folds)

**PU-in-train vs. no-PU**: 0.6036 vs. 0.6133 — Δ = -0.0097, well inside
fold-to-fold noise. Same "no measurable effect" pattern found for
prostnfound's own PU-in-train ablation. Fold 1 plateaus after epoch 0 in
*both* the PU and no-PU runs (0.5339 vs. 0.5345, nearly identical) —
confirms that weakness is a property of fold 1's own train/val split
composition, not something PU inclusion causes or fixes.

## Comparison vs. prostnfound baselines

| Method | Recipe | Mean val/auc (OPTIMUM stage) |
|---|---|---:|
| prostnfound+ (256px) | no-prompt, PU-in-train | 0.6668 ± 0.039 |
| prostnfound (256px, image_clf off) | no-prompt, PU-in-train | 0.6622 ± 0.037 |
| prostnfound+ (512px) | no-prompt, PU-in-train | 0.6997 ± 0.024 |
| prostnfound (512px, image_clf off) | no-prompt, PU-in-train | 0.6945 ± 0.018 |
| **guideus** | no-prompt, UA+OL only, US-histo alignment | **0.6133 ± 0.047** |

guideus (512px, same encoder as prostnfound) currently trails every
prostnfound variant by a real margin (~0.05-0.09) and has notably higher
fold-to-fold variance (std 0.047 vs. 0.018-0.039) — largely driven by
fold 1's early plateau. Not a like-for-like comparison yet: guideus is
UA+OL-only (no PU-in-train ablation done for it), and its alignment loss
adds a second, competing objective (US-histo alignment) on top of the
same decoder loss prostnfound optimizes alone — worth keeping in mind
before treating this gap as evidence about the alignment mechanism itself
rather than recipe differences.
