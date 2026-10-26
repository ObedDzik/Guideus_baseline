# GuideUS baseline

Prostate cancer (csPCa) detection on micro-ultrasound with a GuideUS-style model: a MedSAM ViT-B adapter encoder with a
SAM-style mask decoder (ProstNFound architecture), trained with a needle-region involvement loss plus an
ultrasound–histopathology alignment loss (needle-pooled ABMIL), optionally with clinical prompts (age, PSA).

The code is a standalone extract of the GuideUS baseline from a larger research codebase: `projects/baseline_methods/guideus/`
plus only the `medAI` library modules it uses (`medAI/`), and one model file from `projects/medibot/`.

## Layout

```
projects/baseline_methods/guideus/
  guideus_pnf_train.py            training / validation script
  eval_paper_metrics.py           core- and patient-level metrics from saved checkpoints
  src/                            data loaders, losses, ABMIL pooling, diagnostics
  guideus_cfg0_*.yaml             configs (NCT2013 stage 1; OPTIMUM fine-tuning; leave-one-centre-out)
  train_nct*.sh, finetune_*.sh    SLURM launch scripts
  RESULTS.md                      results notes
medAI/                            library modules used by GuideUS (models, datasets, transforms, evaluator)
resources/G7.npy                  B-mode colormap used by the NCT2013 loader
```

## Setup

Python 3.11 with PyTorch (CUDA), numpy, pandas, scikit-learn, scipy, omegaconf, einops, timm, opencv-python,
Pillow, tqdm and wandb. Run everything from the repository root with it on the Python path:

```bash
export PYTHONPATH=$PWD
```

Environment variables read by the code and launch scripts:

| Variable | Meaning |
|---|---|
| `WANDB_API_KEY` | Weights & Biases key (or set `WANDB_MODE=disabled`) |
| `JOB_ID` | run identifier used in checkpoint paths (set automatically under SLURM) |
| `MEDSAM_CHECKPOINT_DIR`, `MEDSAM_CHECKPOINT` | MedSAM ViT-B weights |
| `NCT_RAW_DATA_DIR`, `NCT_METADATA_PATH`, `EXACTVU_PCA_DATA_ROOT` | NCT2013 data |

Data paths for OPTIMUM, histopathology embeddings and MRI embeddings are set in the YAML configs (`data.root_dir_c3`,
`histo_emb_dir`, ...); edit them to your local copies. No data is included in this repository.

## Training

Stage 1 on NCT2013 (one run per fold, 5 folds), with prompts:

```bash
python -m projects.baseline_methods.guideus.guideus_pnf_train \
    -c projects/baseline_methods/guideus/guideus_cfg0_nct_prompt.yaml fold=0
```

Stage 2 on OPTIMUM, initialised from a stage-1 checkpoint:

```bash
python -m projects.baseline_methods.guideus.guideus_pnf_train \
    -c projects/baseline_methods/guideus/guideus_cfg0_optimum_prompt.yaml fold=0 \
    model_checkpoint=/path/to/stage1/best.pth
```

Any config value can be overridden on the command line as `key=value` (OmegaConf dotlist). The checkpoint is selected on
`val/auc`, the core-level csPCa AUROC of the mean heatmap inside the needle region.

## Notes

- csPCa is ISUP grade group ≥ 2. The auxiliary image-level classification head in `image_clf_mode: cspca` is trained on
  grade group ≥ 3 vs. the rest (`src/loss_new.py`); checkpoint selection uses the heatmap, not this head.
- `guideus_cfg0_nct_prompt_gg3.yaml` instead trains the image-level head on three classes (GG0 / GG1–2 / GG≥3).
  NCT2013 has no GG1 cores, so its middle class is GG2.
