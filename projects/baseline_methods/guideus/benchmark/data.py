"""Cohorts and inputs of the lab's core-level micro-ultrasound benchmark.

Every row of the benchmark tables (MedSAM-UNETR, MedSAM-FT, MicroSegNet, ProstNFound, ProstNFound+, Prost-RL, ...) was
scored on exactly these inputs, so a new baseline is comparable only if it is scored through this module:

* NCT2013, five-fold patient-level cross-validation: `medAI` kfold cohort selection (StratifiedKFold by center, seed 0),
  benign cores of cancer patients removed from training only, first frame, no vertical flip.
* OPTIMUM OL+UA and PU (external, zero-shot): `frames/0000.png` of every cine with a non-empty needle mask and a
  non-empty prostate mask (UA prostate masks may live in a separate root), flipped vertically ("probe_top") for every
  center, PU included; missing PSA / age / PSA density stay NaN and are imputed by the transform with the NCT means.
* One transform for all cohorts (vendored `ProstNFoundTransform`), with the checkpoint's own image size, mask size,
  mean and std.

The vendored files under `vendored/` are verbatim copies of the code that produced the other rows; do not edit them.
"""
from __future__ import annotations

import os
from pathlib import Path

from torchvision.transforms import v2 as T

from .vendored.optimum_adapter import _ProstNFoundDatasetAdapterOptimum
from .vendored.optimum_dataset import OptimumNeedleCoreDataset
from .vendored.prostnfound_transform import ProstNFoundTransform
from .fingerprint import PROMPT_KEYS, collate, compare_digest, id_fingerprint, InputDigest  # noqa: F401  (re-exported)
OPTIMUM_CENTERS = ('OL', 'UA')
PU_CENTERS = ('PU',)
N_FOLDS = 5
# Preprocessing keys taken from the checkpoint's own training config (as for every other row).
PREPROCESSING_KEYS = ['image_size', 'mask_size', 'mean', 'std', 'flip_ud', 'crop_to_prostate', 'first_downsample_size']


def eval_transform(data_cfg):
    """Evaluation transform with the checkpoint's own preprocessing (no augmentation)."""
    assert not data_cfg.get('flip_ud', False), 'the benchmark protocol uses flip_ud: false for NCT'
    return ProstNFoundTransform(
        augment='none',
        image_size=data_cfg.image_size,
        mask_size=data_cfg.mask_size,
        mean=data_cfg.mean,
        std=data_cfg.std,
        crop_to_prostate=data_cfg.get('crop_to_prostate', False),
        first_downsample_size=data_cfg.get('first_downsample_size', None),
        return_raw_images=False,
        grade_group_for_positive_label=1,
    )


# ----------------------------------------------------------------------------------------------------------- NCT2013
def nct_split(fold):
    """(train_core_ids, val_core_ids) of benchmark fold `fold`; training cores already filtered as in training."""
    from medAI.datasets.nct2013.cohort_selection import select_cohort
    train, val, test = select_cohort(fold=fold, n_folds=N_FOLDS, test_center=None,
                                     exclude_benign_cores_from_positive_patients=True, involvement_threshold_pct=None,
                                     undersample_benign_ratio=None, mode='kfold')
    assert not test, 'kfold mode has no test split'
    return list(map(str, train)), list(map(str, val))


def nct_patients(core_ids):
    from medAI.datasets.nct2013.data_access import data_accessor
    meta = data_accessor.get_metadata_table().set_index('core_id').loc[list(core_ids)]
    return sorted(set(map(str, meta.patient_id)))


def nct_val_dataset(fold, data_cfg):
    """NCT2013 validation fold `fold` (every validation core, first frame), plus the training patients for checks."""
    from medAI.datasets.nct2013.bmode_dataset import BModeDatasetV1
    train, val = nct_split(fold)
    train_patients, val_patients = nct_patients(train), nct_patients(val)
    assert not set(train_patients) & set(val_patients), 'NCT train/validation patient overlap'
    ds = BModeDatasetV1(val, eval_transform(data_cfg), rf_as_bmode=False, include_rf=False, flip_ud=False, frames='first')
    assert len(ds) == len(val), 'expected one frame per core'
    return ds, dict(train_patients=train_patients, val_patients=val_patients, val_cores=val)


# ------------------------------------------------------------------------------------------------ OPTIMUM / PU
def optimum_family_dataset(root, centers, data_cfg, prostate_mask_root=None):
    """OPTIMUM-family cores (OL/UA or PU) exactly as the benchmark loader builds them."""
    root = Path(root)
    cases = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name[:2] in centers)
    transform = T.Compose([_ProstNFoundDatasetAdapterOptimum(image_orientation='probe_top'), eval_transform(data_cfg)])
    ds = OptimumNeedleCoreDataset(root_dir=str(root), case_ids=cases, transform=transform, needle_mask_fname='needle_mask.png',
                                  prostate_mask_root=str(prostate_mask_root) if prostate_mask_root else None,
                                  require_prostate_mask=True, require_nonempty_masks=True)
    assert len(ds.core_ids) == len(ds) and len(set(ds.core_ids)) == len(ds), 'expected one frame per core'
    assert set(ds.centers) <= set(centers), (set(ds.centers), centers)
    return ds


def data_roots(args=None):
    """Data roots from CLI arguments or environment variables (see benchmark/README.md)."""
    get = lambda name, env: getattr(args, name, None) or os.environ.get(env)
    roots = dict(nct=os.environ.get('EXACTVU_PCA_DATA_ROOT'),
                 optimum=get('optimum_root', 'BENCHMARK_OPTIMUM_ROOT'),
                 optimum_prostate_masks=get('optimum_prostate_mask_root', 'BENCHMARK_OPTIMUM_PROSTATE_MASK_ROOT'))
    return roots
