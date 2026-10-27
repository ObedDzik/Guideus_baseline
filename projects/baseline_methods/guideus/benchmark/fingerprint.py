"""Pure helpers of the benchmark (no medAI imports): batching, cohort fingerprints and input digests."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

PROMPT_KEYS = ['age', 'psa', 'approx_psa_density', 'base_apex_encoding', 'mid_lateral_encoding']
FINGERPRINTS = Path(__file__).resolve().parent / 'fingerprints.json'
NCT_DATA_FILES = ['nct2013/metadata_with_approx_psa_density.csv', 'nct2013/needle_mask.npy']


class FingerprintError(ValueError):
    pass


def load_fingerprints(path=None):
    """Expected cohorts of the benchmark; refuses to continue when the file is missing or incomplete."""
    path = Path(path or FINGERPRINTS)
    if not path.is_file():
        raise FingerprintError(f'{path} not found: the benchmark checks cannot run without it')
    fp = json.loads(path.read_text())
    ident = lambda x: isinstance(x, dict) and set(x) == {'n', 'sha256'}
    try:
        ok = all(all(ident(fp['nct'][f][k]) for k in ['val_cores', 'val_patients', 'train_patients', 'train_cores_after_filter'])
                 and fp['nct'][f]['digest'] for f in map(str, range(5)))
        ok = ok and all(ident(fp[c]['cores']) and ident(fp[c]['patients']) and fp[c]['digest'] for c in ['optimum', 'pu'])
        ok = ok and set(fp['data_files']) == set(NCT_DATA_FILES) and all(fp['data_files'].values()) and bool(fp['preprocessing'])
    except (KeyError, TypeError) as e:
        raise FingerprintError(f'{path} is incomplete (missing {e!r})') from None
    if not ok:
        raise FingerprintError(f'{path} is incomplete')
    return fp


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def nct_data_file_hashes(root=None):
    """SHA-256 of the NCT2013 files that define the split and the needle mask (None when not found)."""
    root = root or os.environ.get('EXACTVU_PCA_DATA_ROOT')
    return {rel: (sha256_file(Path(root) / rel) if root and (Path(root) / rel).is_file() else None) for rel in NCT_DATA_FILES}


def center_of(core_id):
    """Center code implied by a core identifier (prefix before the first '-', in every cohort)."""
    return str(core_id).split('-')[0]


def patient_of(core_id, cohort):
    """Patient identifier implied by a core identifier (NCT: CENTER-NNNN_LOC; OPTIMUM/PU: CC-NNN-NNN)."""
    core_id = str(core_id)
    return core_id.split('_')[0] if cohort == 'nct' else '-'.join(core_id.split('-')[:2])


# ------------------------------------------------------------------------------------------------ batching
def collate(items):
    """Stack tensors (prompts as [B, 1]); keep identifiers as strings and labels as floats."""
    keep = ['bmode', 'prostate_mask', 'needle_mask', *PROMPT_KEYS]
    out = {}
    for k in keep:
        if k in items[0]:
            out[k] = torch.stack([torch.as_tensor(i[k]).float().reshape(torch.as_tensor(i[k]).shape or (1,)) for i in items])
    for k in ['core_id', 'patient_id', 'center']:
        out[k] = [str(i[k]) for i in items]
    out['grade_group'] = [float(i['grade_group']) for i in items]
    out['involvement'] = [float(i['involvement']) if 'involvement' in i else float('nan') for i in items]
    return out


# ------------------------------------------------------------------------------------------------ fingerprints
def id_fingerprint(ids):
    """Count and SHA-256 of the sorted, newline-joined identifiers (same rule as the benchmark's own exports)."""
    ids = sorted(map(str, ids))
    assert len(ids) == len(set(ids)), 'duplicate identifiers'
    return dict(n=len(ids), sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest())


class InputDigest:
    """Cohort-level averages of the model inputs: catch a wrong flip, mask source, crop or prompt imputation.

    Only cohort means are kept (no per-core values). The image is split into top and bottom halves and the mask
    centroids are measured in normalised coordinates, so a vertical flip changes the digest.
    """
    FIELDS = ['bmode_mean', 'bmode_top_minus_bottom', 'needle_fraction', 'needle_centroid_y', 'needle_centroid_x',
              'prostate_fraction', 'prostate_centroid_y', 'prostate_centroid_x', 'needle_and_prostate_fraction',
              *[f'prompt_{k}' for k in PROMPT_KEYS]]

    def __init__(self):
        self.sums = {k: 0.0 for k in self.FIELDS}
        self.counts = {k: 0 for k in self.FIELDS}

    def _add(self, key, values):
        values = values[torch.isfinite(values)]
        self.sums[key] += float(values.double().sum())
        self.counts[key] += int(values.numel())

    @staticmethod
    def _centroid(mask):
        b, _, h, w = mask.shape
        m = mask[:, 0].double()
        area = m.sum((1, 2))
        ys = torch.linspace(0, 1, h, dtype=torch.float64)[None, :, None]
        xs = torch.linspace(0, 1, w, dtype=torch.float64)[None, None, :]
        cy = (m * ys).sum((1, 2)) / area
        cx = (m * xs).sum((1, 2)) / area
        return torch.where(area > 0, cy, torch.nan), torch.where(area > 0, cx, torch.nan)

    def update(self, batch):
        b = batch['bmode'][:, 0].double()
        h = b.shape[1]
        self._add('bmode_mean', b.mean((1, 2)))
        self._add('bmode_top_minus_bottom', b[:, : h // 2].mean((1, 2)) - b[:, h // 2:].mean((1, 2)))
        needle, prostate = batch['needle_mask'] > 0.5, batch['prostate_mask'] > 0.5
        self._add('needle_fraction', needle.double().mean((1, 2, 3)))
        self._add('prostate_fraction', prostate.double().mean((1, 2, 3)))
        self._add('needle_and_prostate_fraction', (needle & prostate).double().mean((1, 2, 3)))
        for name, mask in [('needle', needle), ('prostate', prostate)]:
            cy, cx = self._centroid(mask)
            self._add(f'{name}_centroid_y', cy)
            self._add(f'{name}_centroid_x', cx)
        for k in PROMPT_KEYS:
            if k in batch:
                self._add(f'prompt_{k}', batch[k].reshape(-1))

    def result(self):
        return {k: (round(self.sums[k] / self.counts[k], 8) if self.counts[k] else None) for k in self.FIELDS}


def compare_digest(observed, expected, tol=1e-4):
    """Fields whose cohort means differ by more than `tol` (absolute)."""
    bad = {}
    for k, v in expected.items():
        o = observed.get(k)
        if (v is None) != (o is None) or (v is not None and not np.isclose(o, v, rtol=0, atol=tol)):
            bad[k] = dict(observed=o, expected=v)
    return bad
