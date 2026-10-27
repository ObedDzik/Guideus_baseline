"""Check a complete five-fold GUIDE-US export before handing it to the benchmark.

    python -m projects.baseline_methods.guideus.benchmark.check_export /path/to/guideus_benchmark_export

Prints PASS only if every fold was exported from a verified benchmark checkpoint, and if the files themselves (not only
their provenance) match the benchmark: identifiers and patients recomputed from each CSV equal the expected cohort,
every core belongs to the patient its identifier names, files are unchanged since export (SHA-256), the recorded input
digests agree with the benchmark's, probabilities are coherent, the NCT validation folds partition the cohort and every
fold model scored the same OPTIMUM and PU cores. As a sanity check it reports NCT validation AUROC (selection folds);
it deliberately computes nothing on OPTIMUM or PU, which must not inform any decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .export import LABEL_COLUMNS, SCORE_COLUMNS, SPLIT_FILES
from .fingerprint import (FINGERPRINTS, FingerprintError, center_of, compare_digest, id_fingerprint, load_fingerprints,
                          patient_of, sha256_file)

ID_COLUMNS = ['core_id', 'patient_id', 'center']
CENTERS = {'nct': None, 'optimum': {'OL', 'UA'}, 'pu': {'PU'}}


def auroc(y, s):
    """Mann-Whitney AUROC with ties counted as one half (no sklearn dependency)."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def check_table(name, t, labelled):
    """Schema and probability coherence of one prediction file."""
    errors = []
    want = ID_COLUMNS + (LABEL_COLUMNS if labelled else []) + SCORE_COLUMNS
    if list(t.columns) != want:
        return [f'{name}: columns {list(t.columns)} != {want}']
    if t.core_id.duplicated().any():
        errors.append(f'{name}: duplicated core_id')
    p = t[['p_benign', 'p_gg12', 'p_gg3']].to_numpy(float)
    if not np.isfinite(p).all() or not np.isfinite(t[['pca', 'gg3']].to_numpy(float)).all():
        errors.append(f'{name}: non-finite classifier probabilities')
    elif (p < 0).any() or (p > 1).any() or np.abs(p.sum(1) - 1).max() > 1e-5:
        errors.append(f'{name}: class probabilities outside [0, 1] or not summing to 1')
    elif np.abs(t.pca - (t.p_gg12 + t.p_gg3)).max() > 1e-6 or np.abs(t.gg3 - t.p_gg3).max() > 1e-7:
        errors.append(f'{name}: pca/gg3 are not P(GG1-2)+P(GG>=3) / P(GG>=3)')
    h = t.heatmap.to_numpy(float)
    if ((h[np.isfinite(h)] < 0) | (h[np.isfinite(h)] > 1)).any():
        errors.append(f'{name}: heatmap outside [0, 1]')
    return errors


def check_identities(name, t, cohort, expected_cores, expected_patients):
    """Identifiers recomputed from the file must be the benchmark cohort, and each core must belong to its patient and
    center. (Exported label columns are informational; the benchmark joins its own labels for every cohort.)"""
    errors = []
    if id_fingerprint(t.core_id) != expected_cores:
        errors.append(f'{name}: core identifiers {id_fingerprint(t.core_id)} != benchmark {expected_cores}')
    if id_fingerprint(set(t.patient_id.astype(str))) != expected_patients:
        errors.append(f'{name}: patient identifiers differ from the benchmark')
    implied = t.core_id.map(lambda c: patient_of(c, cohort))
    if (implied != t.patient_id.astype(str)).any():
        errors.append(f'{name}: {(implied != t.patient_id.astype(str)).sum()} cores attached to the wrong patient')
    wrong_center = t.core_id.map(center_of) != t.center.astype(str)
    if wrong_center.any():
        errors.append(f'{name}: {wrong_center.sum()} cores with a center other than the one in their identifier')
    if CENTERS[cohort] is not None and not set(t.center.astype(str)) <= CENTERS[cohort]:
        errors.append(f'{name}: unexpected centers {sorted(set(t.center.astype(str)) - CENTERS[cohort])}')
    return errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('export_root')
    ap.add_argument('--fingerprints', default=str(FINGERPRINTS), help=argparse.SUPPRESS)  # for tests
    args = ap.parse_args(argv)
    root = Path(args.export_root)
    errors, prov, tables = [], {}, {}
    try:
        expected = load_fingerprints(args.fingerprints)
    except FingerprintError as e:
        print('FAIL\n  -', e)
        sys.exit(1)

    for fold in range(5):
        d = root / f'fold{fold}'
        if not (d / 'COMPLETE').is_file() or (d / 'COMPLETE').read_text().strip() != 'success':
            errors.append(f'fold{fold}: missing or non-success COMPLETE marker')
            continue
        meta = json.loads((d / 'provenance.json').read_text())
        prov[fold] = meta
        protocol = meta.get('protocol') or {}
        if not meta.get('accepted_as_benchmark_result') or meta.get('problems') or meta.get('smoke') \
                or not protocol.get('checked') or not protocol.get('checkpoint_verified'):
            errors.append(f'fold{fold}: not accepted (problems={meta.get("problems")}, smoke={meta.get("smoke")}, '
                          f'protocol checked={protocol.get("checked")}, checkpoint verified={protocol.get("checkpoint_verified")})')
        if meta.get('fold') != fold:
            errors.append(f'fold{fold}: provenance fold {meta.get("fold")}')
        tol = meta.get('digest_tolerance')
        if tol is None or tol > (2e-2 if protocol.get('variant') == 'sensitivity' else 1e-4):
            errors.append(f'fold{fold}: digest tolerance {tol} too loose for variant {protocol.get("variant")}')
        for cohort, fname in SPLIT_FILES.items():
            rec = (meta.get('cohorts') or {}).get(cohort) or {}
            if not (d / fname).is_file():
                errors.append(f'fold{fold}/{fname}: missing')
                continue
            if rec.get('sha256') != sha256_file(d / fname):
                errors.append(f'fold{fold}/{fname}: file changed after export (SHA-256 differs from provenance)')
            t = pd.read_csv(d / fname, dtype={'core_id': str, 'patient_id': str, 'center': str})
            tables[fold, cohort] = t
            errors += check_table(f'fold{fold}/{fname}', t, labelled=cohort != 'pu')
            exp = expected['nct'][str(fold)] if cohort == 'nct' else expected[cohort]
            errors += check_identities(f'fold{fold}/{fname}', t, cohort, exp['val_cores' if cohort == 'nct' else 'cores'],
                                       exp['val_patients' if cohort == 'nct' else 'patients'])
            bad = compare_digest(rec.get('digest') or {}, exp['digest'], tol=tol if tol is not None else 0.0)
            if bad:
                errors.append(f'fold{fold}/{fname}: recorded input digest differs from the benchmark: {sorted(bad)}')

    if len(prov) == 5:
        variants = {(m.get('protocol') or {}).get('variant') for m in prov.values()}
        configs = {m['config']['sha256'] for m in prov.values()}
        ckpts = [m['checkpoint']['sha256'] for m in prov.values()]
        if len(variants) != 1:
            errors.append(f'folds were exported under different protocol variants {variants}')
        if len(configs) != 1:
            errors.append('folds were exported with different config files')
        if len(set(ckpts)) != 5:
            errors.append('two folds share a checkpoint')
    if len(tables) == 15:
        val_ids = [set(tables[f, 'nct'].core_id) for f in range(5)]
        val_pat = [set(tables[f, 'nct'].patient_id) for f in range(5)]
        if any(val_ids[i] & val_ids[j] or val_pat[i] & val_pat[j] for i in range(5) for j in range(i + 1, 5)):
            errors.append('NCT validation folds overlap')
        for cohort in ['optimum', 'pu']:
            if len({tuple(sorted(tables[f, cohort].core_id)) for f in range(5)}) != 1:
                errors.append(f'{cohort}: fold models scored different cores')

    print(f'export: {root}')
    if prov:
        print(f'variant: {sorted({str((m.get("protocol") or {}).get("variant")) for m in prov.values()})}, git: '
              f'{sorted({str((m.get("git") or {}).get("commit"))[:12] for m in prov.values()})}')
    for fold in range(5):
        if (fold, 'nct') in tables:
            t = tables[fold, 'nct']
            gg = t.grade_group.to_numpy(float)
            print(f'fold{fold}: NCT validation (selection fold, all {len(t)} cores) AUROC all-cancer {auroc(gg >= 1, t.pca):.4f}, '
                  f'csPCa {auroc(gg >= 3, t.gg3):.4f}; heatmap NaN cores NCT/OPTIMUM/PU '
                  + '/'.join(str(int(tables[fold, c].heatmap.isna().sum())) for c in SPLIT_FILES if (fold, c) in tables))
    if errors:
        print('FAIL')
        for e in errors:
            print('  -', e)
        sys.exit(1)
    print('PASS: hand the whole export folder to the benchmark maintainers (do not commit it to git).')


if __name__ == '__main__':
    main()
