"""Export per-core predictions of one GUIDE-US fold model for the lab benchmark.

One call scores one NCT fold model on (1) its own NCT2013 validation fold, (2) every OPTIMUM OL+UA core and (3) every
PU core, through the benchmark inputs of `benchmark/data.py`, and writes

    <out>/fold<k>/val_predictions.csv       NCT validation fold (labels included)
    <out>/fold<k>/optimum_predictions.csv   OPTIMUM OL+UA, zero-shot (labels included)
    <out>/fold<k>/pu_predictions.csv        PU, zero-shot (no label columns; the benchmark joins PU labels itself)
    <out>/fold<k>/provenance.json           config, checkpoint hash, cohort fingerprints, input digests, versions
    <out>/fold<k>/COMPLETE

No fitting, calibration, threshold or checkpoint choice happens here. Scores: pca = P(GG1-2) + P(GG>=3) = 1 - P(benign)
and gg3 = P(GG>=3) from the three-class image-level classifier; `heatmap` is the mean heatmap probability inside the
needle-prostate intersection (secondary, reported for completeness).

    python -m projects.baseline_methods.guideus.benchmark.export \
        --config projects/baseline_methods/guideus/guideus_cfg_benchmark_nct.yaml --fold 0 \
        --checkpoint /path/to/fold0/best.pth --out /path/to/guideus_benchmark_export
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from . import data as bd
from .fingerprint import load_fingerprints, nct_data_file_hashes, sha256_file
from .protocol import ProtocolError, check_training_config, verify_checkpoint

HERE = Path(__file__).resolve().parent
SPLIT_FILES = {'nct': 'val_predictions.csv', 'optimum': 'optimum_predictions.csv', 'pu': 'pu_predictions.csv'}
LABEL_COLUMNS = ['grade_group', 'involvement']
SCORE_COLUMNS = ['pca', 'gg3', 'p_benign', 'p_gg12', 'p_gg3', 'heatmap']
SCORES = dict(
    pca='P(GG>=1) = P(GG1-2) + P(GG>=3) from the three-class classifier (all-cancer score)',
    gg3='P(GG>=3) from the three-class classifier (csPCa score)',
    p_benign='softmax probability of class 0 (benign)', p_gg12='softmax probability of class 1 (GG1-2)',
    p_gg3='softmax probability of class 2 (GG>=3)',
    heatmap='mean sigmoid heatmap probability inside needle AND prostate mask (NaN when the intersection is empty)')


def _plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def git_state():
    root = HERE.parents[3]
    try:
        commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout.strip()
        return dict(commit=commit, dirty=bool(dirty), dirty_files=dirty.splitlines()[:50])
    except Exception as e:  # not a git checkout
        return dict(commit=None, error=str(e))


# ----------------------------------------------------------------------------------------------------- model
def load_model(cfg, state, device):
    """ProstNFound part of the trained GUIDE-US meta-model (the ABMIL alignment head is training-only)."""
    from medAI.modeling.registry import create_model
    import medAI.modeling.prostnfound  # noqa: F401  registers the factories

    sd = state['model'] if 'model' in state else state
    model_sd = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')}
    other = sorted(k for k in sd if not k.startswith(('model.', 'abmil.')) and k not in ('temperature', 'bias'))
    assert not other, f'unexpected checkpoint keys {other[:10]}'
    model = create_model(cfg.model, **OmegaConf.to_container(cfg.model_kw, resolve=True))
    model.load_state_dict(model_sd, strict=True)
    temperature = sd.get('temperature', torch.tensor([1.0])).float().to(device)
    bias = sd.get('bias', torch.tensor([0.0])).float().to(device)
    info = dict(epoch=state.get('epoch') if isinstance(state, dict) else None,
                best_score=float(state['best_score']) if isinstance(state, dict) and state.get('best_score') is not None else None,
                temperature=temperature.cpu().tolist(), bias=bias.cpu().tolist(),
                n_abmil_tensors=sum(k.startswith('abmil.') for k in sd))
    model.to(device)
    model.eval()  # ProstNFound.eval() returns None, so no chaining
    model.requires_grad_(False)
    return model, temperature, bias, info


@torch.no_grad()
def predict(model, temperature, bias, dataset, indices, device, batch_size, num_workers, require_classifier=True):
    """Full-precision (no autocast) forward pass; returns a per-core table and the input digest."""
    rows, digest = [], bd.InputDigest()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=bd.collate)
    for batch in loader:
        digest.update(batch)
        bmode = batch['bmode'].to(device)
        prostate, needle = batch['prostate_mask'].to(device), batch['needle_mask'].to(device)
        prompts = {k: batch[k].to(device) for k in model.prompts}
        out = model(bmode, None, prostate, needle, output_mode='all', **prompts)
        logits = out['mask_logits'].float() / temperature[None, None, None, :] + bias[None, None, None, :]
        assert logits.shape[-2:] == needle.shape[-2:], ('heatmap and mask resolutions differ', logits.shape, needle.shape)
        region = ((prostate > 0.5) & (needle > 0.5)).float()
        area = region.sum((1, 2, 3))
        heat = (logits.sigmoid() * region).sum((1, 2, 3)) / area.clamp_min(1)
        heat = torch.where(area > 0, heat, torch.full_like(heat, float('nan')))
        frame = pd.DataFrame(dict(core_id=batch['core_id'], patient_id=batch['patient_id'], center=batch['center'],
                                  grade_group=batch['grade_group'], involvement=batch['involvement']))
        cls = out.get('cls_outputs')
        if cls is not None:
            assert cls[0].shape[-1] == 3, f'expected a three-class classifier, got {cls[0].shape[-1]} outputs'
            p = torch.softmax(cls[0].float(), dim=-1).cpu().numpy()
            frame['p_benign'], frame['p_gg12'], frame['p_gg3'] = p[:, 0], p[:, 1], p[:, 2]
            frame['pca'], frame['gg3'] = p[:, 1] + p[:, 2], p[:, 2]
        else:
            assert not require_classifier, 'the benchmark row is scored by the three-class classifier'
            for c in ['p_benign', 'p_gg12', 'p_gg3', 'pca', 'gg3']:
                frame[c] = np.nan
        frame['heatmap'] = heat.cpu().numpy()
        rows.append(frame)
    table = pd.concat(rows, ignore_index=True)
    assert not table.core_id.duplicated().any()
    return table, digest.result()


# ----------------------------------------------------------------------------------------------------- checks
def check_cohort(name, observed, expected, smoke):
    """Compare a cohort fingerprint with the benchmark's; returns a list of problems."""
    if smoke:
        return []
    if expected is None:
        return [f'{name}: no benchmark fingerprint to compare with']
    return [f'{name}: observed {observed}, expected {expected}'] if observed != expected else []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--config', required=True, help='training config of the checkpoint (the GUIDE-US benchmark yaml)')
    ap.add_argument('--fold', type=int, required=True, choices=range(5))
    ap.add_argument('--checkpoint', required=True, help='best.pth selected on the NCT validation fold by training')
    ap.add_argument('--out', required=True, help='export root; fold<k>/ is created inside and must not exist yet')
    ap.add_argument('--cohorts', default='nct,optimum,pu')
    ap.add_argument('--optimum-root', dest='optimum_root', help='OPTIMUM export with OL-*/UA-*/PU-* case folders '
                    '(default: $BENCHMARK_OPTIMUM_ROOT)')
    ap.add_argument('--optimum-prostate-mask-root', dest='optimum_prostate_mask_root',
                    help='extra UA prostate masks (default: $BENCHMARK_OPTIMUM_PROSTATE_MASK_ROOT)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--smoke', type=int, default=0, help='score only the first N cores per cohort (output is marked smoke)')
    ap.add_argument('--allow-mismatch', action='store_true', help='write outputs even if cohort/input checks fail '
                    '(marked; never accepted as benchmark results)')
    ap.add_argument('--skip-protocol-check', action='store_true', help='for reference/parity runs of other configs '
                    '(marked; never accepted as benchmark results)')
    args = ap.parse_args(argv)

    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, OmegaConf.create({'fold': args.fold}))
    os.environ.setdefault('JOB_ID', 'export')  # only checkpoint_dir uses it, which the export ignores
    expected = load_fingerprints()  # fail closed: no checks, no export
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    protocol = dict(checked=not args.skip_protocol_check)
    if args.skip_protocol_check:
        protocol['checkpoint_verified'] = False
    else:
        protocol['variant'] = check_training_config(cfg)
        problems = verify_checkpoint(cfg, state, Path(args.checkpoint).absolute().parent, expected)
        if problems:
            raise ProtocolError('this checkpoint cannot enter the benchmark:\n  ' + '\n  '.join(problems))
        protocol['checkpoint_verified'] = True
        protocol['training_manifest'] = state['benchmark']['manifest']

    out = Path(args.out) / f'fold{args.fold}'
    if out.exists():
        raise FileExistsError(f'{out} exists; exports are write-once, use a new --out')
    out.mkdir(parents=True)
    try:
        problems = export_fold(args, cfg, protocol, out, state, expected)
    except BaseException:
        print('export failed; partial output moved to', set_aside(out), file=sys.stderr)
        raise
    if problems:
        moved = set_aside(out)
        raise SystemExit(f'cohort/input checks failed; diagnostics in {moved}/provenance.json ("problems"):\n  '
                         + '\n  '.join(problems))


def set_aside(out):
    """Keep a failed fold folder for inspection under a new name, so that a corrected re-run can use fold<k>/."""
    moved = out.with_name(f'{out.name}.failed-{time.strftime("%Y%m%d-%H%M%S")}')
    out.rename(moved)
    return moved


def export_fold(args, cfg, protocol, out, state, expected):
    """Score every requested cohort into `out`; returns the list of check problems (empty when accepted)."""
    cohorts = [c.strip() for c in args.cohorts.split(',') if c.strip()]
    assert set(cohorts) <= set(SPLIT_FILES), cohorts
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model, temperature, bias, ckpt_info = load_model(cfg, state, device)
    roots = bd.data_roots(args)
    meta = dict(method='GUIDE-US', fold=args.fold, smoke=args.smoke, protocol=protocol, argv=sys.argv,
                config=dict(path=str(Path(args.config).resolve()), sha256=sha256_file(args.config),
                            resolved=OmegaConf.to_container(cfg, resolve=False)),
                checkpoint=dict(path=str(Path(args.checkpoint).resolve()), sha256=sha256_file(args.checkpoint), **ckpt_info),
                prompts=list(model.prompts), scores=SCORES, git=git_state(), data_roots=roots,
                data_files=nct_data_file_hashes(), vendored={p.name: sha256_file(p) for p in sorted((HERE / 'vendored').glob('*.py'))},
                versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__),
                device=str(device), settings=dict(
                    nct='NCT2013 validation fold of this checkpoint (the fold that selected it; not an independent test)',
                    optimum='OPTIMUM OL+UA, zero-shot: no fitting, calibration, threshold or selection',
                    pu='PU, zero-shot: no fitting, calibration, threshold or selection'),
                cohorts={}, problems=[])
    problems = []
    # Digests were computed with the benchmark preprocessing; another input size (sensitivity variant) still has to
    # agree closely (a flip or a different mask source moves centroids and halves by far more than 0.02).
    same_preprocessing = all(_plain(cfg.data.get(k)) == v for k, v in (expected.get('preprocessing') or {}).items())
    digest_tol = 1e-4 if same_preprocessing else 2e-2
    meta['digest_tolerance'] = digest_tol
    for f, want in (expected.get('data_files') or {}).items():
        if not args.smoke and meta['data_files'].get(f) != want:
            problems.append(f'data file {f}: sha256 {meta["data_files"].get(f)} != {want}')

    for cohort in cohorts:
        t0 = time.time()
        if cohort == 'nct':
            ds, info = bd.nct_val_dataset(args.fold, cfg.data)
            exp = (expected.get('nct') or {}).get(str(args.fold), {})
            fps = dict(val_cores=bd.id_fingerprint(info['val_cores']), val_patients=bd.id_fingerprint(info['val_patients']),
                       train_patients=bd.id_fingerprint(info['train_patients']))
            for k, v in fps.items():
                problems += check_cohort(f'nct fold{args.fold} {k}', v, exp.get(k), args.smoke)
        else:
            assert roots['optimum'], 'set --optimum-root or $BENCHMARK_OPTIMUM_ROOT'
            centers = bd.OPTIMUM_CENTERS if cohort == 'optimum' else bd.PU_CENTERS
            ds = bd.optimum_family_dataset(roots['optimum'], centers, cfg.data, roots['optimum_prostate_masks'])
            exp = expected.get(cohort) or {}
            fps = dict(cores=bd.id_fingerprint(ds.core_ids), patients=bd.id_fingerprint(set(ds.patient_ids)))
            for k, v in fps.items():
                problems += check_cohort(f'{cohort} {k}', v, exp.get(k), args.smoke)
        indices = list(range(len(ds)))[: args.smoke] if args.smoke else list(range(len(ds)))
        table, digest = predict(model, temperature, bias, ds, indices, device, args.batch_size, args.num_workers,
                                require_classifier=not args.skip_protocol_check)
        if cohort == 'nct':
            assert not set(table.patient_id) & set(info['train_patients']), 'scored a training patient'
        if not args.smoke and exp.get('digest'):
            bad = bd.compare_digest(digest, exp['digest'], tol=digest_tol)
            problems += [f'{cohort} input digest {k}: {v}' for k, v in bad.items()]
        columns = ['core_id', 'patient_id', 'center'] + (LABEL_COLUMNS if cohort != 'pu' else []) + SCORE_COLUMNS
        table[columns].to_csv(out / SPLIT_FILES[cohort], index=False, float_format='%.9g')
        meta['cohorts'][cohort] = dict(file=SPLIT_FILES[cohort], sha256=sha256_file(out / SPLIT_FILES[cohort]),
                                       n_scored=len(table), n_patients=int(table.patient_id.nunique()),
                                       fingerprints=fps, digest=digest, n_nan_heatmap=int(table.heatmap.isna().sum()),
                                       n_nonfinite_classifier=int((~np.isfinite(table[['pca', 'gg3']].values)).any(1).sum()),
                                       seconds=round(time.time() - t0, 1))
        print(cohort, {k: meta['cohorts'][cohort][k] for k in ['n_scored', 'n_patients', 'n_nan_heatmap']}, flush=True)

    meta['problems'] = problems
    meta['accepted_as_benchmark_result'] = bool(not problems and not args.smoke and protocol['checked']
                                                and protocol.get('checkpoint_verified') and set(cohorts) == set(SPLIT_FILES))
    (out / 'provenance.json').write_text(json.dumps(meta, indent=1, default=str) + '\n')
    if problems and not args.allow_mismatch:
        return problems
    (out / 'COMPLETE').write_text('smoke\n' if args.smoke else 'success\n')
    print('wrote', out, 'accepted' if meta['accepted_as_benchmark_result'] else 'NOT accepted as a benchmark result')
    return []




if __name__ == '__main__':
    main()
