"""Rules a GUIDE-US training config must satisfy to enter the lab benchmark tables (see benchmark/README.md).

`check_training_config(cfg)` raises ProtocolError listing every violation. The trainer calls it when the config sets
`benchmark_protocol: <variant>`, and the exporter calls it on the config of every checkpoint it scores.

Two groups of rules:
* SPLIT_RULES always apply: NCT2013 only, the five benchmark folds, NCT validation selection, no initialisation
  from another trained checkpoint, three-class head. Breaking one of these makes the row incomparable.
* RECIPE_RULES fix the training recipe shared by the benchmark's retrained MedSAM baselines (ProstNFound+,
  ProstNFound, MedSAM-FT, ... : 256-px input, 64-px masks, AdamW schedule, 35 epochs, selection on NCT
  high-involvement AUROC). They apply to the `primary` and `image_only` variants; a `sensitivity` variant may change
  them but must keep SPLIT_RULES, and is reported as a secondary row only.
"""
from __future__ import annotations

VARIANTS = {
    'primary': dict(prompts=['age', 'psa', 'psad', 'pos']),
    'image_only': dict(prompts=[]),
    'sensitivity': dict(prompts=None),  # recipe may differ (e.g. 512 px); never the main row
}

_MISSING = object()

SPLIT_RULES = {
    'experiment_type': 'guideus',
    'train_mode': 'us_histo',
    'model_checkpoint': None,
    'run_val': True,
    'run_test': False,
    'save_best_weights': True,
    'add_image_clf': True,
    'data.dataset': 'nct',
    'data.include_nct': True,
    'data.include_optimum_c23': False,
    'data.include_optimum_c3': False,
    'data.include_pu_in_train': (False, _MISSING),
    'data.cohort_selection_mode': 'kfold',
    'data.n_folds': 5,
    'data.test_center': None,
    'data.exclude_benign_cores_from_positive_patients': True,
    'data.involvement_threshold_pct': None,
    'data.undersample_benign_ratio': None,
    'data.limit_train_data': None,
    'data.frames': 'first',
    'data.flip_ud': False,
    'data.rf_as_bmode': False,
    'data.include_rf': False,
    'model': 'prostnfound_adapter_medsam_legacy',
    'model_kw.use_class_decoder': True,
    'model_kw.class_output_dim': 3,
    'image_clf_mode': ('gg3', 'three_class'),
}

# The GUIDE-US alignment branch as implemented in this repository (frozen in every variant): needle-token ABMIL on the
# 256-d encoder tokens, triplet loss to PANDA (Radboud) slide embeddings. loss_type / lambda_* / use_mmd / supcon_only /
# propbce are not read by guideus_pnf_train.py's GUIDE-US path; they are frozen so that nobody changes them believing
# they switch something.
GUIDEUS_RULES = {
    'input_dim': 256,
    'proj_dim': 768,
    'hidden_dim': 512,
    'use_acmil': False,
    'pooling': ('attention', _MISSING),
    'data_provider': 'radboud',
    'data.hist_data_provider': 'radboud',
    'loss_type': 'triplet',
    'lambda_within': 1.0,
    'lambda_cross': 0.0,
    'lambda_mri': 0.0,
    'use_mmd': False,
    'supcon_only': False,
    'propbce': False,
}

# Keys that may differ between the training run and the export call (machine paths, logging, run naming).
MACHINE_KEYS = ('base_dir', 'chkpt_dir', 'checkpoint_dir', 'histo_emb_dir', 'mri_emb_dir', 'hist_csv', 'mri_csv',
                'data.root_dir_c23', 'data.root_dir_c3', 'data.histo_emb_dir', 'data.mri_emb_dir', 'data.hist_csv',
                'data.mri_csv', 'data.num_workers', 'wandb', 'wandb_url', 'job_id', 'project', 'run_name', 'name',
                'device', 'debug', 'save_checkpoint_wandb', 'evaluator')

RECIPE_RULES = {
    'data.image_size': 256,
    'data.mask_size': 64,
    'data.mean': [0, 0, 0],
    'data.std': [1, 1, 1],
    'data.crop_to_prostate': False,
    'data.first_downsample_size': None,
    'data.augmentations': 'translate',
    'data.batch_size': 8,
    'model_kw.backbone_kw.mask_decoder_kw.output_upscaling_version': 'default',
    'loss': 'mil_prop_bce_entropy_reg',
    'image_clf_class_weight': 'balanced',
    'optimizer': 'adamw',
    'lr': 2.0e-5,
    'encoder_lr': 1.0e-5,
    'warmup_lr': 5.0e-5,
    'cnn_lr': 1.0e-5,
    'warmup_epochs': 1,
    'wd': 1.0e-3,
    'epochs': 35,
    'accumulate_grad_steps': 1,
    'cutoff_epoch': None,
    'freeze_encoder': False,
    'seed': 30,
    'tracked_metric': 'val/auc_high_involvement',
}


class ProtocolError(ValueError):
    pass


def _get(cfg, dotted):
    node = cfg
    for part in dotted.split('.'):
        try:
            node = node[part]
        except Exception:  # missing key (dict or OmegaConf), or indexing into a scalar/list
            return _MISSING
    return node


def _plain(value):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    return value


def _ok(value, expected):
    allowed = expected if isinstance(expected, tuple) else (expected,)
    value = _plain(value)
    for a in allowed:
        if a is _MISSING and value is _MISSING:
            return True
        if isinstance(a, float) and isinstance(value, (int, float)) and not isinstance(value, bool):
            if abs(value - a) <= 1e-12 * max(1.0, abs(a)):
                return True
        elif (a is None and value is None) or (value == a and type(value) is type(a)):
            return True
        elif isinstance(a, list) and isinstance(value, list) and [float(x) for x in value] == [float(x) for x in a]:
            return True
    return False


def variant_of(cfg):
    variant = _plain(_get(cfg, 'benchmark_protocol'))
    if variant not in VARIANTS:
        raise ProtocolError(f'benchmark_protocol must be one of {sorted(VARIANTS)}, got {variant!r}')
    return variant


def violations(cfg):
    variant = variant_of(cfg)
    rules = dict(SPLIT_RULES, **GUIDEUS_RULES)
    if variant != 'sensitivity':
        rules.update(RECIPE_RULES)
    bad = [f'{k} = {_plain(_get(cfg, k))!r}, required {v!r}' for k, v in rules.items() if not _ok(_get(cfg, k), v)]
    prompts = _plain(_get(cfg, 'model_kw.prompts'))
    want = VARIANTS[variant]['prompts']
    if want is not None and prompts != want:
        bad.append(f'model_kw.prompts = {prompts!r}, required {want!r} for variant {variant!r}')
    if variant == 'sensitivity' and prompts not in (VARIANTS['primary']['prompts'], VARIANTS['image_only']['prompts']):
        bad.append(f'model_kw.prompts = {prompts!r}: a sensitivity variant must use the primary or image-only prompts')
    tracked = _plain(_get(cfg, 'tracked_metric'))
    if not (isinstance(tracked, str) and tracked.startswith('val/')):
        bad.append(f'tracked_metric = {tracked!r}: checkpoints must be selected on the NCT validation fold (val/...)')
    fold = _plain(_get(cfg, 'fold'))
    if fold not in range(5) or _plain(_get(cfg, 'data.fold')) != fold:
        bad.append(f'fold = {fold!r}, data.fold = {_plain(_get(cfg, "data.fold"))!r}: both must be the same integer 0-4')
    return bad


def check_training_config(cfg):
    bad = violations(cfg)
    if bad:
        raise ProtocolError('config breaks the benchmark protocol (benchmark/README.md):\n  ' + '\n  '.join(bad))
    return variant_of(cfg)


# --------------------------------------------------------------------------------------- training manifest / checkpoint
def method_settings(cfg):
    """Flat {dotted key: value} of every setting except machine-specific ones (paths, logging, run names)."""
    from omegaconf import OmegaConf
    tree = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    flat = {}

    def walk(node, prefix):
        for k, v in node.items():
            key = f'{prefix}{k}'
            if key in MACHINE_KEYS:
                continue
            if isinstance(v, dict):
                walk(v, key + '.')
            else:
                flat[key] = v
    walk(tree, '')
    return flat


def training_manifest(cfg, loaders, fingerprints=None):
    """Fingerprints of the cores the trainer actually loads; raises ProtocolError if they are not the benchmark's.

    Called by guideus_pnf_train.py before the first epoch. The manifest is stored inside every saved checkpoint and
    verified again by the exporter.
    """
    from medAI.datasets.nct2013.data_access import data_accessor
    from .fingerprint import id_fingerprint, load_fingerprints, nct_data_file_hashes
    expected = fingerprints or load_fingerprints()
    meta = data_accessor.get_metadata_table().set_index('core_id')
    cohorts, bad = {}, []
    for split in ('train', 'val'):
        ds = loaders[split].dataset
        subs = getattr(ds, 'sub_datasets', [(ds, None)])
        if len(subs) != 1 or subs[0][1] is not None or not hasattr(subs[0][0], 'core_ids'):
            raise ProtocolError(f'the {split} loader must hold exactly one NCT2013 dataset (found {len(subs)} sub-datasets)')
        cores = sorted(map(str, subs[0][0].core_ids))
        missing = sorted(set(cores) - set(map(str, meta.index)))
        if missing:
            raise ProtocolError(f'{len(missing)} {split} cores are not NCT2013 cores, e.g. {missing[:3]}')
        cohorts[f'{split}_cores'] = id_fingerprint(cores)
        cohorts[f'{split}_patients'] = id_fingerprint(set(map(str, meta.loc[cores].patient_id)))
    want = expected['nct'][str(cfg.fold)]
    for got_key, want_key in [('train_cores', 'train_cores_after_filter'), ('val_cores', 'val_cores'),
                              ('train_patients', 'train_patients'), ('val_patients', 'val_patients')]:
        if cohorts[got_key] != want[want_key]:
            bad.append(f'NCT fold {cfg.fold} {got_key}: {cohorts[got_key]} != benchmark {want[want_key]}')
    data_files = nct_data_file_hashes()
    bad += [f'{f}: sha256 {data_files.get(f)} != benchmark {h}' for f, h in expected['data_files'].items() if data_files.get(f) != h]
    if bad:
        raise ProtocolError('training data are not the benchmark NCT2013 fold:\n  ' + '\n  '.join(bad))
    return dict(fold=int(cfg.fold), variant=variant_of(cfg), cohorts=cohorts, data_files=data_files)


def verify_checkpoint(cfg, state, run_dir, fingerprints):
    """Problems that keep a checkpoint out of the benchmark (empty list = verified). `cfg` is the config given to the
    exporter (fold already set), `state` the loaded best.pth, `run_dir` its folder."""
    import json
    from pathlib import Path
    from omegaconf import OmegaConf
    problems = []
    bench = state.get('benchmark') if isinstance(state, dict) else None
    if not isinstance(bench, dict) or 'config' not in bench or 'manifest' not in bench:
        return ['checkpoint carries no benchmark record: it was not trained by guideus_pnf_train.py with benchmark_protocol set']
    trained = OmegaConf.create(bench['config'])
    try:
        trained_variant = check_training_config(trained)
    except ProtocolError as e:
        return [f'the config the checkpoint was trained with breaks the protocol: {e}']
    if trained_variant != variant_of(cfg):
        problems.append(f'trained as variant {trained_variant!r}, exported as {variant_of(cfg)!r}')
    a, b = method_settings(trained), method_settings(cfg)
    problems += [f'setting {k}: trained {a.get(k, "<absent>")!r}, export config {b.get(k, "<absent>")!r}'
                 for k in sorted(set(a) | set(b)) if a.get(k, _MISSING) != b.get(k, _MISSING)]
    man = bench['manifest']
    want = fingerprints['nct'][str(cfg.fold)]
    if man.get('fold') != cfg.fold:
        problems.append(f'manifest fold {man.get("fold")} != export fold {cfg.fold}')
    for got_key, want_key in [('train_cores', 'train_cores_after_filter'), ('val_cores', 'val_cores'),
                              ('train_patients', 'train_patients'), ('val_patients', 'val_patients')]:
        if (man.get('cohorts') or {}).get(got_key) != want[want_key]:
            problems.append(f'manifest {got_key} differs from the benchmark fold')
    if man.get('data_files') != fingerprints['data_files']:
        problems.append('manifest NCT data-file hashes differ from the benchmark')
    sd = state.get('model', {})
    if not any(k.startswith('abmil.') for k in sd):
        problems.append('checkpoint has no GUIDE-US ABMIL alignment head (abmil.* tensors)')
    done = Path(run_dir) / 'training_complete.json'
    if not done.is_file():
        problems.append(f'{done} not found: training did not finish')
    else:
        rec = json.loads(done.read_text())
        if not rec.get('completed') or rec.get('epochs_run') != trained.epochs:
            problems.append(f'training incomplete: {rec}')
        if rec.get('best_epoch') != state.get('epoch'):
            problems.append(f'best.pth is from epoch {state.get("epoch")}, the run reports best epoch {rec.get("best_epoch")}')
    return problems
