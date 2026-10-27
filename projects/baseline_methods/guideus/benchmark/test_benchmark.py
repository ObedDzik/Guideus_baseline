"""CPU unit tests of the benchmark code path; no data needed.

    python -m unittest projects.baseline_methods.guideus.benchmark.test_benchmark -v
"""
import contextlib
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image

from . import data as bd
from . import export
from .check_export import main as check_export_main
from .fingerprint import FingerprintError, id_fingerprint, load_fingerprints, sha256_file
from .protocol import ProtocolError, check_training_config, verify_checkpoint
from .vendored.optimum_adapter import _ProstNFoundDatasetAdapterOptimum
from .vendored import prostnfound_transform as pt
from projects.baseline_methods.guideus.src.loss_new import ImageLevelClassificationLoss, ProportionBCE

GUIDEUS = Path(__file__).resolve().parents[1]
PRIMARY = GUIDEUS / 'guideus_cfg_benchmark_nct.yaml'
IMAGE_ONLY = GUIDEUS / 'guideus_cfg_benchmark_nct_image_only.yaml'
os.environ.setdefault('JOB_ID', 'test')
DATA_CFG = OmegaConf.create(dict(image_size=64, mask_size=16, mean=[0, 0, 0], std=[1, 1, 1], flip_ud=False,
                                 crop_to_prostate=False, first_downsample_size=None))


def cfg_for(path, fold=0, **changes):
    cfg = OmegaConf.merge(OmegaConf.load(path), OmegaConf.create({'fold': fold}))
    for k, v in changes.items():
        OmegaConf.update(cfg, k, v, force_add=True)
    return cfg


class TestProtocol(unittest.TestCase):
    def test_shipped_configs_pass_for_every_fold(self):
        for fold in range(5):
            self.assertEqual(check_training_config(cfg_for(PRIMARY, fold)), 'primary')
            self.assertEqual(check_training_config(cfg_for(IMAGE_ONLY, fold)), 'image_only')

    def test_split_violations_fail_in_every_variant(self):
        for variant in ['primary', 'sensitivity']:
            for key, value in [('data.include_optimum_c3', True), ('data.include_pu_in_train', True),
                               ('data.n_folds', 4), ('data.cohort_selection_mode', 'nested_kfold'),
                               ('model_checkpoint', '/x/best.pth'), ('tracked_metric', 'test/auc'),
                               ('model_kw.class_output_dim', 2), ('data.exclude_benign_cores_from_positive_patients', False),
                               ('data.flip_ud', True), ('experiment_type', 'pnf')]:
                with self.subTest(variant=variant, key=key), self.assertRaises(ProtocolError):
                    check_training_config(cfg_for(PRIMARY, benchmark_protocol=variant, **{key: value}))

    def test_recipe_is_fixed_except_in_sensitivity_variant(self):
        with self.assertRaises(ProtocolError):
            check_training_config(cfg_for(PRIMARY, **{'data.image_size': 512}))
        with self.assertRaises(ProtocolError):
            check_training_config(cfg_for(PRIMARY, lr=1e-4))
        self.assertEqual(check_training_config(cfg_for(PRIMARY, benchmark_protocol='sensitivity', **{'data.image_size': 512})),
                         'sensitivity')

    def test_prompts_must_match_the_variant(self):
        with self.assertRaises(ProtocolError):
            check_training_config(cfg_for(PRIMARY, **{'model_kw.prompts': ['age', 'psa']}))
        with self.assertRaises(ProtocolError):
            check_training_config(cfg_for(IMAGE_ONLY, **{'model_kw.prompts': ['age', 'psa', 'psad', 'pos']}))

    def test_guideus_branch_is_frozen_in_every_variant(self):
        for variant in ['primary', 'sensitivity']:
            for key, value in [('use_acmil', True), ('proj_dim', 512), ('train_mode', 'us_only'), ('data_provider', 'karolinska')]:
                with self.subTest(variant=variant, key=key), self.assertRaises(ProtocolError):
                    check_training_config(cfg_for(PRIMARY, benchmark_protocol=variant, **{key: value}))

    def test_unknown_variant_and_fold_mismatch_fail(self):
        with self.assertRaises(ProtocolError):
            check_training_config(cfg_for(PRIMARY, benchmark_protocol='main'))
        with self.assertRaises(ProtocolError):
            check_training_config(cfg_for(PRIMARY, **{'data.fold': 3}))


def optimum_item(gg=float('nan'), psa=float('nan'), age=61.0, sample='RBL', center='UA'):
    img = np.zeros((40, 60, 3), np.uint8)
    img[:20] = 200  # bright top half on disk
    needle = np.zeros((40, 60), np.uint8)
    needle[2:8, 10:50] = 255  # needle near the top on disk
    prostate = np.zeros((40, 60), np.uint8)
    prostate[0:30, 5:55] = 255
    info = {'cine_id': f'{center}-001-001', 'case': f'{center}-001', 'center': center, 'Sample ID': sample, 'GG': gg,
            '% Cancer': float('nan'), 'Diagnosis': 'Benign', 'psa': psa, 'age': age}
    return {'image': Image.fromarray(img), 'needle_mask': Image.fromarray(needle),
            'prostate_mask': Image.fromarray(prostate), 'info': info}


class TestInputs(unittest.TestCase):
    def test_adapter_flips_every_optimum_center_and_keeps_nan(self):
        for center in ['OL', 'UA', 'PU']:
            out = _ProstNFoundDatasetAdapterOptimum('probe_top')(optimum_item(center=center))
            self.assertEqual(out['bmode'][-1, 0], 200)  # bright half now at the bottom
            self.assertEqual(out['bmode'][0, 0], 0)
            self.assertTrue(out['needle_mask'][-5].any() and not out['needle_mask'][5].any())
            self.assertTrue(math.isnan(out['psa']) and math.isnan(out['approx_psa_density']))
            self.assertEqual(out['grade_group'], 0)

    def test_transform_imputes_with_nct_means_and_encodes_location(self):
        t = bd.eval_transform(DATA_CFG)
        out = t(_ProstNFoundDatasetAdapterOptimum('probe_top')(optimum_item(sample='RBL')))
        self.assertAlmostEqual(out['psa'].item(), (pt.psa_avg - pt.psa_min) / (pt.psa_max - pt.psa_min), places=6)
        self.assertAlmostEqual(out['approx_psa_density'].item(), (pt.approx_psa_density_avg - pt.approx_psa_density_min)
                               / (pt.approx_psa_density_max - pt.approx_psa_density_min), places=6)
        self.assertAlmostEqual(out['age'].item(), (61 - pt.age_min) / (pt.age_max - pt.age_min), places=6)
        self.assertEqual(out['base_apex_encoding'].item(), -1.0)  # B = base
        self.assertEqual(out['mid_lateral_encoding'].item(), -1.0)  # L = lateral
        self.assertEqual(tuple(out['bmode'].shape), (3, 64, 64))
        self.assertEqual(tuple(out['needle_mask'].shape), (1, 16, 16))
        self.assertAlmostEqual(out['bmode'].max().item(), 1.0)  # per-image min-max scaling

    def test_digest_detects_a_vertical_flip(self):
        t = bd.eval_transform(DATA_CFG)
        good = bd.collate([t(_ProstNFoundDatasetAdapterOptimum('probe_top')(optimum_item()))])
        bad = bd.collate([t(_ProstNFoundDatasetAdapterOptimum('none')(optimum_item()))])
        d1, d2 = bd.InputDigest(), bd.InputDigest()
        d1.update(good)
        d2.update(bad)
        diff = bd.compare_digest(d2.result(), d1.result())
        self.assertIn('bmode_top_minus_bottom', diff)
        self.assertIn('needle_centroid_y', diff)
        self.assertEqual(bd.compare_digest(d1.result(), d1.result()), {})

    def test_id_fingerprint(self):
        a = bd.id_fingerprint(['b', 'a', 'c'])
        self.assertEqual(a, bd.id_fingerprint(['c', 'a', 'b']))
        self.assertEqual(a['n'], 3)
        with self.assertRaises(AssertionError):
            bd.id_fingerprint(['a', 'a'])


class TestLosses(unittest.TestCase):
    def test_three_class_labels_and_balanced_weights(self):
        gg = torch.tensor([0, 0, 0, 0, 2, 2, 3, 5])
        logits = torch.randn(8, 3)
        data = {'grade_group': gg, 'image_level_classification_outputs': [logits], 'label': (gg > 0).long()}
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
        w = torch.tensor([8 / (3 * 4), 8 / (3 * 2), 8 / (3 * 2)])
        for mode in ['gg3', 'three_class']:
            got = ImageLevelClassificationLoss(mode, class_weight='balanced')(data)
            self.assertTrue(torch.allclose(got, torch.nn.functional.cross_entropy(logits, labels, weight=w)))
        plain = ImageLevelClassificationLoss('gg3')(data)
        self.assertTrue(torch.allclose(plain, torch.nn.functional.cross_entropy(logits, labels)))

    def test_balanced_weight_ignores_absent_classes(self):
        gg = torch.tensor([0, 0, 3])
        logits = torch.randn(3, 3)
        data = {'grade_group': gg, 'image_level_classification_outputs': [logits], 'label': (gg > 0).long()}
        w = torch.tensor([3 / (3 * 2), 1.0, 3 / (3 * 1)])
        got = ImageLevelClassificationLoss('gg3', class_weight='balanced')(data)
        self.assertTrue(torch.allclose(got, torch.nn.functional.cross_entropy(logits, torch.tensor([0, 0, 2]), weight=w)))

    def test_proportion_bce_matches_naive_formula_and_is_finite_when_saturated(self):
        torch.manual_seed(0)
        for _ in range(5):
            logits = torch.randn(17, dtype=torch.float64) * 2
            q = torch.rand((), dtype=torch.float64)
            p = logits.sigmoid()
            naive = -q * p.mean().log() - (1 - q) * (1 - p.mean()).log()
            naive = naive + 0.01 * (-p * p.log() - (1 - p) * (1 - p).log()).mean()
            got = ProportionBCE(entropy_penalty_lambda=0.01)(logits.float(), q.float())
            self.assertAlmostEqual(got.item(), naive.item(), places=5)
        saturated = torch.full((9,), 30.0, dtype=torch.float16)
        self.assertTrue(torch.isfinite(ProportionBCE(entropy_penalty_lambda=0.01)(saturated, torch.tensor(0.2))))


class FakeModel(torch.nn.Module):
    prompts = ['age', 'psa']

    def forward(self, image, rf, prostate_mask, needle_mask, output_mode=None, **prompts):
        b = image.shape[0]
        assert set(prompts) == set(self.prompts) and all(v.shape == (b, 1) for v in prompts.values())
        mask_logits = torch.zeros(b, 1, *needle_mask.shape[-2:])
        cls = torch.stack([torch.tensor([2.0, 0.0, 1.0])] * b)
        return dict(mask_logits=mask_logits, cls_outputs=[cls])


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, empty_needle_at=()):
        self.empty = set(empty_needle_at)

    def __len__(self):
        return 3

    def __getitem__(self, i):
        needle = torch.zeros(1, 8, 8)
        if i not in self.empty:
            needle[0, 2:4, 2:6] = 1
        return dict(bmode=torch.rand(3, 32, 32), prostate_mask=torch.ones(1, 8, 8), needle_mask=needle,
                    age=torch.tensor([0.5]), psa=torch.tensor([0.1]), core_id=f'X-{i}', patient_id='X',
                    center='UA', grade_group=float(i), involvement=torch.tensor(0.0))


class TestExport(unittest.TestCase):
    def test_predict_scores(self):
        table, digest = export.predict(FakeModel(), torch.tensor([1.0]), torch.tensor([0.0]), FakeDataset(empty_needle_at=[1]),
                                       [0, 1, 2], 'cpu', batch_size=2, num_workers=0)
        p = torch.softmax(torch.tensor([2.0, 0.0, 1.0]), 0).numpy()
        np.testing.assert_allclose(table.p_benign, p[0], rtol=1e-6)
        np.testing.assert_allclose(table.pca, 1 - p[0], rtol=1e-6)
        np.testing.assert_allclose(table.gg3, p[2], rtol=1e-6)
        np.testing.assert_allclose(table.heatmap[[0, 2]], 0.5)  # sigmoid(0) inside the region
        self.assertTrue(np.isnan(table.heatmap[1]))  # empty needle-prostate intersection
        self.assertEqual(list(table.core_id), ['X-0', 'X-1', 'X-2'])
        self.assertIsNotNone(digest['bmode_mean'])

    def test_check_table_flags_bad_probabilities_and_pu_labels(self):
        from .check_export import check_table
        t = pd.DataFrame(dict(core_id=['a'], patient_id=['p'], center=['UA'], grade_group=[0.0], involvement=[0.0],
                              pca=[0.3], gg3=[0.1], p_benign=[0.7], p_gg12=[0.2], p_gg3=[0.1], heatmap=[np.nan]))
        self.assertEqual(check_table('ok', t, labelled=True), [])
        bad = t.copy()
        bad['p_gg3'] = 0.3
        self.assertTrue(check_table('bad', bad, labelled=True))
        self.assertTrue(check_table('pu', t, labelled=False))  # label columns in a PU file


DIGEST = {'bmode_mean': 0.3, 'needle_centroid_y': 0.4}
IDENT = {'n': 1, 'sha256': '0' * 64}


def nct_cores(fold):
    return [f'C{fold}-{i:04d}_LBM' for i in range(6)]


def external_cores(cohort):
    return ['PU-001-001', 'PU-002-001'] if cohort == 'pu' else ['UA-001-001', 'UA-001-002', 'OL-002-001']


def patients(cores, cohort):
    return {c.split('_')[0] if cohort == 'nct' else '-'.join(c.split('-')[:2]) for c in cores}


def synthetic_fingerprints():
    fp = dict(preprocessing=dict(image_size=256), data_files={'nct2013/metadata_with_approx_psa_density.csv': 'a',
                                                              'nct2013/needle_mask.npy': 'b'}, nct={})
    for f in range(5):
        fp['nct'][str(f)] = dict(val_cores=id_fingerprint(nct_cores(f)), val_patients=id_fingerprint(patients(nct_cores(f), 'nct')),
                                 train_patients=IDENT, train_cores_after_filter=IDENT, digest=DIGEST)
    for c in ['optimum', 'pu']:
        fp[c] = dict(cores=id_fingerprint(external_cores(c)), patients=id_fingerprint(patients(external_cores(c), c)), digest=DIGEST)
    return fp


class TestCheckpointVerification(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name)
        self.cfg = cfg_for(PRIMARY, 0)
        self.fp = synthetic_fingerprints()
        want = self.fp['nct']['0']
        self.state = dict(model={'model.w': torch.zeros(1), 'abmil.proj.weight': torch.zeros(1)}, epoch=3, best_score=0.8,
                          benchmark=dict(config=OmegaConf.to_container(self.cfg, resolve=True), manifest=dict(
                              fold=0, variant='primary', data_files=self.fp['data_files'],
                              cohorts=dict(train_cores=want['train_cores_after_filter'], val_cores=want['val_cores'],
                                           train_patients=want['train_patients'], val_patients=want['val_patients']))))
        self.done(completed=True, epochs_run=35, best_epoch=3)

    def tearDown(self):
        self.tmp.cleanup()

    def done(self, **rec):
        (self.run_dir / 'training_complete.json').write_text(json.dumps(rec))

    def problems(self, state=None):
        return verify_checkpoint(self.cfg, state or self.state, self.run_dir, self.fp)

    def test_verified_checkpoint(self):
        self.assertEqual(self.problems(), [])

    def test_checkpoint_without_benchmark_record_or_abmil_head(self):
        self.assertTrue(self.problems({k: v for k, v in self.state.items() if k != 'benchmark'}))
        self.assertTrue(self.problems(dict(self.state, model={'model.w': torch.zeros(1)})))

    def test_trained_config_is_checked_not_only_the_export_config(self):
        for key, value in [('data.include_optimum_c3', True), ('model_checkpoint', '/x/optimum/best.pth'), ('lambda_aem', 0.2)]:
            trained = OmegaConf.create(self.state['benchmark']['config'])
            OmegaConf.update(trained, key, value, force_add=True)
            bench = dict(self.state['benchmark'], config=OmegaConf.to_container(trained))
            with self.subTest(key=key):
                self.assertTrue(self.problems(dict(self.state, benchmark=bench)))

    def test_manifest_must_be_the_benchmark_fold(self):
        man = json.loads(json.dumps(self.state['benchmark']['manifest']))
        man['cohorts']['val_cores'] = IDENT
        self.assertTrue(self.problems(dict(self.state, benchmark=dict(self.state['benchmark'], manifest=man))))

    def test_training_must_have_finished_and_best_epoch_must_match(self):
        self.done(completed=False, epochs_run=20, best_epoch=3)
        self.assertTrue(self.problems())
        self.done(completed=True, epochs_run=35, best_epoch=7)
        self.assertTrue(self.problems())
        (self.run_dir / 'training_complete.json').unlink()
        self.assertTrue(self.problems())


class TestCheckExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'export'
        self.fp_path = Path(self.tmp.name) / 'fingerprints.json'
        self.fp_path.write_text(json.dumps(synthetic_fingerprints()))
        for f in range(5):
            d = self.root / f'fold{f}'
            d.mkdir(parents=True)
            cohorts = {}
            for cohort, fname in export.SPLIT_FILES.items():
                cores = nct_cores(f) if cohort == 'nct' else external_cores(cohort)
                t = pd.DataFrame(dict(core_id=cores, patient_id=[c.split('_')[0] if cohort == 'nct' else '-'.join(c.split('-')[:2]) for c in cores],
                                      center=[c[:2] for c in cores]))
                if cohort != 'pu':
                    t['grade_group'] = [0.0, 3.0, 2.0, 0.0, 4.0, 0.0][:len(t)]
                    t['involvement'] = 0.0
                t['pca'], t['gg3'], t['p_benign'], t['p_gg12'], t['p_gg3'] = 0.4, 0.1, 0.6, 0.3, 0.1
                t['heatmap'] = 0.2
                t.loc[0, ['pca', 'gg3', 'p_benign', 'p_gg12', 'p_gg3']] = [0.9, 0.5, 0.1, 0.4, 0.5]
                t.to_csv(d / fname, index=False)
                cohorts[cohort] = dict(sha256=sha256_file(d / fname), digest=DIGEST)
            (d / 'provenance.json').write_text(json.dumps(dict(
                fold=f, accepted_as_benchmark_result=True, problems=[], smoke=0, digest_tolerance=1e-4,
                protocol=dict(checked=True, checkpoint_verified=True, variant='primary'), config=dict(sha256='c'),
                checkpoint=dict(sha256=f'k{f}'), cohorts=cohorts, git={})))
            (d / 'COMPLETE').write_text('success\n')

    def tearDown(self):
        self.tmp.cleanup()

    def run_check(self, fingerprints=None):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                check_export_main([str(self.root), '--fingerprints', str(fingerprints or self.fp_path)])
                code = 0
            except SystemExit as e:
                code = e.code
        return code, out.getvalue()

    def rewrite(self, fold, fname, edit):
        """Edit a CSV and refresh its recorded checksum, so that only the identity checks can catch the edit."""
        d = self.root / f'fold{fold}'
        t = pd.read_csv(d / fname)
        edit(t)
        t.to_csv(d / fname, index=False)
        meta = json.loads((d / 'provenance.json').read_text())
        meta['cohorts'][{v: k for k, v in export.SPLIT_FILES.items()}[fname]]['sha256'] = sha256_file(d / fname)
        (d / 'provenance.json').write_text(json.dumps(meta))

    def test_pass(self):
        code, out = self.run_check()
        self.assertEqual(code, 0, out)
        self.assertIn('PASS', out)

    def test_altered_identity_fails_even_with_refreshed_checksum(self):
        self.rewrite(2, 'optimum_predictions.csv', lambda t: t.__setitem__('core_id', t.core_id.str.replace('UA-001-002', 'UA-001-009')))
        self.assertEqual(self.run_check()[0], 1)

    def test_core_attached_to_wrong_patient_fails(self):
        self.rewrite(1, 'pu_predictions.csv', lambda t: t.__setitem__('patient_id', t.patient_id[::-1].values))
        self.assertEqual(self.run_check()[0], 1)

    def test_core_with_wrong_center_fails(self):
        self.rewrite(0, 'optimum_predictions.csv', lambda t: t.__setitem__('center', ['OL'] * len(t)))  # UA cores marked OL
        self.assertEqual(self.run_check()[0], 1)

    def test_file_edited_after_export_fails(self):
        p = self.root / 'fold3' / 'val_predictions.csv'
        p.write_text(p.read_text().replace('0.4', '0.41', 1))
        code, out = self.run_check()
        self.assertEqual(code, 1)
        self.assertIn('changed after export', out)

    def test_missing_or_incomplete_fingerprints_fail(self):
        self.assertEqual(self.run_check(Path(self.tmp.name) / 'absent.json')[0], 1)
        bad = Path(self.tmp.name) / 'bad.json'
        bad.write_text(json.dumps({k: v for k, v in synthetic_fingerprints().items() if k != 'pu'}))
        self.assertEqual(self.run_check(bad)[0], 1)
        with self.assertRaises(FingerprintError):
            load_fingerprints(bad)

    def test_unverified_or_loosely_checked_fold_fails(self):
        d = self.root / 'fold4'
        meta = json.loads((d / 'provenance.json').read_text())
        meta['digest_tolerance'] = 0.02
        (d / 'provenance.json').write_text(json.dumps(meta))
        self.assertEqual(self.run_check()[0], 1)


if __name__ == '__main__':
    unittest.main()
