"""Meaningful leakage boundaries and integrity checks; no clinical estimates."""
import importlib.util
import ast
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "nested_pipeline.py"
spec = importlib.util.spec_from_file_location("nested_pipeline", SCRIPT)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


class NestedBoundaries(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(777)
        self.X = pd.DataFrame({"signal": rng.normal(size=120), "noise": rng.normal(size=120),
                               "missing": np.r_[np.full(21, np.nan), rng.normal(size=99)]}, index=np.arange(500, 620))
        self.y = (self.X.signal.to_numpy() + rng.normal(scale=.6, size=120) > 0).astype(int)
        self.config = {"all_features": False, "lasso_cs": [.01, .1, 1., 100.], "lasso_folds": 2, "model_folds": 2}

    def test_holdout_values_do_not_change_fitted_preprocessing_or_selection(self):
        tr, va = np.arange(90), np.arange(90, 120)
        prep1, selected1, audit1 = q.fit_representation(self.X.iloc[tr], self.y[tr], self.config, features=list(self.X))
        perturbed = self.X.copy()
        perturbed.iloc[va] = 1e10
        y2 = self.y.copy()
        y2[va] = 1 - y2[va]
        prep2, selected2, audit2 = q.fit_representation(perturbed.iloc[tr], y2[tr], self.config, features=list(self.X))
        self.assertEqual(selected1, selected2)
        self.assertEqual(q.canonical(audit1), q.canonical(audit2))
        np.testing.assert_array_equal(prep1.transform(self.X.iloc[tr]), prep2.transform(self.X.iloc[tr]))

    def test_missing_rate_boundary_uses_only_current_training_fold(self):
        prep = q.FoldPreprocessor(list(self.X)).fit(self.X.iloc[:90])
        self.assertNotIn("missing", prep.kept)  # 21/90 > .15
        another = q.FoldPreprocessor(list(self.X)).fit(self.X.iloc[30:])
        self.assertIn("missing", another.kept)  # 0/90
        self.assertNotEqual(prep.meta["fit_rows_sha256"], another.meta["fit_rows_sha256"])

    def test_lasso_cv_preprocessing_rows_match_own_training_members(self):
        _, _, audit = q.fit_representation(self.X, self.y, self.config, features=list(self.X))
        for fold in audit["lasso_cv_folds"]:
            self.assertEqual(fold["train_members_sha256"], fold["preprocessing"]["fit_rows_sha256"])
            self.assertNotEqual(fold["train_members_sha256"], fold["valid_members_sha256"])
            self.assertEqual(fold["preprocessing"]["n_fit"], 60)

    def test_model_cv_feature_selection_uses_only_inner_training_members(self):
        cache, audit = q.prepare_inner_folds(self.X, self.y, self.config, features=list(self.X))
        self.assertEqual(len(cache), 2)
        for matrices, fold in zip(cache, audit):
            representation = fold["representation"]
            self.assertEqual(fold["train_members_sha256"], representation["preprocessing"]["fit_rows_sha256"])
            self.assertEqual(representation["preprocessing"]["n_fit"], 60)
            self.assertEqual(matrices[0].shape[1], len(representation["selected"]))
            for lasso_fold in representation["lasso_cv_folds"]:
                self.assertEqual(lasso_fold["preprocessing"]["n_fit"], 30)

    def test_inner_validation_perturbation_does_not_change_its_selection(self):
        cv = q.StratifiedKFold(2, shuffle=True, random_state=q.SEED)
        tr, va = next(cv.split(self.X, self.y))
        _, chosen1, audit1 = q.fit_representation(self.X.iloc[tr], self.y[tr], self.config, features=list(self.X))
        altered = self.X.copy()
        altered.iloc[va] = np.nan
        _, chosen2, audit2 = q.fit_representation(altered.iloc[tr], self.y[tr], self.config, features=list(self.X))
        self.assertEqual(chosen1, chosen2)
        self.assertEqual(q.canonical(audit1), q.canonical(audit2))

    def test_subtype_cannot_affect_split_or_candidate_features(self):
        a = self.X.assign(**{"冠心病类型": np.arange(120) % 4})
        b = self.X.assign(**{"冠心病类型": np.arange(120)[::-1] % 3})
        tr1, te1, meta1 = q.split_members(a, self.y, 0)
        tr2, te2, meta2 = q.split_members(b, self.y, 0)
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(te1, te2)
        self.assertEqual(meta1, meta2)
        self.assertNotIn("冠心病类型", q.FEATURES)
        self.assertNotIn("patient_ID", q.FEATURES)

    def test_full_grid_preserves_113_candidates_and_49_features(self):
        sizes = {name: len(q.ParameterGrid(grid)) for name, (_, grid) in q.MODELS.items()}
        self.assertEqual(list(sizes.values()), [4, 8, 6, 16, 6, 32, 32, 9])
        self.assertEqual(sum(sizes.values()), 113)
        self.assertEqual(len(q.FEATURES), 49)

    def test_frozen_source_constants_and_grids_match_historical_script(self):
        historical = SCRIPT.parents[2] / "legacy_clinical" / "reanalysis" / "scripts" / "01_q3_analysis.py"
        nodes = ast.parse(historical.read_text(encoding="utf-8-sig")).body
        assignments = {target.id: node.value for node in nodes if isinstance(node, ast.Assign)
                       for target in node.targets if isinstance(target, ast.Name)}
        self.assertEqual(ast.literal_eval(assignments["FEATURES_49"]), q.FEATURES)
        self.assertEqual(ast.literal_eval(assignments["RANGE"]), q.RANGES)
        self.assertEqual(set(ast.literal_eval(assignments["CATEGORICAL"])), q.CATEGORICAL)
        old_models = assignments["MODELS"].args[0]  # inspect AST, never import side-effectful old module
        grids = {ast.literal_eval(key): ast.literal_eval(value.elts[1])
                 for key, value in zip(old_models.keys, old_models.values)}
        self.assertEqual(grids, {name: grid for name, (_, grid) in q.MODELS.items()})

    def test_modern_lasso_api_is_exactly_equivalent_to_historical_penalty(self):
        from sklearn.linear_model import LogisticRegression
        X = self.X[["signal", "noise"]].to_numpy()
        for c in [.01, .1, 1., 10.]:
            with warnings.catch_warnings(record=True) as old_warnings:
                warnings.simplefilter("always")
                old = LogisticRegression(C=c, penalty="l1", solver="liblinear", max_iter=5000, random_state=42).fit(X, self.y)
            with warnings.catch_warnings(record=True) as new_warnings:
                warnings.simplefilter("always")
                new = LogisticRegression(C=c, l1_ratio=1., solver="liblinear", max_iter=5000, random_state=42).fit(X, self.y)
            np.testing.assert_array_equal(old.coef_, new.coef_)
            np.testing.assert_array_equal(old.intercept_, new.intercept_)
            self.assertGreater(len(old_warnings), 0)
            self.assertEqual(len(new_warnings), 0)

    def test_resume_fails_closed_on_changed_config(self):
        with tempfile.TemporaryDirectory() as folder:
            config = {"code_sha256": "original", "grid": [1, 2]}
            q.initialize_run(folder, config, {})
            q.initialize_run(folder, config, {})
            with self.assertRaisesRegex(RuntimeError, "Resume refused"):
                q.initialize_run(folder, {**config, "code_sha256": "changed"}, {})

    def test_checkpoint_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            q.atomic_json(root / "data.json", {"value": 1})
            q.atomic_json(root / "manifest.json", {"run_hash": "known", "state": "complete", "output_sha256": {"data.json": q.file_hash(root / "data.json")}})
            q.verify_checkpoint(root, "known")
            q.atomic_json(root / "data.json", {"value": 2})
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                q.verify_checkpoint(root, "known")


if __name__ == "__main__":
    unittest.main(verbosity=2)
