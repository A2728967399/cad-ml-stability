import copy
import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "analysis/audit_training_events.py"
spec = importlib.util.spec_from_file_location("training_events", SCRIPT)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def warnings(total=0, convergence=0, future=0, user=0):
    categories = {name: value for name, value in [("ConvergenceWarning", convergence), ("FutureWarning", future), ("UserWarning", user)] if value}
    return {"total": total, "convergence": convergence, "categories": categories, "examples": []}


def representation(warning=None, additional=0):
    return {"method": "nested LASSO 1SE", "warnings": warning or warnings(),
            "path": {"fold_auc": [[.5, .6], [.5, .7]]}, "empty_model_fallback_steps": additional,
            "preprocessing": {"n_fit": 40}, "selected": ["x"], "C_initial_1se": .1, "C": 1. if additional else .1,
            "lasso_cv_folds": [{"warnings": warnings(2, convergence=2)}]}


class EventCounts(unittest.TestCase):
    def setUp(self):
        self.context = {"run": "main", "split": 0, "checkpoint": "split_000", "development": "holdout_final"}
        self.development = {
            "warnings": [],  # Verifier/audit empty list is not fitting evidence.
            "final_representation": representation(warnings(3, convergence=1, future=2), 2),
            "model_cv_folds": [{"fold": 0, "representation": representation()}],
            "candidates": {"M": [
                {"params": {"C": 1}, "mean_auc": .6, "fold_auc": [.5, .7], "warnings": warnings(2, convergence=2)},
                {"params": {"C": 2}, "mean_auc": .8, "fold_auc": [.7, .9], "warnings": warnings(3, convergence=3)}]},
        }
        self.metrics = [{"model": "M", "scheme": "retuned", "params": {"C": 2}, "fit_warnings": warnings(1, user=1), "calibration_converged": True}]

    def test_disjoint_aggregate_buckets_not_parent_child_or_cache_recounts(self):
        buckets, fallbacks = [], []
        q.collect_development(self.development, self.context, ["M"], self.metrics, buckets, fallbacks)
        total = q.tally(buckets, fallbacks)
        self.assertEqual(total["warning_events"], 9)
        self.assertEqual(total["convergence_warning_events"], 6)
        self.assertEqual(total["future_warning_events"], 2)
        self.assertEqual(total["completed_development_fit_calls"], 17)
        self.assertEqual(total["lasso_additional_refits"], 2)
        self.assertEqual(total["lasso_empty_fallback_selectors"], 1)

    def test_selected_unselected_cv_and_final_are_separate(self):
        buckets, fallbacks = [], []
        q.collect_development(self.development, self.context, ["M"], self.metrics, buckets, fallbacks)
        total = q.tally(buckets, fallbacks)
        self.assertEqual(total["selected_candidate_cv_convergence_warning_events"], 3)
        self.assertEqual(total["unselected_candidate_cv_convergence_warning_events"], 2)
        self.assertEqual(total["final_classifier_convergence_warning_events_recorded"], 0)
        self.assertEqual(total["lasso_convergence_warning_events_aggregate"], 1)

    def test_oof_final_warning_absence_is_unknown_not_zero(self):
        buckets, fallbacks = [], []
        q.collect_development(self.development, {**self.context, "development": "oof_fold_0"}, ["M"], None, buckets, fallbacks)
        total = q.tally(buckets, fallbacks)
        self.assertEqual(total["unrecorded_oof_final_fit_calls"], 1)
        final = [row for row in buckets if row["stage"] == "final_classifier"][0]
        self.assertFalse(final["record_available"])
        self.assertIsNone(final["warning_events"])
        final_summary = q.warning_tables(buckets)[2][0]
        self.assertIsNone(final_summary["warning_events"])

    def test_parameter_selection_mismatch_fails(self):
        corrupted = copy.deepcopy(self.metrics)
        corrupted[0]["params"] = {"C": 1}
        with self.assertRaisesRegex(AssertionError, "Selected CV candidate"):
            q.collect_development(self.development, self.context, ["M"], corrupted, [], [])

    def test_all_features_empty_placeholder_is_not_lasso_fit(self):
        buckets, fallbacks = [], []
        q.collect_representation({"method": "all eligible features", "warnings": warnings()}, self.context, "full", buckets, fallbacks)
        self.assertEqual(buckets, [])
        self.assertEqual(fallbacks, [])

    def test_warning_category_total_mismatch_fails(self):
        with self.assertRaisesRegex(AssertionError, "sum to total"):
            q.validate_warning_record(warnings(3, convergence=1))

    def test_log_copy_does_not_invent_retry_and_warning_not_failure(self):
        text = "[2026-01-01T00:00:00] split_001 started\nConvergenceWarning: lbfgs failed to converge"
        a = q.scan_log_text(text, "main", "a.log")
        b = q.scan_log_text(text, "main", "copy.log")
        self.assertEqual(q.observed_restarts(a + b), [])
        self.assertEqual(a[1]["signal"], "library_warning_text")
        c = q.scan_log_text("[2026-01-01T01:00:00] split_001 started", "main", "c.log")
        self.assertEqual(q.observed_restarts(a + b + c)[0]["n_distinct_starts"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
