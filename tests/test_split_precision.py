"""Synthetic tests only; no clinical or aggregate study files are opened."""
import importlib.util
import itertools
from pathlib import Path
import unittest

import numpy as np
from scipy.stats import spearmanr

spec = importlib.util.spec_from_file_location("split_precision", Path(__file__).resolve().parents[1] / "tools/split_precision.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20731)
        self.a = self.rng.uniform(.4, .8, (12, 8))
        self.b = self.rng.uniform(.4, .8, (12, 8))

    def test_rank_kernel_matches_scipy_with_ties(self):
        self.a[0, :3] = .65
        kernel, _ = module.correlation_kernel(self.a)
        for i, j in itertools.combinations(range(12), 2):
            self.assertAlmostEqual(kernel[i, j], spearmanr(self.a[i], self.a[j]).statistic, places=14)

    def test_explicit_deletion(self):
        kernel, _ = module.correlation_kernel(self.a)
        result, leave_one = module.jackknife_pair_mean(kernel)
        direct = []
        for i in range(12):
            selected = np.delete(np.arange(12), i)
            small = kernel[np.ix_(selected, selected)]
            direct.append(small[np.triu_indices(11, 1)].mean())
        np.testing.assert_allclose(leave_one, direct, atol=1e-14)
        self.assertAlmostEqual(result["mcse"], np.sqrt(11 * np.var(direct, ddof=0)), places=14)

    def test_identical_strategies_zero_paired_error(self):
        result, _ = module.paired_precision(self.a, self.a)
        self.assertEqual(result["fixed_minus_retuned"]["estimate"], 0)
        self.assertEqual(result["fixed_minus_retuned"]["mcse"], 0)

    def test_contrast_recomputes_joint_deletions(self):
        result, kernels = module.paired_precision(self.a, self.b)
        _, left = module.jackknife_pair_mean(kernels[0])
        _, right = module.jackknife_pair_mean(kernels[1])
        direct = np.sqrt(11 * np.var(left - right, ddof=0))
        self.assertAlmostEqual(result["fixed_minus_retuned"]["mcse"], direct, places=14)

    def test_common_split_permutation_invariant(self):
        original, _ = module.paired_precision(self.a, self.b)
        order = self.rng.permutation(12)
        reordered, _ = module.paired_precision(self.a[order], self.b[order])
        for name in ("fixed", "retuned", "fixed_minus_retuned"):
            for field in ("estimate", "mcse"):
                self.assertAlmostEqual(original[name][field], reordered[name][field], places=14)

    def test_global_rank_reversal_invariant(self):
        first, _ = module.correlation_kernel(self.a)
        second, _ = module.correlation_kernel(1 - self.a)
        np.testing.assert_allclose(first, second, atol=1e-14)

    def test_constant_ranking_precision(self):
        matrix = np.tile(np.linspace(.4, .8, 8), (12, 1))
        result, _ = module.paired_precision(matrix, matrix)
        self.assertAlmostEqual(result["fixed"]["estimate"], 1)
        self.assertAlmostEqual(result["fixed"]["mcse"], 0)
        self.assertFalse(result["fixed"]["near_degenerate_first_order_warning"])

    def test_degenerate_influence_is_flagged(self):
        # Regular cycle: identical row sums but nonconstant pair entries.
        kernel = np.eye(6)
        for i in range(6):
            kernel[i, (i + 1) % 6] = kernel[(i + 1) % 6, i] = 1
        result, _ = module.jackknife_pair_mean(kernel)
        self.assertTrue(result["near_degenerate_first_order_warning"])

    def test_invalid_inputs_rejected(self):
        for matrix in (np.ones((4, 8)), np.full((4, 8), np.nan), np.full((4, 8), 2)):
            with self.assertRaises(ValueError):
                module.correlation_kernel(matrix)
        with self.assertRaises(ValueError):
            module.paired_precision(self.a, self.b[:10])
        with self.assertRaises(ValueError):
            module.jackknife_pair_mean(np.eye(3))

    def test_generated_macro_namespace(self):
        report, _ = module.paired_precision(self.a, self.b)
        tex = module.tex_macros(report)
        self.assertEqual(tex.count(r"\csname result@precision-"), 6)
        self.assertNotIn("val@", tex)


if __name__ == "__main__":
    unittest.main()
