"""Artificial fixtures only; these tests never run clinical or model training."""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "synthetic_smoke.py"
specification = importlib.util.spec_from_file_location("synthetic_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(specification)
specification.loader.exec_module(smoke)


class SyntheticSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cad_ml_tool_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "nested_pipeline.py"
        features = ["性别", "年龄", "室壁运动评分", "总胆固醇", "高密度脂蛋白",
                    "低密度脂蛋白", "dNLR", "TC-HDLDL"] + [f"artificial_{i}" for i in range(41)]
        self.source.write_text(
            "raise RuntimeError('AST inspection must not execute this line')\n"
            f"FEATURES = {features!r}\n"
            "CATEGORICAL = {'性别', '室壁运动评分'}\n"
            "RANGES = {'artificial_0': (0, 100)}\n"
            f"MODELS = {dict.fromkeys([f'model_{i}' for i in range(8)])!r}\n",
            encoding="utf-8")
        self.spec = smoke.read_spec(self.source)

    def csv(self, path, rows):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def fixture_outputs(self):
        output = self.root / "run"
        checkpoint = output / "checkpoints" / "split_000"
        checkpoint.mkdir(parents=True)
        run_hash = "artificial-run-hash"
        (output / "run_manifest.json").write_text(json.dumps({"run_hash": run_hash, "config": {
            "mode": "smoke", "publication_eligible": False, "historical_params_sha256": None,
            "primary_seed42": False, "strict_primary_oof_requested": False}}), encoding="utf-8")
        (output / "run_status.json").write_text(json.dumps({"run_hash": run_hash, "state": "complete",
            "n_completed": 1, "n_planned": 1}), encoding="utf-8")
        self.csv(checkpoint / "membership.csv", [{"source_row": i, "partition": "test" if i < 30 else "train",
            "y": i % 2} for i in range(120)])
        self.csv(checkpoint / "predictions.csv", [{"source_row": i, "y": i % 2, "p": .5, "score": .5,
            "scheme": "retuned", "model": model, "split": 0} for model in self.spec["MODELS"] for i in range(30)])
        for name in ("metrics.json", "audit.json"):
            (checkpoint / name).write_text("{}", encoding="utf-8")
        self.refresh_manifest(checkpoint, run_hash)
        return output, checkpoint

    def refresh_manifest(self, checkpoint, run_hash="artificial-run-hash"):
        hashes = {name: smoke.file_sha(checkpoint / name)
                  for name in ("membership.csv", "predictions.csv", "metrics.json", "audit.json")}
        (checkpoint / "manifest.json").write_text(json.dumps({"run_hash": run_hash, "state": "complete",
            "mode": "smoke", "output_sha256": hashes}), encoding="utf-8")

    def test_constants_are_read_without_executing_source(self):
        self.assertEqual(len(self.spec["HEADERS"]), 52)
        self.assertEqual(len(self.spec["MODELS"]), 8)

    def test_generated_values_are_deterministic_and_interface_valid(self):
        rows = smoke.artificial_rows(self.spec)
        self.assertEqual(rows, smoke.artificial_rows(self.spec))
        path = self.root / "artificial.csv"
        smoke.write_artificial_csv(path, self.spec, rows)
        smoke.check_artificial_csv(path, self.spec, 240)
        self.assertTrue(all(set(row) == set(self.spec["HEADERS"]) for row in rows))

    def test_invalid_row_count_is_rejected(self):
        for count in (0, 119, 121):
            with self.assertRaises(ValueError):
                smoke.artificial_rows(self.spec, count)

    def test_default_help_does_not_inspect_source_or_start_process(self):
        output = io.StringIO()
        with patch.object(smoke, "check") as check, contextlib.redirect_stdout(output):
            self.assertEqual(smoke.main([]), 0)
        check.assert_not_called()
        self.assertIn("--check-only", output.getvalue())

    def test_arbitrary_patient_input_option_is_not_accepted(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            smoke.main(["--raw", "not-a-real-file.csv"])

    def test_check_only_never_launches_subprocess_and_prints_only_pass(self):
        output = io.StringIO()
        before = smoke.file_sha(self.source)
        with patch.object(smoke, "source_path", return_value=self.source), \
             patch.object(smoke.subprocess, "run") as run, contextlib.redirect_stdout(output):
            self.assertEqual(smoke.main(["--check-only"]), 0)
        run.assert_not_called()
        self.assertEqual(output.getvalue(), "PASS\n")
        self.assertEqual(smoke.file_sha(self.source), before)

    def test_subprocess_arguments_are_smoke_only_and_temporary_files_are_removed(self):
        observed = {}
        def fake_run(command, **kwargs):
            observed["command"] = command
            observed["folder"] = Path(kwargs["cwd"])
            self.assertTrue(observed["folder"].is_dir())
            self.assertEqual(command[command.index("--mode") + 1], "smoke")
            self.assertNotIn("--historical-params", command)
            for option in ("--raw", "--output"):
                self.assertTrue(Path(command[command.index(option) + 1]).is_relative_to(observed["folder"]))
            self.assertTrue(kwargs["capture_output"])
            return subprocess.CompletedProcess(command, 0, stdout="internal log", stderr="")
        with patch.object(smoke.subprocess, "run", side_effect=fake_run), \
             patch.object(smoke, "validate_outputs") as validate:
            smoke.check(self.source, execute=True)
        validate.assert_called_once()
        self.assertFalse(observed["folder"].exists())

    def test_failure_also_removes_temporary_files_without_echoing_logs(self):
        observed = {}
        def fail(command, **kwargs):
            observed["folder"] = Path(kwargs["cwd"])
            return subprocess.CompletedProcess(command, 7, stdout="hidden-process-text", stderr="hidden-error-text")
        output, error = io.StringIO(), io.StringIO()
        with patch.object(smoke, "source_path", return_value=self.source), \
             patch.object(smoke.subprocess, "run", side_effect=fail), \
             contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            self.assertEqual(smoke.main(["--run"]), 1)
        self.assertFalse(observed["folder"].exists())
        self.assertEqual(output.getvalue(), "")
        self.assertNotIn("hidden-", error.getvalue())

    def test_complete_artificial_checkpoint_is_accepted(self):
        output, _ = self.fixture_outputs()
        smoke.validate_outputs(output, self.spec, 120)

    def test_publication_eligible_output_is_rejected(self):
        output, _ = self.fixture_outputs()
        path = output / "run_manifest.json"
        value = json.loads(path.read_text())
        value["config"]["publication_eligible"] = True
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "publication eligible"):
            smoke.validate_outputs(output, self.spec, 120)

    def test_prediction_tampering_is_rejected_even_with_updated_file_hash(self):
        output, checkpoint = self.fixture_outputs()
        path = checkpoint / "predictions.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["source_row"] = "119"
        self.csv(path, rows)
        self.refresh_manifest(checkpoint)
        with self.assertRaisesRegex(ValueError, "membership mismatch"):
            smoke.validate_outputs(output, self.spec, 120)

    def test_checkpoint_hash_tampering_is_rejected(self):
        output, checkpoint = self.fixture_outputs()
        (checkpoint / "audit.json").write_text('{"changed": true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            smoke.validate_outputs(output, self.spec, 120)


if __name__ == "__main__":
    unittest.main(verbosity=2)
