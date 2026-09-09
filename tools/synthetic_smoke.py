#!/usr/bin/env python3
"""Isolated software checks using generated data only, never clinical evidence.

No arguments prints help. --check-only checks the artificial CSV interface without
importing the analysis or its dependencies. --run additionally executes the frozen
pipeline's reduced smoke mode. All generated files live in one temporary directory
and are removed on success or failure; only PASS is printed on success.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile


SYNTHETIC_SEED = 961010
SYNTHETIC_ROWS = 240


def source_path():
    return (Path(__file__).resolve().parents[1] / "source" /
            "strict_development" / "analysis" / "nested_pipeline.py")


def read_spec(path):
    """Read only literal constants and model-name keys; never execute the source."""
    nodes = ast.parse(Path(path).read_text(encoding="utf-8-sig")).body
    assignments = {target.id: node.value for node in nodes if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}
    spec = {key: ast.literal_eval(assignments[key])
            for key in ("FEATURES", "CATEGORICAL", "RANGES")}
    models = assignments["MODELS"]
    if not isinstance(models, ast.Dict):
        raise ValueError("The frozen model dictionary is not a literal dictionary")
    spec["MODELS"] = [ast.literal_eval(key) for key in models.keys]
    features = spec["FEATURES"]
    if len(features) != 49 or len(set(features)) != 49 or len(spec["MODELS"]) != 8:
        raise ValueError("Unexpected frozen feature or model schema")
    if not set(spec["RANGES"]) <= set(features) or not set(spec["CATEGORICAL"]) <= set(features):
        raise ValueError("Feature metadata does not match the frozen schema")
    required = {"性别", "年龄", "室壁运动评分", "总胆固醇", "高密度脂蛋白",
                "低密度脂蛋白", "dNLR", "TC-HDLDL"}
    if not required <= set(features):
        raise ValueError("Required artificial interface fields are missing")
    spec["HEADERS"] = features + ["Gensini评分", "白细胞计数", "中性粒细胞计数"]
    if len(set(spec["HEADERS"])) != 52:
        raise ValueError("Expected exactly 52 unique input columns")
    return spec


def artificial_rows(spec, count=SYNTHETIC_ROWS):
    """Invent every value from a local PRNG; no fitted or observed data are used.

    A deliberately strong age signal keeps the tiny LASSO smoke test nonempty.
    This design is an interface fixture, not a realistic cohort or a simulation
    of the study's clinical performance. No missing values are needed here.
    """
    if count < 120 or count % 8:
        raise ValueError("Synthetic row count must be a multiple of 8 and at least 120")
    rng = random.Random(SYNTHETIC_SEED)
    rows = []
    for i in range(count):
        event = i % 2
        row = {}
        for feature in spec["FEATURES"]:
            if feature in spec["CATEGORICAL"]:
                row[feature] = rng.randrange(2)
            elif feature in spec["RANGES"]:
                low, high = spec["RANGES"][feature]
                row[feature] = low + (high - low) * rng.uniform(.2, .8)
            else:
                row[feature] = rng.uniform(5., 95.)
        row["年龄"] = 30. + 40. * event + rng.uniform(-8., 8.)
        row["室壁运动评分"] = rng.randrange(1, 3)
        row["Gensini评分"] = 60. if event else 20.
        row["白细胞计数"] = rng.uniform(6., 10.)
        row["中性粒细胞计数"] = rng.uniform(1., 5.)
        row["总胆固醇"] = rng.uniform(4., 6.)
        row["高密度脂蛋白"] = rng.uniform(.8, 1.2)
        row["低密度脂蛋白"] = rng.uniform(1., 2.)
        row["dNLR"] = row["中性粒细胞计数"] / (row["白细胞计数"] - row["中性粒细胞计数"])
        row["TC-HDLDL"] = row["总胆固醇"] - row["高密度脂蛋白"] - row["低密度脂蛋白"]
        rows.append(row)
    return rows


def write_artificial_csv(path, spec, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=spec["HEADERS"])
        writer.writeheader()
        writer.writerows(rows)


def check_artificial_csv(path, spec, expected_rows):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != spec["HEADERS"]:
            raise ValueError("Artificial input headers changed")
        rows = list(reader)
    if len(rows) != expected_rows:
        raise ValueError("Artificial input row count changed")
    for row in rows:
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise ValueError("Artificial fixture has non-finite values")
        if not float(row["白细胞计数"]) > float(row["中性粒细胞计数"]) >= 0:
            raise ValueError("Artificial dNLR inputs are invalid")
    if sum(float(row["Gensini评分"]) > 37 for row in rows) != expected_rows // 2:
        raise ValueError("Artificial classes are not balanced")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_outputs(output, spec, expected_rows):
    """Inspect only this call's temporary artificial checkpoint, not old results."""
    output = Path(output)
    run = read_json(output / "run_manifest.json")
    status = read_json(output / "run_status.json")
    config = run["config"]
    if config["mode"] != "smoke" or config["publication_eligible"] is not False:
        raise ValueError("Reduced run must never be publication eligible")
    if config["historical_params_sha256"] is not None or config["primary_seed42"] or config["strict_primary_oof_requested"]:
        raise ValueError("Unexpected clinical or historical option in smoke run")
    if status["state"] != "complete" or status["n_completed"] != 1 or status["n_planned"] != 1:
        raise ValueError("Synthetic smoke run is incomplete")
    checkpoint = output / "checkpoints" / "split_000"
    manifest = read_json(checkpoint / "manifest.json")
    if manifest["state"] != "complete" or manifest["mode"] != "smoke":
        raise ValueError("Synthetic checkpoint is not complete smoke output")
    if manifest["run_hash"] != run["run_hash"] or status["run_hash"] != run["run_hash"]:
        raise ValueError("Synthetic checkpoint identity mismatch")
    if not {"membership.csv", "predictions.csv", "metrics.json", "audit.json"} <= set(manifest["output_sha256"]):
        raise ValueError("Synthetic checkpoint lacks required files")
    for name, digest in manifest["output_sha256"].items():
        if Path(name).name != name or (checkpoint / name).is_symlink():
            raise ValueError("Unsafe synthetic checkpoint filename")
        if file_sha(checkpoint / name) != digest:
            raise ValueError("Synthetic checkpoint hash mismatch")
    with (checkpoint / "membership.csv").open(encoding="utf-8-sig", newline="") as handle:
        members = list(csv.DictReader(handle))
    ids = [int(row["source_row"]) for row in members]
    if len(ids) != expected_rows or len(set(ids)) != expected_rows or set(ids) != set(range(expected_rows)):
        raise ValueError("Synthetic membership is incomplete or duplicated")
    if {row["partition"] for row in members} != {"train", "test"}:
        raise ValueError("Synthetic partitions are invalid")
    if any(int(row["y"]) != int(row["source_row"]) % 2 for row in members):
        raise ValueError("Synthetic membership labels do not match the fixture")
    test = {int(row["source_row"]): int(row["y"]) for row in members if row["partition"] == "test"}
    if len(test) != expected_rows // 4:
        raise ValueError("Synthetic holdout size is wrong")
    with (checkpoint / "predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    by_model = {name: [] for name in spec["MODELS"]}
    for row in predictions:
        if row["model"] not in by_model or row["scheme"] != "retuned" or int(row["split"]) != 0:
            raise ValueError("Unexpected synthetic prediction group")
        index = int(row["source_row"])
        if index not in test or int(row["y"]) != test[index]:
            raise ValueError("Synthetic prediction membership mismatch")
        if not math.isfinite(float(row["score"])) or not 0 <= float(row["p"]) <= 1:
            raise ValueError("Invalid synthetic prediction values")
        by_model[row["model"]].append(index)
    if any(len(indices) != len(test) or set(indices) != set(test) for indices in by_model.values()):
        raise ValueError("Synthetic prediction coverage is incomplete or duplicated")


def check(source, execute=False, timeout=900):
    source = Path(source).resolve()
    spec = read_spec(source)
    original_sha = file_sha(source)
    with tempfile.TemporaryDirectory(prefix="cad_ml_synthetic_") as folder:
        temporary = Path(folder).resolve()
        raw = temporary / "artificial_only.csv"
        output = temporary / "run"
        write_artificial_csv(raw, spec, artificial_rows(spec))
        check_artificial_csv(raw, spec, SYNTHETIC_ROWS)
        if execute:
            command = [sys.executable, str(source), "--mode", "smoke", "--splits", "1",
                       "--workers", "1", "--split-workers", "1", "--output", str(output),
                       "--raw", str(raw)]
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            result = subprocess.run(command, cwd=temporary, env=environment, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=timeout,
                                    check=False)
            if result.returncode != 0:
                # Do not echo paths, predictions, or arbitrary captured process logs.
                raise RuntimeError("Artificial smoke subprocess failed; no clinical analysis was run")
            validate_outputs(output, spec, SYNTHETIC_ROWS)
        if file_sha(source) != original_sha:
            raise RuntimeError("Frozen analysis source changed during the software check")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--check-only", action="store_true", help="Check artificial CSV interface; no analysis imports or training")
    choice.add_argument("--run", action="store_true", help="Run reduced eight-model smoke software check on artificial data")
    parser.add_argument("--timeout", type=int, default=900, help="Smoke subprocess timeout in seconds (1–3600)")
    args = parser.parse_args(argv)
    if not args.check_only and not args.run:
        parser.print_help()
        return 0
    if not 1 <= args.timeout <= 3600:
        parser.error("timeout must be between 1 and 3600 seconds")
    try:
        check(source_path(), execute=args.run, timeout=args.timeout)
    except (OSError, ValueError, KeyError, TypeError, SyntaxError, RuntimeError, subprocess.TimeoutExpired):
        print("FAIL: isolated artificial-data software check did not complete; no result is publication evidence.", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
