#!/usr/bin/env python3
"""Audit recorded training warning events without refitting or double counting.

Warning events are events inside producer catch_warnings scopes, which include
some fit/scoring calls. A warning is not an algorithm failure. Candidate CV and
LASSO records are aggregates: they cannot identify the number of affected fits.
The five primary OOF final-model warning records were not persisted by the
producer; this audit reports those 40 fit calls as unknown, never zero.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def atomic_csv(path, rows, columns):
    path = Path(path)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def validate_warning_record(record):
    if record is None:
        return None
    categories = record["categories"]
    assert isinstance(categories, dict)
    assert all(isinstance(v, int) and v >= 0 for v in categories.values())
    assert sum(categories.values()) == record["total"], "Warning categories do not sum to total"
    assert 0 <= record["convergence"] <= record["total"]
    assert record["convergence"] >= categories.get("ConvergenceWarning", 0)
    assert isinstance(record["examples"], list)
    return record


def warning_bucket(context, stage, model, role, fit_calls, record, **extra):
    record = validate_warning_record(record)
    available = record is not None
    return {**context, "stage": stage, "model": model, "selection_role": role,
            "fit_calls": int(fit_calls), "record_available": available,
            "warning_events": record["total"] if available else None,
            "convergence_warning_events": record["convergence"] if available else None,
            "future_warning_events": record["categories"].get("FutureWarning", 0) if available else None,
            "categories_json": canonical(record["categories"]) if available else "",
            "examples_json": canonical(record["examples"]) if available else "", **extra}


def collect_representation(rep, context, location, buckets, fallbacks):
    if rep["method"] == "all eligible features":
        # The empty producer record is a placeholder, not a captured LASSO fit.
        assert rep["warnings"]["total"] == 0 and "path" not in rep
        return
    assert rep["method"] == "nested LASSO 1SE"
    path = rep["path"]["fold_auc"]
    cv_fits = sum(len(row) for row in path)
    additional = rep["empty_model_fallback_steps"]
    assert isinstance(additional, int) and additional >= 0
    nfits = cv_fits + 1 + additional
    # Only this aggregate is consumed. Do not recursively sum parent+child
    # warning dictionaries; the selector already aggregates its CV and refits.
    buckets.append(warning_bucket(context, "lasso_cv_and_refit_aggregate", "LASSO", "all_C_and_refit_not_separable", nfits,
                                  rep["warnings"], representation_scope=location))
    fallbacks.append({**context, "representation_scope": location,
                      "n_training_rows": rep["preprocessing"]["n_fit"], "n_selected": len(rep["selected"]),
                      "C_initial_1se": rep["C_initial_1se"], "C_final": rep["C"],
                      "empty_model_fallback_steps": additional, "fallback_triggered": additional > 0,
                      "lasso_cv_fit_calls": cv_fits, "lasso_refit_calls": 1 + additional})


def collect_development(development, context, model_names, final_metrics, buckets, fallbacks):
    collect_representation(development["final_representation"], context, "full_development", buckets, fallbacks)
    for fold in development["model_cv_folds"]:
        collect_representation(fold["representation"], context, f"model_cv_train_fold_{fold['fold']}", buckets, fallbacks)
    selected = {}
    for model in model_names:
        candidates = development["candidates"][model]
        assert len(candidates)
        for candidate in candidates:
            assert len(candidate["fold_auc"]) and math.isfinite(candidate["mean_auc"])
            assert abs(sum(candidate["fold_auc"]) / len(candidate["fold_auc"]) - candidate["mean_auc"]) < 1e-12
        selected_index = max(range(len(candidates)), key=lambda i: candidates[i]["mean_auc"])
        selected[model] = candidates[selected_index]["params"]
        for index, candidate in enumerate(candidates):
            buckets.append(warning_bucket(context, "model_grid_cv", model,
                           "selected_candidate" if index == selected_index else "unselected_candidate",
                           len(candidate["fold_auc"]), candidate["warnings"], candidate_index=index,
                           params_json=canonical(candidate["params"]), cv_mean_auc=candidate["mean_auc"]))
    if final_metrics is None:
        # develop_and_evaluate returns metrics, but strict_primary_oof discards
        # those final-estimator records. Candidate CV records are not a proxy.
        for model in model_names:
            buckets.append(warning_bucket(context, "final_classifier", model, "retuned", 1, None,
                           params_json=canonical(selected[model]),
                           missing_reason="OOF final estimator fit_warnings not persisted by producer"))
        return
    for metric in final_metrics:
        if metric["scheme"] == "retuned":
            assert canonical(metric["params"]) == canonical(selected[metric["model"]]), "Selected CV candidate differs from final retuned parameters"
        buckets.append(warning_bucket(context, "final_classifier", metric["model"], metric["scheme"], 1,
                       metric.get("fit_warnings"), params_json=canonical(metric["params"]),
                       calibration_diagnostic_converged=metric.get("calibration_converged")))


def tally(buckets, fallbacks):
    available = [row for row in buckets if row["record_available"]]
    def event_sum(stage=None, role=None, key="convergence_warning_events"):
        return sum(row[key] for row in available if (stage is None or row["stage"] == stage) and (role is None or row["selection_role"] == role))
    return {"completed_development_fit_calls": sum(r["fit_calls"] for r in buckets),
            "fit_calls_with_structured_warning_record": sum(r["fit_calls"] for r in available),
            "fit_calls_without_structured_warning_record": sum(r["fit_calls"] for r in buckets if not r["record_available"]),
            "warning_record_buckets": len(available), "positive_warning_record_buckets": sum(r["warning_events"] > 0 for r in available),
            "warning_events": event_sum(key="warning_events"), "convergence_warning_events": event_sum(),
            "future_warning_events": event_sum(key="future_warning_events"),
            "selected_candidate_cv_convergence_warning_events": event_sum("model_grid_cv", "selected_candidate"),
            "unselected_candidate_cv_convergence_warning_events": event_sum("model_grid_cv", "unselected_candidate"),
            "final_classifier_convergence_warning_events_recorded": event_sum("final_classifier"),
            "lasso_convergence_warning_events_aggregate": event_sum("lasso_cv_and_refit_aggregate"),
            "lasso_selectors": len(fallbacks), "lasso_empty_fallback_selectors": sum(r["fallback_triggered"] for r in fallbacks),
            "lasso_additional_refits": sum(r["empty_model_fallback_steps"] for r in fallbacks),
            "unrecorded_oof_final_fit_calls": sum(r["fit_calls"] for r in buckets if not r["record_available"] and r["development"].startswith("oof_")),
            "calibration_diagnostic_nonconvergence_records": sum(r.get("calibration_diagnostic_converged") is False for r in available)}


def warning_tables(buckets):
    categories = defaultdict(lambda: {"events": 0, "buckets": 0})
    coverage = defaultdict(lambda: Counter())
    finals = defaultdict(lambda: Counter())
    for row in buckets:
        key = (row["run"], row["stage"], row["model"], row["selection_role"])
        group = coverage[key]
        group["fit_calls"] += row["fit_calls"]
        group["recorded_fit_calls" if row["record_available"] else "unrecorded_fit_calls"] += row["fit_calls"]
        if row["record_available"]:
            group["recorded_warning_buckets"] += 1
            for category, count in json.loads(row["categories_json"]).items():
                c = categories[key + (category,)]
                c["events"] += count
                c["buckets"] += count > 0
        if row["stage"] == "final_classifier":
            scope = "OOF_validation" if row["development"].startswith("oof_") else "holdout_test"
            group = finals[(row["run"], scope, row["model"], row["selection_role"])]
            group["recorded_final_fit_calls" if row["record_available"] else "unrecorded_final_fit_calls"] += row["fit_calls"]
            if row["record_available"]:
                for name in ["warning_events", "convergence_warning_events", "future_warning_events"]:
                    group[name] += row[name]
    category_rows = [{"run": k[0], "stage": k[1], "model": k[2], "selection_role": k[3], "warning_category": k[4],
                      "warning_events": v["events"], "positive_warning_buckets": v["buckets"]} for k, v in sorted(categories.items())]
    coverage_rows = [{"run": k[0], "stage": k[1], "model": k[2], "selection_role": k[3],
                      **{name: v[name] for name in ["fit_calls", "recorded_fit_calls", "unrecorded_fit_calls", "recorded_warning_buckets"]}} for k, v in sorted(coverage.items())]
    final_rows = [{"run": k[0], "evaluation_scope": k[1], "model": k[2], "scheme": k[3],
                   **{name: v[name] for name in ["recorded_final_fit_calls", "unrecorded_final_fit_calls"]},
                   **{name: v[name] if v["recorded_final_fit_calls"] else None for name in ["warning_events", "convergence_warning_events", "future_warning_events"]}}
                  for k, v in sorted(finals.items())]
    return category_rows, coverage_rows, final_rows


def scan_log_text(text, run, source):
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        start = re.search(r"\[([^]]+)\]\s+(split_(\d+)|primary_42) started", line)
        complete = re.search(r"(split_(\d+)|primary_42) COMPLETE", line)
        if start:
            kind, split, timestamp = "split_start", int(start[3]) if start[3] else 0, start[1]
        elif complete:
            kind, split, timestamp = "split_completion", int(complete[2]) if complete[2] else 0, ""
        elif "pending_this_invocation=" in line:
            kind, split, timestamp = "run_invocation_or_resume", "", ""
        elif re.search(r"\b[A-Za-z]*Warning\b", line):
            kind, split, timestamp = "library_warning_text", "", ""
        elif re.search(r"Traceback|RuntimeError|ValueError|AssertionError|BrokenProcessPool|KeyboardInterrupt|SIGTERM|Terminated|\bfailed\b", line, flags=re.I):
            kind, split, timestamp = "failure_or_interruption_text_not_attempt_count", "", ""
        else:
            continue
        rows.append({"run": run, "source": source, "line": number, "signal": kind, "split": split,
                     "timestamp": timestamp, "text": line[:1200]})
    return rows


def observed_restarts(log_signals):
    starts = defaultdict(set)
    for row in log_signals:
        if row["signal"] == "split_start":
            starts[(row["run"], row["split"])].add(row["timestamp"])
    return [{"run": key[0], "split": key[1], "event": "multiple_distinct_start_times_in_available_logs",
             "n_distinct_starts": len(times), "timestamps_json": canonical(sorted(times)),
             "interpretation": "At least this many logged starts; not a count of algorithm failures"}
            for key, times in sorted(starts.items()) if len(times) > 1]


class Reader:
    def __init__(self):
        self.sources = {}

    def content(self, path, expected_hash=None, mutable=False):
        path = Path(path).resolve()
        content = path.read_bytes()
        digest = sha_bytes(content)
        if expected_hash is not None:
            assert digest == expected_hash, f"Checkpoint file hash mismatch: {path}"
        self.sources[str(path)] = {"sha256_snapshot": digest, "bytes_snapshot": len(content), "may_change_during_live_run": mutable}
        return content

    def json(self, path, expected_hash=None, mutable=False):
        return json.loads(self.content(path, expected_hash, mutable).decode("utf-8-sig"))


def audit(args):
    output = Path(args.output).resolve()
    run_paths = {"main": Path(args.main_run).resolve(), "all_features": Path(args.all_features_run).resolve(), "primary": Path(args.primary_run).resolve()}
    assert len(set(run_paths.values())) == 3
    assert output.is_relative_to((ROOT / "results").resolve()) and output != (ROOT / "results").resolve(), "Output must be a separate child of Q4/results"
    assert all(not output.is_relative_to(path) for path in run_paths.values()), "Audit output must not alter a scientific run directory"
    reader, buckets, fallbacks, runs, operational, logs = Reader(), [], [], {}, [], []
    for label, path in run_paths.items():
        rm = reader.json(path / "run_manifest.json")
        config = rm["config"]
        assert sha_bytes(canonical(config).encode()) == rm["run_hash"], "Run configuration hash mismatch"
        assert config["mode"] == "full", "Only the three full scientific runs belong in this audit"
        assert config["n_splits_planned"] == (1 if label == "primary" else 200)
        assert bool(config["all_features"]) == (label == "all_features")
        assert bool(config["primary_seed42"]) == (label == "primary")
        assert bool(config["strict_primary_oof_requested"]) == (label == "primary")
        assert bool(config["historical_params_sha256"]) == (label == "main")
        completed = []
        start_buckets, start_fallbacks = len(buckets), len(fallbacks)
        folders = sorted(p for p in (path / "checkpoints").iterdir() if p.is_dir() and not p.name.startswith(".") and (p / "manifest.json").is_file())
        for folder in folders:
            cm = reader.json(folder / "manifest.json")
            assert cm["state"] == "complete" and cm["run_hash"] == rm["run_hash"]
            completed.append(int(cm["split"]))
            development = reader.json(folder / "audit.json", cm["output_sha256"]["audit.json"])["development"]
            final_metrics = reader.json(folder / "metrics.json", cm["output_sha256"]["metrics.json"])
            context = {"run": label, "split": int(cm["split"]), "checkpoint": folder.name, "development": "holdout_final"}
            collect_development(development, context, config["models"], final_metrics, buckets, fallbacks)
            if config["strict_primary_oof_requested"]:
                outer = reader.json(folder / "oof_audit.json", cm["output_sha256"]["oof_audit.json"])
                assert len(outer) == 5 and {r["oof_fold"] for r in outer} == set(range(5))
                for fold in outer:
                    collect_development(fold["development"], {**context, "development": f"oof_fold_{fold['oof_fold']}"},
                                        config["models"], None, buckets, fallbacks)
        assert len(set(completed)) == len(completed)
        expected_ids = set(range(config["start_split"], config["start_split"] + config["n_splits_planned"]))
        assert set(completed) <= expected_ids
        failed = []
        for status_file in sorted((path / "status").glob("*.json")):
            status = reader.json(status_file, mutable=True)
            if status.get("state") == "failed":
                event = {"run": label, "split": status.get("split"), "event": "current_failed_status_at_snapshot",
                         "source": str(status_file), "error": status.get("error"), "timestamp": status.get("failed_utc")}
                failed.append(event)
                operational.append(event)
        run_logs = list(path.glob("*.log"))
        external_patterns = {"main": ["main_run*.log", "main_train*.log"], "all_features": ["all_features*.log"], "primary": ["primary_oof*.log"]}
        for pattern in external_patterns[label]:
            run_logs += list((ROOT / "logs").glob(pattern))
        log_manifest = []
        for logfile in sorted(set(p.resolve() for p in run_logs)):
            content = reader.content(logfile, mutable=True)
            log_manifest.append({"source": str(logfile), "bytes_snapshot": len(content)})
            logs.extend(scan_log_text(content.decode("utf-8-sig", errors="replace"), label, str(logfile)))
        totals = tally(buckets[start_buckets:], fallbacks[start_fallbacks:])
        runs[label] = {"directory": str(path), "run_hash": rm["run_hash"], "mode": config["mode"],
                       "completed_checkpoints": len(completed), "planned_checkpoints": config["n_splits_planned"],
                       "complete": set(completed) == expected_ids, "completed_split_ids": completed,
                       "current_failed_status_records": len(failed), "logs_available": log_manifest,
                       **totals}
    for evidence_path in sorted((ROOT / "audit").glob("all_features_stop_4workers_*.json")):
        evidence = reader.json(evidence_path)
        if evidence.get("state") != "stopped_and_lock_archived" or evidence.get("run_hash") != runs["all_features"]["run_hash"]:
            continue
        for record in evidence["running_splits_before"]:
            operational.append({"run": "all_features", "split": record["split"], "event": "authorized_operational_interruption_for_worker_resize",
                                "timestamp": evidence["utc"], "source": str(evidence_path), "old_worker_pid": record["pid"],
                                "resumed_completed_at_snapshot": record["split"] in runs["all_features"]["completed_split_ids"],
                                "interpretation": "4-to-6 worker resize; not an algorithm failure; interrupted partial fit-call count was not recorded"})
    operational += observed_restarts(logs)
    totals = tally(buckets, fallbacks)
    totals["current_failed_status_records"] = sum(run["current_failed_status_records"] for run in runs.values())
    totals["documented_operationally_interrupted_attempts"] = sum(r["event"] == "authorized_operational_interruption_for_worker_resize" for r in operational)
    categories, coverage, final_rows = warning_tables(buckets)
    complete = all(run["complete"] for run in runs.values())
    summary = {"schema_version": 1, "snapshot_utc": datetime.now(timezone.utc).isoformat(),
               "status": "complete" if complete else "partial", "complete": complete,
               "structured_estimator_warning_capture_complete": totals["fit_calls_without_structured_warning_record"] == 0,
               "all_warning_capture_complete": False,
               "counting_unit": {
                   "warning_events": "Producer-captured warning emissions inside fit/scoring scopes; repeated emissions count repeatedly, not number of failed/affected models",
                   "warning_record_buckets": "One disjoint selector aggregate, one candidate-five-fold aggregate, or one final-estimator capture; no parent/child totals added twice",
                   "fit_calls": "Top-level estimator.fit calls in completed developments only, including LASSO refits; cache reuse does not multiply counts",
                   "selection": "Selected candidate is first maximum mean CV AUC in stored candidate order, cross-checked against final retuned parameters where available",
                   "future_warning": "Library/API FutureWarning emissions, reported separately from ConvergenceWarning",
                   "operational_events": "Current failure statuses, distinct logged starts and explicit resize evidence; log text lines are not independent failed attempts",
               },
               "runs": runs, "totals": totals,
               "limitations": [
                   "40 final estimator fit_warnings in primary five-fold OOF were not persisted; unknown is not zero.",
                   "LASSO warning aggregate combines all C-fold fits and final/refit steps; selected-C versus unselected-C localization is unavailable.",
                   "Candidate CV aggregates combine five folds; the number/identity of affected folds or fits cannot be recovered from event totals.",
                   "Preprocessing and calibration-diagnostic optimization run outside these estimator warning capture scopes; their warning absence is not certified.",
                   "The verifier's own empty warnings list is not used as evidence of warning-free estimator fitting.",
                   "Missing or overwritten historical failed statuses and incomplete/empty logs limit attempt-history recovery; no zero-lifetime-failure claim is made.",
                   "Partially completed work during authorized interruptions is not counted as fully completed fit calls; four explicit interrupted splits are listed separately.",
                   "A convergence warning in an unselected candidate may still affect hyperparameter selection and is retained; it is not automatically a final-model failure.",
               ],
               "source_files": reader.sources, "audit_script_sha256": sha_bytes(Path(__file__).read_bytes())}
    output.mkdir(parents=True, exist_ok=True)
    common = ["run", "split", "checkpoint", "development"]
    exports = {
        "warning_records.csv": (buckets, common + ["stage", "model", "selection_role", "representation_scope", "candidate_index", "params_json", "cv_mean_auc", "fit_calls", "record_available", "warning_events", "convergence_warning_events", "future_warning_events", "categories_json", "examples_json", "missing_reason", "calibration_diagnostic_converged"]),
        "warning_summary.csv": (categories, ["run", "stage", "model", "selection_role", "warning_category", "warning_events", "positive_warning_buckets"]),
        "fit_coverage.csv": (coverage, ["run", "stage", "model", "selection_role", "fit_calls", "recorded_fit_calls", "unrecorded_fit_calls", "recorded_warning_buckets"]),
        "final_model_warnings.csv": (final_rows, ["run", "evaluation_scope", "model", "scheme", "recorded_final_fit_calls", "unrecorded_final_fit_calls", "warning_events", "convergence_warning_events", "future_warning_events"]),
        "lasso_fallback.csv": (fallbacks, common + ["representation_scope", "n_training_rows", "n_selected", "C_initial_1se", "C_final", "empty_model_fallback_steps", "fallback_triggered", "lasso_cv_fit_calls", "lasso_refit_calls"]),
        "operational_events.csv": (operational, ["run", "split", "event", "timestamp", "source", "old_worker_pid", "resumed_completed_at_snapshot", "n_distinct_starts", "timestamps_json", "error", "interpretation"]),
        "log_signals.csv": (logs, ["run", "source", "line", "signal", "split", "timestamp", "text"]),
    }
    supplement = [{"run": label, **{k: run[k] for k in ["completed_checkpoints", "planned_checkpoints", "complete", "completed_development_fit_calls", "warning_events", "convergence_warning_events", "future_warning_events", "selected_candidate_cv_convergence_warning_events", "unselected_candidate_cv_convergence_warning_events", "final_classifier_convergence_warning_events_recorded", "lasso_convergence_warning_events_aggregate", "unrecorded_oof_final_fit_calls", "lasso_empty_fallback_selectors", "lasso_additional_refits", "current_failed_status_records"]}} for label, run in runs.items()]
    exports["supplement_table.csv"] = (supplement, list(supplement[0]))
    for name, (rows, columns) in exports.items():
        atomic_csv(output / name, rows, columns)
    summary["output_sha256"] = {name: sha_bytes((output / name).read_bytes()) for name in exports}
    atomic_json(output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "complete": complete,
                      "run_checkpoints": {k: f"{r['completed_checkpoints']}/{r['planned_checkpoints']}" for k, r in runs.items()},
                      "totals": totals, "output": str(output)}, ensure_ascii=False))
    if args.require_complete and not complete:
        raise SystemExit(3)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-run", default=str(ROOT / "results/strict_lasso_200"))
    parser.add_argument("--all-features-run", default=str(ROOT / "results/strict_all_features_200"))
    parser.add_argument("--primary-run", default=str(ROOT / "results/strict_primary_oof"))
    parser.add_argument("--output", default=str(ROOT / "results/training_events"))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    try:
        audit(args)
    except Exception as error:
        output = Path(args.output).resolve()
        protected = [Path(p).resolve() for p in [args.main_run, args.all_features_run, args.primary_run]]
        if output.is_relative_to((ROOT / "results").resolve()) and output != (ROOT / "results").resolve() and all(not output.is_relative_to(p) for p in protected):
            atomic_json(output / "summary.json", {"schema_version": 1, "status": "failed", "complete": False,
                        "snapshot_utc": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
