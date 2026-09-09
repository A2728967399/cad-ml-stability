#!/usr/bin/env python3
"""Append-only study-number namespace; no training or manuscript writes.

    python -B analysis/study_numbers_v1.py write
    python -B analysis/study_numbers_v1.py check

The reviewed source digests below are an approval boundary, not a cache.
Changing a source requires a separately reviewed version, never regeneration
of this version. ``check`` is strictly read-only, including its JSON report.
Only aggregate cohort checks are emitted; source_row is used only in memory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
Q4 = "strict_development"
ZH = "supplementary_analysis"
EN = "reporting"
PRIMARY = Q4 + "/results/strict_primary_oof"
CP = PRIMARY + "/checkpoints/primary_42"
ORIGINAL = ZH + "/results/original_metrics"
TEX = ROOT / "generated/study_numbers_v1.tex"
AUDIT = ROOT / "audit/study_numbers_v1_sources.json"
PREFIX = "study-v1-"

# Approved on 2026-09-08. Never replace these automatically from live files.
SOURCES = {
    "frequency": (Q4 + "/results/strict_summary/feature_frequency.csv", "030123c7e9864990d6fc00768635679324bbe70910e88f51e27b90f93c40f6a0"),
    "repeated": (Q4 + "/results/strict_summary/summary.json", "9a8016c640e00140453bccc96de1c9b14b537ca45b6ac2da81464a3172f6f03b"),
    "all_features": (Q4 + "/results/strict_all_features_summary/all_features_vs_lasso_summary.json", "702dd3268245b5f2ea2ba02cfdd0fa889271b3b1be49954bdbc513b24321d0fe"),
    "primary_audit": (CP + "/audit.json", "216cc81dfc75ebd6c9798be9dad60f29433c9cda5bd11c7959f84be49f0e07f6"),
    "categories": (ORIGINAL + "/clinical_category_counts.csv", "f5437f0be6b2f002248038a6d3f0d6adbd45ddee421c5f37593a7c5c75c565e6"),
    "subtypes": (ORIGINAL + "/clinical_subtype_baseline.csv", "635d3f1bbcfc0f01b746f3beb3e4f0fdc6d48d567ae2fe5c16cb9d2a90532d97"),
    "baseline": (ORIGINAL + "/baseline_long.csv", "d8aee678739d3c528d4af8679fa3e3c9ab6ce1cbe2965d6708c7fc00e6bc4662"),
    "simulation_complete": (Q4 + "/results/simulation/COMPLETE.json", "1f01c2f44e67ad7baa2114135f0db6f9a45ed430134b26ee80afb582885ca195"),
    "simulation": (Q4 + "/results/simulation/summary.json", "16762550a029ed5540d3a9f1eebd49c6e94227363567bc5e60e7afae9f2ad78e"),
    "raw": ("clinical_pipeline/data/raw/source_clean.csv", "c1f000031fc81def013265bf472a32003ffb22891b12fca39b2f907b0dfcb49e"),
    "run_manifest": (PRIMARY + "/run_manifest.json", "8dae041f480ee34d345e7738b71957608b29a51a46a15eb179d5f5bd614f8368"),
    "checkpoint_manifest": (CP + "/manifest.json", "31d20e27db977f6b613eb0f6f4ef0e8d17caf66f60c187afb11c55097ff19989"),
    "membership": (CP + "/membership.csv", "c5d2b00f59c3ec08ad552aaf5e56ae4252c7a9027f7c260d84a6557e6c3f609e"),
    "predictions": (CP + "/predictions.csv", "eb47aa5152fc1f8d2439911d458a5ae0182b657d4eab719942ebe349caac20b5"),
    "subgroup_auc": (ORIGINAL + "/subgroup_auc.csv", "1fdbaf5fc9433e00bac39de9317730f7ff7c55c0571b85b6fe687bb495a32823"),
}
BASE_MACROS = {
    "numbers": (EN + "/generated/numbers.tex", "e29bd61e9c597bd5ed2318a5b8d93c6cf07044fa76e62ab3b2d418360203cfe5", 632),
    "editorial_numbers": (EN + "/generated/editorial_numbers.tex", "00f70f27aa41218dc2e2b22eb34788c849bd9a2b567c901e9c25fd715a8b6c67", 50),
}
MODELS = {"Logistic回归", "随机森林", "K近邻", "梯度提升", "SVM", "XGBoost", "LightGBM", "多层感知机"}
SUBTYPE = {1: "STEMI", 2: "NSTEMI", 3: "不稳定型心绞痛", 4: "稳定型心绞痛", None: "分型未知"}
APPROVED_DISPLAY = {
    "repeat200-wms-selection-pct": "99.5", "repeat200-ctnt-selection-pct": "100.0",
    "repeat200-siri-selection-pct": "87.0", "repeat200-ck-selection-pct": "68.0",
    "repeat200-ldl-selection-pct": "59.5", "repeat200-lvef-selection-pct": "60.0",
    "allfeatures-lr-delta-signed": "+0.0133", "allfeatures-lr-delta-abs": "0.0133",
    "allfeatures-rf-delta-signed": "+0.0074", "allfeatures-rf-delta-abs": "0.0074",
    "allfeatures-knn-delta-signed": "-0.0121", "allfeatures-knn-delta-abs": "0.0121",
    "cohort-acs-pct": "90.6", "cohort-mi-pct": "38.3", "highgs-ctnt-q3": "1.29",
    "train-events": "484", "candidate-count": "49", "selected-count": "13",
    "candidate-epv": "9.88", "selected-epv": "37.2", "unknown-all-n": "53",
    "unknown-train-n": "43", "unknown-test-n": "10", "test-mi-n": "131",
    "test-nonmi-n": "185", "simulation-completed": "400", "simulation-scenarios": "8",
    "cohort-n": "1304", "train-n": "978", "test-n": "326", "repeat-completed": "200",
    "repeat-selected-ge80-count": "3",
}
MACRO_RE = re.compile(r"\\expandafter\\def\\csname result@([^\\]+)\\endcsname\{([^{}]*)\}")


class GateError(RuntimeError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def object_hash(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8"))


def members_hash(rows):
    return object_hash(sorted(int(row) for row in rows))


def need(condition, label):
    if not condition:
        # Labels deliberately never include an individual source_row or record.
        raise GateError(label)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def unique(rows, **selector):
    matches = [r for r in rows if all(r.get(k) == v for k, v in selector.items())]
    need(len(matches) == 1, "missing or ambiguous aggregate selector")
    return matches[0]


def pinned_bytes():
    values = {}
    for name, spec in {**SOURCES, **BASE_MACROS}.items():
        path, expected = spec[:2]
        content = (WORK / path).read_bytes()
        need(digest(content) == expected, "approved source hash changed: " + name)
        values[name] = content
    return values


def load_csv(values, name):
    return list(csv.DictReader(io.StringIO(values[name].decode("utf-8-sig"))))


def verify_cohort(values, reports):
    """Independent label/count verification, never patient-level disclosure."""
    run = json.loads(values["run_manifest"])
    cp = json.loads(values["checkpoint_manifest"])
    audit = json.loads(values["primary_audit"])
    config, cohort = run["config"], run["cohort"]
    need(run["run_hash"] == cp["run_hash"] == object_hash(config), "run/config identity")
    need(cp["state"] == "complete" and cp["key"] == "primary_42" and cp["split"] == 0
         and cp["mode"] == config["mode"] == "full", "completed primary checkpoint identity")
    need(config["raw_sha256"] == SOURCES["raw"][1] and config["primary_seed42"] is True
         and config["split_seed_rule"] == "42" and config["outcome_cutoff_locked"] == 37
         and config["exclude_gs_lt2"] is False and config["all_features"] is False
         and config["train_ratio"] == 0.75, "locked primary specification")
    need(config["stratification"] == "outcome only; clinical subtype neither read nor used"
         and cohort["clinical_subtype_read_or_used"] is False, "outcome-only development metadata")
    need(set(config["models"]) == MODELS and len(config["models"]) == 8, "eight models in run")
    for name, filename in [("primary_audit", "audit.json"), ("membership", "membership.csv"),
                           ("predictions", "predictions.csv")]:
        need(cp["output_sha256"][filename] == SOURCES[name][1], "checkpoint output digest: " + name)
    need(audit["membership"] == cp["membership"], "audit/checkpoint membership equality")
    mem = cp["membership"]
    need(mem["seed"] == 42 and mem["stratification"] == "outcome only", "seed not split index")

    # Whole-file bytes are hashed, but only these two data columns are retained.
    reader = csv.reader(io.StringIO(values["raw"].decode("utf-8-sig")))
    header = next(reader)
    need(header.count("Gensini评分") == header.count("冠心病类型") == 1, "raw column uniqueness")
    gi, si = header.index("Gensini评分"), header.index("冠心病类型")
    raw = {}
    source_count = excluded = low_positive = 0
    for source_row, record in enumerate(reader):
        source_count += 1
        gs = float(record[gi])
        need(math.isfinite(gs) and gs >= 0, "valid nonnegative raw GS")
        st = record[si].strip()
        subtype = None if st == "" else float(st)
        need(subtype is None or subtype in (1, 2, 3, 4), "valid raw clinical subtype codes")
        subtype = None if subtype is None else int(subtype)
        excluded += int(gs == 0)
        low_positive += int(0 < gs < 2)
        if gs != 0:
            raw[source_row] = (int(gs > 37), subtype)
    need(source_count == cohort["n_source"] == 1313 and excluded == cohort["excluded_gs_eq0"] == 9
         and low_positive == cohort["n_gs_positive_lt2_source"] == 3, "raw eligibility aggregates")
    need(len(raw) == cohort["n_analyzed"] == 1304
         and sum(v[0] for v in raw.values()) == cohort["events"] == 645
         and len(raw) - cohort["events"] == cohort["non_events"] == 659, "raw working cohort aggregates")
    need(cohort["source_row_convention"] == "zero-based data record index, excluding CSV header"
         and members_hash(raw) == cohort["cohort_members_sha256"], "zero-based cohort membership hash")

    membership = load_csv(values, "membership")
    partitions = {"train": set(), "test": set()}
    seen = set()
    for row in membership:
        sid, y, part = int(row["source_row"]), int(row["y"]), row["partition"]
        need(part in partitions and sid in raw and sid not in seen, "unique valid membership records")
        need(y == raw[sid][0], "membership outcome matches raw GS")
        seen.add(sid)
        partitions[part].add(sid)
    need(seen == set(raw) and not partitions["train"] & partitions["test"], "exhaustive disjoint partitions")
    for part in ("train", "test"):
        ids = partitions[part]
        need(len(ids) == mem["n_" + part]
             and sum(raw[s][0] for s in ids) == mem["events_" + part]
             and members_hash(ids) == mem[part + "_members_sha256"], "partition counts/labels/hash: " + part)

    predictions = load_csv(values, "predictions")
    need(set(r["model"] for r in predictions) == MODELS, "prediction model names")
    for model in sorted(MODELS):
        rows = [r for r in predictions if r["model"] == model]
        ids = [int(r["source_row"]) for r in rows]
        need(len(ids) == len(set(ids)) == mem["n_test"] and set(ids) == partitions["test"],
             "each model has exactly the same held-out test set")
        need(all(r["scheme"] == "retuned" and r["split"] == "0" for r in rows), "prediction scheme/split index")
        need(all(int(r["y"]) == raw[int(r["source_row"])][0] for r in rows), "prediction labels match raw GS")
        need(all(math.isfinite(float(r["p"])) and 0 <= float(r["p"]) <= 1 for r in rows),
             "valid stored probabilities; no recomputation")

    groups = {"总体": set(raw), "单次开发集": partitions["train"], "单次测试集": partitions["test"],
              "GS≤37": {s for s in raw if raw[s][0] == 0}, "GS>37": {s for s in raw if raw[s][0] == 1}}
    cat_codes = {"ACS": {1, 2, 3}, "MI": {1, 2}, "nonMI": {3, 4}, "unknown": {None}}
    categories, subtypes = load_csv(values, "categories"), load_csv(values, "subtypes")
    need(len(categories) == 20 and len(subtypes) == 25, "complete category/subtype summary shape")
    counts = {}
    for group, ids in groups.items():
        counts[group] = {cat: sum(raw[s][1] in codes for s in ids) for cat, codes in cat_codes.items()}
        for category, count in counts[group].items():
            row = unique(categories, group=group, category=category)
            need(int(row["count"]) == count and int(row["denominator"]) == len(ids)
                 and close(row["fraction"], count / len(ids)), "raw/category summary agreement")
        need(sum(counts[group][cat] for cat in ("MI", "nonMI", "unknown")) == len(ids), "exhaustive clinical categories")
        for code, label in SUBTYPE.items():
            row = unique(subtypes, group=group, subtype=label)
            count = sum(raw[s][1] == code for s in ids)
            need(int(row["count"]) == count and int(row["denominator"]) == len(ids)
                 and close(row["fraction"], count / len(ids)), "raw/subtype summary agreement")
    subgroup_rows = load_csv(values, "subgroup_auc")
    need(len(subgroup_rows) == 24, "eight-model three-subgroup summary shape")
    subgroup_counts = {}
    for group in ("MI", "nonMI", "unknown"):
        ids = {s for s in partitions["test"] if raw[s][1] in cat_codes[group]}
        events = sum(raw[s][0] for s in ids)
        subgroup_counts[group] = {"n": len(ids), "events": events, "non_events": len(ids) - events}
        for model in sorted(MODELS):
            row = unique(subgroup_rows, subgroup=group, model=model)
            need(int(row["n"]) == len(ids) and int(row["events"]) == events
                 and int(row["non_events"]) == len(ids) - events, "raw/predictions/subgroup summary counts")
    reports["cohort_verification"] = {
        "status": "PASS", "raw_data_columns_used": ["Gensini评分", "冠心病类型"],
        "source_row_convention": cohort["source_row_convention"], "run_hash": run["run_hash"],
        "checkpoint_key": cp["key"], "split_index": 0, "seed": 42,
        "n_source": source_count, "excluded_gs_eq0": excluded, "positive_gs_lt2_retained_pending_eligibility_review": low_positive,
        "membership": mem, "cohort_members_sha256": cohort["cohort_members_sha256"],
        "prediction_models_checked": len(MODELS), "prediction_rows_checked": len(predictions),
        "identical_test_members_and_raw_outcomes_all_models": True,
        "category_summary_cells_checked": len(categories), "subtype_summary_cells_checked": len(subtypes),
        "subgroup_model_cells_checked": len(subgroup_rows), "test_subgroups": subgroup_counts,
        "clinical_category_counts": counts, "patient_level_information_emitted": False,
        "scope": "Read-only post-hoc count verification; clinical subtype was not used in model development or stratification.",
    }
    return run, audit, counts


def generate():
    values = pinned_bytes()
    reports = {}
    base_keys = []
    for name, (_, _, count) in BASE_MACROS.items():
        keys = [m[0] for m in MACRO_RE.findall(values[name].decode("utf-8"))]
        need(len(keys) == len(set(keys)) == count, "base macro definition count: " + name)
        base_keys.extend(keys)
    need(len(base_keys) == len(set(base_keys)) == 682, "old 682 namespace uniqueness")
    need(not any(k.startswith(PREFIX) for k in base_keys), "new namespace absent from protected macros")
    run, audit, counts = verify_cohort(values, reports)
    macros = {}

    def add(key, value, fmt, source, selector, calculation="stored value", **extra):
        display = format(value, fmt)
        need(key in APPROVED_DISPLAY and display == APPROVED_DISPLAY[key], "approved display mismatch: " + key)
        need(PREFIX + key not in macros, "duplicate new macro")
        macros[PREFIX + key] = {"raw_value": value, "display": display, "format_spec": fmt,
                                "source": source, "selector": selector, "calculation": calculation, **extra}

    repeated = json.loads(values["repeated"])
    need(repeated["complete"] is True and repeated["status"] == "complete"
         and repeated["n_completed"] == repeated["expected"] == 200
         and repeated["all_features"] is False and repeated["primary_seed42"] is False, "complete 200 LASSO repetitions")
    freq = load_csv(values, "frequency")
    features = run["cohort"]["predictors"]
    need(len(freq) == len(features) == len(set(features)) == 49
         and set(r["feature"] for r in freq) == set(features), "49 repeated feature rows")
    for row in freq:
        count, denominator = int(row["count"]), int(row["denominator"])
        need(denominator == 200 and 0 <= count <= denominator
             and close(row["frequency"], count / denominator), "frequency/count denominator agreement")
    for key, feature in [("wms", "室壁运动评分"), ("ctnt", "血清肌钙蛋白T"), ("siri", "SIRI"),
                         ("ck", "肌酸激酶"), ("ldl", "低密度脂蛋白"), ("lvef", "左室射血分数")]:
        row = unique(freq, feature=feature)
        add("repeat200-" + key + "-selection-pct", int(row["count"]) / int(row["denominator"]) * 100,
            ".1f", ["frequency", "repeated"], {"feature": feature}, "100 * count / denominator",
            numerator=int(row["count"]), denominator=int(row["denominator"]), stored_frequency=row["frequency"],
            scope="200 complete repeated developments; not the 300 conditional bootstrap samples")
    comparison = json.loads(values["all_features"])
    need(comparison["direction"] == "all features minus common LASSO; retuned workflows on identical train/test membership"
         and comparison["complete"] is True and comparison["membership_sha256_agrees_for_all_paired_splits"] is True
         and comparison["n_paired_splits"] == comparison["current_n"] == comparison["reference_n"] == 200
         and comparison["current_only_splits"] == comparison["reference_only_splits"] == [], "complete paired directional comparison")
    for key, model, direction in [("lr", "Logistic回归", 1), ("rf", "随机森林", 1), ("knn", "K近邻", -1)]:
        row = unique(comparison["model_results"], model=model)
        delta = float(row["mean"])
        need(row["n"] == 200 and math.isfinite(delta) and direction * delta > 0, "signed delta direction gate: " + key)
        metadata = {"signed_source_value": delta, "direction": "all features minus common LASSO", "expected_sign": direction}
        add("allfeatures-" + key + "-delta-signed", delta, "+.4f", ["all_features"], {"model": model, "field": "mean"}, **metadata)
        add("allfeatures-" + key + "-delta-abs", abs(delta), ".4f", ["all_features"], {"model": model, "field": "mean"},
            "abs(stored signed mean); use only with matching increase/decrease wording", **metadata)
    n = run["cohort"]["n_analyzed"]
    for key, category in [("acs", "ACS"), ("mi", "MI")]:
        count = counts["总体"][category]
        add("cohort-" + key + "-pct", count / n * 100, ".1f", ["raw", "categories", "subtypes", "membership", "run_manifest"],
            {"group": "总体", "category": category}, "100 * raw verified count / working cohort n", numerator=count, denominator=n)
    ctnt = unique(load_csv(values, "baseline"), source_column="血清肌钙蛋白T", group="GS>37")
    need(int(ctnt["n"]) == 645 and ctnt["role"] == "continuous"
         and int(ctnt["observed_n"]) + int(ctnt["missing_n"]) == 645, "cTnT observed denominator")
    add("highgs-ctnt-q3", float(ctnt["q3"]), ".2f", ["baseline"], {"source_column": "血清肌钙蛋白T", "group": "GS>37", "field": "q3"},
        "format stored unrounded binary64 value at original two-decimal precision", source_decimal_string=ctnt["q3"],
        note="Preserves the reviewed display 1.29; this is not a fresh quantile calculation or a 1.30 rounding change.")
    mem = audit["membership"]
    selected = audit["development"]["final_representation"]["selected"]
    need(len(selected) == len(set(selected)) == 13 and set(selected) <= set(features)
         and run["config"]["features"] == features, "candidate and selected feature identity")
    add("train-events", mem["events_train"], "d", ["raw", "membership", "primary_audit", "checkpoint_manifest"], {"partition": "train", "field": "events_train"})
    add("candidate-count", len(features), "d", ["run_manifest", "frequency"], {"field": "cohort.predictors"}, "length of unique candidate list")
    add("selected-count", len(selected), "d", ["primary_audit"], {"field": "development.final_representation.selected"}, "length of unique final selected list")
    for key, denominator, fmt in [("candidate", len(features), ".2f"), ("selected", len(selected), ".1f")]:
        add(key + "-epv", mem["events_train"] / denominator, fmt, ["primary_audit", "run_manifest", "raw", "membership"],
            {"numerator": "events_train", "denominator": key + " count"}, "training events / variable count; descriptive ratio, not a sample-size adequacy proof",
            numerator=mem["events_train"], denominator=denominator)
    for key, group in [("all", "总体"), ("train", "单次开发集"), ("test", "单次测试集")]:
        add("unknown-" + key + "-n", counts[group]["unknown"], "d", ["raw", "membership", "categories", "subtypes", "predictions", "subgroup_auc"],
            {"group": group, "category": "unknown"}, "raw subtype missing within verified working-cohort partition")
    for key, category in [("mi", "MI"), ("nonmi", "nonMI")]:
        add("test-" + key + "-n", counts["单次测试集"][category], "d", ["raw", "membership", "predictions", "categories", "subtypes", "subgroup_auc"],
            {"group": "单次测试集", "category": category}, "raw subtype category within verified held-out test set")
    simulation, completed = json.loads(values["simulation"]), json.loads(values["simulation_complete"])
    need(len(simulation) == len({row["scenario"] for row in simulation}) == 8
         and all(row["replicates"] == 50 for row in simulation)
         and sum(row["replicates"] for row in simulation) == completed["records"] == 400, "eight scenarios and 400 completed simulation replicates")
    add("simulation-completed", completed["records"], "d", ["simulation", "simulation_complete"], {"field": "records"}, "sum of completed replicates across eight simulation scenarios")
    add("simulation-scenarios", len(simulation), "d", ["simulation"], {"field": "scenario"}, "number of distinct data-generating scenarios, not number of clinical models")
    add("cohort-n", n, "d", ["raw", "membership", "run_manifest"], {"field": "cohort.n_analyzed"})
    for part in ("train", "test"):
        add(part + "-n", mem["n_" + part], "d", ["raw", "membership", "primary_audit"], {"field": "n_" + part})
    add("repeat-completed", repeated["n_completed"], "d", ["repeated"], {"field": "n_completed"})
    ge80 = sum(int(r["count"]) / int(r["denominator"]) >= 0.8 for r in freq)
    need(ge80 == repeated["features_at_least_080"], "at-least-80-percent feature count")
    add("repeat-selected-ge80-count", ge80, "d", ["frequency", "repeated"], {"field": "features_at_least_080"}, "count features with count / denominator >= 0.8")
    need(set(macros) == {PREFIX + k for k in APPROVED_DISPLAY} and len(macros) == 32, "approved exact 32-key namespace")
    tex = ("% Study-number v1 additions only. Original 682 definitions remain byte-for-byte protected.\n"
           "% Generated by analysis/study_numbers_v1.py write; check is strictly read-only.\n"
           + "".join("\\expandafter\\def\\csname result@" + key + "\\endcsname{" + macros[key]["display"] + "}\n"
                     for key in sorted(macros))).encode("utf-8")
    report = {
        "schema_version": 1, "namespace": PREFIX, "status": "PASS", "approved_on": "2026-09-08",
        "scope": "32 narrowly selected additional study-number definitions; not all manuscript numbers are automated.",
        "generator": {"path": "analysis/study_numbers_v1.py", "sha256": digest(Path(__file__).read_bytes())},
        "approved_source_policy": "Pinned digests are an explicit approval boundary. Changed sources require a separately reviewed version; no silent regeneration.",
        "approved_sources": {name: {"workspace_relative_path": path, "sha256": sha} for name, (path, sha) in sorted(SOURCES.items())},
        "protected_base_macros": {name: {"workspace_relative_path": path, "sha256": sha, "definition_count": count}
                                  for name, (path, sha, count) in sorted(BASE_MACROS.items())},
        "protected_definition_count": 682, "new_definition_count": 32, "combined_definition_count": 714,
        "macros": dict(sorted(macros.items())), "aggregate_checks": reports,
        "output": {"path": "generated/study_numbers_v1.tex", "sha256": digest(tex), "encoding": "UTF-8", "line_ending": "LF"},
        "limitations": ["No model fitting, selection, bootstrap, or quantile recomputation was performed.",
                        "EPV is a descriptive events-per-variable calculation, not proof of adequate sample size.",
                        "Empirical repeated-split values are not independent confidence intervals.",
                        "Literature numbers and methodological constants remain literal and outside this namespace.",
                        "Clinical subtype is a post-hoc cohort descriptor; missing subtype is not a new diagnosis.",
                        "This report verifies evidence and definitions, not manuscript context or final PDF layout."],
    }
    report_bytes = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    # Fail if a source moved while it was being checked, before any output write.
    need(all(values[k] == v for k, v in pinned_bytes().items()), "input bytes changed during verification")
    return tex, report_bytes, report


def execute(mode):
    tex, report_bytes, report = generate()
    outputs = [(TEX, tex), (AUDIT, report_bytes)]
    # Preflight both outputs before creating either; never overwrite a stale file.
    for path, expected in outputs:
        if path.exists():
            need(path.read_bytes() == expected, "existing output differs; reviewed version required: " + path.name)
        else:
            need(mode == "write", "missing generated output: " + path.name)
    created = []
    if mode == "write":
        for path, expected in outputs:
            if not path.exists():
                need(path.parent.is_dir(), "expected output directory missing")
                with path.open("xb") as handle:
                    handle.write(expected)
                created.append(path.relative_to(ROOT).as_posix())
    need(all(path.read_bytes() == expected for path, expected in outputs), "output byte verification")
    return {"status": "PASS", "mode": mode, "write_policy": "first creation only; changed bytes fail closed",
            "created_files": created, "new_definition_count": 32, "protected_definition_count": 682,
            "combined_definition_count": 714, "approved_source_count": len(SOURCES),
            "generator_sha256": report["generator"]["sha256"], "generated_tex_sha256": digest(tex),
            "source_report_sha256": digest(report_bytes), "cohort_gate": "PASS",
            "check_performs_no_writes": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("write", "check"))
    args = parser.parse_args()
    try:
        result = execute(args.mode)
    except (GateError, OSError, ValueError, KeyError, TypeError) as exc:
        # Do not print tracebacks containing patient data.
        result = {"status": "FAIL", "mode": args.mode,
                  "error": str(exc) if isinstance(exc, (GateError, OSError)) else type(exc).__name__,
                  "check_performs_no_writes": True}
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
