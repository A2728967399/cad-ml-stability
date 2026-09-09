#!/usr/bin/env python3
"""Software versions from the three frozen clinical runs, never the build host.

render: emit deterministic TeX to stdout; generate --authorized: explicitly create that TeX;
check: validate sources, all 401 checkpoint memberships, exact TeX bytes and the
seven English macro calls against the Chinese version sentence (zero writes);
selftest: pure in-memory negative tests (zero writes).
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
Q4 = WORK / "strict_development"
OUTPUT = ROOT / "generated/software_environment_v1.tex"
ENGLISH = ROOT / "sections/02_methods.tex"
CHINESE = WORK / "supplementary_analysis/sections/02_methods.tex"
KEYS = ("python", "scikit-learn", "xgboost", "lightgbm", "numpy", "pandas", "scipy")
LABELS = {"python": "Python", "scikit-learn": "scikit-learn", "xgboost": "XGBoost",
          "lightgbm": "LightGBM", "numpy": "NumPy", "pandas": "pandas", "scipy": "SciPy"}
RUN_SPECS = {
    "strict_lasso_200": {"sha256": "7c0cde5efc4dd86754ba20616a71237aff09ce85204ff9008705f0fcd80b8e1c", "count": 200, "primary": False, "all_features": False},
    "strict_all_features_200": {"sha256": "7164ffb00328edd7296afc19d28210c548406b014ac917550959a2bb02c9c588", "count": 200, "primary": False, "all_features": True},
    "strict_primary_oof": {"sha256": "8dae041f480ee34d345e7738b71957608b29a51a46a15eb179d5f5bd614f8368", "count": 1, "primary": True, "all_features": False},
}
RUN_PATHS = tuple(Q4 / "results" / name / "run_manifest.json" for name in RUN_SPECS)
SOFTWARE_RE = re.compile(r"\\SoftwareVersion\{([^{}]*)\}")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8"))


def validate_run_payloads(payloads, checkpoint_records):
    """Pure validation; the loader separately enforces approved file digests.

    Only run identity, runtime, and checkpoint key/split/state are examined.
    No source rows, memberships, predictions, or clinical data are opened.
    """
    require(set(payloads) == set(RUN_SPECS) == set(checkpoint_records), "Missing or extra formal run information")
    shared_runtime = None
    verified = {}
    for name, spec in RUN_SPECS.items():
        run = payloads[name]
        require(isinstance(run, dict) and isinstance(run.get("config"), dict), "Missing run configuration: " + name)
        config = run["config"]
        require(isinstance(config.get("runtime"), dict), "Missing recorded runtime: " + name)
        runtime = config["runtime"]
        require(isinstance(runtime.get("python"), str) and isinstance(runtime.get("packages"), dict)
                and isinstance(runtime.get("platform"), str), "Incomplete recorded runtime: " + name)
        require(set(KEYS[1:]) <= set(runtime["packages"]), "Missing scientific package version: " + name)
        require(run.get("run_hash") == canonical_hash(config), "Forged or stale canonical run_hash: " + name)
        require(config.get("mode") == "full" and config.get("publication_eligible") is True
                and config.get("n_splits_planned") == spec["count"]
                and config.get("primary_seed42") is spec["primary"]
                and config.get("all_features") is spec["all_features"], "Formal run specification mismatch: " + name)
        if shared_runtime is None:
            shared_runtime = runtime
        require(runtime == shared_runtime, "Formal runs have different recorded software environments")
        records = checkpoint_records[name]
        require(len(records) == spec["count"], "Incomplete checkpoint count: " + name)
        expected_keys = {"primary_42"} if spec["primary"] else {f"split_{i:03d}" for i in range(spec["count"])}
        keys, splits = [], []
        for record in records:
            require(isinstance(record, dict) and record.get("state") == "complete"
                    and record.get("run_hash") == run["run_hash"] and record.get("mode") == "full",
                    "Incomplete or foreign checkpoint: " + name)
            keys.append(record.get("key")); splits.append(record.get("split"))
        require(len(set(keys)) == len(keys) and set(keys) == expected_keys
                and len(set(splits)) == len(splits) and set(splits) == set(range(spec["count"])),
                "Checkpoint key/split coverage mismatch: " + name)
        verified[name] = {"run_hash": run["run_hash"], "completed_checkpoints": len(records),
                          "canonical_run_hash_matches": True, "checkpoint_run_identity_matches": True}
    match = re.fullmatch(r"(\d+\.\d+\.\d+)(?:\s.*)?", shared_runtime["python"], re.S)
    require(match is not None, "Unparseable recorded Python version")
    versions = {"python": match[1], **{key: shared_runtime["packages"][key] for key in KEYS[1:]}}
    require(all(isinstance(value, str) and re.fullmatch(r"\d+\.\d+\.\d+", value) for value in versions.values()),
            "Unparseable recorded package version")
    require(sum(row["completed_checkpoints"] for row in verified.values()) == 401, "Expected exactly 401 formal checkpoints")
    return {"versions": versions, "recorded_runtime": shared_runtime, "runs": verified,
            "completed_checkpoints": 401, "all_run_environments_identical": True}


def source_evidence():
    payloads, checkpoints, sources, checkpoint_hashes = {}, {}, {}, {}
    for name, spec in RUN_SPECS.items():
        directory = Q4 / "results" / name
        path = directory / "run_manifest.json"
        raw = path.read_bytes()
        require(digest(raw) == spec["sha256"], "Approved run manifest digest changed: " + name)
        payloads[name] = json.loads(raw)
        paths = sorted((directory / "checkpoints").glob("*/manifest.json"))
        records, hashes = [], {}
        for cp in paths:
            content = cp.read_bytes()
            record = json.loads(content)
            require(record.get("key") == cp.parent.name, "Checkpoint directory/key mismatch: " + name)
            records.append({key: record.get(key) for key in ("key", "split", "state", "run_hash", "mode")})
            hashes[cp.parent.name] = digest(content)
        checkpoints[name] = records
        checkpoint_hashes[name] = canonical_hash(hashes)
        sources[name] = {"workspace_relative_path": path.relative_to(WORK).as_posix(), "sha256": digest(raw)}
    result = validate_run_payloads(payloads, checkpoints)
    result.update({"source_manifests": sources, "checkpoint_manifest_hashes_digest": checkpoint_hashes,
                   "current_execution_environment_used_as_scientific_source": False,
                   "patient_level_data_read": False})
    return result


def render_tex(versions):
    require(set(versions) == set(KEYS), "Exact seven software keys required")
    require(all(re.fullmatch(r"\d+\.\d+\.\d+", value) for value in versions.values()), "Invalid software version text")
    lines = ["% Generated only from three approved frozen clinical run manifests.",
             "% Not from the current Python/build environment; original 714 result macros unchanged."]
    lines += ["% " + name + " SHA256 " + RUN_SPECS[name]["sha256"] for name in RUN_SPECS]
    lines += [r"\expandafter\def\csname softwareversion@" + key + r"\endcsname{" + versions[key] + "}" for key in KEYS]
    lines += [r"\newcommand{\SoftwareVersion}[1]{%",
              r"  \ifcsname softwareversion@#1\endcsname%",
              r"    \csname softwareversion@#1\endcsname%",
              r"  \else%",
              r"    \PackageError{software-environment-v1}{Unknown software version key: #1}{Use one of the seven approved software keys.}%",
              r"  \fi%", "}"]
    return ("\n".join(lines) + "\n").encode("utf-8")


def validate_tex_bytes(actual, versions):
    require(actual == render_tex(versions), "Generated software TeX differs from exact deterministic source-derived bytes")


def expand_software(value, versions):
    def replace(match):
        require(match[1] in versions, "Unknown software macro key: " + match[1])
        return versions[match[1]]
    expanded = SOFTWARE_RE.sub(replace, value)
    require(r"\SoftwareVersion" not in expanded, "Malformed or unresolved SoftwareVersion call")
    return expanded


def check_versions_in_chapters(english, chinese, versions):
    english = re.sub(r"(?<!\\)%[^\n]*", "", english)
    chinese = re.sub(r"(?<!\\)%[^\n]*", "", chinese)
    calls = SOFTWARE_RE.findall(english)
    require(Counter(calls) == Counter(KEYS) and english.count(r"\SoftwareVersion") == 7,
            "English must contain each of the seven software macros exactly once")
    expanded = expand_software(english, versions)
    found = {}
    for key in KEYS:
        label = LABELS[key]
        pattern = re.escape(label) + r"\s+(\d+\.\d+\.\d+)"
        en_values, zh_values = re.findall(pattern, expanded), re.findall(pattern, chinese)
        require(en_values == zh_values == [versions[key]], "English/Chinese recorded software version mismatch: " + key)
        # Prevent a permutation of otherwise individually valid macro keys.
        require(re.search(re.escape(label) + r"\s+\\SoftwareVersion\{" + re.escape(key) + r"\}", english),
                "Software label is bound to the wrong macro key: " + key)
        found[key] = {"label": label, "version": versions[key], "English_macro_occurrences": 1, "Chinese_occurrences": 1}
    return found


def check():
    evidence = source_evidence()
    validate_tex_bytes(OUTPUT.read_bytes(), evidence["versions"])
    evidence["chapter_version_checks"] = check_versions_in_chapters(
        ENGLISH.read_text(encoding="utf-8-sig"), CHINESE.read_text(encoding="utf-8-sig"), evidence["versions"])
    require(source_evidence() == {k: v for k, v in evidence.items() if k != "chapter_version_checks"},
            "Source evidence changed during the read-only check")
    return {"schema": "software-environment-v1", "status": "PASS", "read_only": True,
            "generated_tex_sha256": digest(OUTPUT.read_bytes()), "generator_sha256": digest(Path(__file__).read_bytes()),
            **evidence}


def generate():
    evidence = source_evidence()
    expected = render_tex(evidence["versions"])
    if OUTPUT.exists():
        validate_tex_bytes(OUTPUT.read_bytes(), evidence["versions"])
        created = False
    else:
        require(OUTPUT.parent.is_dir(), "Generated directory missing")
        with OUTPUT.open("xb") as handle:
            handle.write(expected)
        created = True
    return {"status": "PASS", "created": created, "output": OUTPUT.relative_to(ROOT).as_posix(),
            "sha256": digest(expected), "versions": evidence["versions"], "completed_checkpoints": evidence["completed_checkpoints"]}


def selftest():
    # Synthetic metadata only: no fitting, clinical records, output files or imports of scientific packages.
    runtime = {"python": "3.11.16 | test fixture", "platform": "test fixture",
               "packages": {"numpy": "2.4.6", "pandas": "3.0.5", "scipy": "1.17.1",
                            "scikit-learn": "1.9.0", "xgboost": "3.2.0", "lightgbm": "4.7.0"}}
    runs, cps = {}, {}
    for name, spec in RUN_SPECS.items():
        config = {"runtime": copy.deepcopy(runtime), "mode": "full", "publication_eligible": True,
                  "n_splits_planned": spec["count"], "primary_seed42": spec["primary"], "all_features": spec["all_features"]}
        runs[name] = {"config": config, "run_hash": canonical_hash(config)}
        cps[name] = [{"key": "primary_42" if spec["primary"] else f"split_{i:03d}", "split": i,
                      "state": "complete", "mode": "full", "run_hash": runs[name]["run_hash"]} for i in range(spec["count"])]
    checks = {}
    def rejects(name, fn):
        try:
            fn()
        except (RuntimeError, FileNotFoundError):
            checks[name] = True
        else:
            checks[name] = False
    result = validate_run_payloads(runs, cps)
    checks["401_valid_synthetic_checkpoints_pass"] = result["completed_checkpoints"] == 401
    wrong = copy.deepcopy(runs); wrong["strict_lasso_200"]["run_hash"] = "forged"
    rejects("forged_run_hash_rejected", lambda: validate_run_payloads(wrong, cps))
    missing = copy.deepcopy(runs); missing["strict_primary_oof"]["config"].pop("runtime")
    rejects("missing_runtime_rejected", lambda: validate_run_payloads(missing, cps))
    partial = copy.deepcopy(cps); partial["strict_lasso_200"].pop()
    rejects("missing_checkpoint_rejected", lambda: validate_run_payloads(runs, partial))
    foreign = copy.deepcopy(cps); foreign["strict_lasso_200"][0]["run_hash"] = "another-run"
    rejects("foreign_checkpoint_rejected", lambda: validate_run_payloads(runs, foreign))
    other = copy.deepcopy(runs); changed = copy.deepcopy(cps)
    other["strict_all_features_200"]["config"]["runtime"]["packages"]["numpy"] = "1.26.4"
    other["strict_all_features_200"]["run_hash"] = canonical_hash(other["strict_all_features_200"]["config"])
    for row in changed["strict_all_features_200"]:
        row["run_hash"] = other["strict_all_features_200"]["run_hash"]
    rejects("different_package_even_with_valid_hash_rejected", lambda: validate_run_payloads(other, changed))
    versions = result["versions"]
    english = "; ".join(LABELS[k] + r" \SoftwareVersion{" + k + "}" for k in KEYS)
    chinese = "; ".join(LABELS[k] + " " + versions[k] for k in KEYS)
    checks["seven_correct_calls_pass"] = len(check_versions_in_chapters(english, chinese, versions)) == 7
    rejects("missing_macro_rejected", lambda: check_versions_in_chapters(english.replace(r"\SoftwareVersion{python}", "3.11.16"), chinese, versions))
    rejects("unknown_macro_rejected", lambda: expand_software(r"\SoftwareVersion{unknown}", versions))
    rejects("wrong_generated_value_rejected", lambda: validate_tex_bytes(render_tex(versions).replace(b"3.11.16", b"3.11.15"), versions))
    rejects("extra_tex_instruction_rejected", lambda: validate_tex_bytes(render_tex(versions) + b"\\input{unexpected}\n", versions))
    rejects("wrong_chinese_version_rejected", lambda: check_versions_in_chapters(english, chinese.replace("3.11.16", "3.11.15"), versions))
    rejects("swapped_macro_label_rejected", lambda: check_versions_in_chapters(english.replace("{python}", "{TEMP}").replace("{numpy}", "{python}").replace("{TEMP}", "{numpy}"), chinese, versions))
    checks["tex_missing_key_raises_package_error"] = b"\\PackageError{software-environment-v1}" in render_tex(versions)
    require(all(checks.values()), "Software environment negative selftest failed")
    return {"status": "PASS", "read_only": True, "checks_count": len(checks), "checks": checks}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("render", "generate", "check", "selftest"))
    parser.add_argument("--authorized", action="store_true", help="Required for explicit generation; check never writes")
    args = parser.parse_args()
    try:
        require(args.mode != "generate" or args.authorized, "generate requires --authorized; no output was written")
        if args.mode == "render":
            sys.stdout.write(render_tex(source_evidence()["versions"]).decode("utf-8"))
            return 0
        result = {"generate": generate, "check": check, "selftest": selftest}[args.mode]()
    except Exception as exc:
        result = {"status": "FAIL", "mode": args.mode, "error": str(exc), "read_only": args.mode != "generate"}
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
