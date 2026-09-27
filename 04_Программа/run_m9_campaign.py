from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "04_Программа" / "run_m7.py"
CAMPAIGN_PATH = ROOT / "07_Испытания" / "campaign.yaml"
RUNS_ROOT = ROOT / "07_Испытания" / "runs"
STATE_PATH = ROOT / "07_Испытания" / "campaign_state.jsonl"
METADATA_PATH = ROOT / "07_Испытания" / "campaign_metadata.json"

OPTION_FLAGS = {
    "camera_failure": "--camera-failure",
    "placement_camera_failure_at_verify": "--placement-camera-failure-at-verify",
    "emergency_stop_after_hold": "--emergency-stop-after-hold",
    "payload_loss_after_hold": "--payload-loss-after-hold",
    "disable_gripper_actuator": "--disable-gripper-actuator",
    "release_actuator_failure": "--release-actuator-failure",
}
OPTION_VALUES = {
    "stale_camera_age_s": "--stale-camera-age-s",
    "payload_loss_impulse_ns": "--payload-loss-impulse-ns",
    "gripper_friction_scale": "--gripper-friction-scale",
    "lighting_scale": "--lighting-scale",
    "rgb_gain": "--rgb-gain",
    "rgb_noise_sigma": "--rgb-noise-sigma",
    "sensor_noise_seed": "--sensor-noise-seed",
    "calibration_bias_x_m": "--calibration-bias-x-m",
    "calibration_bias_y_m": "--calibration-bias-y-m",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    )
    return result.stdout.strip()


def _expand_jobs(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = campaign.get("seed_sets", {})
    jobs: list[dict[str, Any]] = []
    for experiment in campaign.get("experiments", []):
        seed_key = experiment.get("seeds")
        if seed_key not in seeds:
            raise ValueError(f"Unknown seed set {seed_key!r}")
        variants = experiment.get("variants", [])
        if not variants:
            raise ValueError(f"Experiment {experiment.get('experiment_id')} has no variants")
        for variant in variants:
            for seed in seeds[seed_key]:
                trial_id = (
                    f"{experiment['experiment_id']}__{variant['variant_id']}__{int(seed)}"
                )
                jobs.append({
                    "trial_id": trial_id,
                    "experiment_id": experiment["experiment_id"],
                    "variant_id": variant["variant_id"],
                    "scenario": experiment["scenario"],
                    "scenario_dir": experiment.get("scenario_dir"),
                    "seed": int(seed),
                    "expected_kind": variant["expected_kind"],
                    "expected_reason": variant.get("expected_reason"),
                    "expected_track_status": variant.get("expected_track_status"),
                    "expected_failure_reason": variant.get("expected_failure_reason"),
                    "expected_object_outcomes": dict(variant.get("expected_object_outcomes", {})),
                    "required_perception_checks": dict(variant.get("required_perception_checks", {})),
                    "tags": list(variant.get("tags", [])),
                    "parameters": dict(variant.get("parameters", {})),
                    "paired_control": experiment.get("paired_control"),
                })
    for fixed in campaign.get("fixed_trials", []):
        jobs.append({
            "trial_id": fixed["trial_id"],
            "experiment_id": fixed["trial_id"],
            "variant_id": "fixed",
            "scenario": fixed["scenario"],
            "scenario_dir": fixed.get("scenario_dir"),
            "seed": fixed.get("seed"),
            "expected_kind": fixed["expected_kind"],
            "expected_reason": fixed.get("expected_reason"),
            "expected_track_status": fixed.get("expected_track_status"),
            "expected_failure_reason": fixed.get("expected_failure_reason"),
            "expected_object_outcomes": dict(fixed.get("expected_object_outcomes", {})),
            "required_perception_checks": dict(fixed.get("required_perception_checks", {})),
            "tags": list(fixed.get("tags", [])),
            "parameters": dict(fixed.get("parameters", {})),
            "paired_control": None,
        })
    ids = [job["trial_id"] for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("Campaign expands to duplicate trial IDs")
    return jobs


def _command_for(job: dict[str, Any], out_root: Path) -> list[str]:
    command = [
        sys.executable, str(RUNNER), "--mode", "batch",
        "--scenario", str(job["scenario"]),
        "--physics-profile", "nominal",
        "--out", str(out_root),
    ]
    if job.get("scenario_dir"):
        command.extend(("--scenario-dir", str((ROOT / job["scenario_dir"]).resolve())))
    if job.get("seed") is not None:
        command.extend(("--seed", str(int(job["seed"]))))
    parameters = job.get("parameters", {})
    for name, flag in OPTION_FLAGS.items():
        if parameters.get(name) is True:
            command.append(flag)
        elif name in parameters and parameters[name] not in (False, None):
            raise ValueError(f"Boolean campaign option {name} must be true/false")
    for name, flag in OPTION_VALUES.items():
        if name in parameters:
            command.extend((flag, str(parameters[name])))
    return command


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _completed_trials(path: Path, campaign_hash: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("campaign_sha256") != campaign_hash:
                continue
            if record.get("status") == "completed" and record.get("run_id"):
                completed[record["trial_id"]] = record
    return completed


def _run_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Runner produced no manifest at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence_files(run_dir: Path, manifest: dict[str, Any]) -> list[Path]:
    candidates: set[Path] = set()
    for key, relative in manifest.get("relative_outputs", {}).items():
        if key.startswith("recording_") or not relative:
            continue
        path = (run_dir / relative).resolve()
        if not path.is_relative_to(run_dir.resolve()):
            continue
        if path.is_dir():
            candidates.update(file for file in path.rglob("*") if file.is_file())
        elif path.is_file():
            candidates.add(path)
    for relative in (
        "evaluator_private/scene_truth.json",
        "snapshots/m7_integrated_model.xml",
        "snapshots/ssot.yaml",
        "snapshots/scenario.yaml",
        "snapshots/runtime_config.yaml",
        "snapshots/perception_config.yaml",
    ):
        path = run_dir / relative
        if path.is_file():
            candidates.add(path.resolve())
    return sorted(candidates)


def _hash_evidence(run_dir: Path, manifest: dict[str, Any]) -> dict[str, str]:
    return {
        path.relative_to(run_dir.resolve()).as_posix(): _sha256(path)
        for path in _evidence_files(run_dir, manifest)
    }


def _compress_evidence(run_dir: Path, evidence_hashes: dict[str, str]) -> dict[str, dict[str, str]]:
    """Losslessly gzip bulky text logs after preserving their uncompressed SHA-256."""
    compressed: dict[str, dict[str, str]] = {}
    for relative in evidence_hashes:
        source = run_dir / relative
        if not source.is_file() or not (
            source.suffix in {".csv", ".jsonl"}
            or (source.suffix == ".json" and "controller/plans/" in relative.replace("\\", "/"))
        ):
            continue
        target = source.with_name(source.name + ".gz")
        with source.open("rb") as input_stream, target.open("wb") as output_stream:
            with gzip.GzipFile(fileobj=output_stream, mode="wb", compresslevel=6, mtime=0) as zipped:
                shutil.copyfileobj(input_stream, zipped)
        compressed[relative] = {
            "path": target.relative_to(run_dir).as_posix(),
            "uncompressed_sha256": evidence_hashes[relative],
            "compressed_sha256": _sha256(target),
        }
        source.unlink()
    return compressed


def _outputs_present(run_dir: Path, manifest: dict[str, Any]) -> bool:
    for key, relative in manifest.get("relative_outputs", {}).items():
        if key.startswith("recording_") or not relative:
            continue
        path = run_dir / relative
        if path.is_dir():
            continue
        if path.is_file() or path.with_name(path.name + ".gz").is_file():
            continue
        return False
    return True


def _compress_verified_repeatability_pair(
    campaign: dict[str, Any], campaign_hash: str, state_path: Path,
) -> None:
    verification_path = ROOT / "07_Испытания" / "tables" / "Проверка_повторяемости_V14.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("accepted") is not True:
        raise ValueError("V14 repeatability comparison must pass before compressing its raw pair")
    completed = _completed_trials(state_path, campaign_hash)
    pair_ids = (
        "V03_RANDOM_MIXED__nominal__20261001",
        "v14_repeat_seed_20261001",
    )
    for trial_id in pair_ids:
        state_record = completed.get(trial_id)
        if not state_record:
            raise ValueError(f"Missing completed repeatability trial {trial_id}")
        run_dir = ROOT / state_record["run_directory"]
        trial_path = run_dir / "campaign_trial.json"
        trial = json.loads(trial_path.read_text(encoding="utf-8"))
        manifest = _run_manifest(run_dir)
        compressed = _compress_evidence(run_dir, trial["evidence_hashes_sha256"])
        trial["compressed_evidence"] = {**trial.get("compressed_evidence", {}), **compressed}
        trial_path.write_text(json.dumps(trial, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
        if not _outputs_present(run_dir, manifest):
            raise RuntimeError(f"Evidence is incomplete after compression: {trial_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute the preregistered M9 virtual test campaign")
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN_PATH)
    parser.add_argument("--out", type=Path, default=RUNS_ROOT)
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--trial-id", action="append", dest="trial_ids",
                        help="Run only a named preregistered trial (for an explicitly recorded resume/debug pass)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Do not skip trials already completed with this exact campaign hash")
    parser.add_argument("--compress-verified-repeatability-pair", action="store_true",
                        help="Losslessly gzip V14 raw logs only after its comparison passed")
    args = parser.parse_args()

    campaign_path = args.campaign.resolve()
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    if not isinstance(campaign, dict) or campaign.get("schema_version") != "M9-campaign-v1.2":
        raise ValueError("Unsupported M9 campaign schema")
    campaign_hash = _sha256(campaign_path)
    jobs = _expand_jobs(campaign)
    if args.compress_verified_repeatability_pair:
        _compress_verified_repeatability_pair(campaign, campaign_hash, args.state.resolve())
        print("Verified V14 pair compressed; uncompressed hashes remain in trial records.")
        return 0
    if args.trial_ids:
        unknown = set(args.trial_ids) - {job["trial_id"] for job in jobs}
        if unknown:
            raise ValueError(f"Unregistered trial IDs: {sorted(unknown)}")
        selected = set(args.trial_ids)
        jobs = [job for job in jobs if job["trial_id"] in selected]
    out_root = args.out.resolve()
    if args.dry_run:
        for job in jobs:
            print(f"{job['trial_id']} | {job['scenario']} | seed={job.get('seed')} | {job['expected_kind']}")
        print(f"jobs={len(jobs)} campaign_sha256={campaign_hash}")
        return 0

    args.state.parent.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    revision = _git_revision()
    metadata_path = METADATA_PATH
    metadata = {
        "schema_version": "M9-campaign-metadata-v1",
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": campaign_hash,
        "protocol_path": campaign_path.relative_to(ROOT).as_posix()
        if campaign_path.is_relative_to(ROOT) else str(campaign_path),
        "software_revision": revision,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "runtime": campaign.get("runtime", {}),
        "independent_units": {
            "V03_RANDOM_MIXED": "seeded three-object batch",
            "V07_RGB_NOISE": "paired seed, one randomized red object per level",
            "V10_GRIP_FRICTION": "same scene seeds paired against V07_RGB_NOISE/sigma_0",
        },
        "commands_are_saved_per_trial": True,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    completed = {} if args.no_resume else _completed_trials(args.state, campaign_hash)
    log_root = ROOT / "07_Испытания" / "runner_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    for index, job in enumerate(jobs, start=1):
        previous = completed.get(job["trial_id"])
        if previous:
            previous_run = out_root / previous["run_id"]
            try:
                previous_manifest = _run_manifest(previous_run)
                if _outputs_present(previous_run, previous_manifest):
                    print(f"[{index}/{len(jobs)}] SKIP {job['trial_id']} ({previous['run_id']})")
                    continue
            except (OSError, ValueError, json.JSONDecodeError):
                pass

        command = _command_for(job, out_root)
        started = datetime.now(timezone.utc).isoformat()
        print(f"[{index}/{len(jobs)}] RUN {job['trial_id']}", flush=True)
        env = os.environ.copy()
        program_dir = str(ROOT / "04_Программа")
        env["PYTHONPATH"] = program_dir + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            command, cwd=ROOT, env=env, text=True, capture_output=True, check=False
        )
        log_base = log_root / job["trial_id"]
        log_base.with_suffix(".stdout.txt").write_text(result.stdout, encoding="utf-8")
        log_base.with_suffix(".stderr.txt").write_text(result.stderr, encoding="utf-8")
        try:
            runner_output = json.loads(result.stdout)
            run_id = str(runner_output["run_id"])
            run_dir = out_root / run_id
            manifest = _run_manifest(run_dir)
        except (json.JSONDecodeError, KeyError, OSError, ValueError) as exc:
            record = {
                "trial_id": job["trial_id"],
                "experiment_id": job["experiment_id"],
                "campaign_sha256": campaign_hash,
                "software_revision": revision,
                "status": "invalid_infrastructure",
                "runner_returncode": result.returncode,
                "error": f"{type(exc).__name__}: {exc}",
                "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }
            _append_jsonl(args.state, record)
            print(f"ERROR {job['trial_id']}: {record['error']}", file=sys.stderr)
            return 2

        run_manifest_path = run_dir / "manifest.json"
        evidence_hashes = _hash_evidence(run_dir, manifest)
        run_manifest_sha256 = _sha256(run_manifest_path)
        keep_for_repeatability = job["trial_id"] in {
            "V03_RANDOM_MIXED__nominal__20261001", "v14_repeat_seed_20261001",
        }
        keep_for_recording_parity = job["trial_id"] == "v01_red"
        compressed_evidence = (
            {} if keep_for_repeatability or keep_for_recording_parity
            else _compress_evidence(run_dir, evidence_hashes)
        )

        run_record = {
            "schema_version": "M9-trial-record-v1",
            **job,
            "campaign_id": campaign["campaign_id"],
            "campaign_sha256": campaign_hash,
            "software_revision": revision,
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "runner_returncode": result.returncode,
            "run_id": run_id,
            "run_manifest_sha256": run_manifest_sha256,
            "run_directory": run_dir.relative_to(ROOT).as_posix()
            if run_dir.is_relative_to(ROOT) else str(run_dir),
            "input_hashes_sha256": manifest["input_hashes_sha256"],
            "evidence_hashes_sha256": evidence_hashes,
            "compressed_evidence": compressed_evidence,
            "scenario_seed": manifest["scenario_seed"],
            "final_state": manifest["controller_summary"]["final_state"],
            "simulation_time_s": manifest["controller_summary"].get("simulation_time_s"),
            "fault_injection": manifest.get("fault_injection", {}),
            "public_input_conditions": manifest.get("m9_public_input_conditions", {}),
            "evaluator_summary": manifest.get("evaluator_summary", {}),
            "command": command,
        }
        (run_dir / "campaign_trial.json").write_text(
            json.dumps(run_record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        status = "completed" if result.returncode in (0, 2) else "runner_error"
        record = {
            "trial_id": job["trial_id"],
            "experiment_id": job["experiment_id"],
            "variant_id": job["variant_id"],
            "campaign_sha256": campaign_hash,
            "software_revision": revision,
            "status": status,
            "runner_returncode": result.returncode,
            "run_id": run_id,
            "run_directory": run_record["run_directory"],
            "final_state": run_record["final_state"],
            "finished_utc": run_record["finished_utc"],
        }
        _append_jsonl(args.state, record)
        print(
            f"[{index}/{len(jobs)}] {status.upper()} {run_id} "
            f"state={run_record['final_state']} sim={run_record['simulation_time_s']}",
            flush=True,
        )
        if status != "completed":
            print(f"Runner failed for {job['trial_id']}; raw output is retained.", file=sys.stderr)
            return result.returncode or 2
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["state_file"] = args.state.relative_to(ROOT).as_posix()
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(f"Campaign jobs completed: {len(jobs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
