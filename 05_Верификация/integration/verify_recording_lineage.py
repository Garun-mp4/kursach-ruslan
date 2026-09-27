#!/usr/bin/env python3
"""Verify that an M8 recording and its source run artifacts match their SHA-256 lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(path: str) -> Path:
    candidate = (REPO_ROOT / Path(path)).resolve()
    candidate.relative_to(REPO_ROOT)
    return candidate


def _check(name: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def _verify_source_group(
    run_dir: Path,
    group_name: str,
    relative_output: str,
    logged_group: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    expected_path = (run_dir / Path(relative_output.rstrip("/\\"))).resolve()
    expected_path.relative_to(run_dir.resolve())

    if expected_path.is_dir():
        actual_files = {
            path.relative_to(REPO_ROOT).as_posix(): _sha256(path)
            for path in expected_path.rglob("*")
            if path.is_file()
        }
        logged_files = {
            entry["path"]: entry["sha256"] for entry in logged_group.get("files", [])
        }
        logged_directory = logged_group.get("path", "").rstrip("/\\")
        passed = (
            logged_directory == expected_path.relative_to(REPO_ROOT).as_posix()
            and
            logged_group.get("file_count") == len(actual_files)
            and logged_files == actual_files
        )
        return passed, {
            "kind": "directory",
            "file_count": len(actual_files),
            "logged_file_count": logged_group.get("file_count"),
            "files_match": logged_files == actual_files,
        }

    actual_hash = _sha256(expected_path) if expected_path.is_file() else None
    logged_path = logged_group.get("path")
    logged_hash = logged_group.get("sha256")
    passed = (
        expected_path.is_file()
        and logged_path == expected_path.relative_to(REPO_ROOT).as_posix()
        and logged_hash == actual_hash
    )
    return passed, {
        "kind": "file",
        "path": expected_path.relative_to(REPO_ROOT).as_posix(),
        "exists": expected_path.is_file(),
        "sha256_matches": logged_hash == actual_hash,
    }


def verify(run_id: str, recording_manifest_path: Path) -> dict[str, Any]:
    media = json.loads(recording_manifest_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    run_dir = _repo_path(media["source_run_directory"])
    run_manifest_path = run_dir / "manifest.json"
    source_manifest_path = _repo_path(media["source_run_manifest"]["path"])
    state_stream_path = _repo_path(media["source_state_stream"])
    video_path = _repo_path(media["video"]["path"])

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_outputs = source_manifest.get("relative_outputs", {})
    logged_outputs = media.get("source_logs", {})
    expected_groups = set(source_outputs)

    checks.append(
        _check(
            "run_identity",
            media.get("run_id") == run_id
            and source_manifest.get("run_id") == run_id
            and run_manifest_path.is_file(),
            {
                "requested_run_id": run_id,
                "recording_run_id": media.get("run_id"),
                "source_run_id": source_manifest.get("run_id"),
            },
        )
    )

    expected_source_manifest_hash = media["source_run_manifest"]["sha256"]
    actual_source_manifest_hash = _sha256(source_manifest_path)
    checks.append(
        _check(
            "source_run_manifest_sha256",
            expected_source_manifest_hash == actual_source_manifest_hash
            and media.get("source_run_manifest_sha256") == actual_source_manifest_hash,
            {
                "expected": expected_source_manifest_hash,
                "actual": actual_source_manifest_hash,
            },
        )
    )

    expected_state_hash = media["source_state_stream_sha256"]
    actual_state_hash = _sha256(state_stream_path) if state_stream_path.is_file() else None
    checks.append(
        _check(
            "state_stream_sha256",
            state_stream_path.is_file() and expected_state_hash == actual_state_hash,
            {"expected": expected_state_hash, "actual": actual_state_hash},
        )
    )

    expected_video_hash = media["video"]["sha256"]
    actual_video_hash = _sha256(video_path) if video_path.is_file() else None
    checks.append(
        _check(
            "video_sha256",
            video_path.is_file() and expected_video_hash == actual_video_hash,
            {"expected": expected_video_hash, "actual": actual_video_hash},
        )
    )

    checks.append(
        _check(
            "source_output_catalog",
            expected_groups == set(logged_outputs),
            {
                "expected_groups": sorted(expected_groups),
                "logged_groups": sorted(logged_outputs),
            },
        )
    )

    for group_name in sorted(expected_groups):
        passed, details = _verify_source_group(
            run_dir,
            group_name,
            source_outputs[group_name],
            logged_outputs.get(group_name, {}),
        )
        checks.append(_check(f"source_output:{group_name}", passed, details))

    capture = cv2.VideoCapture(str(video_path))
    decoded_frames = 0
    while True:
        ok, _frame = capture.read()
        if not ok:
            break
        decoded_frames += 1
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    expected_video = media["video"]
    expected_resolution = expected_video["resolution_px"]
    state_manifest_path = state_stream_path.with_name("state_stream_manifest.json")
    state_manifest = json.loads(state_manifest_path.read_text(encoding="utf-8"))
    with np.load(state_stream_path, allow_pickle=False) as state_archive:
        state_count = int(state_archive["timeline_states"].shape[0])
    timeline = state_manifest["timeline"]
    timeline_count = len(timeline)
    timeline_matches = (
        timeline_count == state_count
        and state_manifest.get("run_id") == run_id
        and state_manifest.get("dropped_frame_count") == expected_video["dropped_frames"]
        and float(state_manifest.get("fps", 0.0)) == float(expected_video["fps"])
        and abs(float(timeline[0]["simulation_time_s"]) - expected_video["source_simulation_time_start_s"]) <= 1e-12
        and abs(float(timeline[-1]["simulation_time_s"]) - expected_video["source_simulation_time_end_s"]) <= 1e-12
    )
    decode_passed = (
        decoded_frames == expected_video["frame_count"]
        == expected_video["decoded_frame_count"]
        and fps == float(expected_video["fps"])
        and [width, height] == expected_resolution
        and decoded_frames == state_count
        and timeline_matches
    )
    checks.append(
        _check(
            "video_decode_and_format",
            decode_passed,
            {
                "decoded_frames": decoded_frames,
                "expected_frames": expected_video["frame_count"],
                "state_count": state_count,
                "timeline_count": timeline_count,
                "timeline_matches_run": timeline_matches,
                "fps": fps,
                "resolution_px": [width, height],
            },
        )
    )

    return {
        "schema_version": "M8-recording-lineage-audit-v1",
        "run_id": run_id,
        "recording_manifest": recording_manifest_path.resolve().relative_to(REPO_ROOT).as_posix(),
        "source_run_manifest": source_manifest_path.relative_to(REPO_ROOT).as_posix(),
        "source_output_group_count": len(expected_groups),
        "check_count": len(checks),
        "checks": checks,
        "accepted": all(check["passed"] for check in checks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="M8 simulation run ID")
    parser.add_argument(
        "--recording-manifest",
        type=Path,
        help="Recording manifest path; defaults to the M8 media folder for RUN_ID",
    )
    parser.add_argument("--output", type=Path, help="Optional audit JSON output path")
    args = parser.parse_args()

    default_media_dir = REPO_ROOT / "06_Медиа" / "проверка_записи" / args.run_id
    recording_manifest_path = (
        args.recording_manifest.resolve()
        if args.recording_manifest
        else default_media_dir / "Запись_и_связанные_логи.json"
    )
    report = verify(args.run_id, recording_manifest_path)
    output_path = (
        args.output.resolve()
        if args.output
        else default_media_dir / "Проверка_целостности.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"accepted": report["accepted"], "check_count": report["check_count"], "output": str(output_path)}))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
