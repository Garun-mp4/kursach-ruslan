from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NONDETERMINISTIC_KEYS = {
    "planner_wall_time_s",
    "wall_time_s",
    "wall_clock_elapsed_s",
    "started_utc",
    "finished_utc",
}


def _normalize(value: Any, run_ids: tuple[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize(item, run_ids)
            for key, item in value.items()
            if key not in NONDETERMINISTIC_KEYS and key != "performance"
        }
    if isinstance(value, list):
        return [_normalize(item, run_ids) for item in value]
    if isinstance(value, str):
        for run_id in run_ids:
            value = value.replace(run_id, "<RUN_ID>")
        return value
    return value


def _json_lines(path: Path, run_ids: tuple[str, str]) -> list[Any]:
    return [
        _normalize(json.loads(line), run_ids)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _csv_rows(path: Path, run_ids: tuple[str, str]) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    return [
        [
            next(
                (cell.replace(run_id, "<RUN_ID>") for run_id in run_ids if run_id in cell),
                cell,
            )
            for cell in row
        ]
        for row in rows
    ]


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "M7-run-manifest-v1.1":
        raise ValueError(f"Unsupported or stale M7 run manifest: {path}")
    return manifest


def compare_runs(first_dir: Path, second_dir: Path, *, require_mode_pair: bool) -> dict[str, Any]:
    first = _read_manifest(first_dir)
    second = _read_manifest(second_dir)
    run_ids = (first["run_id"], second["run_id"])
    first_summary = first["controller_summary"]
    second_summary = second["controller_summary"]
    first_evaluator = first_dir / first["relative_outputs"]["evaluator_report"]
    second_evaluator = second_dir / second["relative_outputs"]["evaluator_report"]

    checks: dict[str, bool] = {
        "same_scenario_and_seed": (
            first.get("scenario_id"), first.get("scenario_seed")
        ) == (second.get("scenario_id"), second.get("scenario_seed")),
        "same_physics_and_runtime_configuration": (
            first.get("physics"), first.get("runtime_configuration")
        ) == (second.get("physics"), second.get("runtime_configuration")),
        "same_input_hashes": first.get("input_hashes_sha256") == second.get("input_hashes_sha256"),
        "same_requested_modes": (
            {first.get("mode"), second.get("mode")} == {"batch", "interactive"}
            if require_mode_pair else first.get("mode") == second.get("mode")
        ),
        "same_simulation_duration": first_summary.get("simulation_time_s") == second_summary.get("simulation_time_s"),
        "same_control_ticks": first_summary.get("control_ticks") == second_summary.get("control_ticks"),
        "same_controller_summary_except_wall_timing": (
            _normalize(first_summary, run_ids) == _normalize(second_summary, run_ids)
        ),
        "same_controller_events": _json_lines(
            first_dir / first["relative_outputs"]["controller_events"], run_ids
        ) == _json_lines(
            second_dir / second["relative_outputs"]["controller_events"], run_ids
        ),
        "same_public_rgb_observations": _json_lines(
            first_dir / first["relative_outputs"]["perception_snapshots"], run_ids
        ) == _json_lines(
            second_dir / second["relative_outputs"]["perception_snapshots"], run_ids
        ),
        "same_controller_telemetry": _csv_rows(
            first_dir / first["relative_outputs"]["controller_telemetry"], run_ids
        ) == _csv_rows(
            second_dir / second["relative_outputs"]["controller_telemetry"], run_ids
        ),
        "same_plan_artifacts": True,
        "same_evaluator_report": _normalize(
            json.loads(first_evaluator.read_text(encoding="utf-8")), run_ids
        ) == _normalize(
            json.loads(second_evaluator.read_text(encoding="utf-8")), run_ids
        ),
    }

    first_plans = first_dir / first["relative_outputs"]["m5_plan_samples"]
    second_plans = second_dir / second["relative_outputs"]["m5_plan_samples"]
    first_files = sorted(path.relative_to(first_plans) for path in first_plans.rglob("*") if path.is_file())
    second_files = sorted(path.relative_to(second_plans) for path in second_plans.rglob("*") if path.is_file())
    checks["same_plan_artifacts"] = first_files == second_files and all(
        _normalize(json.loads((first_plans / relative).read_text(encoding="utf-8")), run_ids)
        == _normalize(json.loads((second_plans / relative).read_text(encoding="utf-8")), run_ids)
        if relative.suffix.lower() == ".json"
        else (first_plans / relative).read_bytes() == (second_plans / relative).read_bytes()
        for relative in first_files
    )

    first_state = np.load(first_dir / first["private_evaluator_inputs"]["final_state"], allow_pickle=False)
    second_state = np.load(second_dir / second["private_evaluator_inputs"]["final_state"], allow_pickle=False)
    state_differences = {
        key: float(np.max(np.abs(first_state[key] - second_state[key])))
        for key in ("qpos", "qvel", "simulation_time_s")
    }
    checks["same_final_state_within_1e-12"] = all(
        difference <= 1e-12 for difference in state_differences.values()
    )
    checks["same_physical_sort_outcome"] = (
        first.get("evaluator_summary", {}).get("physical_sort_success_count"),
        first.get("evaluator_summary", {}).get("physical_sort_failure_count"),
        first.get("evaluator_summary", {}).get("controller_done_and_physical_success"),
    ) == (
        second.get("evaluator_summary", {}).get("physical_sort_success_count"),
        second.get("evaluator_summary", {}).get("physical_sort_failure_count"),
        second.get("evaluator_summary", {}).get("controller_done_and_physical_success"),
    )

    accepted = all(checks.values())
    return {
        "schema_version": "M7-run-comparison-v1.0",
        "run_ids": list(run_ids),
        "scenario_id": first.get("scenario_id"),
        "seed": first.get("scenario_seed"),
        "modes": [first.get("mode"), second.get("mode")],
        "comparison_method": (
            "Compare reproducible configuration/source hashes and normalized controller, "
            "public RGB, plan, telemetry, evaluator, and final physics data. Exclude only "
            "wall-clock measurements and run-ID strings from functional equivalence."
        ),
        "checks": checks,
        "accepted": accepted,
        "simulation_time_s": [
            first_summary.get("simulation_time_s"), second_summary.get("simulation_time_s")
        ],
        "control_ticks": [first_summary.get("control_ticks"), second_summary.get("control_ticks")],
        "wall_clock_elapsed_s": [
            first.get("wall_clock_elapsed_s"), second.get("wall_clock_elapsed_s")
        ],
        "performance": [
            first_summary.get("performance"), second_summary.get("performance")
        ],
        "physical_sort_success_count": [
            first.get("evaluator_summary", {}).get("physical_sort_success_count"),
            second.get("evaluator_summary", {}).get("physical_sort_success_count"),
        ],
        "final_state_max_abs_difference": state_differences,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two reproducible M7 virtual runs")
    parser.add_argument("first_run", type=Path)
    parser.add_argument("second_run", type=Path)
    parser.add_argument("--mode-pair", action="store_true",
                        help="Require one batch and one interactive run")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    first_dir = args.first_run if args.first_run.is_absolute() else ROOT / args.first_run
    second_dir = args.second_run if args.second_run.is_absolute() else ROOT / args.second_run
    result = compare_runs(first_dir, second_dir, require_mode_pair=args.mode_pair)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
