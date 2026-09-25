from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
PROGRAM = ROOT / "04_Программа"
TEST_DIR = PROGRAM / "controller" / "tests"
VERIFY = ROOT / "05_Верификация" / "controller"
EVENTS = VERIFY / "events"
SPEC = ROOT / "02_Спецификация"
DIAGRAMS = ROOT / "03_Модель_и_схемы" / "algorithm"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def installed_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def run_suite(name: str, test_path: Path) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(test_path), "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    transcript = result.stdout + result.stderr
    log_path = VERIFY / f"{name}_unittest.log"
    log_path.write_text(transcript, encoding="utf-8")
    match = re.search(r"Ran (\d+) tests? in ([\d.]+)s", transcript)
    return {
        "name": name,
        "command": [sys.executable, "-m", "unittest", "discover", "-s", str(test_path), "-v"],
        "exit_code": result.returncode,
        "tests_run": int(match.group(1)) if match else None,
        "duration_s": float(match.group(2)) if match else None,
        "log": str(log_path.relative_to(ROOT)).replace("\\", "/"),
        "passed": result.returncode == 0 and "OK" in transcript,
    }


def write_nominal_test_double_trace() -> dict:
    if str(TEST_DIR) not in sys.path:
        sys.path.insert(0, str(TEST_DIR))
    import test_controller as fixtures

    fixtures.ControllerTests.setUpClass()
    test_case = fixtures.ControllerTests()
    controller, scene, _, output = test_case.start_with_plan(run_id="M6-controlled-nominal")
    test_case.complete_nominal_cycle(controller, scene, output)
    events = controller.event_log
    event_types = {event.event_type for event in events}
    required = {"TARGET_SELECTED", "PLAN_ACCEPTED", "HOLD_CONFIRMED", "PLACEMENT_CONTROLLER_CONFIRMED", "CYCLE_CLOSED"}
    if not required.issubset(event_types):
        raise RuntimeError(f"Nominal test-double trace missed required events: {sorted(required - event_types)}")

    ssot = SPEC / "параметры_системы.yaml"
    model = ROOT / "03_Модель_и_схемы" / "source" / "scara_color_sorter_m2.xml"
    records = [{
        "record_type": "trace_metadata",
        "run_id": controller.run_id,
        "evidence_class": "CONTROLLED_TEST_DOUBLE",
        "planner": "M5 test fixture; no M7 executor or MuJoCo dynamic grasp",
        "random_seed": None,
        "physical_build_or_test": False,
        "controller_policy_version": controller.config.policy_version,
        "ssot_version": controller.config.config_version,
        "ssot_sha256": sha256(ssot),
        "model_sha256": sha256(model),
        "note": "Policy trace only; never report as M9 or physical/simulation dynamics result.",
    }]
    for event in events:
        value = asdict(event)
        value["state"] = event.state.value
        records.append({"record_type": "controller_event", **value})
    trace_path = EVENTS / "M6_nominal_controlled_double.jsonl"
    trace_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
                          encoding="utf-8")
    summary = {
        "run_id": controller.run_id,
        "evidence_class": "CONTROLLED_TEST_DOUBLE",
        "final_state": controller.state.value,
        "track_statuses": {str(k): v.status.value for k, v in controller.track_records.items()},
        "slot_states": [{"slot_id": slot.slot_id, "status": slot.status.value}
                        for slot in controller.slot_states if slot.status.value != "FREE"],
        "event_count": len(events),
        "event_types": sorted(event_types),
        "success_contract": "PLACED_CONTROLLER_CONFIRMED from public evidence only; not evaluator-confirmed physical sorting",
        "trace": str(trace_path.relative_to(ROOT)).replace("\\", "/"),
    }
    (EVENTS / "M6_nominal_controlled_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def check_exports() -> dict:
    from PIL import Image

    result = {}
    for stem in ("fsm_state", "sorting_flow"):
        source = DIAGRAMS / f"{stem}.drawio"
        svg = DIAGRAMS / f"{stem}.svg"
        png = DIAGRAMS / f"{stem}.png"
        pdf = DIAGRAMS / f"{stem}.pdf"
        ET.parse(source)
        ET.parse(svg)
        with Image.open(png) as image:
            image.verify()
        with Image.open(png) as image:
            width, height = image.size
        pdf_header = pdf.read_bytes()[:5]
        if pdf_header != b"%PDF-":
            raise RuntimeError(f"Diagram PDF export is missing or invalid: {pdf}")
        if width < 1000 or height < 500:
            raise RuntimeError(f"Diagram export is unexpectedly small: {stem} {width}x{height}")
        result[stem] = {"source_bytes": source.stat().st_size, "svg_bytes": svg.stat().st_size,
                        "png_bytes": png.stat().st_size, "pdf_bytes": pdf.stat().st_size,
                        "png_size_px": [width, height], "valid_xml": True,
                        "valid_png": True, "valid_pdf": True}
    return result


def main() -> int:
    VERIFY.mkdir(parents=True, exist_ok=True)
    EVENTS.mkdir(parents=True, exist_ok=True)
    suites = [
        run_suite("controller", PROGRAM / "controller" / "tests"),
        run_suite("planning", PROGRAM / "planning" / "tests"),
        run_suite("perception", PROGRAM / "perception" / "tests"),
    ]
    trace = write_nominal_test_double_trace()
    export_checks = check_exports()
    source_files = [
        SPEC / "FSM.md", SPEC / "Политика_сортировки.md", SPEC / "Инварианты_FSM.md",
        SPEC / "Матрица_ошибок_и_реакций.csv", SPEC / "Контракт_M7_исполнения.md",
        SPEC / "Контракт_M9_телеметрии.md", VERIFY / "transition_coverage.csv",
        VERIFY / "state_handler_coverage.csv", VERIFY / "event_taxonomy.csv",
        VERIFY / "ground_truth_boundary_audit.md",
    ]
    implementation_files = sorted((PROGRAM / "controller").glob("*.py"))
    implementation_files += sorted((PROGRAM / "controller" / "tests").glob("*.py"))
    implementation_files += [
        Path(__file__).resolve(),
        Path(__file__).with_name("generate_m6_artifacts.py"),
        PROGRAM / "controller" / "requirements-verification.txt",
    ]
    report = {
        "report_schema": "M6-v1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "project_root": str(ROOT),
        "controller_policy_version": "M6-v1.0",
        "verification_requirements_sha256": sha256(
            PROGRAM / "controller" / "requirements-verification.txt"
        ),
        "environment_packages": {
            "mujoco": installed_version("mujoco"),
            "numpy": installed_version("numpy"),
            "opencv-python-headless": installed_version("opencv-python-headless"),
            "Pillow": installed_version("Pillow"),
            "PyYAML": installed_version("PyYAML"),
        },
        "ssot_sha256": sha256(SPEC / "параметры_системы.yaml"),
        "mjcf_sha256": sha256(ROOT / "03_Модель_и_схемы" / "source" / "scara_color_sorter_m2.xml"),
        "implementation_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                                   for path in implementation_files},
        "suite_results": suites,
        "controlled_trace": trace,
        "diagram_exports": export_checks,
        "specification_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                                  for path in source_files},
        "scope_note": "M6 controller contract/unit verification only; no M7 physics integration or M9 campaign.",
        "passed": all(item["passed"] for item in suites)
                   and all(all(export.get(key) for key in ("valid_xml", "valid_png", "valid_pdf"))
                           for export in export_checks.values()),
    }
    out = VERIFY / "M6_verification_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out), "suite_results": suites,
                      "controlled_trace": trace, "diagram_exports": export_checks,
                      "passed": report["passed"]}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
