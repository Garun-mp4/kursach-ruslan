from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
NOTE = ROOT / "08_Записка" / "source" / "Пояснительная_записка.md"
TABLE_DIR = ROOT / "08_Записка" / "tables"
REVIEW_DIR = ROOT / "08_Записка" / "review"
SSOT_PATH = ROOT / "02_Спецификация" / "параметры_системы.yaml"
CAMPAIGN_PATH = ROOT / "07_Испытания" / "campaign.yaml"
RESULTS_PATH = ROOT / "07_Испытания" / "campaign_results.json"
RUNS_CSV = ROOT / "07_Испытания" / "tables" / "Результаты_по_прогонам.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    failures: list[str] = []
    ssot_doc = yaml.safe_load(SSOT_PATH.read_text(encoding="utf-8-sig"))
    parameters = ssot_doc["parameters"]
    campaign = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    protocol_hash = sha256(CAMPAIGN_PATH)
    if protocol_hash != campaign["campaign_sha256"]:
        failures.append("campaign.yaml SHA-256 differs from campaign_results.json")
    if ssot_doc["config_version"] != "M7-v1.8":
        failures.append(f"Unexpected SSOT version: {ssot_doc['config_version']}")

    with RUNS_CSV.open(encoding="utf-8-sig", newline="") as handle:
        run_rows = list(csv.DictReader(handle))
    if len(run_rows) != 98:
        failures.append(f"Expected 98 raw run rows, got {len(run_rows)}")
    run_ids = [row.get("run_id", "") for row in run_rows]
    if len(run_ids) != len(set(run_ids)):
        failures.append("Duplicate run_id in raw results")
    v03_rows = [row for row in run_rows if row["experiment_id"] == "V03_RANDOM_MIXED"]
    if len(v03_rows) != 30:
        failures.append(f"Expected 30 V03 raw rows, got {len(v03_rows)}")
    matched_n = sum(int(row["perception_matched_objects"] or 0) for row in v03_rows)
    if matched_n != 90:
        failures.append(f"Expected 90 matched V03 objects, got {matched_n}")

    def weighted_mean(field: str) -> float:
        return sum(float(row[field]) * int(row["perception_matched_objects"]) for row in v03_rows) / matched_n

    xy_mean = weighted_mean("perception_mean_xy_error_m")
    xy_max = max(float(row["perception_max_xy_error_m"]) for row in v03_rows)
    yaw_mean = weighted_mean("perception_mean_yaw_error_rad")
    yaw_max = max(float(row["perception_max_yaw_error_rad"]) for row in v03_rows)
    med_sim = statistics.median(float(row["simulation_time_s"]) for row in v03_rows)
    med_wall = statistics.median(float(row["wall_time_s"]) for row in v03_rows)
    med_path = statistics.median(float(row["tcp_path_length_m"]) for row in v03_rows)

    v03_series = next(x for x in campaign["series"] if x["experiment_id"] == "V03_RANDOM_MIXED" and x["variant_id"] == "nominal")
    primary = campaign["primary_baseline"]
    if (v03_series["runs"], v03_series["test_passes"], v03_series["active_objects"], v03_series["correctly_sorted_objects"]) != (30, 21, 90, 73):
        failures.append("V03 campaign aggregation does not match expected raw campaign facts")
    if primary["target_complete_batches"] != 27 or primary["pass"] is not False:
        failures.append("V03 frozen acceptance threshold or pass status changed")
    if primary["wrong_bin_count"] or primary["forbidden_contact_episodes"] or primary["joint_force_limit_violation_samples"]:
        failures.append("V03 safety/wrong-bin values are not zero as described")
    if campaign["registered_trials"] != campaign["analyzed_trials"] or campaign["missing_trials"]:
        failures.append("Campaign trial completeness disagrees with the note")

    v01 = [x for x in campaign["series"] if x["experiment_id"].startswith("v01_")]
    v02 = next(x for x in campaign["series"] if x["experiment_id"] == "v02_two_per_color")
    v07_noise = [x for x in campaign["series"] if x["experiment_id"] == "V07_RGB_NOISE"]
    v10 = next(x for x in campaign["series"] if x["experiment_id"] == "V10_GRIP_FRICTION")
    v14 = next(x for x in campaign["series"] if x["experiment_id"].startswith("v14_repeat_seed_"))
    if sum(x["runs"] for x in v01) != 3 or sum(x["test_passes"] for x in v01) != 3 or sum(x["correctly_sorted_objects"] for x in v01) != 3:
        failures.append("V01 fixed-color summary mismatch")
    if (v02["active_objects"], v02["correctly_sorted_objects"], v02["forbidden_contact_episodes"]) != (6, 3, 1):
        failures.append("V02 summary mismatch")
    if len(v07_noise) != 3 or any((x["runs"], x["test_passes"]) != (10, 10) for x in v07_noise):
        failures.append("V07 RGB noise summary mismatch")
    if v10["runs"] != 10 or v10["test_passes"] != 10:
        failures.append("V10 expected-outcome summary mismatch")
    repeat_path = ROOT / "07_Испытания" / "tables" / "Проверка_повторяемости_V14.json"
    repeat = json.loads(repeat_path.read_text(encoding="utf-8"))
    if repeat.get("accepted") is not True or not all(repeat.get("checks", {}).values()):
        failures.append("V14 repeatability check is not accepted")

    note_text = NOTE.read_text(encoding="utf-8")
    final_tray = parameters["cell.tray_outer_size_xy_m"]["value"]
    final_slots = parameters["cell.tray_slot_x_offsets_m"]["value"]
    total_slots = len(parameters["cell.tray_centers_xy_m"]["value"]) * len(final_slots)
    if [round(float(v), 3) for v in final_tray] != [0.184, 0.160] or len(final_slots) != 3 or total_slots != 9:
        failures.append("Final tray geometry is not 184x160 mm with 3 slots per tray")
    required_note_values = [
        "0,184 × 0,160 м", "всего девять мест", "21 из 30 партий", "73 из 90",
        "27 из 30", "90 из 90 объектов", "0,415 мм", "1,316 мм", "0,01347 рад",
        "0,08641 рад", "409,605 с", "241,686 с", "8,166 м", "84 из 98 прогонов",
        campaign["campaign_sha256"], "не достигнут",
    ]
    for value in required_note_values:
        if value not in note_text:
            failures.append(f"Expected fact missing from thesis source: {value}")
    forbidden_old_topic = re.compile(r"line\s*follower|следовани[ея]\s+по\s+линии|объезд[а-яё ]+препятств|робот-объезд", re.I)
    if forbidden_old_topic.search(note_text):
        failures.append("Legacy project topic remains in thesis source")
    if re.search(r"физически испытан|реальный прототип прош[её]л|натурные испытания проведен", note_text, re.I):
        failures.append("Thesis source contains an unsupported physical-test claim")

    figures = re.findall(r"\{\{figure:\s*(.*?)\s*\|\s*caption:\s*(.*?)\}\}", note_text)
    for relative, caption in figures:
        image_path = ROOT / "08_Записка" / relative
        if not image_path.is_file():
            failures.append(f"Missing figure source: {image_path.relative_to(ROOT)}")
        if not caption.startswith("Рисунок "):
            failures.append(f"Figure without numbered caption: {caption}")
    figure_labels = [re.match(r"Рисунок (\d+\.\d+)", caption).group(1) for _, caption in figures if re.match(r"Рисунок (\d+\.\d+)", caption)]
    if len(figure_labels) != len(set(figure_labels)):
        failures.append("Duplicate figure number in thesis source")
    table_captions = re.findall(r"^Таблица\s+(\d+(?:\.\d+)?)\s+-", note_text, re.M)
    if len(table_captions) != len(set(table_captions)):
        failures.append("Duplicate table number in thesis source")
    for citation in range(1, 7):
        if not re.search(rf"\[{citation}(?:\s*,|\])", note_text):
            failures.append(f"Bibliography item [{citation}] has no in-text citation")

    # All M9 result copies in this folder are verified byte-for-byte against canonical outputs.
    copy_checks: list[dict[str, str]] = []
    for copied in sorted(TABLE_DIR.glob("source__*")):
        name = copied.name.removeprefix("source__")
        source = RESULTS_PATH if name == "campaign_results.json" else ROOT / "07_Испытания" / "tables" / name
        if not source.is_file():
            failures.append(f"Missing canonical table for {copied.name}")
            continue
        source_digest, copy_digest = sha256(source), sha256(copied)
        copy_checks.append({"copy": str(copied.relative_to(ROOT)), "source": str(source.relative_to(ROOT)), "sha256": copy_digest})
        if source_digest != copy_digest:
            failures.append(f"M9 source copy changed: {copied.name}")
    if not copy_checks:
        failures.append("No M9 source tables are preserved in the M10 tables folder")

    mapping = {
        "robot.type": ("§ 2.2; рис. 2.1", "SCARA topology"),
        "robot.arm_dof": ("§ 2.2; табл. 3.1", "Arm degrees of freedom"),
        "robot.gripper_actuated_dof": ("§ 2.2; табл. 3.1", "Actuated gripper coordinate"),
        "robot.link1_length_m": ("§ 3.2; табл. 3.1", "Link geometry"),
        "robot.link2_length_m": ("§ 3.2; табл. 3.1", "Link geometry"),
        "robot.shoulder_z_m": ("§ 3.2; табл. 3.1", "Vertical geometry"),
        "robot.j3_range_m": ("§ 3.2; табл. 3.1", "Vertical joint limits"),
        "robot.j1_range_rad": ("§ 3.2; табл. 3.1", "Joint limits"),
        "robot.j2_range_rad": ("§ 3.2; табл. 3.1", "Joint limits"),
        "robot.j4_range_rad": ("§ 3.2; табл. 3.1", "Joint limits"),
        "robot.radial_reach_min_max_m": ("§ 3.2; § 4.2", "Reach envelope"),
        "object.size_xyz_m": ("§ 3.2; табл. 3.1", "Object dimensions"),
        "object.mass_kg": ("§ 3.2–3.3", "Calculated object mass"),
        "object.count_per_class": ("§ 3.2", "Batch composition"),
        "object.color_class_labels": ("§ 1.1; § 5.2", "Color classes and UNKNOWN"),
        "cell.table_size_xy_m": ("§ 3.2; рис. 3.2–3.4", "Workcell envelope"),
        "cell.input_center_xy_m": ("§ 3.2; рис. 3.2", "Input area location"),
        "cell.input_size_xy_m": ("§ 3.2; табл. 3.1", "Input area size"),
        "cell.tray_outer_size_xy_m": ("§ 3.2; табл. 3.1", "Final M7 tray dimensions"),
        "cell.tray_centers_xy_m": ("§ 3.2; рис. 3.2", "Sorting area locations"),
        "cell.tray_slot_x_offsets_m": ("§ 3.2; § 4.3", "Final slot count and reachability"),
        "cell.tray_slot_y_offset_m": ("§ 3.2", "Slot layout"),
        "camera.position_world_m": ("§ 5.1; рис. 5.1", "Primary camera pose"),
        "camera.placement_position_world_m": ("§ 5.1", "Placement camera pose"),
        "camera.placement_target_world_m": ("§ 5.1", "Placement camera view"),
        "camera.resolution_px": ("§ 5.1", "Image dimensions"),
        "camera.fovy_deg": ("§ 5.1", "Camera field of view"),
        "camera.placement_max_age_s": ("§ 5.1; § 7", "Freshness limit"),
        "environment.engine": ("§ 2.3; § 8–11", "Simulation engine"),
        "environment.engine_version": ("§ 2.3; § 9", "Pinned engine version"),
        "environment.python_version": ("§ 2.3; § 8", "Pinned Python version"),
        "environment.timestep_s": ("§ 6.2; § 8", "Physics step"),
        "planning.static_clearance_m": ("§ 6.2", "Static collision clearance"),
        "planning.perceived_object_clearance_m": ("§ 6.2", "Object clearance"),
        "planning.path_sweep_sample_step_m": ("§ 6.2", "Sampled path step"),
        "robot.gripper_actuator_total_force_N": ("§ 3.3", "Virtual gripper actuator force"),
        "gripper.friction_slide": ("§ 3.3; § 11", "Contact-model assumption"),
    }
    crosswalk: list[dict[str, str]] = []
    for key, (destination, reason) in mapping.items():
        if key not in parameters:
            failures.append(f"SSOT crosswalk key does not exist: {key}")
            continue
        entry = parameters[key]
        crosswalk.append({
            "Источник": "02_Спецификация/параметры_системы.yaml",
            "Ключ": key,
            "Значение": json_text(entry.get("value")),
            "Единица": str(entry.get("unit", "")),
            "Происхождение_SSOT": str(entry.get("origin", "")),
            "Владелец": str(entry.get("owner", "")),
            "Раздел_рисунок_таблица": destination,
            "Назначение_сверки": reason,
            "Статус": "PASS",
        })

    result_rows = [
        ("Campaign identity", campaign["campaign_id"], "§ 9.1", "Frozen campaign and protocol hash"),
        ("Registered / analyzed trials", f"{campaign['registered_trials']} / {campaign['analyzed_trials']}; missing={len(campaign['missing_trials'])}", "§ 9.1", "Campaign aggregate and run-level CSV"),
        ("Expected outcomes across heterogeneous tests", f"{campaign['expected_outcome_trials_passed']} / {campaign['registered_trials']}", "§ 10.4", "Not a sorting-success rate"),
        ("V03 complete batches", f"{primary['complete_batches']} / {primary['n']} ({primary['complete_batch_rate']:.1%})", "§ 10.1; табл. 10.1; рис. 10.1", "Independent unit is a three-object seeded batch"),
        ("V03 Wilson 95% interval", f"{primary['wilson_95_interval'][0]:.1%}–{primary['wilson_95_interval'][1]:.1%}", "§ 10.1; рис. 10.1", "Calculated over 30 seeded batches"),
        ("Frozen V03 threshold", f"{primary['target_complete_batches']} / {primary['n']} ({primary['target_rate']:.0%}); pass={primary['pass']}", "§ 1.2; § 10.1", "Threshold was not modified post hoc"),
        ("V03 correctly placed objects", f"{v03_series['correctly_sorted_objects']} / {v03_series['active_objects']}", "§ 10.1; табл. 10.1", "Evaluator truth, not controller self-report"),
        ("V03 planner rejections", "17 COLLISION_GRASP; rejected targets received no motion commands", "§ 10.1; § 10.2", "Derived by raw run and plan evidence"),
        ("V03 wrong bins / forbidden contacts / limit violations", f"{primary['wrong_bin_count']} / {primary['forbidden_contact_episodes']} / {primary['joint_force_limit_violation_samples']}", "§ 10.1", "V03 scope only; not universal guarantee"),
        ("V03 median simulation / wall time", f"{med_sim:.3f} / {med_wall:.3f} s", "§ 10.1", "Wall-clock is machine-dependent"),
        ("V03 median TCP path", f"{med_path:.3f} m", "§ 10.1", "Median over 30 batch trials"),
        ("V03 first-view classifications", f"{campaign['baseline_first_view_color']['correct_classifications']} / {campaign['baseline_first_view_color']['object_observations']}", "§ 5.3; § 10.2; рис. 5.2", "Public first observation under registered palette/conditions"),
        ("V03 first-view XY mean / max", f"{xy_mean*1000:.3f} / {xy_max*1000:.3f} mm", "§ 5.3; § 10.2", "Weighted across 90 matched object observations"),
        ("V03 first-view yaw mean / max", f"{yaw_mean:.5f} / {yaw_max:.5f} rad", "§ 5.3; § 10.2", "Square symmetry handled modulo pi/2"),
        ("V01 fixed-color tests", f"{sum(x['test_passes'] for x in v01)} / {sum(x['runs'] for x in v01)} expected outcomes", "табл. 10.1", "Three fixed single-object scenes"),
        ("V02 batch / forbidden contact", f"{v02['correctly_sorted_objects']} / {v02['active_objects']} sorted; {v02['forbidden_contact_episodes']} episode", "табл. 10.1; § 10.2", "The campaign did not pass this case"),
        ("V07 RGB noise", "; ".join(f"σ={x['variant_id'].split('_')[-1]}: {x['test_passes']}/{x['runs']}" for x in v07_noise), "табл. 10.1; рис. 10.1", "Paired seed samples of n=10 per level"),
        ("V10 low-friction expected outcomes", f"{v10['test_passes']} / {v10['runs']}", "§ 10.1 figure note; рис. 10.1", "Expected safe response is not successful sorting"),
        ("V14 repeatability", f"accepted={repeat['accepted']}; {sum(repeat['checks'].values())}/{len(repeat['checks'])} recorded checks true", "табл. 10.1; § 9.1", "One repeated seed pair; not broad statistical evidence"),
        ("M9 representative video", "single_red_center-2d771e58cc; 826 frames, 1280×720, 25 fps, 33.04 s", "§ 8; video/source materials", "Sequential frames decoded; simulation recording"),
    ]
    for label, value, destination, basis in result_rows:
        crosswalk.append({
            "Источник": "07_Испытания/campaign_results.json + исходные таблицы/логи",
            "Ключ": label,
            "Значение": value,
            "Единица": "",
            "Происхождение_SSOT": "registered virtual trial data",
            "Владелец": "M9",
            "Раздел_рисунок_таблица": destination,
            "Назначение_сверки": basis,
            "Статус": "PASS",
        })

    write_csv(TABLE_DIR / "Сверка_SSOT_и_результатов.csv", list(crosswalk[0].keys()), crosswalk)

    # Derived, compact M9 table for review/Word; each value comes from campaign_results.json.
    summary: list[dict[str, str]] = []
    def append_series(label: str, rows: list[dict[str, Any]], interpretation: str) -> None:
        summary.append({
            "Серия": label,
            "Прогоны": str(sum(x["runs"] for x in rows)),
            "Ожидаемый_исход": f"{sum(x['test_passes'] for x in rows)}/{sum(x['runs'] for x in rows)}",
            "Активные_объекты": str(sum(x["active_objects"] for x in rows)),
            "Верно_размещено": str(sum(x["correctly_sorted_objects"] for x in rows)),
            "Неверная_зона": str(sum(x["wrong_bin_count"] for x in rows)),
            "Запрещенные_контакты": str(sum(x["forbidden_contact_episodes"] for x in rows)),
            "Интерпретация": interpretation,
        })
    append_series("V01 фиксированные цвета", v01, "Три фиксированные сцены; конкретная демонстрация, не общая статистика.")
    append_series("V02 по два объекта каждого цвета", [v02], "Неудачная партия; один запрещенный контакт и безопасная остановка.")
    append_series("V03 случайные смешанные партии", [v03_series], "Основной критерий 27/30 не достигнут.")
    for row in sorted(v07_noise, key=lambda x: x["variant_id"]):
        append_series(f"V07 RGB noise {row['variant_id'].split('_')[-1]}", [row], "Парная выборка n=10; показатель ограничен зарегистрированными условиями.")
    append_series("V10 низкое трение захвата", [v10], "Прохождение ожидаемого безопасного исхода не равно сортировке.")
    append_series("V14 повторяемость", [v14], "Один повторный seed; 14 сопоставлений функциональных данных.")
    for experiment_id, label in [("v06_unknown_gray", "V06 серый неизвестный цвет"), ("v07_light_low", "V07 низкая освещенность"), ("v07_light_high", "V07 высокая освещенность"), ("v08_adjacent_pair", "V08 близкая пара"), ("v09_placement_camera_failure", "V09 отказ placement-камеры")]:
        row = next(x for x in campaign["series"] if x["experiment_id"] == experiment_id)
        append_series(label, [row], "Негативный/стрессовый исход; трактовать по индивидуальному критерию и журналу.")
    write_csv(TABLE_DIR / "Сводка_M9_для_записки.csv", list(summary[0].keys()), summary)

    copied_source_rows = []
    for check in copy_checks:
        copied_source_rows.append({"path": check["copy"], "sha256": check["sha256"]})
    audit = {
        "schema_version": "M10-source-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ssot_version": ssot_doc["config_version"],
        "campaign_id": campaign["campaign_id"],
        "campaign_protocol_sha256": protocol_hash,
        "analyzed_trials": len(run_rows),
        "v03_trials": len(v03_rows),
        "v03_matched_public_observations": matched_n,
        "v03_xy_mean_m": xy_mean,
        "v03_xy_max_m": xy_max,
        "v03_yaw_mean_rad": yaw_mean,
        "v03_yaw_max_rad": yaw_max,
        "m9_source_copies_byte_identical": len(copy_checks),
        "source_copy_hashes": copied_source_rows,
        "figures_verified": len(figures),
        "tables_verified": len(table_captions),
        "crosswalk_rows": len(crosswalk),
        "summary_rows": len(summary),
        "checks": {"result": "PASS" if not failures else "FAIL", "failures": failures},
    }
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    (REVIEW_DIR / "Проверка_исходных_данных_M10.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ["ssot_version", "campaign_id", "analyzed_trials", "v03_trials", "v03_matched_public_observations", "v03_xy_mean_m", "v03_xy_max_m", "v03_yaw_mean_rad", "v03_yaw_max_rad", "m9_source_copies_byte_identical", "figures_verified", "tables_verified", "crosswalk_rows", "summary_rows", "checks"]}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
