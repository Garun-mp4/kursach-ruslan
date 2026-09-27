from __future__ import annotations

import csv
import ast
from html import escape
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
PROGRAM = ROOT / "04_Программа"
if str(PROGRAM) not in sys.path:
    sys.path.insert(0, str(PROGRAM))

from controller.config import ControllerConfig
from controller.controller import PLAN_FATAL, PLAN_RETRYABLE, PLAN_SLOT_CONFLICT
from controller.fsm_spec import EXECUTION_PHASES, STATE_SPECS, TRANSITIONS
from controller.types import State
from planning.records import Phase, PlanCode


SPEC_DIR = ROOT / "02_Спецификация"
DIAGRAM_DIR = ROOT / "03_Модель_и_схемы" / "algorithm"
VERIFY_DIR = ROOT / "05_Верификация" / "controller"

STATE_RU = {
    State.INIT: "Проверка готовности",
    State.OBSERVE: "Получение кадра",
    State.SELECT: "Выбор объекта и места",
    State.PLAN: "Полный preflight M5",
    State.APPROACH: "Подход к объекту",
    State.DESCEND: "Опускание",
    State.GRASP: "Закрытие пальцев",
    State.VERIFY_HOLD: "Пробный подъём и проверка удержания",
    State.TRANSFER: "Перенос с контролем удержания",
    State.PLACE: "Опускание в зарезервированное место",
    State.RELEASE: "Отпускание",
    State.VERIFY_PLACE: "Проверка места камерой",
    State.RETREAT: "Отход и возврат",
    State.RECOVER: "Ограниченное восстановление",
    State.SAFE_STOP: "Защёлкнутая безопасная остановка",
    State.DONE: "Пакет завершён",
}

TRANSITION_LABEL = {
    (State.INIT, State.OBSERVE): "готовность подтверждена",
    (State.OBSERVE, State.SELECT): "свежая сцена",
    (State.OBSERVE, State.DONE): "3 свежих пустых кадра",
    (State.SELECT, State.PLAN): "объект и слот зарезервированы",
    (State.SELECT, State.OBSERVE): "повтор наблюдения",
    (State.SELECT, State.DONE): "нет допустимых целей",
    (State.PLAN, State.APPROACH): "полный валидный M5 Plan",
    (State.PLAN, State.RECOVER): "обработанный отказ планирования",
    (State.APPROACH, State.DESCEND): "подход завершён",
    (State.APPROACH, State.RECOVER): "сбой без возможного груза",
    (State.DESCEND, State.GRASP): "спуск завершён",
    (State.DESCEND, State.RECOVER): "сбой без возможного груза",
    (State.GRASP, State.VERIFY_HOLD): "закрытие завершено",
    (State.GRASP, State.RECOVER): "свежее подтверждение no-hold",
    (State.VERIFY_HOLD, State.TRANSFER): "свежий hold + источник пуст",
    (State.VERIFY_HOLD, State.RECOVER): "явный no-hold",
    (State.TRANSFER, State.PLACE): "перенос завершён, hold свежий",
    (State.PLACE, State.RELEASE): "опускание завершено, hold свежий",
    (State.RELEASE, State.RETREAT): "свежий датчик подтвердил пустой захват",
    (State.RETREAT, State.VERIFY_PLACE): "отход открыл камере обзор слота",
    (State.VERIFY_PLACE, State.OBSERVE): "исход укладки учтен; запросить новую сцену",
    (State.VERIFY_PLACE, State.RECOVER): "лимит цикла без удерживаемого груза",
    (State.RETREAT, State.RECOVER): "ошибка отхода без груза",
    (State.RECOVER, State.OBSERVE): "восстановление подтверждено",
    (State.SAFE_STOP, State.INIT): "операторский сброс + безопасное разрешение",
}

BEHAVIOR_TESTS = {
    (State.INIT, State.OBSERVE): "test_nominal_cycle_requires_public_hold_and_place_evidence",
    (State.INIT, State.SAFE_STOP): "test_init_fails_closed_without_fresh_empty_gripper_signal",
    (State.OBSERVE, State.SELECT): "test_nominal_cycle_requires_public_hold_and_place_evidence",
    (State.OBSERVE, State.DONE): "test_done_means_controller_terminated_not_all_items_sorted",
    (State.OBSERVE, State.SAFE_STOP): "test_camera_failures_are_bounded_and_latch_safe_stop",
    (State.SELECT, State.PLAN): "test_nominal_cycle_requires_public_hold_and_place_evidence",
    (State.SELECT, State.OBSERVE): "test_unknown_is_never_mapped_to_a_sorting_zone",
    (State.SELECT, State.DONE): "test_all_slots_full_yields_safe_skip_without_planning_or_fake_placement",
    (State.SELECT, State.SAFE_STOP): "test_emergency_stop_latches_from_every_active_state",
    (State.PLAN, State.APPROACH): "test_nominal_cycle_requires_public_hold_and_place_evidence",
    (State.PLAN, State.RECOVER): "test_problematic_candidate_does_not_starve_next_candidate",
    (State.PLAN, State.SAFE_STOP): "test_m5_plan_with_reordered_phases_is_rejected_before_motion",
    (State.APPROACH, State.DESCEND): "test_nominal_cycle_requires_public_hold_and_place_evidence",
    (State.APPROACH, State.RECOVER): "test_execution_failure_without_payload_releases_reservations",
    (State.APPROACH, State.SAFE_STOP): "test_cycle_timeout_without_payload_releases_reservations",
    (State.DESCEND, State.GRASP): "test_nominal_cycle_requires_public_hold_and_place_evidence",
    (State.DESCEND, State.RECOVER): "test_descend_execution_failure_with_fresh_no_hold_enters_recover",
    (State.DESCEND, State.SAFE_STOP): "test_emergency_stop_latches_from_every_active_state",
    (State.GRASP, State.VERIFY_HOLD): "test_nominal_cycle_requires_public_hold_and_place_evidence",
    (State.GRASP, State.RECOVER): "test_grasp_execution_failure_with_fresh_no_hold_enters_recover",
    (State.GRASP, State.SAFE_STOP): "test_grasp_execution_failure_with_unknown_hold_latches_safe_stop",
    (State.VERIFY_HOLD, State.TRANSFER): "test_nominal_cycle_requires_public_hold_and_place_evidence",
    (State.VERIFY_HOLD, State.RECOVER): "test_grasp_attempt_budget_stops_after_one_bounded_retry",
    (State.VERIFY_HOLD, State.SAFE_STOP): "test_hold_verification_timeout_with_no_fresh_sensor_is_safe_stop",
    (State.TRANSFER, State.PLACE): "test_nominal_cycle_requires_public_hold_and_place_evidence",
    (State.TRANSFER, State.SAFE_STOP): "test_lost_payload_during_transfer_quarantines_slot_and_stops",
    (State.PLACE, State.RELEASE): "test_nominal_cycle_requires_public_hold_and_place_evidence",
    (State.PLACE, State.SAFE_STOP): "test_emergency_stop_latches_from_every_active_state",
    (State.RELEASE, State.RETREAT): "test_nominal_cycle_requires_public_hold_and_place_evidence",
    (State.RETREAT, State.VERIFY_PLACE): "test_nominal_cycle_requires_public_hold_and_place_evidence",
    (State.VERIFY_PLACE, State.OBSERVE): "test_nominal_cycle_requires_public_hold_and_place_evidence",
    (State.RELEASE, State.SAFE_STOP): "test_release_confirmation_timeout_preserves_slot_and_stops",
    (State.VERIFY_PLACE, State.RECOVER): "test_verify_place_cycle_timeout_enters_recover_without_claiming_sort",
    (State.VERIFY_PLACE, State.SAFE_STOP): "test_emergency_stop_latches_from_every_active_state",
    (State.RETREAT, State.RECOVER): "test_retreat_execution_failure_never_claims_placement",
    (State.RETREAT, State.SAFE_STOP): "test_emergency_stop_latches_from_every_active_state",
    (State.RECOVER, State.OBSERVE): "test_recovery_budget_resets_after_completed_safe_recovery",
    (State.RECOVER, State.SAFE_STOP): "test_recovery_failure_and_timeout_escalate_to_safe_stop",
    (State.SAFE_STOP, State.INIT): "test_safe_stop_reset_requires_explicit_safe_disposition",
}

STATE_HANDLER_TESTS = {
    State.INIT: "test_init_fails_closed_without_fresh_empty_gripper_signal; test_nominal_cycle_requires_public_hold_and_place_evidence",
    State.OBSERVE: "test_camera_failures_are_bounded_and_latch_safe_stop; test_valid_empty_scene_is_not_treated_as_camera_failure; test_stale_frame_is_rejected_before_selection",
    State.SELECT: "test_unknown_is_never_mapped_to_a_sorting_zone; test_all_slots_full_yields_safe_skip_without_planning_or_fake_placement; test_multiple_objects_are_processed_once_across_batch_cycles",
    State.PLAN: "test_transient_planner_failure_has_one_bounded_retry; test_problematic_candidate_does_not_starve_next_candidate; test_m5_plan_with_reordered_phases_is_rejected_before_motion",
    State.APPROACH: "test_nominal_cycle_requires_public_hold_and_place_evidence; test_execution_failure_without_payload_releases_reservations",
    State.DESCEND: "test_nominal_cycle_requires_public_hold_and_place_evidence; test_descend_execution_failure_with_fresh_no_hold_enters_recover; test_emergency_stop_latches_from_every_active_state",
    State.GRASP: "test_nominal_cycle_requires_public_hold_and_place_evidence; test_grasp_execution_failure_with_fresh_no_hold_enters_recover; test_grasp_execution_failure_with_unknown_hold_latches_safe_stop",
    State.VERIFY_HOLD: "test_hold_verification_timeout_with_no_fresh_sensor_is_safe_stop; test_grasp_attempt_budget_stops_after_one_bounded_retry",
    State.TRANSFER: "test_lost_payload_during_transfer_quarantines_slot_and_stops; test_stale_hold_signal_during_transfer_latches_safe_stop",
    State.PLACE: "test_nominal_cycle_requires_public_hold_and_place_evidence; test_transfer_invariant_requires_fresh_public_hold",
    State.RELEASE: "test_release_confirmation_timeout_preserves_slot_and_stops; test_multiple_objects_are_processed_once_across_batch_cycles",
    State.VERIFY_PLACE: "test_nominal_cycle_requires_public_hold_and_place_evidence; test_evaluator_only_placement_evidence_is_ignored; test_place_timeout_quarantines_instead_of_counting_success; test_verify_place_cycle_timeout_enters_recover_without_claiming_sort",
    State.RETREAT: "test_nominal_cycle_requires_public_hold_and_place_evidence; test_retreat_execution_failure_never_claims_placement",
    State.RECOVER: "test_recovery_budget_resets_after_completed_safe_recovery; test_recovery_failure_and_timeout_escalate_to_safe_stop",
    State.SAFE_STOP: "test_safe_stop_holding_payload_preserves_slot_and_never_opens_gripper; test_safe_stop_reset_requires_explicit_safe_disposition",
    State.DONE: "test_done_means_controller_terminated_not_all_items_sorted",
}

ERROR_ROWS = [
    ("ERR-001", "Нет/невалиден/устарел кадр M4", "OBSERVE получает batch со статусом/временем вне контракта.", "Ограниченный повтор; после 3 последовательных ошибок защёлкнуть SAFE_STOP.", "Активных резервов нет; если ошибка в активном цикле — M6 не открывает захват.", "PERCEPTION_FAILURE / CAMERA_FAILURE_LIMIT", "3 последовательных ошибки; валидная сцена сбрасывает счётчик.", "test_camera_failures_are_bounded_and_latch_safe_stop; test_stale_frame_is_rejected_before_selection", "M7 предоставляет монотонные frame_id и simulation_time."),
    ("ERR-002", "Сцена содержит OCCLUDED или конфликтующие треки", "M4 batch отмечен неполным либо дублирующий track ID имеет разные данные.", "Повторно наблюдать; дубликаты схлопнуть только при согласованном треке; конфликт закрыть отказом.", "До SELECT резерв не создаётся.", "SCENE_INCOMPLETE / DUPLICATE_TRACK_COLLAPSED", "Ограничено max_scene_uncertainty_frames=3.", "test_conflicting_duplicate_track_fails_closed_instead_of_double_counting; test_same_m4_track_duplicates_collapse_before_reservation", "Не использовать evaluator object ID для track repair."),
    ("ERR-003", "Пустая валидная сцена", "Свежий M4 кадр имеет OK/NO_CANDIDATES и ноль detections.", "Считать только distinct frame_id; DONE после 3 подряд.", "Нет.", "EMPTY_SCENE_OBSERVED / BATCH_EMPTY_CONFIRMED", "Счетчик сбрасывается обнаружением объектов; timeout не равен пустой сцене.", "test_valid_empty_scene_is_not_treated_as_camera_failure; test_done_means_controller_terminated_not_all_items_sorted", "Это завершение batch, не 100% sorting success."),
    ("ERR-004", "Цвет UNKNOWN", "status/class_label M4 равны UNKNOWN.", "Не выбирать и не связывать с зоной; ждать 2 distinct кадра, затем SKIPPED_UNKNOWN.", "Нет.", "UNKNOWN_COLOR_OBSERVED / TRACK_SKIPPED", "Повтор ограничен unknown_confirm_frames=2.", "test_unknown_is_never_mapped_to_a_sorting_zone", "M5 допускает такой трек только как геометрическое препятствие, не как target."),
    ("ERR-005", "Нет свободного слота нужного цвета", "До захвата free_slots(class)=empty или M5 сообщает slot conflict.", "Пропустить цель/класс; не выдавать motion-команду.", "Освободить track; при SLOT_INVALID/OCCUPIED зарезервированный слот карантинировать.", "TRACK_SKIPPED / PLAN_REJECTED", "Автоматический повтор без изменения публичного состояния запрещен.", "test_all_slots_full_yields_safe_skip_without_planning_or_fake_placement", "M7 должен передавать публичную начальную занятость слотов."),
    ("ERR-006", "Временный отказ M5", "TIMEOUT, STALE_DETECTION или SCENE_UNCERTAIN.", "Освободить резервы, обновить наблюдение, повторить максимум один раз.", "Освободить оба резерва.", "PLAN_RETRY_SCHEDULED", "controller.max_plan_retries=1 на track.", "test_transient_planner_failure_has_one_bounded_retry", "M5 compute wall-time и M6 simulation-time watchdog различаются."),
    ("ERR-007", "Постоянный отказ M5: unreachable/коллизия/лимит/маршрут", "Неуспешный PlanResult вне transient/fatal/slot-conflict групп.", "Не двигаться; отметить SKIPPED_UNREACHABLE и рассмотреть следующую цель.", "Освободить track и slot.", "PLAN_REJECTED / TRACK_SKIPPED", "Не повторять детерминированный геометрический отказ.", "test_permanent_planner_failure_releases_reservations_and_skips_track; test_problematic_candidate_does_not_starve_next_candidate", "M5 остается единственным planner."),
    ("ERR-008", "Фатальная ошибка/несовпадение контракта M5", "Некорректная модель/SSOT/версия/порядок фаз/события/геометрия Plan.", "Немедленно SAFE_STOP до первого motion command.", "Без груза — освободить; неизвестный/удерживаемый груз — сохранить.", "SAFE_STOP", "Повтора нет без нового run/config.", "test_m5_plan_with_reordered_phases_is_rejected_before_motion; test_m5_plan_without_release_marker_is_rejected_before_motion", "Plan must match exact active frame, track, class, slot and hashes."),
    ("ERR-009", "No-hold после тестового подъёма", "Свежий разрешенный датчик сообщает holding=false после LIFT.", "Ограниченный OPEN_AND_RETREAT_NO_PAYLOAD, освободить резервы, повторить цель до бюджета.", "Освободить; slot не занят.", "GRASP_FAILED / RECOVERY_REQUESTED", "Максимум 2 grasp attempts; recovery один раз на recovery episode.", "test_grasp_failure_uses_bounded_recovery_without_claiming_hold; test_grasp_attempt_budget_stops_after_one_bounded_retry", "Использовать только finger contact/effort proxy."),
    ("ERR-010", "Hold неизвестен/устарел после подъема", "Нет свежего разрешенного сигнала или источник запрещен.", "SAFE_STOP и удержание захвата; no-hold не выводить из старого false.", "Сохранить track и slot.", "HOLD_EVIDENCE_TIMEOUT_UNKNOWN / SAFE_STOP", "Повтор захвата запрещен без свежего false.", "test_hold_verification_timeout_with_no_fresh_sensor_is_safe_stop; test_grasp_execution_failure_with_unknown_hold_latches_safe_stop", "M7 обязан помечать timestamp/source/valid."),
    ("ERR-011", "Удержание потеряно/устарело при TRANSFER или PLACE", "Fresh false либо срок hold evidence превысил 0.04 s.", "Защелкнуть SAFE_STOP, отменить дальнейшее движение и не отпускать автоматически.", "Slot QUARANTINED; при unknown цикл остается диагностически активен.", "PUBLIC_HOLD_SIGNAL_LOST / PUBLIC_HOLD_SIGNAL_STALE_OR_UNKNOWN", "Без повтора.", "test_lost_payload_during_transfer_quarantines_slot_and_stops; test_stale_hold_signal_during_transfer_latches_safe_stop", "Не подменять потерю по evaluator ground truth."),
    ("ERR-012", "Сбой выполнения движения", "command_id/status/timestamp в ExecutionFeedback.", "В approach/descent — recovery только при свежем подтверждении пустого захвата; возможный груз — SAFE_STOP.", "No-payload recovery releases; uncertain hold preserves.", "EXECUTION_FAILED / SAFE_STOP", "State/cycle timeouts конечны; stale command ID игнорируется.", "test_execution_failure_without_payload_releases_reservations; test_descend_execution_failure_with_fresh_no_hold_enters_recover; test_grasp_execution_failure_with_fresh_no_hold_enters_recover; test_grasp_execution_failure_with_unknown_hold_latches_safe_stop; test_retreat_execution_failure_never_claims_placement", "M7 обеспечивает stop/hold adapter."),
    ("ERR-013", "Захват не подтвержден датчиком после закрытия", "Свежий публичный сенсорный сигнал false; закрытие пальцев само по себе не считается успехом.", "Только тогда выполнить bounded recovery; stale/unknown — SAFE_STOP.", "Освободить на подтвержденном no-hold; сохранить на uncertain.", "GRASP_FAILED / HOLD_CONFIRMED", "2 попытки на цель.", "test_grasp_attempt_budget_stops_after_one_bounded_retry", "Не читать истинную позицию объекта или контакты evaluator."),
    ("ERR-014", "Release не подтвержден пустым захватом", "После команды RELEASE не поступает свежий false.", "Остаться в RELEASE до timeout, затем SAFE_STOP; не объявлять VERIFY_PLACE.", "Сохранить slot reservation.", "RELEASE_WAITING_FOR_EMPTY_GRIPPER / RELEASE_STATE_UNCERTAIN", "controller.gripper_release_confirmation_timeout_s=0.25 s.", "test_release_confirmation_timeout_preserves_slot_and_stops", "M7 public sensor must cover release transition."),
    ("ERR-015", "Placement evidence запрещен/устарел/неоднозначен", "Источник evaluator, старый frame_id, не-новый кадр или низкая уверенность.", "Игнорировать и ждать публичную камеру до bounded timeout.", "Slot сохраняется RESERVED до исхода; при timeout — QUARANTINED.", "PLACE_EVIDENCE_IGNORED / PLACE_EVIDENCE_INCONCLUSIVE", "Не повторять release/placement.", "test_evaluator_only_placement_evidence_is_ignored; test_place_timeout_quarantines_instead_of_counting_success", "Evaluator полностью отделен от controller."),
    ("ERR-016", "Неверные class/zone/slot или placement timeout", "Публичное наблюдение не совпало с резервом или не появилось до 0.5 s.", "Не подтверждать сортировку; quarantine slot; безопасно отойти/зафиксировать отказ.", "Slot QUARANTINED.", "PLACEMENT_NOT_CONFIRMED / PLACE_NOT_VERIFIED", "Без повторного отпускания.", "test_place_timeout_quarantines_instead_of_counting_success", "Истинная зона используется только независимым evaluator."),
    ("ERR-017", "Cycle или batch timeout", "simulation_time превышает deadline цикла/пакета.", "No-payload — bounded recovery; возможный payload — SAFE_STOP.", "При no-payload освободить; при held/unknown сохранить.", "CYCLE_TIMEOUT_NO_PAYLOAD / SAFE_STOP", "Cycle = duration_M5*1.5+30 s; batch = 1800 s.", "test_cycle_timeout_without_payload_releases_reservations; test_verify_place_cycle_timeout_enters_recover_without_claiming_sort", "Не использовать wall clock/FPS."),
    ("ERR-018", "Время симуляции нечисловое или пошло назад", "Input.simulation_time_s не finite/monotonic.", "Защелкнуть SAFE_STOP.", "По состоянию публичного удержания.", "INVALID_SIMULATION_TIME / SIMULATION_TIME_REGRESSION", "Сброс только оператором после безопасного разрешения.", "test_simulation_clock_regression_latches_stop", "Время всех интерфейсов — единые simulation seconds."),
    ("ERR-019", "Recovery не завершился или состояние руки неизвестно", "Feedback FAILED, timeout, invalid arm/gripper state или holding не false.", "SAFE_STOP; отметить исход объекта SAFE_FAILURE.", "Не считать slot свободным при удержании/неопределенности.", "RECOVERY_FAILED_OR_STATE_UNKNOWN / RECOVERY_TIMEOUT", "Одна попытка на episode; новая допускается после успешного recovery.", "test_recovery_failure_and_timeout_escalate_to_safe_stop", "План recovery физически реализуется только M7."),
    ("ERR-020", "Аварийная остановка", "Внешний emergency_stop из любого активного состояния.", "Защелкнуть SAFE_STOP, очистить активный command, не выдавать plan/release.", "При held сохранять резерв; при подтвержденном empty освобождать; unknown не трактовать как empty.", "SAFE_STOP", "Сброс только с operator_reset + payload_safely_resolved + valid state.", "test_emergency_stop_latches_from_every_active_state; test_safe_stop_reset_requires_explicit_safe_disposition", "Физическая остановка M7 adapter не реализуется в M6."),
]


def fmt(value: float | int) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", ",")


def md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def write_csv(path: Path, headers: list[str], rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def controller_timeout_rows(config: ControllerConfig) -> list[tuple[str, str, str]]:
    by_state = {
        State.INIT: f"{fmt(config.p('init_timeout_s'))} s",
        State.OBSERVE: f"{fmt(config.p('observation_timeout_s'))} s",
        State.SELECT: f"{fmt(config.p('select_timeout_s'))} s",
        State.PLAN: f"{fmt(config.p('plan_response_timeout_s'))} s",
        State.APPROACH: "max(min, сумма M5 phase durations × factor + margin)",
        State.DESCEND: "max(min, M5 DESCENT duration × factor + margin)",
        State.GRASP: "max(min, M5 GRASP_CLOSE duration × factor + margin)",
        State.VERIFY_HOLD: (
            f"LIFT: динамический budget; затем {fmt(config.p('hold_verification_timeout_s'))} s"
        ),
        State.TRANSFER: "max(min, сумма M5 TRANSFER + PREPLACE × factor + margin)",
        State.PLACE: "max(min, M5 PLACE_DESCENT duration × factor + margin)",
        State.RELEASE: (
            f"M5 RELEASE; после команды без fresh empty-signal: "
            f"{fmt(config.p('gripper_release_confirmation_timeout_s'))} s"
        ),
        State.VERIFY_PLACE: f"{fmt(config.p('place_verification_timeout_s'))} s",
        State.RETREAT: "max(min, сумма M5 RETREAT/SAFE_RETURN/SAFE_HOME × factor + margin)",
        State.RECOVER: f"{fmt(config.p('recovery_timeout_s'))} s",
        State.SAFE_STOP: "нет; защелкнут до явного разрешенного reset",
        State.DONE: "нет; терминальное состояние",
    }
    return [(state.value, STATE_RU[state], by_state[state]) for state in State]


def write_fsm_doc(config: ControllerConfig) -> None:
    lines = [
        "# Конечный автомат контроллера цветовой сортировки",
        "",
        f"Версия политики: **{config.policy_version}**",
        "",
        f"Версия SSOT, применённая для текущей M7-интеграции: **{config.config_version}**",
        "",
        f"SHA-256 SSOT на генерации: `{config.ssot_sha256}`",
        f"SHA-256 MJCF модели: `{config.model_sha256}`",
        "",
        "Контроллер — независимый логический слой: perception, M5 planner и исполнительные операции приходят через типизированные интерфейсы. На M6 FSM проверялась управляемыми входными последовательностями и test doubles; на M7 исполнительные интерфейсы связаны с контактной моделью MuJoCo и публичными RGB-адаптерами (приёмка — `01_Управление/приемка/M07.md`).",
        "",
        "## Диаграммы",
        "",
        "- Состояния: [`fsm_state.drawio`](../03_Модель_и_схемы/algorithm/fsm_state.drawio), [SVG](../03_Модель_и_схемы/algorithm/fsm_state.svg), [PNG](../03_Модель_и_схемы/algorithm/fsm_state.png).",
        "- Поток сортировки: [`sorting_flow.drawio`](../03_Модель_и_схемы/algorithm/sorting_flow.drawio), [SVG](../03_Модель_и_схемы/algorithm/sorting_flow.svg), [PNG](../03_Модель_и_схемы/algorithm/sorting_flow.png).",
        "",
        "## Состояния",
        "",
        "| State | Назначение | Вход / guard | Действие при входе | Успешный выход | Ошибка / timeout | Повтор и резерв |",
        "|---|---|---|---|---|---|---|",
    ]
    for spec in STATE_SPECS:
        cells = [
            f"`{spec.state.value}` — {STATE_RU[spec.state]}",
            spec.purpose,
            spec.inputs,
            f"{spec.entry_action}; команда: {spec.command}",
            spec.success,
            f"{spec.failure}; timeout: {spec.timeout}",
            f"{spec.retry}; {spec.reservation}",
        ]
        lines.append("| " + " | ".join(md_cell(value) for value in cells) + " |")

    lines.extend([
        "",
        "## Разрешенные переходы",
        "",
        "Список ниже берется из `controller/fsm_spec.py`; `SortController._transition()` отклоняет любую дугу, отсутствующую в нем.",
        "",
        "| Из | В | Причина / условие |",
        "|---|---|---|",
    ])
    for source in State:
        for target in sorted(TRANSITIONS[source], key=lambda state: list(State).index(state)):
            label = TRANSITION_LABEL.get((source, target),
                                         "аварийное событие / явный guard соответствующего handler")
            lines.append(f"| `{source.value}` | `{target.value}` | {label} |")

    lines.extend([
        "",
        "## Таймеры и retry budgets",
        "",
        "Все значения времени — секунды симуляции. Контроллер не читает wall-clock и FPS; переходы происходят только по входам `ControllerInput.simulation_time_s`.",
        "",
        "| Состояние | Тайм-аут / watchdog |",
        "|---|---|",
    ])
    for state, label, timeout in controller_timeout_rows(config):
        lines.append(f"| `{state}` — {label} | {timeout} |")
    lines.extend([
        "",
        f"- Выполнение M5-фаз: `max({fmt(config.p('execution_timeout_min_s'))} s, phase_duration × {fmt(config.p('execution_timeout_factor'))} + {fmt(config.p('execution_timeout_margin_s'))} s)`. В duration входят времена M5 samples соответствующих фаз.",
        f"- Полный объектный цикл: `M5_plan.duration_s × {fmt(config.p('cycle_timeout_factor'))} + {fmt(config.p('cycle_timeout_margin_s'))} s`; отсчет начинается только после принятия полного Plan.",
        f"- Пакет: {fmt(config.p('batch_timeout_s'))} s simulation time.",
        f"- Perception: до {config.p('max_camera_failures')} последовательных отказов; частичная сцена — до {config.p('max_scene_uncertainty_frames')} свежих кадров.",
        f"- Неизвестный класс: {config.p('unknown_confirm_frames')} distinct свежих кадра; пустая партия: {config.p('empty_scene_confirm_frames')} distinct свежих пустых кадров.",
        f"- M5 transient result: не более {config.p('max_plan_retries')} повторов на track; захват: не более {config.p('max_grasp_attempts')} полных попыток на track; цикл: не более {config.p('max_cycle_attempts_per_track')} попыток.",
        f"- Recovery: одна попытка на episode (`max_recovery_attempts`), счетчик обнуляется только после подтвержденного безопасного восстановления.",
        "- Тайм-аут без свежего подтверждения отсутствия груза не является доказательством `holding=False`: применяется SAFE_STOP.",
        "- M5 собственный compute wall-time budget является отдельным ограничением planner; он не заменяет M6 simulation-time ожидание результата.",
        "",
        "## M5 Plan preflight",
        "",
        "До `APPROACH` контроллер сверяет target track/class, зарезервированный slot, M2/M5 SSOT и модельные хеши, frame ID и время M4, полный упорядоченный список фаз, монотонность временных samples, конечность координат, фазовую последовательность, согласованность duration и обязательные `OBJECT_ATTACHED` в `LIFT` / `OBJECT_RELEASED` в `RELEASE`. Любое несовпадение приводит к `SAFE_STOP` до команды движения.",
        "",
        "## Публичные доказательства удержания и укладки",
        "",
        "- Допустимые источники удержания: `FINGER_CONTACT_SENSOR` или `GRIPPER_EFFORT_PROXY`; максимальный возраст — 0,04 s. Поза/ID/контакт из evaluator недопустимы.",
        "- `VERIFY_HOLD` требует завершения тестового `LIFT`, свежего сенсорного сигнала и нового M4 кадра, который не показывает цель в зоне источника. Только затем разрешен `TRANSFER`.",
        "- Перед и при `TRANSFER`/`PLACE` hold должен оставаться свежим. `false`, устаревшее или отсутствующее подтверждение приводит к SAFE_STOP и quarantine места.",
        "- `RELEASE` не доказывает укладку. Для controller-level подтверждения нужен новый публичный кадр с совпадающими class/zone/slot и confidence не ниже 0,75. `VERIFY_PLACE` не использует evaluator.",
        "- Текущий M4 ROI исключает зоны лотков. В M7 необходимо реализовать отдельный публичный placement-perception adapter без ground-truth доступа.",
        "",
        "## Состояния объекта и ячейки",
        "",
        "- Объект: `PENDING → RESERVED → PLACED_CONTROLLER_CONFIRMED`, либо `SKIPPED_UNKNOWN`, `SKIPPED_UNREACHABLE`, `SKIPPED_FULL_ZONE`, `SAFE_FAILURE`. До controller evidence не используется термин «отсортирован». `DONE` — завершение цикла batch controller, а не 100% успешность.",
        "- Slot: `FREE → RESERVED → OCCUPIED` только после публичного совпадающего наблюдения; неопределенность/потеря/ошибка результата переводит его в `QUARANTINED`; no-payload planning/grasp failure освобождает reserve.",
        "- `ReservationManager` обеспечивает одновременно максимум один active track и один слот для однорукого цикла; их владелец должен совпадать.",
        "",
        "## Структурированные события и инварианты",
        "",
        "События `ControllerEvent` содержат monotonically increasing event_id, `simulation_time_s`, state, стабильный `reason_code`, track/slot (если применимо) и структурированный `details`. Список статических event type извлекается из исходника и сохраняется в verification log.",
        "",
        "| Инвариант | Автоматическая проверка |",
        "|---|---|",
        "| Не более одной пары active track/slot reservation; владелец совпадает. | `ReservationManager.assert_invariants()` и `SortController._assert_invariants()`. |",
        "| Track с резервом имеет статус `RESERVED`; неиспользуемый slot не имеет владельца. | Каждый `step()` после обработки состояния. |",
        "| `TRANSFER` и `PLACE` требуют свежего публичного hold-сигнала; `PLACE` требует зарезервированный слот. | Каждый `step()`; нарушенный инвариант приводит к ошибке теста. |",
        "| Confirmed placement соответствует занятому slot, а не эвристике внутреннего FSM. | `_placed_slot_by_track` consistency check. |",
        "| `DONE` не удерживает резервации и команду движения; `SAFE_STOP` не хранит active command. | Каждый `step()` и regression tests. |",
        "",
        "## Граница доказательств M6",
        "",
        "M6 подтверждает логику переходов и контрактов на controlled sequences. Он не подтверждает силовые контакты, физическое удержание, точность реальной камеры или фактическую укладку в MuJoCo-физике: это предмет M7, а итоговые виртуальные испытания — M9.",
        "",
    ])
    (SPEC_DIR / "FSM.md").write_text("\n".join(lines), encoding="utf-8")


def flowchart_cells() -> tuple[list[dict], list[dict]]:
    nodes = [
        ("step1", "<b>01 · ИНИЦИАЛИЗАЦИЯ И НАБЛЮДЕНИЕ</b><br>Проверить готовность системы; получить новый M4 frame.<br>Ошибка камеры ограниченно повторяется; пустой кадр не считается ошибкой.", 65, 120, 470, 130, "process"),
        ("step2", "<b>02 · ПРОВЕРКА СЦЕНЫ</b><br>Только свежие согласованные кадры.<br>3 distinct пустых кадра → DONE.<br>UNKNOWN: подтвердить 2 кадрами, не назначать зону.", 645, 120, 470, 130, "decision"),
        ("step3", "<b>03 · ВЫБОР И РЕЗЕРВИРОВАНИЕ</b><br>Выбрать известный класс; детерминированная оценка и tie-break по track_id.<br>Зарезервировать один track и один свободный slot.", 1225, 120, 470, 130, "process"),
        ("step4", "<b>04 · ПОЛНЫЙ M5 PREFLIGHT</b><br>Проверить все фазы цикла, свежесть M4, SSOT/model hash, геометрию и события.<br>До полного SUCCESS команды движения запрещены.", 1225, 350, 470, 130, "process"),
        ("step5", "<b>05 · ПОДХОД И ЗАХВАТ</b><br>APPROACH → DESCEND → GRASP_CLOSE → test LIFT.<br>Закрытие захвата не доказывает удержание: нужны свежий публичный датчик и новая сцена у источника.", 645, 350, 470, 130, "process"),
        ("step6", "<b>06 · ПЕРЕНОС И ОТПУСКАНИЕ</b><br>TRANSFER → PLACE → RELEASE.<br>Контролировать свежее удержание; движение и release выполняются только по принятому M5 plan и в зарезервированный slot.", 65, 350, 470, 130, "process"),
        ("step7", "<b>07 · ОТХОД ОТ ЛОТКА</b><br>После свежего подтверждения пустого захвата выполнить M5 RETREAT/SAFE_RETURN/SAFE_HOME.<br>Затем запросить новый кадр placement-камеры.", 65, 580, 470, 130, "process"),
        ("step8", "<b>08 · ПУБЛИЧНАЯ ПРОВЕРКА УКЛАДКИ</b><br>Сверить класс, зону, slot и порог confidence по новому RGB-кадру после отхода.<br>Evaluator ground truth контроллеру недоступен.", 645, 580, 470, 130, "decision"),
        ("step9", "<b>09 · ФИКСАЦИЯ ИСХОДА</b><br>Совпадение → PLACED_CONTROLLER_CONFIRMED и slot OCCUPIED.<br>Иначе → SAFE_FAILURE; slot QUARANTINED. Track history не допускает повторного учета.", 1225, 580, 470, 130, "success"),
        ("policy_retry", "<b>ПОВТОР / ПРОПУСК</b><br>Transient M5: не более 1 retry с новым кадром.<br>Нет slot → SKIPPED_FULL_ZONE.<br>Постоянный unreachable/route failure → SKIPPED_UNREACHABLE.<br>Не-starvation: рассматривается следующий кандидат.", 65, 850, 510, 205, "warning"),
        ("policy_recover", "<b>ОГРАНИЧЕННОЕ ВОССТАНОВЛЕНИЕ</b><br>Только свежий подтвержденный no-hold позволяет открыть захват, отойти и повторить цикл в заданном бюджете.<br>После подтвержденного recovery бюджет episode сбрасывается.", 645, 850, 510, 205, "warning"),
        ("policy_stop", "<b>SAFE_STOP / КАРАНТИН</b><br>Unknown/stale payload, потеря груза, fatal contract mismatch, release ambiguity, timeout с неизвестным грузом или emergency: отменить движение, сохранить безопасное удержание, не засчитывать сортировку.", 1225, 850, 510, 205, "stop"),
    ]
    edges = [
        ("step1", "step2", "валидный кадр"), ("step2", "step3", "есть цель + слот"),
        ("step3", "step4", "track и slot зарезервированы"),
        ("step4", "step5", "полный план принят"),
        ("step5", "step6", "свежий hold подтвержден"),
        ("step6", "step7", "release подтвержден"),
        ("step7", "step8", "новый public frame"),
        ("step8", "step9", "исход цикла записан"),
    ]
    return nodes, edges


def xml_cell_object(parent: ET.Element, cell_id: str, label: str, style: str,
                    *, x: int | None = None, y: int | None = None,
                    width: int | None = None, height: int | None = None,
                    source: str | None = None, target: str | None = None,
                    edge: bool = False, metadata: dict[str, str] | None = None,
                    waypoints: tuple[tuple[int, int], ...] = ()) -> None:
    attrs = {"id": cell_id, "label": label}
    if metadata:
        attrs.update(metadata)
    obj = ET.SubElement(parent, "object", attrs)
    cell_attrs = {"style": style, "parent": "1"}
    if edge:
        cell_attrs["edge"] = "1"
        if source:
            cell_attrs["source"] = source
        if target:
            cell_attrs["target"] = target
    else:
        cell_attrs["vertex"] = "1"
    cell = ET.SubElement(obj, "mxCell", cell_attrs)
    if edge:
        geom = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        if waypoints:
            points = ET.SubElement(geom, "Array", {"as": "points"})
            for point_x, point_y in waypoints:
                ET.SubElement(points, "mxPoint", {"x": str(point_x), "y": str(point_y)})
    else:
        geom = ET.SubElement(cell, "mxGeometry", {
            "x": str(x), "y": str(y), "width": str(width), "height": str(height),
            "as": "geometry",
        })


def mxfile(title: str, page_id: str, width: int, height: int,
           build_cells) -> bytes:
    root = ET.Element("mxfile", {
        "host": "app.diagrams.net", "modified": "2026-09-25T00:00:00.000Z",
        "agent": "Codex", "version": "24.7.17", "type": "device",
    })
    diagram = ET.SubElement(root, "diagram", {"id": page_id, "name": title})
    model = ET.SubElement(diagram, "mxGraphModel", {
        "dx": str(width), "dy": str(height), "grid": "1", "gridSize": "10",
        "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
        "fold": "1", "page": "1", "pageScale": "1", "pageWidth": str(width),
        "pageHeight": str(height), "math": "0", "shadow": "0",
    })
    graph_root = ET.SubElement(model, "root")
    ET.SubElement(graph_root, "mxCell", {"id": "0"})
    ET.SubElement(graph_root, "mxCell", {"id": "1", "parent": "0"})
    build_cells(graph_root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def write_state_diagram() -> None:
    # The success path is laid out as a three-row snake.  Exceptional arcs are
    # deliberately summarized in a separate policy panel; the exact legal
    # source/target pairs remain machine-derived in FSM.md and coverage CSV.
    layout = {
        State.INIT: (50, 125), State.OBSERVE: (340, 125),
        State.SELECT: (630, 125), State.PLAN: (920, 125),
        State.APPROACH: (1210, 125), State.DESCEND: (1210, 330),
        State.GRASP: (920, 330), State.VERIFY_HOLD: (630, 330),
        State.TRANSFER: (340, 330), State.PLACE: (50, 330),
        State.RELEASE: (50, 535), State.RETREAT: (340, 535),
        State.VERIFY_PLACE: (630, 535), State.DONE: (50, 755),
        State.RECOVER: (460, 755), State.SAFE_STOP: (870, 755),
    }
    fills = {
        State.SAFE_STOP: "#FCE8E6", State.DONE: "#E6F4EA",
        State.RECOVER: "#FFF4E5",
    }
    main_path = (
        State.INIT, State.OBSERVE, State.SELECT, State.PLAN,
        State.APPROACH, State.DESCEND, State.GRASP, State.VERIFY_HOLD,
        State.TRANSFER, State.PLACE, State.RELEASE, State.RETREAT,
        State.VERIFY_PLACE,
    )
    short_state = {
        State.INIT: "readiness + safe state",
        State.OBSERVE: "fresh M4 frame",
        State.SELECT: "track + slot reservation",
        State.PLAN: "full M5 preflight",
        State.APPROACH: "pregrasp route",
        State.DESCEND: "planned descent",
        State.GRASP: "GRASP_CLOSE",
        State.VERIFY_HOLD: "LIFT + fresh public evidence",
        State.TRANSFER: "hold monitored",
        State.PLACE: "reserved slot only",
        State.RELEASE: "fresh empty-gripper proof",
        State.VERIFY_PLACE: "fresh public class/zone/slot after retreat",
        State.RETREAT: "M5 safe retreat clears tray view",
        State.DONE: "batch ended; success rate is separate",
        State.RECOVER: "bounded, feedback-confirmed",
        State.SAFE_STOP: "latched; no automatic release",
    }

    def build(root: ET.Element) -> None:
        for state in State:
            x, y = layout[state]
            label = (f"<b>{escape(state.value)}</b><br>"
                     f"<font style=\"font-size:11px\">{escape(short_state[state])}</font>")
            fill = fills.get(state, "#EAF2F8")
            stroke = "#B42318" if state == State.SAFE_STOP else "#1F4E79"
            xml_cell_object(root, f"state_{state.value}", label,
                            f"rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor={fill};strokeColor={stroke};strokeWidth=2;fontColor=#172B34;fontSize=14;align=center;verticalAlign=middle;shadow=0;",
                            x=x, y=y, width=230, height=82,
                            metadata={"state": state.value})
        main_edges = list(zip(main_path, main_path[1:]))
        main_edges.append((State.VERIFY_PLACE, State.OBSERVE))
        for index, (source, target) in enumerate(main_edges, 1):
            is_cycle_return = (source, target) == (State.VERIFY_PLACE, State.OBSERVE)
            style = (
                "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
                "endArrow=block;endFill=1;strokeColor=#385D7A;strokeWidth=2;"
                "exitX=1;exitY=0.5;entryX=1;entryY=0.5;"
            ) if is_cycle_return else (
                "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
                "endArrow=block;endFill=1;strokeColor=#385D7A;strokeWidth=2;"
            )
            xml_cell_object(root, f"main_{index:02d}", "",
                            style,
                            source=f"state_{source.value}", target=f"state_{target.value}", edge=True,
                            metadata={"transition": f"{source.value}->{target.value}"},
                            waypoints=((1620, 576), (1620, 230), (570, 230)) if is_cycle_return else ())
        xml_cell_object(root, "exception_policy",
                        "<b>Отклонения от основного пути</b><br>" 
                        "DONE: валидная пустая сцена подтверждена 3 раз.<br>" 
                        "RECOVER: только bounded действие при подтверждённом состоянии без груза.<br>" 
                        "SAFE_STOP: неизвестный/удерживаемый груз, fatal mismatch, исчерпанный recovery или emergency.<br>" 
                        "Неразрешённые дуги не предполагаются: переходы ограничены TRANSITIONS.",
                        "rounded=1;whiteSpace=wrap;html=1;fillColor=#F6F9FA;strokeColor=#AAB7C4;fontColor=#172B34;fontSize=12;align=left;verticalAlign=middle;spacing=10;",
                        x=1210, y=720, width=390, height=160)
        xml_cell_object(root, "fsm_note",
                        "Карта показывает 13 основных состояний цикла. Отдельные исходы DONE / RECOVER / SAFE_STOP указаны ниже; точные допустимые рёбра, guard, тайм-ауты и причины отказа полностью перечислены в FSM.md и transition_coverage.csv.",
                        "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#AAB7C4;fontColor=#172B34;fontSize=11;align=left;verticalAlign=middle;spacing=8;",
                        x=50, y=925, width=1550, height=62)

    (DIAGRAM_DIR / "fsm_state.drawio").write_bytes(mxfile("FSM states", "m6-fsm", 1660, 1030, build))


def write_flowchart() -> None:
    nodes, edges = flowchart_cells()
    colors = {
        "start": ("#E6F4EA", "#2E7D32", "ellipse"),
        "end": ("#E6F4EA", "#2E7D32", "ellipse"),
        "process": ("#EAF2F8", "#1F4E79", "rounded=1"),
        "decision": ("#FFF4E5", "#C47A00", "rounded=1"),
        "warning": ("#FFF4E5", "#C47A00", "rounded=1"),
        "stop": ("#FCE8E6", "#B42318", "rounded=1"),
        "success": ("#E6F4EA", "#2E7D32", "rounded=1"),
    }

    def build(root: ET.Element) -> None:
        for node_id, label, x, y, width, height, kind in nodes:
            fill, stroke, shape = colors[kind]
            style_shape = shape if shape in {"ellipse", "rhombus"} else "rounded=1"
            html_label = "<div style=\"text-align:left;white-space:normal;padding:4px\">" + label + "</div>"
            xml_cell_object(root, node_id, html_label,
                            f"{style_shape};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;fontColor=#172B34;fontSize=13;align=left;verticalAlign=middle;spacing=12;",
                            x=x, y=y, width=width, height=height)
        for index, (source, target, label) in enumerate(edges, 1):
            xml_cell_object(root, f"flow_edge_{index:02d}", escape(label),
                            "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#385D7A;strokeWidth=2;fontColor=#385D7A;fontSize=10;labelBackgroundColor=#FFFFFF;",
                            source=source, target=target, edge=True,
                            metadata={"flow": f"{source}->{target}"})
        xml_cell_object(root, "flow_note",
                        "<b>Границы исходов:</b> `DONE` означает завершение обработки batch, а не 100% сортировку. `SAFE_FAILURE`, `SKIPPED_*` и controller-confirmed placement регистрируются раздельно. Физическое исполнение контактов не подтверждается на M6.",
                        "rounded=1;whiteSpace=wrap;html=1;fillColor=#F6F9FA;strokeColor=#AAB7C4;fontColor=#172B34;fontSize=12;align=left;verticalAlign=middle;spacing=10;",
                        x=65, y=1090, width=1630, height=58)

    (DIAGRAM_DIR / "sorting_flow.drawio").write_bytes(mxfile("Sorting flow", "m6-flow", 1780, 1180, build))


def write_invariants() -> None:
    (SPEC_DIR / "Инварианты_FSM.md").write_text(
        "# Инварианты контроллера M6\n\n"
        "Проверяются автоматически при каждом штатном завершении `SortController.step()` через `_assert_invariants()`; нарушение поднимает `ControllerContractError` и проваливает тест.\n\n"
        "1. Существует не более одной зарезервированной M4 track и одного slot; owner совпадает.\n"
        "2. Зарезервированный track имеет lifecycle `RESERVED` и равен `_active_track_id`.\n"
        "3. В `TRANSFER` и `PLACE` есть свежее публичное evidence `holding=True`; его возраст не превышает 0,04 s simulation time.\n"
        "4. `PLACE` дополнительно требует активный reserved slot того же track.\n"
        "5. Controller-confirmed placement всегда сохраняет `OCCUPIED` status соответствующего slot.\n"
        "6. `DONE` не удерживает track/slot reservation и active execution command.\n"
        "7. При SAFE_STOP с подтвержденным грузом обе резервации сохраняются; SAFE_STOP не содержит active execution command.\n"
        "8. Любой переход обязан принадлежать таблице `TRANSITIONS`; запрещенные переходы поднимают `IllegalTransitionError`.\n",
        encoding="utf-8",
    )


def write_sorting_policy(config: ControllerConfig) -> None:
    weights = config.score_weights
    classes = "\n".join(
        f"| `{label}` | `{zone}` | "
        f"{', '.join(f'`{slot}`' for slot in config.slot_ids if config.slot_classes[slot] == label)} |"
        for label, zone in sorted(config.class_to_zone.items())
    )
    retry_codes = ", ".join(code.value for code in sorted(PLAN_RETRYABLE, key=lambda item: item.value))
    fatal_codes = ", ".join(code.value for code in sorted(PLAN_FATAL, key=lambda item: item.value))
    slot_codes = ", ".join(code.value for code in sorted(PLAN_SLOT_CONFLICT, key=lambda item: item.value))
    lines = [
        "# Политика сортировки и принятия решений M6",
        "",
        f"Версия controller policy: `{config.policy_version}`. Версия конфигурации: `{config.config_version}`.",
        f"SSOT SHA-256: `{config.ssot_sha256}`; MJCF SHA-256: `{config.model_sha256}`.",
        "",
        "Этот документ фиксирует логику высокого уровня, реализованную в `04_Программа/controller/`. M6 проверяет решения контроллера на контролируемых интерфейсных последовательностях и тестовых doubles. Исполнение контактов, динамическая физика, реальное удержание и окончательная кампания виртуальных испытаний сюда не входят.",
        "",
        "## Разрешённые данные и интерфейсы",
        "",
        "- Perception предоставляет `DetectionBatch`: кадр, simulation timestamp, статус сцены, собственные track ID, оценённые класс/позицию/yaw/confidence/неопределённость/валидность. Устаревшие, невалидные и противоречивые данные не становятся целями.",
        "- Kinematic feasibility и полный цикл планирования принадлежат M3/M5. M6 не подменяет IK, collision checking или planner.",
        "- M5 получает только выбранный публичный batch, состояние суставов/захвата и зарезервированный слот. Команда движения создаётся только после полного preflight `PlanCode.SUCCESS` и проверки полного плана.",
        "- Исполнение, сенсор захвата, camera perception лотков, recovery и emergency stop приходят через типизированные порты M7. Контроллер не импортирует evaluator и не получает истинные цвет/позицию/контакт.",
        "",
        "## Основной функциональный цикл",
        "",
        "`INIT → OBSERVE → SELECT → PLAN → APPROACH → DESCEND → GRASP → VERIFY_HOLD → TRANSFER → PLACE → RELEASE → RETREAT → VERIFY_PLACE → OBSERVE`.",
        "",
        "1. `INIT` проверяет конфигурацию, зависимости, arm/gripper state, конечность/актуальность времени и отсутствие зависших резервов. До успешной инициализации движение не выдаётся.",
        "2. `OBSERVE` запрашивает кадр с `frame_id` новее последнего принятого. Ошибка камеры, частичная сцена, валидная пустая сцена и неполное измерение различаются.",
        f"3. Пустая партия завершается только после {config.p('empty_scene_confirm_frames')} различных свежих пустых кадров. `UNKNOWN` требует {config.p('unknown_confirm_frames')} различных свежих наблюдений; он никогда не получает сортировочную зону.",
        "4. `SELECT` выбирает только известный поддерживаемый класс, ещё не имеющий терминального результата, и до планирования резервирует ровно один controller track и один свободный слот соответствующего класса.",
        "5. `PLAN` отправляет полный цикл M5. Проверяются track/class/slot, версия и хеш SSOT/MJCF, source frame/time, точная последовательность M5 фаз, монотонность samples, конечность q/TCP/gripper, согласование duration и уникальные согласованные маркеры attachment/release. До принятия полного плана motion запрещён.",
        "6. Исполнение идёт только фазовыми группами из принятого плана. Закрытие пальцев само по себе не считается захватом. `VERIFY_HOLD` требует завершённый test lift, свежий разрешённый hold sensor и новый M4 кадр, в котором объект отсутствует у source estimate.",
        "7. На `TRANSFER` и `PLACE` hold evidence проверяется вновь и должно оставаться свежим. Потеря, устаревание или неизвестный груз вызывают `SAFE_STOP`; контроллер не пытается автоматически открыть захват.",
        "8. После команды `RELEASE` нужен свежий public sensor `holding=False`; затем исполнитель выполняет M5 retreat/return, чтобы освободить поле зрения над лотком.",
        "9. Только после завершения отхода `VERIFY_PLACE` запрашивает новый public RGB frame. Успех требует совпадающих class/zone/slot и confidence выше порога; подтвержденный слот становится `OCCUPIED`, неоднозначный или неверный результат переводится в `QUARANTINED`. `RELEASE` или завершение FSM сами по себе не доказывают sorting.",
        "",
        "## Детерминированный выбор цели и сортировочного места",
        "",
        "Меньшая оценка предпочтительнее. Веса поступают из SSOT и в этой версии равны:",
        "",
        "```text",
        f"score = {weights[0]:g}·(1-confidence) + {weights[1]:g}·min(1, σ_xy/σ_xy_limit) + ",
        f"        {weights[2]:g}·min(1, σ_yaw/σ_yaw_limit) + {weights[3]:g}·distance/max_candidate_distance + ",
        f"        {weights[4]:g}·min(1, prior_failure_count/max_cycle_attempts_per_track)",
        "tie-break: lower M4 track_id",
        "```",
        "",
        "Если TCP XY неизвестен, motion contribution равен нулю для всех кандидатов; остальные слагаемые и tie-break сохраняют детерминизм. M5 остаётся единственным источником решения о геометрической достижимости.",
        "",
        "| Класс | Зона | Слоты в конфигурационном порядке |",
        "|---|---|---|",
        classes,
        "",
        "Слоты выбираются из SSOT в фиксированном порядке. `FREE → RESERVED → OCCUPIED` разрешён только при подтверждении; неоднозначный/неверный исход переводит резерв в `QUARANTINED`. Ошибка без груза освобождает резерв. Зарезервированный или занятый слот не объявляется свободным.",
        "",
        "## Классификация результата M5",
        "",
        f"- Временные коды `{retry_codes}`: освободить обе резервации, запросить новое наблюдение и повторить не более {config.p('max_plan_retries')} раза на track. Повтор не исполняет старый план.",
        f"- Коды нехватки/конфликта места `{slot_codes}`: пропустить цель как `SKIPPED_FULL_ZONE`; некорректный/уже занятый заявленный slot карантинировать.",
        "- Прочие постоянные геометрические отказы: `SKIPPED_UNREACHABLE`, без движения и без повторения неизменённого запроса; далее рассматривается другая допустимая цель.",
        f"- Фатальные контрактные коды `{fatal_codes}` и успешный код без корректного полного плана: `SAFE_STOP` до первого motion command.",
        "",
        "## Учёт исходов и резерваций",
        "",
        "| Уровень | Значения | Смысл |",
        "|---|---|---|",
        "| Object/controller | `PENDING`, `RESERVED`, `PLACED_CONTROLLER_CONFIRMED`, `SKIPPED_UNKNOWN`, `SKIPPED_UNREACHABLE`, `SKIPPED_FULL_ZONE`, `SAFE_FAILURE` | Измеримый исход controller policy; это не ground-truth truth label. |",
        "| Placement slot | `FREE`, `RESERVED`, `OCCUPIED`, `QUARANTINED` | `OCCUPIED` только после подтверждённого нового public-camera evidence. |",
        "| Batch/controller | `DONE`, `SAFE_STOP` | `DONE` = batch controller остановил обработку; не означает 100% сортировку. `SAFE_STOP` — защёлкнутый safety outcome. |",
        "| Evaluator | отдельные truth-классы и физические контакты | M9-only независимая оценка, не подмешивается в решения и controller result. |",
        "",
        "## Retry budgets, таймеры и безопасные исходы",
        "",
        f"Все controller timers используют только секунды симуляции. Wall-clock/FPS не являются FSM-часами. Лимиты из SSOT: планирование retry={config.p('max_plan_retries')}; grasp attempts={config.p('max_grasp_attempts')}; полный cycle attempts/track={config.p('max_cycle_attempts_per_track')}; recovery attempts/episode={config.p('max_recovery_attempts')}; camera failures={config.p('max_camera_failures')}; uncertain-scene frames={config.p('max_scene_uncertainty_frames')}.",
        "",
        "| Timeout / порог | Значение SSOT | Реакция/владелец |",
        "|---|---:|---|",
        f"| INIT / OBSERVE / SELECT / PLAN | {fmt(config.p('init_timeout_s'))} / {fmt(config.p('observation_timeout_s'))} / {fmt(config.p('select_timeout_s'))} / {fmt(config.p('plan_response_timeout_s'))} s | Simulation-time watchdogs. |",
        f"| Hold evidence max age | {fmt(config.p('hold_evidence_max_age_s'))} s | Stale/unknown при переносе → SAFE_STOP. |",
        f"| Minimum placement confidence | {fmt(config.p('place_evidence_min_confidence'))} | Ниже порога evidence не подтверждает slot. |",
        f"| VERIFY_PLACE / RELEASE-confirmation / RECOVER | {fmt(config.p('place_verification_timeout_s'))} / {fmt(config.p('gripper_release_confirmation_timeout_s'))} / {fmt(config.p('recovery_timeout_s'))} s | Timeout policy в FSM.md. |",
        f"| Full cycle / batch | `M5 duration × {fmt(config.p('cycle_timeout_factor'))} + {fmt(config.p('cycle_timeout_margin_s'))} s` / {fmt(config.p('batch_timeout_s'))} s | No-payload: bounded recovery; payload unknown/held: SAFE_STOP. |",
        "| Execution phase | `max(min, phase duration × factor + margin)` | Коэффициенты/M5 samples берутся из SSOT/принятого плана. |",
        "",
        "## События и воспроизводимость",
        "",
        "`ControllerEvent` фиксирует последовательный `event_id`, `simulation_time_s`, state, `event_type`, стабильный `reason_code`, track/slot при наличии и типизированный details object. Каждая команда имеет `command_id`; запросы M4/M5/recovery имеют request ID. Это контролируемые воспроизводимые policy traces, а не результаты M9. M6 не использует randomness; run seed равен `null`.",
        "",
        "## Известные границы",
        "",
        "M4 input ROI не наблюдает sorting trays; M7 добавляет отдельную фиксированную public RGB placement view и `PlacementEvidence` adapter. Проверка выполняется только после M5 отхода и не использует evaluator data. M7 acceptance проверяет camera freshness, классификацию и сопоставление всех 9 слотов; геометрическая точность камеры проверена на виртуальной калибровочной сетке.",
        "",
    ]
    (SPEC_DIR / "Политика_сортировки.md").write_text("\n".join(lines), encoding="utf-8")


def write_forward_contracts(config: ControllerConfig) -> None:
    m7 = [
        "# Контракт интеграции M7 — исполнение и публичные сенсоры",
        "",
        f"Baseline controller policy `{config.policy_version}` / SSOT `{config.config_version}` / SSOT SHA-256 `{config.ssot_sha256}`.",
        "",
        "Документ фиксировал будущую границу модулей на M6; M7 интегрировал MuJoCo execution, public RGB placement view, grasp evidence и evaluator-isolated run artifacts.",
        "",
        "## Вход M6 controller из M7 ports",
        "",
        "- Каждый `ControllerInput.simulation_time_s` — конечное неубывающее simulation time; все event, request и feedback timestamps используют ту же шкалу.",
        "- `SystemHealth` сообщает readiness, валидность измеренной конфигурации руки и известность состояния захвата. Суставное состояние имеет ровно 4 конечных координаты, gripper position конечна; это измеренное состояние исполнительной модели, не evaluator truth.",
        "- `DetectionBatch` приходит от M4 и соблюдает `frame_id`, status, class/UNKNOWN, оценённые pose/confidence/uncertainty/freshness. Никакой simulator object ID/истинный цвет не добавляется в perception adapter.",
        "- `PlanResponse` коррелирует по request ID и возвращает M5 full-cycle result. Контроллер повторно проверяет SSOT/model hashes, source frame, phases, samples и events.",
        "- `ExecutionFeedback` коррелирует по точному `command_id`; статусы RUNNING/COMPLETED/FAILED и timestamp реальны для выполненного M7 motion adapter. Устаревший ID/time не продвигает FSM.",
        "- `GraspEvidence` допускает только `FINGER_CONTACT_SENSOR` либо `GRIPPER_EFFORT_PROXY`, а также `valid`, nullable holding и simulation timestamp. Закрытая команда пальцев не равна контакту.",
        "- `PlacementEvidence` поступает только из `PUBLIC_CAMERA_PERCEPTION`, содержит новый frame, status, class/zone/slot/confidence/reason/timestamp. M7 реализовал отдельную фиксированную RGB-view над лотками и публичный адаптер; truth из evaluator в контроллер не передается.",
        "- `RecoveryFeedback` коррелирует по recovery request ID и подтверждает arm state, known gripper state и `holding=False`; без этого controller не разрешит no-payload retry.",
        "- `emergency_stop` передаётся при любом аварийном событии; операторский reset несёт отдельные `operator_reset` и `payload_safely_resolved` interlocks.",
        "",
        "## Выход M6 controller в M7 ports",
        "",
        "- `ExecutionCommand` включает устойчивый run-scoped command ID, state, упорядоченный список M5 фаз и принятый полный `Plan`. Исполнитель не изменяет фазу, слоты, target или trajectory по собственной эвристике.",
        "- `safety_hold_requested=True` означает немедленно прекратить заданное движение и удерживать текущее закрытие gripper. SAFE_STOP не разрешает автоматическое открытие.",
        "- `RecoveryRequest.action=OPEN_AND_RETREAT_NO_PAYLOAD` допускается только после свежего сенсорного `holding=False`; UNKNOWN/held состояние должно завершиться остановкой/удержанием, не эвакуацией с открытым gripper.",
        "- `ObservationRequest.minimum_frame_id` обязателен: публикация этого или более нового кадра должна быть после соответствующего запроса, на том же simulation clock.",
        "",
        "## Непереходимые границы",
        "",
        "1. Вход контроллера строится из public sensors/perception/актуального actuator state; запрещено подавать напрямую истинные цвет, координаты, object ID, контакты или идеальную segmentation mask из MuJoCo.",
        "2. Evaluator truth хранится отдельно и вызывается только после завершения цикла для независимой проверки. Он не корректирует target selection, plan, grasp, release или FSM transition.",
        "3. Executor обязан публиковать ошибки контакта и фактическую потерю груза открыто; успешная команда и событие M5 `OBJECT_ATTACHED/OBJECT_RELEASED` не заменяют измерение выполнения.",
        "4. Если физическое состояние противоречит plan marker/sensor evidence, адаптер сообщает FAILED/uncertain state; controller fail-closed policy остаётся обязательной.",
        "5. M7 интеграционные проверки подтверждают корреляцию request/command IDs, свежесть сигналов, cancel/hold semantics и public placement camera; они не заменяют итоговую M9 кампанию.",
        "",
    ]
    (SPEC_DIR / "Контракт_M7_исполнения.md").write_text("\n".join(m7), encoding="utf-8")

    m9 = [
        "# Контракт телеметрии M9 для controller/M7",
        "",
        "M6 фиксирует поля наблюдений, необходимые будущей виртуальной кампании. Это не описание уже проведённых испытаний.",
        "",
        "| Запись | Обязательные поля | Зачем |",
        "|---|---|---|",
        "| Run manifest | run_id, scenario_id, config/SSOT version+hash, model hash, start conditions, simulation seed или `null`, software revision | Повторить один и тот же запуск и связать медиа. |",
        "| ControllerEvent | event_id, simulation_time_s, state, event_type, reason_code, track_id, slot_id, details | Восстановить решения FSM и тайминги. |",
        "| M4 observation | request_id, frame_id, timestamp/status, tracks/classes/confidence/pose/uncertainty/reason | Проверить использованные public inputs и freshness. |",
        "| M5 plan | request_id, code, plan_id, phases, duration, target/slot, SSOT/model hash, reject reason | Проверить acceptance/rejection и preflight. |",
        "| M7 execution | command_id, state/phase group, feedback status/time, joint/gripper state, public hold evidence/source/time | Сопоставить запрос и факт динамического исполнения. |",
        "| Placement | frame_id/source/time, observed class/zone/slot/confidence, controller decision | Подтвердить/отклонить controller-level evidence. |",
        "| Evaluator-only outcome | truth object/run IDs, true class, final bin/slot, contacts, collision/failure metrics | Независимо оценить конечный результат; недоступно controller. |",
        "",
        "Минимальная связь данных: `run_id → event_id → request_id/command_id → track_id/slot_id → frame_id → evaluator outcome`. Числа для M9 выводятся из raw run log и evaluator post-run, не печатаются вручную. Видео/кадры хранят ссылку на `run_id` и ту же конфигурацию.",
        "",
        "## Производные показатели M9",
        "",
        "Кандидаты для утверждения в M9: доля истинно правильных укладок; coverage объектов (placed/skipped/failure отдельно); доля безопасных отказов; wrong-bin count; collision/contact events; grasp retention; unknown-class handling; число retries; cycle/batch simulation time; path length/joint motion. Численные acceptance limits владеет M9 и их нельзя выводить из unit tests M6.",
        "",
        "## Доказательная граница",
        "",
        "В M6 нет M9 trial logs, измеренных успешностей или заявлений о динамической симуляции. M6 event trace — только запись одного тестового контролируемого последовательного сценария с M5 test double; он доказывает реакцию policy, не качество физики.",
        "",
    ]
    (SPEC_DIR / "Контракт_M9_телеметрии.md").write_text("\n".join(m9), encoding="utf-8")


def write_ground_truth_audit() -> None:
    sources = sorted((ROOT / "04_Программа" / "controller").glob("*.py"))
    forbidden = ("evaluator", "ground_truth", "truth_position", "true_color", "object_weld")
    findings = []
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imported = ast.unparse(node).lower()
                if any(token in imported for token in forbidden):
                    findings.append(f"Forbidden import pattern in {path.name}:{node.lineno}: {imported}")
            elif isinstance(node, ast.Attribute) and any(token in node.attr.lower() for token in forbidden):
                findings.append(f"Forbidden truth attribute in {path.name}:{node.lineno}: {node.attr}")
    if findings:
        raise RuntimeError("Ground-truth boundary audit failed: " + "; ".join(findings))
    report = [
        "# Аудит информационной границы controller M6",
        "",
        f"Проверено модулей controller: {len(sources)}.",
        "",
        "## Статическая проверка",
        "",
        "AST audit controller production modules: нет imports evaluator/ground_truth, нет доступа к полям true_color/truth_position/object_weld и нет обращений к evaluator-only атрибутам.",
        "",
        "## Контрактные проверки",
        "",
        "- Входные типы controller содержат M4 оценки и public evidence; `GraspEvidence`/`PlacementEvidence` требуют разрешённый source.",
        "- `test_evaluator_only_placement_evidence_is_ignored` передаёт evidence с evaluator source и подтверждает, что его нельзя засчитать.",
        "- Controller result сохраняет `evaluator_result: None`; M6 state/status не является truth label.",
        "- `_handle_verify_hold` проверяет новый M4 source observation относительно M4 source estimate, а не читает истинное положение объекта.",
        "",
        "## Вывод и граница",
        "",
        "Внутренний `PLACED_CONTROLLER_CONFIRMED` означает только совпадающее новое public-camera class/zone/slot evidence. Он не равен независимому evaluator-confirmed правильному цвету физически удержанного объекта. Evaluator остаётся будущим post-run компонентом M9.",
        "",
    ]
    (VERIFY_DIR / "ground_truth_boundary_audit.md").write_text("\n".join(report), encoding="utf-8")


def write_event_taxonomy() -> None:
    path = ROOT / "04_Программа" / "controller" / "controller.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    event_sites: dict[str, list[tuple[str, int]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "_emit" or len(node.args) < 2:
            continue
        event = ast.literal_eval(node.args[0]) if isinstance(node.args[0], ast.Constant) else ast.unparse(node.args[0])
        reason = ast.literal_eval(node.args[1]) if isinstance(node.args[1], ast.Constant) else ast.unparse(node.args[1])
        event_sites.setdefault(str(event), []).append((str(reason), node.lineno))
    rows = []
    for event, sites in sorted(event_sites.items()):
        reasons = sorted({reason for reason, _ in sites})
        lines = sorted({line for _, line in sites})
        rows.append((event, "; ".join(reasons), "controller.py:" + ",".join(map(str, lines)),
                     "Structured ControllerEvent; details schema is event-specific.",
                     "M6 controller policy; no evaluator truth."))
    write_csv(VERIFY_DIR / "event_taxonomy.csv",
              ["event_type", "reason_code_or_expression", "source_site", "semantic", "boundary"], rows)


def validate_generated_artifacts() -> None:
    transition_path = VERIFY_DIR / "transition_coverage.csv"
    with transition_path.open(encoding="utf-8-sig", newline="") as handle:
        transition_rows = list(csv.DictReader(handle))
    expected = {(source.value, target.value)
                for source, targets in TRANSITIONS.items() for target in targets}
    if set(BEHAVIOR_TESTS) != {
        (source, target) for source, targets in TRANSITIONS.items() for target in targets
    }:
        missing = expected - {(source.value, target.value) for source, target in BEHAVIOR_TESTS}
        extra = {(source.value, target.value) for source, target in BEHAVIOR_TESTS} - expected
        raise RuntimeError(f"Every allowed transition needs behavioral coverage; missing={sorted(missing)}, extra={sorted(extra)}")
    guard_only_test = "test_transition_guard_covers_all_declared_and_forbidden_edges"
    if guard_only_test in BEHAVIOR_TESTS.values():
        raise RuntimeError("Graph guard cannot substitute for behavioral transition coverage")
    test_source = (PROGRAM / "controller" / "tests" / "test_controller.py").read_text(encoding="utf-8")
    defined_tests = set(re.findall(r"^\s+def (test_[A-Za-z0-9_]+)\(", test_source, re.MULTILINE))
    handler_test_names = {
        name.strip()
        for test_list in STATE_HANDLER_TESTS.values()
        for name in test_list.split(";")
    }
    missing_tests = sorted((set(BEHAVIOR_TESTS.values()) | handler_test_names) - defined_tests)
    if missing_tests:
        raise RuntimeError(f"Transition coverage refers to missing behavior tests: {missing_tests}")
    actual = {(row["From"], row["To"]) for row in transition_rows}
    if actual != expected or len(transition_rows) != len(expected):
        raise RuntimeError("Generated transition coverage does not match fsm_spec.TRANSITIONS")
    if any(row["Behavioral path"] != "Да" or row["Test case"] == guard_only_test
           for row in transition_rows):
        raise RuntimeError("Every allowed transition must have a named behavioral test")
    state_path = VERIFY_DIR / "state_handler_coverage.csv"
    with state_path.open(encoding="utf-8-sig", newline="") as handle:
        state_rows = list(csv.DictReader(handle))
    if {row["State"] for row in state_rows} != {state.value for state in State}:
        raise RuntimeError("Generated state coverage does not match the State enum")
    error_path = SPEC_DIR / "Матрица_ошибок_и_реакций.csv"
    with error_path.open(encoding="utf-8-sig", newline="") as handle:
        error_rows = list(csv.DictReader(handle))
    error_ids = [row["ID"] for row in error_rows]
    if len(error_ids) != len(set(error_ids)) or len(error_rows) != len(ERROR_ROWS):
        raise RuntimeError("Generated error matrix has missing or duplicate IDs")
    for filename in ("fsm_state.drawio", "sorting_flow.drawio"):
        path = DIAGRAM_DIR / filename
        root = ET.fromstring(path.read_bytes())
        cells = {obj.get("id"): obj for obj in root.iter("object")}
        if filename == "fsm_state.drawio":
            state_ids = {obj.get("state") for obj in cells.values() if obj.get("state")}
            if state_ids != {state.value for state in State}:
                raise RuntimeError("FSM diagram state nodes differ from State enum")
            expected_main_edges = {
                f"{source.value}->{target.value}"
                for source, target in zip((
                    State.INIT, State.OBSERVE, State.SELECT, State.PLAN,
                    State.APPROACH, State.DESCEND, State.GRASP, State.VERIFY_HOLD,
                    State.TRANSFER, State.PLACE, State.RELEASE, State.RETREAT,
                    State.VERIFY_PLACE,
                ), (
                    State.OBSERVE, State.SELECT, State.PLAN, State.APPROACH,
                    State.DESCEND, State.GRASP, State.VERIFY_HOLD, State.TRANSFER,
                    State.PLACE, State.RELEASE, State.RETREAT, State.VERIFY_PLACE,
                    State.OBSERVE,
                ))
            }
            actual_main_edges = {obj.get("transition") for obj in cells.values()
                                 if obj.get("transition")}
            if actual_main_edges != expected_main_edges:
                raise RuntimeError("FSM diagram main cycle differs from the implemented state path")
        elif not all(f"step{index}" in cells for index in range(1, 10)):
            raise RuntimeError("Sorting flowchart is missing one of its nine pipeline steps")
    for filename in ("FSM.md", "Инварианты_FSM.md", "Политика_сортировки.md",
                     "Контракт_M7_исполнения.md", "Контракт_M9_телеметрии.md"):
        if not (SPEC_DIR / filename).is_file() or (SPEC_DIR / filename).stat().st_size < 300:
            raise RuntimeError(f"Generated specification missing or unexpectedly small: {filename}")


def write_coverage() -> None:
    edge_rows = []
    for source in State:
        for target in sorted(TRANSITIONS[source], key=lambda item: list(State).index(item)):
            behavior = BEHAVIOR_TESTS.get((source, target), "")
            edge_rows.append((
                source.value, target.value,
                "Да" if behavior else "Графовый guard",
                behavior or "test_transition_guard_covers_all_declared_and_forbidden_edges",
                "P0" if target == State.SAFE_STOP else "P1",
                "Для каждой разрешённой дуги указан отдельный behavioral test case; общий guard отдельно проверяет объявленные и запрещённые переходы.",
            ))
    write_csv(VERIFY_DIR / "transition_coverage.csv",
              ["From", "To", "Behavioral path", "Test case", "Priority", "Coverage note"], edge_rows)
    state_rows = []
    for state in State:
        tests = STATE_HANDLER_TESTS.get(state, "")
        state_rows.append((state.value, STATE_RU[state], "Да" if tests else "Нет",
                           tests or "No direct handler test", "test_transition_guard_covers_all_declared_and_forbidden_edges"))
    write_csv(VERIFY_DIR / "state_handler_coverage.csv",
              ["State", "Handler", "Behavior test", "Test cases", "Guard test"], state_rows)


def write_error_matrix() -> None:
    write_csv(SPEC_DIR / "Матрица_ошибок_и_реакций.csv",
              ["ID", "Ошибка/сигнал", "Обнаружение", "Реакция", "Резервации", "Событие/reason", "Retry/эскалация", "Test case", "M7 interface note"],
              ERROR_ROWS)


def write_diagrams() -> None:
    write_state_diagram()
    write_flowchart()


def main() -> None:
    for folder in (SPEC_DIR, DIAGRAM_DIR, VERIFY_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    config = ControllerConfig.load()
    write_fsm_doc(config)
    write_invariants()
    write_sorting_policy(config)
    write_forward_contracts(config)
    write_coverage()
    write_error_matrix()
    write_diagrams()
    write_event_taxonomy()
    write_ground_truth_audit()
    validate_generated_artifacts()
    print(json.dumps({
        "policy_version": config.policy_version,
        "ssot_version": config.config_version,
        "ssot_sha256": config.ssot_sha256,
        "states": len(State),
        "allowed_transitions": sum(len(targets) for targets in TRANSITIONS.values()),
        "errors": len(ERROR_ROWS),
        "artifacts": [
            str(SPEC_DIR / "FSM.md"), str(SPEC_DIR / "Инварианты_FSM.md"),
            str(SPEC_DIR / "Матрица_ошибок_и_реакций.csv"),
            str(DIAGRAM_DIR / "fsm_state.drawio"), str(DIAGRAM_DIR / "sorting_flow.drawio"),
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
