from __future__ import annotations

from dataclasses import dataclass

from .types import State


@dataclass(frozen=True)
class StateSpec:
    state: State
    purpose: str
    inputs: str
    entry_action: str
    command: str
    success: str
    failure: str
    timeout: str
    retry: str
    reservation: str


STATE_SPECS: tuple[StateSpec, ...] = (
    StateSpec(State.INIT, "Проверка готовности и пустых внутренних резервов.",
              "SSOT/config, SystemHealth, q, координата захвата, emergency.",
              "Проверить монотонное конечное время, зависимости, состояние руки и свободные резервы.",
              "Нет команды движения.", "Все проверки пройдены.", "Любая критическая проверка не пройдена.",
              "controller.init_timeout_s = 0,5 s simulation time.", "Повтора нет; остановка защелкивается.",
              "Резервы должны отсутствовать."),
    StateSpec(State.OBSERVE, "Получить новый свежий снимок M4.",
              "DetectionBatch; валидный пустой кадр является отдельным исходом.",
              "Запросить кадр с frame_id новее последнего принятого.",
              "ObservationRequest; физического движения нет.",
              "Свежий согласованный снимок OK или NO_CANDIDATES.",
              "NO_FRAME/INVALID/STALE, время кадра из будущего, конфликт сцены.",
              "controller.observation_timeout_s по simulation time.",
              "Ограниченные повторные запросы; после лимита SAFE_STOP.",
              "Активный цикл к этому моменту очищен; при SAFE_STOP с грузом резерв сохраняется."),
    StateSpec(State.SELECT, "Детерминированно выбрать доступный наблюдаемый объект.",
              "Свежий DetectionBatch, история TrackRecord, текущая TCP XY и статусы слотов.",
              "Учитывать валидность, класс, качество, стоимость движения и попытки.",
              "Резервировать один M4 track ID и один свободный slot до планирования/захвата.",
              "Избран известный цвет, есть слот и резервы согласованы.",
              "Поврежденная/неполная сцена, отсутствие допустимых кандидатов.",
              "controller.select_timeout_s по simulation time.",
              "UNKNOWN пересматривается по свежим кадрам ограниченное число раз; затем пропуск.",
              "Успех: track и slot зарезервированы; иначе резерва нет."),
    StateSpec(State.PLAN, "Запросить полный безопасный цикл у M5.",
              "PlanRequest с M4 batch, target track, q, gripper и только зарезервированным slot.",
              "Создать request_id; начать ожидание результата M5.",
              "Асинхронный запрос M5; команда движения запрещена.",
              "Полный Plan SUCCESS совпадает по track/class/slot/SSOT/MJCF и содержит все фазы.",
              "Любой PlanCode с классификацией; SUCCESS без полного Plan — системная ошибка.",
              "controller.plan_response_timeout_s по simulation time; M5 отдельно ограничивает compute wall-time.",
              "Только коды transient допускают один повтор после нового кадра; остальные отказывают.",
              "Отказ/повтор: освободить оба резерва; конфликт slot переводит его в QUARANTINED."),
    StateSpec(State.APPROACH, "Выполнить M5 pregrasp маршрут.",
              "Полный принятый Plan и execution feedback.",
              "Выпустить команду только для фаз начального подхода/pregrasp.",
              "Исполнить INITIAL_APPROACH, ORIENTATION_ALIGN, PREGRASP_ROUTE, PREGRASP.",
              "Точное выполнение command_id подтверждено адаптером.", "FAILED/timeout/аварийный останов.",
              "max(минимум, фазы M5 * factor + margin), simulation time.",
              "Одна ограниченная recovery-попытка; неизвестное положение -> SAFE_STOP.",
              "Сохранить обе резервации во время активного переноса/цикла."),
    StateSpec(State.DESCEND, "Опуститься по M5 вертикальному контактному сегменту.",
              "Принятый Plan, актуальное состояние руки, execution feedback.",
              "Выпустить фазу DESCENT.", "Исполнить DESCENT.", "Команда завершена без ошибки.",
              "FAILED/timeout; перед GRASP объект удерживаться не предполагается.",
              "Динамический timeout из M5 phase duration и SSOT factor/margin.", "Ограниченное recovery.",
              "Сохранить track и slot до подтверждения результата цикла."),
    StateSpec(State.GRASP, "Закрыть пальцы по M5 команде захвата.",
              "M5 GRASP_CLOSE и наблюдаемое состояние захвата.",
              "Выпустить команду закрытия, не считать закрытие доказательством удержания.",
              "Исполнить GRASP_CLOSE.", "Команда закрытия завершена; перейти к тестовому подъему.",
              "FAILED/timeout/invalid feedback.", "Динамический timeout из M5 phase duration и SSOT factor/margin.",
              "Глобальный grasp budget ограничивает повторы.", "Резервации сохраняются до результата; при no-hold освобождаются перед recovery."),
    StateSpec(State.VERIFY_HOLD, "Проверить захват только разрешенными сенсорными данными после тестового подъема.",
              "M5 LIFT completion, свежий finger-contact/effort evidence и новый M4 кадр источника.",
              "Выпустить команду тестового подъема; ждать свежих доказательств.",
              "Исполнить LIFT; не читать truth pose/contact evaluator.",
              "Достоверный hold sensor=holding и свежая M4 сцена не показывает объект в исходной области.",
              "NO_HOLD — ограниченная retry; противоречивое/устаревшее доказательство — recovery или stop.",
              "Динамический timeout LIFT + margin по simulation time.",
              "Не более controller.max_grasp_attempts; после исчерпания SAFE_FAILURE.",
              "При no-hold резервации освобождаются перед reobserve; при неопределенном hold SAFE_STOP сохраняет их."),
    StateSpec(State.TRANSFER, "Перенести удерживаемый объект к назначенной зоне.",
              "M5 TRANSFER/PREPLACE и свежий signal удержания.",
              "Проверить сигнал удержания перед запуском.", "Исполнить TRANSFER и PREPLACE.",
              "Команда завершена, сигнал удержания свеж и подтвержден.",
              "Потеря hold или ошибка/timeout -> SAFE_STOP с сохранением захвата.",
              "Динамический timeout M5 фаз и cycle deadline.",
              "Автоматический повтор запрещен при неопределенном грузе.",
              "Обе резервации удерживаются; неизвестная укладка не освобождает slot."),
    StateSpec(State.PLACE, "Опустить объект в заранее зарезервированный слот.",
              "M5 PLACE_DESCENT, slot reservation, hold signal.",
              "Проверить, что слот все еще зарезервирован для этого track.",
              "Исполнить PLACE_DESCENT.", "Команда завершена; перейти к отпусканию.",
              "Reservation mismatch, потеря груза, FAILED/timeout.",
              "Динамический timeout M5 phase и cycle deadline.", "Повтор укладки без нового плана запрещен.",
              "Слот остается RESERVED до public placement confirmation."),
    StateSpec(State.RELEASE, "Открыть захват в точке укладки.",
              "M5 RELEASE и проверенное размещение в фазе PLACE.",
              "Выпустить команду раскрытия только после PLACE completion.",
              "Исполнить RELEASE; открытие не является доказательством сортировки.",
              "Команда завершена и M7 public sensor state сообщает released.",
              "FAILED/timeout -> SAFE_STOP или безопасная recovery с сохранением причины.",
              "Динамический timeout M5 phase и cycle deadline.",
              "Повтор release без подтверждения/нового плана запрещен.",
              "Слот остается RESERVED до VERIFY_PLACE; затем OCCUPIED или QUARANTINED."),
    StateSpec(State.VERIFY_PLACE, "Проверить размещение независимой от evaluator наблюдаемой перцепцией.",
              "Свежий public-camera PlacementEvidence после release.",
              "Запросить/ожидать новое визуальное подтверждение зоны, слота и цвета.",
              "Нового движения нет.",
              "Только CONFIRMED + ожидаемые class/zone/slot и confidence threshold.",
              "Нет/неоднозначно/неверно — не засчитывать; slot quarantine.",
              "controller.place_verification_timeout_s по simulation time.",
              "Повторить наблюдение до timeout; действие не повторять.",
              "Верно: slot OCCUPIED; неопределенно/ошибка: QUARANTINED; обе не считаются свободными."),
    StateSpec(State.RETREAT, "Завершить предусмотренные M5 отход и безопасный возврат.",
              "M5 RETREAT, SAFE_RETURN, SAFE_HOME, execution feedback.",
              "Выпустить отход только после release и решения VERIFY_PLACE.",
              "Исполнить RETREAT/SAFE_RETURN/SAFE_HOME.", "Команда завершена; зафиксировать исход цикла.",
              "FAILED/timeout -> SAFE_STOP.", "Динамический timeout M5 фаз и cycle deadline.",
              "Повтор не запускается автоматически.", "При подтвержденном месте slot остается OCCUPIED."),
    StateSpec(State.RECOVER, "Ограниченно обработать отказ текущего цикла.",
              "RecoveryContext и датчики, требуемые безопасным адаптером M7.",
              "Отменить активное движение; выдать recovery request лишь для явно заданного безопасного действия.",
              "Без planner/evaluator решений; при необходимости OPEN_AND_RETREAT только при no-hold подтверждении.",
              "Адаптер вернул COMPLETED, известное состояние и безопасное положение.",
              "Неуспех, timeout, неизвестное удержание -> SAFE_STOP.",
              "controller.recovery_timeout_s по simulation time.",
              "controller.max_recovery_attempts ограничивает повторы.",
              "Освободить до retry/skip; сохранять оба резерва при безопасной остановке с грузом."),
    StateSpec(State.SAFE_STOP, "Защелкнутая остановка без новых motion/release команд.",
              "Emergency, fatal error, exhausted retries, unsafe uncertainty.",
              "Отменить motion; выдать HOLD_CURRENT_GRIPPER; сохранить причину и состояние груза.",
              "Только безопасное удержание; произвольное раскрытие запрещено.",
              "Не терминально; явный operator reset с payload disposition подтверждением.",
              "Никакой автоматической попытки сброса.", "Не ограничивается таймером.",
              "Сброс только после ручного безопасного разрешения.",
              "При удержании сохраняются reservation; иначе не должно быть зависших slot reservation."),
    StateSpec(State.DONE, "Терминальное завершение batch с раздельным учетом исходов.",
              "Подтвержденный пустой batch или отсутствие допустимых объектов после bounded policy.",
              "Сформировать итоговый счетчик; больше не выполнять motion.", "Нет команды.",
              "Batch завершен независимо от доли успеха.",
              "Новый валидный объект после DONE требует нового run_id.", "Нет таймера.",
              "Повтор запрещен; новый batch создает новый controller.",
              "Активные reservation запрещены."),
)


TRANSITIONS: dict[State, frozenset[State]] = {
    State.INIT: frozenset({State.OBSERVE, State.SAFE_STOP}),
    State.OBSERVE: frozenset({State.SELECT, State.DONE, State.SAFE_STOP}),
    State.SELECT: frozenset({State.PLAN, State.OBSERVE, State.DONE, State.SAFE_STOP}),
    State.PLAN: frozenset({State.APPROACH, State.RECOVER, State.SAFE_STOP}),
    State.APPROACH: frozenset({State.DESCEND, State.RECOVER, State.SAFE_STOP}),
    State.DESCEND: frozenset({State.GRASP, State.RECOVER, State.SAFE_STOP}),
    State.GRASP: frozenset({State.VERIFY_HOLD, State.RECOVER, State.SAFE_STOP}),
    State.VERIFY_HOLD: frozenset({State.TRANSFER, State.RECOVER, State.SAFE_STOP}),
    State.TRANSFER: frozenset({State.PLACE, State.SAFE_STOP}),
    State.PLACE: frozenset({State.RELEASE, State.SAFE_STOP}),
    State.RELEASE: frozenset({State.VERIFY_PLACE, State.SAFE_STOP}),
    State.VERIFY_PLACE: frozenset({State.RETREAT, State.RECOVER, State.SAFE_STOP}),
    State.RETREAT: frozenset({State.OBSERVE, State.RECOVER, State.SAFE_STOP}),
    State.RECOVER: frozenset({State.OBSERVE, State.SAFE_STOP}),
    State.SAFE_STOP: frozenset({State.INIT}),
    State.DONE: frozenset(),
}


EXECUTION_PHASES: dict[State, tuple[str, ...]] = {
    State.APPROACH: ("INITIAL_APPROACH", "ORIENTATION_ALIGN", "PREGRASP_ROUTE", "PREGRASP"),
    State.DESCEND: ("DESCENT",),
    State.GRASP: ("GRASP_CLOSE",),
    State.VERIFY_HOLD: ("LIFT",),
    State.TRANSFER: ("TRANSFER", "PREPLACE"),
    State.PLACE: ("PLACE_DESCENT",),
    State.RELEASE: ("RELEASE",),
    State.RETREAT: ("RETREAT", "SAFE_RETURN", "SAFE_HOME"),
}


PLAN_REQUIRED_PHASES = frozenset(phase for phases in EXECUTION_PHASES.values() for phase in phases)


def transition_is_allowed(source: State, target: State) -> bool:
    return target in TRANSITIONS[source]
