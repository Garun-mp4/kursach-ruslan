# Конечный автомат контроллера цветовой сортировки

Версия политики: **M6-v1.0**  
Версия аппаратной/геометрической SSOT, передаваемая в M5: **M5-v1.1**  
SHA-256 SSOT на генерации: `51fbd34bb6bd5dc3da74744e44ae78adc9ee21aab69fec1662677b716b8d8627`  
SHA-256 MJCF модели: `c76f67828dd5ffd6f3dc543b302729c394579e2d4b393df67e8bcae557bc748b`

Контроллер — независимый логический слой: perception, M5 planner и исполнительные операции приходят через типизированные интерфейсы. В M6 физический исполнитель не реализуется; проверка FSM выполняется управляемыми входными последовательностями и test doubles.

## Диаграммы

- Состояния: [`fsm_state.drawio`](../03_Модель_и_схемы/algorithm/fsm_state.drawio), [SVG](../03_Модель_и_схемы/algorithm/fsm_state.svg), [PNG](../03_Модель_и_схемы/algorithm/fsm_state.png).
- Поток сортировки: [`sorting_flow.drawio`](../03_Модель_и_схемы/algorithm/sorting_flow.drawio), [SVG](../03_Модель_и_схемы/algorithm/sorting_flow.svg), [PNG](../03_Модель_и_схемы/algorithm/sorting_flow.png).

## Состояния

| State | Назначение | Вход / guard | Действие при входе | Успешный выход | Ошибка / timeout | Повтор и резерв |
|---|---|---|---|---|---|---|
| `INIT` — Проверка готовности | Проверка готовности и пустых внутренних резервов. | SSOT/config, SystemHealth, q, координата захвата, emergency. | Проверить монотонное конечное время, зависимости, состояние руки и свободные резервы.; команда: Нет команды движения. | Все проверки пройдены. | Любая критическая проверка не пройдена.; timeout: controller.init_timeout_s = 0,5 s simulation time. | Повтора нет; остановка защелкивается.; Резервы должны отсутствовать. |
| `OBSERVE` — Получение кадра | Получить новый свежий снимок M4. | DetectionBatch; валидный пустой кадр является отдельным исходом. | Запросить кадр с frame_id новее последнего принятого.; команда: ObservationRequest; физического движения нет. | Свежий согласованный снимок OK или NO_CANDIDATES. | NO_FRAME/INVALID/STALE, время кадра из будущего, конфликт сцены.; timeout: controller.observation_timeout_s по simulation time. | Ограниченные повторные запросы; после лимита SAFE_STOP.; Активный цикл к этому моменту очищен; при SAFE_STOP с грузом резерв сохраняется. |
| `SELECT` — Выбор объекта и места | Детерминированно выбрать доступный наблюдаемый объект. | Свежий DetectionBatch, история TrackRecord, текущая TCP XY и статусы слотов. | Учитывать валидность, класс, качество, стоимость движения и попытки.; команда: Резервировать один M4 track ID и один свободный slot до планирования/захвата. | Избран известный цвет, есть слот и резервы согласованы. | Поврежденная/неполная сцена, отсутствие допустимых кандидатов.; timeout: controller.select_timeout_s по simulation time. | UNKNOWN пересматривается по свежим кадрам ограниченное число раз; затем пропуск.; Успех: track и slot зарезервированы; иначе резерва нет. |
| `PLAN` — Полный preflight M5 | Запросить полный безопасный цикл у M5. | PlanRequest с M4 batch, target track, q, gripper и только зарезервированным slot. | Создать request_id; начать ожидание результата M5.; команда: Асинхронный запрос M5; команда движения запрещена. | Полный Plan SUCCESS совпадает по track/class/slot/SSOT/MJCF и содержит все фазы. | Любой PlanCode с классификацией; SUCCESS без полного Plan — системная ошибка.; timeout: controller.plan_response_timeout_s по simulation time; M5 отдельно ограничивает compute wall-time. | Только коды transient допускают один повтор после нового кадра; остальные отказывают.; Отказ/повтор: освободить оба резерва; конфликт slot переводит его в QUARANTINED. |
| `APPROACH` — Подход к объекту | Выполнить M5 pregrasp маршрут. | Полный принятый Plan и execution feedback. | Выпустить команду только для фаз начального подхода/pregrasp.; команда: Исполнить INITIAL_APPROACH, ORIENTATION_ALIGN, PREGRASP_ROUTE, PREGRASP. | Точное выполнение command_id подтверждено адаптером. | FAILED/timeout/аварийный останов.; timeout: max(минимум, фазы M5 * factor + margin), simulation time. | Одна ограниченная recovery-попытка; неизвестное положение -> SAFE_STOP.; Сохранить обе резервации во время активного переноса/цикла. |
| `DESCEND` — Опускание | Опуститься по M5 вертикальному контактному сегменту. | Принятый Plan, актуальное состояние руки, execution feedback. | Выпустить фазу DESCENT.; команда: Исполнить DESCENT. | Команда завершена без ошибки. | FAILED/timeout; перед GRASP объект удерживаться не предполагается.; timeout: Динамический timeout из M5 phase duration и SSOT factor/margin. | Ограниченное recovery.; Сохранить track и slot до подтверждения результата цикла. |
| `GRASP` — Закрытие пальцев | Закрыть пальцы по M5 команде захвата. | M5 GRASP_CLOSE и наблюдаемое состояние захвата. | Выпустить команду закрытия, не считать закрытие доказательством удержания.; команда: Исполнить GRASP_CLOSE. | Команда закрытия завершена; перейти к тестовому подъему. | FAILED/timeout/invalid feedback.; timeout: Динамический timeout из M5 phase duration и SSOT factor/margin. | Глобальный grasp budget ограничивает повторы.; Резервации сохраняются до результата; при no-hold освобождаются перед recovery. |
| `VERIFY_HOLD` — Пробный подъём и проверка удержания | Проверить захват только разрешенными сенсорными данными после тестового подъема. | M5 LIFT completion, свежий finger-contact/effort evidence и новый M4 кадр источника. | Выпустить команду тестового подъема; ждать свежих доказательств.; команда: Исполнить LIFT; не читать truth pose/contact evaluator. | Достоверный hold sensor=holding и свежая M4 сцена не показывает объект в исходной области. | NO_HOLD — ограниченная retry; противоречивое/устаревшее доказательство — recovery или stop.; timeout: Динамический timeout LIFT + margin по simulation time. | Не более controller.max_grasp_attempts; после исчерпания SAFE_FAILURE.; При no-hold резервации освобождаются перед reobserve; при неопределенном hold SAFE_STOP сохраняет их. |
| `TRANSFER` — Перенос с контролем удержания | Перенести удерживаемый объект к назначенной зоне. | M5 TRANSFER/PREPLACE и свежий signal удержания. | Проверить сигнал удержания перед запуском.; команда: Исполнить TRANSFER и PREPLACE. | Команда завершена, сигнал удержания свеж и подтвержден. | Потеря hold или ошибка/timeout -> SAFE_STOP с сохранением захвата.; timeout: Динамический timeout M5 фаз и cycle deadline. | Автоматический повтор запрещен при неопределенном грузе.; Обе резервации удерживаются; неизвестная укладка не освобождает slot. |
| `PLACE` — Опускание в зарезервированное место | Опустить объект в заранее зарезервированный слот. | M5 PLACE_DESCENT, slot reservation, hold signal. | Проверить, что слот все еще зарезервирован для этого track.; команда: Исполнить PLACE_DESCENT. | Команда завершена; перейти к отпусканию. | Reservation mismatch, потеря груза, FAILED/timeout.; timeout: Динамический timeout M5 phase и cycle deadline. | Повтор укладки без нового плана запрещен.; Слот остается RESERVED до public placement confirmation. |
| `RELEASE` — Отпускание | Открыть захват в точке укладки. | M5 RELEASE и проверенное размещение в фазе PLACE. | Выпустить команду раскрытия только после PLACE completion.; команда: Исполнить RELEASE; открытие не является доказательством сортировки. | Команда завершена и M7 public sensor state сообщает released. | FAILED/timeout -> SAFE_STOP или безопасная recovery с сохранением причины.; timeout: Динамический timeout M5 phase и cycle deadline. | Повтор release без подтверждения/нового плана запрещен.; Слот остается RESERVED до VERIFY_PLACE; затем OCCUPIED или QUARANTINED. |
| `VERIFY_PLACE` — Проверка места камерой | Проверить размещение независимой от evaluator наблюдаемой перцепцией. | Свежий public-camera PlacementEvidence после release. | Запросить/ожидать новое визуальное подтверждение зоны, слота и цвета.; команда: Нового движения нет. | Только CONFIRMED + ожидаемые class/zone/slot и confidence threshold. | Нет/неоднозначно/неверно — не засчитывать; slot quarantine.; timeout: controller.place_verification_timeout_s по simulation time. | Повторить наблюдение до timeout; действие не повторять.; Верно: slot OCCUPIED; неопределенно/ошибка: QUARANTINED; обе не считаются свободными. |
| `RETREAT` — Отход и возврат | Завершить предусмотренные M5 отход и безопасный возврат. | M5 RETREAT, SAFE_RETURN, SAFE_HOME, execution feedback. | Выпустить отход только после release и решения VERIFY_PLACE.; команда: Исполнить RETREAT/SAFE_RETURN/SAFE_HOME. | Команда завершена; зафиксировать исход цикла. | FAILED/timeout -> SAFE_STOP.; timeout: Динамический timeout M5 фаз и cycle deadline. | Повтор не запускается автоматически.; При подтвержденном месте slot остается OCCUPIED. |
| `RECOVER` — Ограниченное восстановление | Ограниченно обработать отказ текущего цикла. | RecoveryContext и датчики, требуемые безопасным адаптером M7. | Отменить активное движение; выдать recovery request лишь для явно заданного безопасного действия.; команда: Без planner/evaluator решений; при необходимости OPEN_AND_RETREAT только при no-hold подтверждении. | Адаптер вернул COMPLETED, известное состояние и безопасное положение. | Неуспех, timeout, неизвестное удержание -> SAFE_STOP.; timeout: controller.recovery_timeout_s по simulation time. | controller.max_recovery_attempts ограничивает повторы.; Освободить до retry/skip; сохранять оба резерва при безопасной остановке с грузом. |
| `SAFE_STOP` — Защёлкнутая безопасная остановка | Защелкнутая остановка без новых motion/release команд. | Emergency, fatal error, exhausted retries, unsafe uncertainty. | Отменить motion; выдать HOLD_CURRENT_GRIPPER; сохранить причину и состояние груза.; команда: Только безопасное удержание; произвольное раскрытие запрещено. | Не терминально; явный operator reset с payload disposition подтверждением. | Никакой автоматической попытки сброса.; timeout: Не ограничивается таймером. | Сброс только после ручного безопасного разрешения.; При удержании сохраняются reservation; иначе не должно быть зависших slot reservation. |
| `DONE` — Пакет завершён | Терминальное завершение batch с раздельным учетом исходов. | Подтвержденный пустой batch или отсутствие допустимых объектов после bounded policy. | Сформировать итоговый счетчик; больше не выполнять motion.; команда: Нет команды. | Batch завершен независимо от доли успеха. | Новый валидный объект после DONE требует нового run_id.; timeout: Нет таймера. | Повтор запрещен; новый batch создает новый controller.; Активные reservation запрещены. |

## Разрешенные переходы

Список ниже берется из `controller/fsm_spec.py`; `SortController._transition()` отклоняет любую дугу, отсутствующую в нем.

| Из | В | Причина / условие |
|---|---|---|
| `INIT` | `OBSERVE` | готовность подтверждена |
| `INIT` | `SAFE_STOP` | аварийное событие / явный guard соответствующего handler |
| `OBSERVE` | `SELECT` | свежая сцена |
| `OBSERVE` | `SAFE_STOP` | аварийное событие / явный guard соответствующего handler |
| `OBSERVE` | `DONE` | 3 свежих пустых кадра |
| `SELECT` | `OBSERVE` | повтор наблюдения |
| `SELECT` | `PLAN` | объект и слот зарезервированы |
| `SELECT` | `SAFE_STOP` | аварийное событие / явный guard соответствующего handler |
| `SELECT` | `DONE` | нет допустимых целей |
| `PLAN` | `APPROACH` | полный валидный M5 Plan |
| `PLAN` | `RECOVER` | обработанный отказ планирования |
| `PLAN` | `SAFE_STOP` | аварийное событие / явный guard соответствующего handler |
| `APPROACH` | `DESCEND` | подход завершён |
| `APPROACH` | `RECOVER` | сбой без возможного груза |
| `APPROACH` | `SAFE_STOP` | аварийное событие / явный guard соответствующего handler |
| `DESCEND` | `GRASP` | спуск завершён |
| `DESCEND` | `RECOVER` | сбой без возможного груза |
| `DESCEND` | `SAFE_STOP` | аварийное событие / явный guard соответствующего handler |
| `GRASP` | `VERIFY_HOLD` | закрытие завершено |
| `GRASP` | `RECOVER` | свежее подтверждение no-hold |
| `GRASP` | `SAFE_STOP` | аварийное событие / явный guard соответствующего handler |
| `VERIFY_HOLD` | `TRANSFER` | свежий hold + источник пуст |
| `VERIFY_HOLD` | `RECOVER` | явный no-hold |
| `VERIFY_HOLD` | `SAFE_STOP` | аварийное событие / явный guard соответствующего handler |
| `TRANSFER` | `PLACE` | перенос завершён, hold свежий |
| `TRANSFER` | `SAFE_STOP` | аварийное событие / явный guard соответствующего handler |
| `PLACE` | `RELEASE` | опускание завершено, hold свежий |
| `PLACE` | `SAFE_STOP` | аварийное событие / явный guard соответствующего handler |
| `RELEASE` | `VERIFY_PLACE` | захват пуст по свежему датчику |
| `RELEASE` | `SAFE_STOP` | аварийное событие / явный guard соответствующего handler |
| `VERIFY_PLACE` | `RETREAT` | публичная проверка или безопасный исход |
| `VERIFY_PLACE` | `RECOVER` | лимит цикла без удерживаемого груза |
| `VERIFY_PLACE` | `SAFE_STOP` | аварийное событие / явный guard соответствующего handler |
| `RETREAT` | `OBSERVE` | цикл закрыт |
| `RETREAT` | `RECOVER` | ошибка отхода без груза |
| `RETREAT` | `SAFE_STOP` | аварийное событие / явный guard соответствующего handler |
| `RECOVER` | `OBSERVE` | восстановление подтверждено |
| `RECOVER` | `SAFE_STOP` | аварийное событие / явный guard соответствующего handler |
| `SAFE_STOP` | `INIT` | операторский сброс + безопасное разрешение |

## Таймеры и retry budgets

Все значения времени — секунды симуляции. Контроллер не читает wall-clock и FPS; переходы происходят только по входам `ControllerInput.simulation_time_s`.

| Состояние | Тайм-аут / watchdog |
|---|---|
| `INIT` — Проверка готовности | 0,5 s |
| `OBSERVE` — Получение кадра | 0,25 s |
| `SELECT` — Выбор объекта и места | 0,5 s |
| `PLAN` — Полный preflight M5 | 120 s |
| `APPROACH` — Подход к объекту | max(min, сумма M5 phase durations × factor + margin) |
| `DESCEND` — Опускание | max(min, M5 DESCENT duration × factor + margin) |
| `GRASP` — Закрытие пальцев | max(min, M5 GRASP_CLOSE duration × factor + margin) |
| `VERIFY_HOLD` — Пробный подъём и проверка удержания | LIFT: динамический budget; затем 0,25 s |
| `TRANSFER` — Перенос с контролем удержания | max(min, сумма M5 TRANSFER + PREPLACE × factor + margin) |
| `PLACE` — Опускание в зарезервированное место | max(min, M5 PLACE_DESCENT duration × factor + margin) |
| `RELEASE` — Отпускание | M5 RELEASE; после команды без fresh empty-signal: 0,25 s |
| `VERIFY_PLACE` — Проверка места камерой | 0,5 s |
| `RETREAT` — Отход и возврат | max(min, сумма M5 RETREAT/SAFE_RETURN/SAFE_HOME × factor + margin) |
| `RECOVER` — Ограниченное восстановление | 5 s |
| `SAFE_STOP` — Защёлкнутая безопасная остановка | нет; защелкнут до явного разрешенного reset |
| `DONE` — Пакет завершён | нет; терминальное состояние |

- Выполнение M5-фаз: `max(1 s, phase_duration × 1,5 + 2 s)`. В duration входят времена M5 samples соответствующих фаз.
- Полный объектный цикл: `M5_plan.duration_s × 1,5 + 30 s`; отсчет начинается только после принятия полного Plan.
- Пакет: 1800 s simulation time.
- Perception: до 3 последовательных отказов; частичная сцена — до 3 свежих кадров.
- Неизвестный класс: 2 distinct свежих кадра; пустая партия: 3 distinct свежих пустых кадров.
- M5 transient result: не более 1 повторов на track; захват: не более 2 полных попыток на track; цикл: не более 2 попыток.
- Recovery: одна попытка на episode (`max_recovery_attempts`), счетчик обнуляется только после подтвержденного безопасного восстановления.
- Тайм-аут без свежего подтверждения отсутствия груза не является доказательством `holding=False`: применяется SAFE_STOP.
- M5 собственный compute wall-time budget является отдельным ограничением planner; он не заменяет M6 simulation-time ожидание результата.

## M5 Plan preflight

До `APPROACH` контроллер сверяет target track/class, зарезервированный slot, M2/M5 SSOT и модельные хеши, frame ID и время M4, полный упорядоченный список фаз, монотонность временных samples, конечность координат, фазовую последовательность, согласованность duration и обязательные `OBJECT_ATTACHED` в `LIFT` / `OBJECT_RELEASED` в `RELEASE`. Любое несовпадение приводит к `SAFE_STOP` до команды движения.

## Публичные доказательства удержания и укладки

- Допустимые источники удержания: `FINGER_CONTACT_SENSOR` или `GRIPPER_EFFORT_PROXY`; максимальный возраст — 0,04 s. Поза/ID/контакт из evaluator недопустимы.
- `VERIFY_HOLD` требует завершения тестового `LIFT`, свежего сенсорного сигнала и нового M4 кадра, который не показывает цель в зоне источника. Только затем разрешен `TRANSFER`.
- Перед и при `TRANSFER`/`PLACE` hold должен оставаться свежим. `false`, устаревшее или отсутствующее подтверждение приводит к SAFE_STOP и quarantine места.
- `RELEASE` не доказывает укладку. Для controller-level подтверждения нужен новый публичный кадр с совпадающими class/zone/slot и confidence не ниже 0,75. `VERIFY_PLACE` не использует evaluator.
- Текущий M4 ROI исключает зоны лотков. В M7 необходимо реализовать отдельный публичный placement-perception adapter без ground-truth доступа.

## Состояния объекта и ячейки

- Объект: `PENDING → RESERVED → PLACED_CONTROLLER_CONFIRMED`, либо `SKIPPED_UNKNOWN`, `SKIPPED_UNREACHABLE`, `SKIPPED_FULL_ZONE`, `SAFE_FAILURE`. До controller evidence не используется термин «отсортирован». `DONE` — завершение цикла batch controller, а не 100% успешность.
- Slot: `FREE → RESERVED → OCCUPIED` только после публичного совпадающего наблюдения; неопределенность/потеря/ошибка результата переводит его в `QUARANTINED`; no-payload planning/grasp failure освобождает reserve.
- `ReservationManager` обеспечивает одновременно максимум один active track и один слот для однорукого цикла; их владелец должен совпадать.

## Структурированные события и инварианты

События `ControllerEvent` содержат monotonically increasing event_id, `simulation_time_s`, state, стабильный `reason_code`, track/slot (если применимо) и структурированный `details`. Список статических event type извлекается из исходника и сохраняется в verification log.

| Инвариант | Автоматическая проверка |
|---|---|
| Не более одной пары active track/slot reservation; владелец совпадает. | `ReservationManager.assert_invariants()` и `SortController._assert_invariants()`. |
| Track с резервом имеет статус `RESERVED`; неиспользуемый slot не имеет владельца. | Каждый `step()` после обработки состояния. |
| `TRANSFER` и `PLACE` требуют свежего публичного hold-сигнала; `PLACE` требует зарезервированный слот. | Каждый `step()`; нарушенный инвариант приводит к ошибке теста. |
| Confirmed placement соответствует занятому slot, а не эвристике внутреннего FSM. | `_placed_slot_by_track` consistency check. |
| `DONE` не удерживает резервации и команду движения; `SAFE_STOP` не хранит active command. | Каждый `step()` и regression tests. |

## Граница доказательств M6

M6 подтверждает логику переходов и контрактов на controlled sequences. Он не подтверждает силовые контакты, физическое удержание, точность реальной камеры или фактическую укладку в MuJoCo-физике: это предмет M7, а итоговые виртуальные испытания — M9.
