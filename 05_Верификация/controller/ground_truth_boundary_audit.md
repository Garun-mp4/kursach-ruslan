# Аудит информационной границы controller M6

Проверено модулей controller: 6.

## Статическая проверка

AST audit controller production modules: нет imports evaluator/ground_truth, нет доступа к полям true_color/truth_position/object_weld и нет обращений к evaluator-only атрибутам.

## Контрактные проверки

- Входные типы controller содержат M4 оценки и public evidence; `GraspEvidence`/`PlacementEvidence` требуют разрешённый source.
- `test_evaluator_only_placement_evidence_is_ignored` передаёт evidence с evaluator source и подтверждает, что его нельзя засчитать.
- Controller result сохраняет `evaluator_result: None`; M6 state/status не является truth label.
- `_handle_verify_hold` проверяет новый M4 source observation относительно M4 source estimate, а не читает истинное положение объекта.

## Вывод и граница

Внутренний `PLACED_CONTROLLER_CONFIRMED` означает только совпадающее новое public-camera class/zone/slot evidence. Он не равен независимому evaluator-confirmed правильному цвету физически удержанного объекта. Evaluator остаётся будущим post-run компонентом M9.
