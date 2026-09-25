# Инварианты контроллера M6

Проверяются автоматически при каждом штатном завершении `SortController.step()` через `_assert_invariants()`; нарушение поднимает `ControllerContractError` и проваливает тест.

1. Существует не более одной зарезервированной M4 track и одного slot; owner совпадает.
2. Зарезервированный track имеет lifecycle `RESERVED` и равен `_active_track_id`.
3. В `TRANSFER` и `PLACE` есть свежее публичное evidence `holding=True`; его возраст не превышает 0,04 s simulation time.
4. `PLACE` дополнительно требует активный reserved slot того же track.
5. Controller-confirmed placement всегда сохраняет `OCCUPIED` status соответствующего slot.
6. `DONE` не удерживает track/slot reservation и active execution command.
7. При SAFE_STOP с подтвержденным грузом обе резервации сохраняются; SAFE_STOP не содержит active execution command.
8. Любой переход обязан принадлежать таблице `TRANSITIONS`; запрещенные переходы поднимают `IllegalTransitionError`.
