# Схема Single Source of Truth

**Назначение:** описать структуру, владельцев и правила изменения единого машинно-читаемого реестра проекта. Геометрический baseline M2 зафиксирован, численные параметры кинематики добавлены в M3; схема продолжает регламентировать расширения M4–M9 и аудит M12.

## Канонические файлы SSOT

Основным машинным источником назначается 02_Спецификация/параметры_системы.yaml. Человекочитаемое представление — 02_Спецификация/Реестр_параметров.md, генерируемое или проверяемое по YAML. Код, MJCF, планы прогонов, графики, записка и презентация не должны иметь независимо редактируемые копии чисел.

Каждое значение обязано иметь ключ, численное значение или явный статус неизвестности, единицу, источник/обоснование, дату/версию и владельца milestone. Неизвестные параметры до измерения задаются как null/UNKNOWN с владельцем, а не нулем.

## Состав SSOT и дальнейшее владение

SSOT был создан в M2 в `02_Спецификация/параметры_системы.yaml` (`schema_version: 1.0`, `geometry_baseline_version: M2-v1.0`). После M3 и M4 его `config_version` равен `M4-v1.3`; под `parameters` остаются 194 ключа геометрии и кинематики, geometry hash MJCF не менялся. M4 runtime profile расположен отдельно в верхнеуровневом `perception.runtime_config` и не считается дополнительным элементом `parameters`. Profile SHA-256 равен `55c87dcd96f6146a1214765c0fcab66a1be452fc55467ed2ae3a19046a7e19f2`; для каждого из 60 листовых значений присутствуют единица, происхождение и владелец `M4` в `perception.profile_parameter_metadata`. `04_Программа/perception/perception_config.yaml` — экспорт профиля, а не независимый источник. Человекочитаемые параметры M4, результаты calibration и error budget описаны в `02_Спецификация/Зрение_и_калибровка.md`; параметры FK/IK — в `02_Спецификация/Кинематика.md`.

Таблица ниже задает владельцев уточнений и расширения. Пометка milestone означает владельца окончательного определения соответствующих данных, а не отсутствие уже заданных базовых значений.

| Группа | Поля, которые будут заведены | Владелец определения |
|---|---|---|
| Идентификация | project_id, тема, schema_version, config_version | Базовые поля M2; версия конфигурации при каждом подтвержденном изменении; аудит M12 |
| Среда | engine, engine_version, python_version, timestep_s, solver settings, gravity_m_s2 | M1 выбрал движок/язык; baseline физики M2, пересмотр контактов M7 |
| Топология | arm_type, ordered_joint_types, arm_dof, gripper_actuated_dof, linkage_type | Архитектура M1; имена, оси и кинематические offsets зафиксированы M2 |
| Геометрия | link_lengths_m, offsets_m, link dimensions, base pose, table height, TCP transform, finger stroke_m | Baseline M2; изменения только с повторной проверкой M3/M5/M7 |
| Ограничения движения | joint limits rad/m, velocity rad_s/m_s, acceleration, actuator limits, stop margins | Baseline M2; достижимость M3, профили M5, исполнительная проверка M7 |
| Захват и объекты | primitive/shape, dimensions_m, mass_kg, object classes, contact parameters, jaw limits | Базовая геометрия M2; динамика захвата и контакты проверяются M7 |
| Ячейка | source area, bins, placement slots, static obstacles, allowed object regions | Геометрия M2; collision/path validation M5/M7 и кампании M9 |
| Камера | fixed pose, resolution_px, FOV/intrinsics, camera-to-world transform, frame names, RGB order | Геометрический baseline M2; calibration/perception parameters M4 |
| Оценка цвета/позы | class labels, threshold/model version, confidence, UNKNOWN rules, XY/yaw conventions, calibration ID | M4; M4-v1.3 profile заморожен, повторная калибровка требует новой версии и полной проверки |
| Кинематика | FK convention, IK method/branches, convergence tolerances and units | M3 |
| Планирование | clearance_m, collision pairs, trajectory interpolation, motion limits, planner version | M5 |
| Управление | state names, timeouts_s, retry limits, safe-stop rules | M6 |
| Испытания | scenario IDs, seeds, acceptance thresholds, metric definitions, evaluation version | M9; критерии должны быть заморожены до доказательной кампании |
| Происхождение | sources, license/asset records, code/model hashes | Каждый milestone; контроль M12 |

## Владение и изменение

1. M1 фиксирует концептуальные решения и версии среды, которые реально проверены.
2. M2 создал первый полный YAML и дал каждому параметру единицу, происхождение, основание и владельца; геометрия служит одной истиной для MJCF и расчетов.
3. M3 добавил численные допуски FK/IK; M4 добавил отдельный perception profile и metadata для каждого его листа. M5–M9 добавляют свои значения в назначенные разделы, не копируя существующие ключи с иным смыслом.
4. Изменение параметра требует версии/причины, перечня зависимых артефактов и повторного запуска соответствующих проверок. Старые экспериментальные результаты сохраняют прежний config_version.
5. Генераторы документов и графиков по возможности читают значения из YAML. Если число вводится вручную, ревью требует явной трассировки к ключу.

Единицы и мировые оси зафиксированы концептуально в Интерфейсах M1. Разрешение камеры тестового стенда, физические силы/трение стенда и размеры его тестового куба не являются параметрами будущего манипулятора.
