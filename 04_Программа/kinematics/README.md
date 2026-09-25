# M3 — аналитическая кинематика SCARA

Модуль содержит независимые ручные преобразования FK, аналитическую IK для обеих ветвей R–R, проверку суставных пределов, классификацию сингулярности и радиальную рабочую область. Математическая библиотека не импортирует MuJoCo; движок используется только в отдельном verification runner для независимого сравнения transforms.

## Воспроизведение

Из корня проекта, CPython 3.12.10 x64:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\04_Программа\kinematics\requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s .\04_Программа\kinematics -v
.\.venv\Scripts\python.exe .\04_Программа\kinematics\verify_kinematics.py
```

Раннер использует SSOT `02_Спецификация/параметры_системы.yaml`, XML M2 и фиксированные seed, создает CSV/JSON/SVG/PNG в `05_Верификация/kinematics/`. Основная выборка содержит 10 000 FK→IK→FK поз; parity выборка содержит 1 000 суставных состояний и сверяет J1/J2/J3/J4/TCP с компилированным MuJoCo.

## API

```python
from scara import load_config, forward_kinematics, inverse_kinematics

config = load_config()
fk = forward_kinematics([q1_rad, q2_rad, q3_m, q4_rad], config)
ik = inverse_kinematics([x_m, y_m, z_m, yaw_rad], config, current_q=q)
```

`IKResult` возвращает статус, причину, все физически допустимые кандидаты, выбранное решение, остатки и флаг коррекции только машинного round-off. Если FK-остаток выбранной пары превышает допуски SSOT (`1e-6 m` или `1e-8 rad`), возвращается `NUMERICAL_FAILURE`, а кандидат остается только для диагностики. Near-singularity помечается, но сама по себе не объявляется невозможностью позы. IK не выполняет collision checking и не утверждает, что между позами существует безопасный путь.
