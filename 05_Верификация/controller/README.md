# Повторная проверка Milestone 6

Эта папка содержит доказательства проверки управляющей политики M6. Controlled traces строятся тестовыми doubles и не являются результатами динамической интеграции M7, виртуальных испытаний полной системы M9 или физических испытаний.

## Воспроизводимая среда

Требуется Python 3.12; приемочный прогон выполнен на Python 3.12.10. Все закрепленные зависимости для M2, M4, M5 и экспорта диаграмм устанавливаются в отдельное виртуальное окружение. Не запускайте runner системным Python: без MuJoCo регрессионные тесты M4/M5 не импортируются.

Из корня проекта:

```powershell
py -3.12 -m venv .venv-m6
$python = '.\.venv-m6\Scripts\python.exe'
& $python -m pip install -r .\04_Программа\controller\requirements-verification.txt
```

`requirements-verification.txt` включает уже закрепленные requirements планировщика M5 и perception M4. В их основе находятся версии из `04_Программа\m2\requirements.txt`.

## Перегенерация спецификации и диаграмм

Сначала обновить FSM-спецификацию, coverage tables и редактируемые draw.io исходники на основе state model:

```powershell
& $python .\04_Программа\controller\tools\generate_m6_artifacts.py
```

Затем экспортировать оба исходника в SVG, PNG и PDF со встроенными данными диаграммы:

```powershell
$drawio = 'C:\Program Files\draw.io\draw.io.exe'
$directory = '.\03_Модель_и_схемы\algorithm'
foreach ($name in @('fsm_state', 'sorting_flow')) {
    foreach ($format in @('svg', 'png', 'pdf')) {
        $source = Join-Path $directory ($name + '.drawio')
        $output = Join-Path $directory ($name + '.' + $format)
        $arguments = @('-x', '-f', $format, '-e', '-b', '16', '-o', $output, $source)
        $process = Start-Process -FilePath $drawio -ArgumentList $arguments -Wait -PassThru -WindowStyle Hidden
        if ($process.ExitCode -ne 0) { throw "draw.io export failed: $name.$format ($($process.ExitCode))" }
    }
}
```

## Повторный прогон

Запустить controller suite, унаследованные M5 planning и M4 perception suites, controlled nominal trace и проверки экспортов:

```powershell
& $python .\04_Программа\controller\tools\run_m6_verification.py
```

Runner использует тот же `sys.executable` для всех suites, сохраняет package versions и SHA-256 requirements/source, а также записывает коды выхода, число тестов, длительность, controlled trace и валидность SVG/PNG/PDF в `M6_verification_report.json`. При неуспешном suite или поврежденном/отсутствующем экспорте процесс завершается с ненулевым кодом.

Отдельно запустить только controller tests:

```powershell
& $python -m unittest discover -s .\04_Программа\controller\tests -v
```

Подробные transcript находятся в `controller_unittest.log`, `planning_unittest.log` и `perception_unittest.log`; распределение переходов и state handler tests — в `transition_coverage.csv` и `state_handler_coverage.csv`.

## Интерпретация

`PLACED_CONTROLLER_CONFIRMED` означает только совпадение нового публичного camera evidence с ожидаемыми class/zone/slot. Это не независимая truth label и не измеренная эффективность сортировки. Фактическая динамика и интеграция относятся к M7, итоговая кампания и evaluator — к M9.
