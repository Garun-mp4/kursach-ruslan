# M5 path planning and collision preflight

The package reads the frozen project SSOT (`M5-v1.1`) and M2 MJCF (`M2-v1.1`). `Planner.plan_cycle` returns a complete checked plan or a structured fail-closed result. It accepts M4 `DetectionBatch` estimates and caller-supplied free slot IDs; it does not access simulator ground truth.

## Run unit and subsystem tests

From the project root, using Python 3.12.10 and the shared M2/M3 dependencies:

```powershell
& .\99_Рабочие_материалы\integration\.venv\Scripts\python.exe -m unittest discover -s .\04_Программа\planning\tests -p "test_*.py" -v
```

The suite uses Python's standard `unittest`; no pytest dependency is required.

## Rebuild M5 verification evidence

```powershell
& .\99_Рабочие_материалы\integration\.venv\Scripts\python.exe .\04_Программа\planning\run_m5_verification.py
```

The deterministic route campaign writes M4-contract inputs, compact plan JSON (waypoints, events, status and provenance), full sampled trajectories and clearance profiles as CSV, convergence and thin-obstacle evidence, and paired SVG/PNG plots into `05_Верификация/planning/`. CSV columns include analytic FK-derived TCP linear/angular rates and accelerations, checked against conservative bounds derived from the configured joint limits and link lengths. The report records package versions and SHA-256 hashes for the configuration, model and relevant source modules. It uses no random seed because it draws no randomness. A nonzero exit means one of the predefined expected outcomes changed or a required convergence/collision check failed.

The M5-v1.1 wall-clock preflight timeout is 90 s. The half-step route took 58.4 s on the recorded PC; the timeout was raised from 60 s to retain bounded runtime headroom. This does not alter route, motion, or collision acceptance limits. The report's sampling-refinement gate is decision-level: the same M4 estimate must produce the same safe outcome, destination, IK branch, phase/event sequence, and TCP path length within the refined 0.5 mm spatial step; the finer run must not discover a collision. The minimum sampled clearance is reported separately as `GRID_SENSITIVE` when it changes with the sample grid. That numerical minimum is not forced to match between grids, and a passing comparison is not a continuous collision-free proof.

The route cases use deterministic synthetic records conforming to the M4 `DetectionBatch` interface. They verify planner behavior from perception estimates; they are not end-to-end camera/perception runs and do not establish batch-sorting performance. A targeted compatibility regression for the M2-v1.1 wrist/finger geometry repeats M4's arm-occlusion fixture against the frozen M4-v1.3 detector and calibration:

```powershell
& .\99_Рабочие_материалы\integration\.venv\Scripts\python.exe .\04_Программа\planning\verify_m4_model_compatibility.py
```

This is a focused interface regression, not a rerun of the full M4 acceptance campaign.

Dependencies are inherited from `04_Программа/m2/requirements.txt` (MuJoCo, NumPy, PyYAML and Pillow). No extra plotting or testing package is required.

## Output interpretation

`minimum_clearance_m` is capped to the nearest narrow-phase pair measurement. `clearance_lower_bound_m` is the conservative all-pair sampled lower bound from AABB broad phase plus signed-distance narrow phase. Neither is a continuous-time collision proof. M5 validates paths and time parameterization only; it does not claim dynamic execution or physical tests.
