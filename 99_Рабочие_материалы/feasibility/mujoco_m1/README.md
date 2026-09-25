# M1 — MuJoCo feasibility probes

This directory contains only isolated tool and contact-model probes used to choose the simulator and gripper architecture. It is not the integrated sorting system and its outputs are not course test results.

## Recreate the clean environment

From this directory in PowerShell with CPython 3.12 x64 installed:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_m1_feasibility.py
```

Direct dependencies are pinned in `requirements.txt`; the exact transitive package snapshot from the clean M1 run is in `requirements-lock.txt`. The clean venv used for the recorded install check is retained locally as `.validation_env`; the portable commands above recreate a separate `.venv`. The probe does not use random sampling, so the seed is not applicable. Offscreen RGB rendering requires a working OpenGL context/graphics driver on the machine.

## What the harness checks

- bounded position control for one revolute and one prismatic joint;
- offscreen 640×480 RGB rendering and a saved diagnostic frame;
- symmetric two-finger contact with one actuated coordinate, lift and release without attaching/welding the object;
- deliberately insufficient friction and actuator-force negative controls;
- sensitivity to a coarser integration step;
- MP4 encoding and full decode.

The grasp fixture is intentionally small and is not the final arm geometry. Its mass, dimensions, contact parameters, force limits, and time step must not be copied into the M2 SSOT without a separate engineering decision.

## Evidence

- `results/feasibility_results.json` — machine-readable outcomes;
- `Feasibility_Matrix.csv` — tabular case-by-case protocol results;
- `results/clean_install_and_rerun.log` — fresh environment installation and repeated probe output;
- `results/m1_manifest.json` — environment details and SHA-256 of probe inputs/results;
- `results/*.png` and `results/gripper_contact_probe.mp4` — diagnostic visual outputs;
- `models/*.xml` — editable isolated test models;
- `../..` — project-level `02_Спецификация` and `01_Управление/приемка/Протокол_M1.md` for interpretation and architecture decisions.

`overall_status=PASS` means the protocol itself completed, including expected failures for deliberately poor grasp settings and the detected coarse-step limitation. It does not mean the sorter is implemented or that the robot has been physically built or tested.
