# M4-v1.3 camera and perception modules

The `camera` package wraps the MuJoCo RGB renderer and returns `CameraFrame` only. The `perception` package consumes that frame, the canonical M4 configuration exported from the SSOT, a checked homography artifact and a static background RGB reference. Runtime perception does not import MuJoCo, the workcell rig or evaluator, and cannot query model bodies, geom colors, object names, poses, internal object IDs or segmentation buffers.

The only supported geometry is the M2 `28 × 28 × 16 mm` cuboid top face at `z=16 mm` viewed by the fixed M2 overhead camera. `DetectionBatch` carries class/pose/quality/status and reason; a future controller must require both `status == VALID` and a supported class (`RED`, `GREEN` or `BLUE`). `UNKNOWN`, `OCCLUDED`, `NO_FRAME`, `STALE` and `INVALID` are not pick commands. The reported confidence is a score, not a calibrated probability. Pose sigma fields are configured allowances, not guaranteed bounds.

Create/use an isolated Python 3.12 environment from the project root, then run:

```powershell
py -3.12 -m venv '99_Рабочие_материалы/m4_perception/.venv'
& '99_Рабочие_материалы/m4_perception/.venv/Scripts/python.exe' -m pip install -r '04_Программа/perception/requirements.txt'
& '99_Рабочие_материалы/m4_perception/.venv/Scripts/python.exe' -m unittest discover -s '04_Программа/perception/tests' -v
& '99_Рабочие_материалы/m4_perception/.venv/Scripts/python.exe' '05_Верификация/perception/run_m4_verification.py'
& '99_Рабочие_материалы/m4_perception/.venv/Scripts/python.exe' '05_Верификация/perception/generate_analysis_artifacts.py'
```

Dependencies are pinned in `requirements.txt` (MuJoCo 3.14.0, opencv-python-headless 4.14.0.94, NumPy 2.5.3, Pillow 12.0.0 and PyYAML 6.0.3). The current verifier used Python 3.12.10. Tuning seeds `20260401…20260424`, blank-scene tuning seeds `20261200…20261226` and final validation seeds `20265000…20265067` are disjoint. The tuning configuration is frozen before the evaluation records are scored.

The verifier stores raw RGB frames, masks, detections JSON/CSV, evaluator manifests, calibration artifacts, metrics and worst-case overlays under `05_Верификация/perception/`. It is a perception-subsystem verifier, not the full sorting controller, trajectory planner or FSM. `generate_analysis_artifacts.py` rebuilds publication-sized XY/yaw error plots, the color confusion matrix and calibration validation map from the saved CSV/JSON artifacts; it does not use simulator truth as an input to perception.
