from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from perception.types import Detection
from rig import GroundTruthObject


CLASSES = ("RED", "GREEN", "BLUE", "UNKNOWN")
MATRIX_LABELS = (*CLASSES, "NO_DETECTION")


@dataclass(frozen=True)
class Match:
    gt_index: int
    detection_index: int
    distance_m: float


def match_detections(detections: Iterable[Detection], truth: list[GroundTruthObject],
                     max_match_distance_m: float = 0.025) -> tuple[list[Match], set[int], set[int]]:
    dets = list(detections)
    candidate_pairs: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(truth):
        for di, det in enumerate(dets):
            dist = float(np.linalg.norm(np.asarray(det.xy_base_m) - [gt.x_m, gt.y_m]))
            if dist <= max_match_distance_m:
                candidate_pairs.append((dist, gi, di))
    used_gt: set[int] = set()
    used_det: set[int] = set()
    matches: list[Match] = []
    for dist, gi, di in sorted(candidate_pairs):
        if gi not in used_gt and di not in used_det:
            used_gt.add(gi)
            used_det.add(di)
            matches.append(Match(gi, di, dist))
    return matches, set(range(len(truth))) - used_gt, set(range(len(dets))) - used_det


def confusion_matrix(rows: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    matrix = {actual: {pred: 0 for pred in MATRIX_LABELS} for actual in CLASSES}
    for actual, predicted in rows:
        matrix.setdefault(actual, {pred: 0 for pred in MATRIX_LABELS})
        matrix[actual].setdefault(predicted, 0)
        matrix[actual][predicted] += 1
    return matrix


def class_metrics(matrix: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    result = {}
    for cls in CLASSES:
        tp = matrix.get(cls, {}).get(cls, 0)
        actual = sum(matrix.get(cls, {}).values())
        predicted = sum(row.get(cls, 0) for row in matrix.values())
        result[cls] = {
            "support": actual,
            "precision": tp / predicted if predicted else 0.0,
            "recall": tp / actual if actual else 0.0,
        }
    return result


def aggregate_metric(records: list[dict], key: str) -> dict[str, float | int | None]:
    values = [float(r[key]) for r in records if r.get(key) is not None]
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    a = np.asarray(values, dtype=float)
    return {"n": len(values), "mean": float(np.mean(a)), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(np.max(a))}
