"""Decision-level comparison for swept-path sampling refinements."""
from __future__ import annotations

import math


_ESTIMATE_FIELDS = (
    "class_label_from_M4_contract",
    "estimated_xy_x_m",
    "estimated_xy_y_m",
    "estimated_yaw_rad",
    "perceived_obstacle_count",
)


def compare_sampling_runs(coarse: dict, refined: dict, *, refined_step_m: float) -> dict:
    """Compare the same request at two collision-sampling resolutions.

    Collision outcome, selected branch/slot and phase/event sequence must agree.
    The path-length integration difference is bounded by the refined spatial
    resolution. Sampled minimum clearance is reported as a grid-sensitive
    diagnostic, not treated as a continuous or resolution-invariant quantity.
    """
    if not math.isfinite(refined_step_m) or refined_step_m <= 0.0:
        raise ValueError("refined_step_m must be finite and positive")

    same_estimate = all(coarse.get(key) == refined.get(key) for key in _ESTIMATE_FIELDS)
    same_result = coarse.get("result") == refined.get("result")
    both_success = bool(coarse.get("success")) and bool(refined.get("success"))
    same_route = all(coarse.get(key) == refined.get(key) for key in (
        "selected_slot_id", "selected_ik_branch", "phase_sequence", "event_sequence"))

    path_delta = abs(float(coarse["tcp_path_length_m"]) - float(refined["tcp_path_length_m"]))
    duration_delta = abs(float(coarse["duration_s"]) - float(refined["duration_s"]))
    coarse_clearance = float(coarse["all_pair_clearance_lower_bound_m"])
    refined_clearance = float(refined["all_pair_clearance_lower_bound_m"])
    clearance_delta = abs(coarse_clearance - refined_clearance)
    new_collision = bool(coarse.get("success")) and not bool(refined.get("success"))

    geometry_stable = path_delta <= refined_step_m
    passed = (same_estimate and same_result and both_success and same_route
              and geometry_stable and coarse_clearance > 0.0 and refined_clearance > 0.0)
    return {
        "status": "PASS" if passed else "FAIL",
        "method": (
            "Run the same M4 estimate and deterministic plan at baseline and half-step collision sampling. "
            "Require the same successful collision-preflight outcome, destination, IK branch, phase/event "
            "sequence, and TCP path length difference no greater than the refined spatial step."
        ),
        "same_m4_estimate": same_estimate,
        "same_collision_outcome": same_result and both_success,
        "new_collision_at_refined_resolution": new_collision,
        "same_planned_route_semantics": same_route,
        "tcp_path_length_stable": geometry_stable,
        "path_length_comparison_tolerance_m": refined_step_m,
        "coarse_case": coarse["case_id"],
        "coarse_step_m": float(coarse["collision_sample_step_m"]),
        "refined_case": refined["case_id"],
        "refined_step_m": float(refined["collision_sample_step_m"]),
        "both_plans_success": both_success,
        "duration_difference_s": duration_delta,
        "tcp_path_length_difference_m": path_delta,
        "coarse_min_clearance_lower_bound_m": coarse_clearance,
        "refined_min_clearance_lower_bound_m": refined_clearance,
        "clearance_lower_bound_difference_m": clearance_delta,
        "clearance_estimate_status": (
            "GRID_SENSITIVE" if clearance_delta > refined_step_m else "STABLE_WITHIN_REFINED_STEP"
        ),
        "note": (
            "PASS is decision-level: the refined full-path preflight found no new collision and the same "
            "geometric route was produced. The minimum sampled clearance lower bound changed by "
            f"{clearance_delta:.9g} m; it is explicitly grid-sensitive because the minimum may occur "
            "between coarse samples. Pair-specific clearance, swept-step and numeric requirements are "
            "checked at every sample. This comparison is not a continuous collision-free proof."
        ),
    }
