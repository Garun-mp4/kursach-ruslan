"""Independent analytic kinematics for the M2 SCARA model."""

from .scara import (
    ArmConfig,
    FKResult,
    IKCandidate,
    IKResult,
    IKStatus,
    Pose,
    classify_singularity,
    forward_kinematics,
    inverse_kinematics,
    jacobian,
    load_config,
    workspace_bounds,
)

__all__ = [
    "ArmConfig",
    "FKResult",
    "IKCandidate",
    "IKResult",
    "IKStatus",
    "Pose",
    "classify_singularity",
    "forward_kinematics",
    "inverse_kinematics",
    "jacobian",
    "load_config",
    "workspace_bounds",
]
