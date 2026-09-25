from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import math

import numpy as np

from .context import PlanningContext
from .records import CartesianWaypoint, Phase, TrajectorySample
from kinematics.scara import check_joint_limits, forward_kinematics


@dataclass(frozen=True)
class SegmentMetric:
    duration_s: float
    samples: tuple[TrajectorySample, ...]
    maximum_velocity_ratio: float
    maximum_acceleration_ratio: float
    swept_bound_m: float
    sample_displacement_bound_m: float


def _quintic(u: float) -> tuple[float, float, float]:
    u2, u3 = u*u, u*u*u
    return 10*u3-15*u3*u+6*u3*u2, 30*u2-60*u3+30*u2*u2, 60*u-180*u2+120*u3


def _swept_bound(delta: np.ndarray, gripper_delta: float, ctx: PlanningContext) -> float:
    r1=float(ctx.p("robot.link1_length_m"))+float(ctx.p("robot.link2_length_m"))+0.07
    r2=float(ctx.p("robot.link2_length_m"))+0.035
    return float(r1*abs(delta[0])+r2*abs(delta[1])+abs(delta[2])+0.07*abs(delta[3])+abs(gripper_delta))


def _duration_rest_to_rest(displacement: np.ndarray, velocity: np.ndarray,
                           acceleration: np.ndarray, speed_scale: float) -> float:
    # Exact extrema of a quintic 10u^3-15u^4+6u^5.
    tv=1.875*np.abs(displacement)/velocity
    ta=np.sqrt((10.0/math.sqrt(3.0))*np.abs(displacement)/acceleration)
    nominal=float(np.max(np.maximum(tv,ta))) if len(displacement) else 0.0
    return nominal/speed_scale if nominal else 0.0


def parameterize_segment(
    q0: Iterable[float], q1: Iterable[float], *, start_time_s: float,
    waypoint: CartesianWaypoint, ctx: PlanningContext, start_gripper_m: float | None = None,
    speed_scale: float = 1.0, sample_step_override_m: float | None = None,
) -> SegmentMetric:
    """Two-point quintic segment utility used by focused trajectory checks."""
    qa,qb=np.asarray(tuple(q0),dtype=float),np.asarray(tuple(q1),dtype=float)
    if qa.shape!=(4,) or qb.shape!=(4,) or not np.all(np.isfinite(qa)) or not np.all(np.isfinite(qb)):
        raise ValueError("Arm segment endpoints must be four finite coordinates")
    if not math.isfinite(start_time_s) or not 0<speed_scale<=1:
        raise ValueError("Invalid start time or speed scale")
    if not check_joint_limits(qa,ctx.arm) or not check_joint_limits(qb,ctx.arm):
        raise ValueError("Arm segment endpoint violates a joint limit")
    g0=float(waypoint.gripper_m if start_gripper_m is None else start_gripper_m)
    g1=float(waypoint.gripper_m)
    glo,ghi=map(float,ctx.p("robot.finger_range_m"))
    if not glo<=g0<=ghi or not glo<=g1<=ghi:
        raise ValueError("Gripper segment endpoint violates its joint range")
    delta=qb-qa; dg=g1-g0
    vlim=np.asarray((*ctx.motion_velocity_limits,float(ctx.p("robot.finger_velocity_m_s"))),dtype=float)
    alim=np.asarray((*ctx.motion_acceleration_limits,float(ctx.p("robot.finger_acceleration_m_s2"))),dtype=float)
    movement=np.asarray((*delta,dg))
    duration=_duration_rest_to_rest(movement,vlim,alim,speed_scale)
    if duration==0:
        fk=forward_kinematics(qb,ctx.arm).pose
        sample=TrajectorySample(start_time_s,waypoint.phase,tuple(qb),(0.,)*4,(0.,)*4,
            (fk.x_m,fk.y_m,fk.z_m,fk.yaw_rad),waypoint.branch_id,g1,waypoint.payload_mode,0.,0.,waypoint.event)
        return SegmentMetric(0.,(sample,),0.,0.,0.,0.)
    swept=_swept_bound(delta,dg,ctx)
    step=float(sample_step_override_m if sample_step_override_m is not None else ctx.p("planning.path_sweep_sample_step_m"))
    if not math.isfinite(step) or step<=0: raise ValueError("Sample step must be positive and finite")
    n=max(1,math.ceil(1.875*swept/step))
    samples=[]; vmax=amax=0.
    for i in range(n+1):
        u=i/n; s,ds,dds=_quintic(u)
        q=qa+delta*s; dq=delta*ds/duration; ddq=delta*dds/duration**2
        g=g0+dg*s; gd=dg*ds/duration; gdd=dg*dds/duration**2
        if not check_joint_limits(q,ctx.arm) or not glo<=g<=ghi: raise ValueError("Interpolation crossed a physical limit")
        vr=float(np.max(np.abs(np.asarray((*dq,gd))/vlim)))
        ar=float(np.max(np.abs(np.asarray((*ddq,gdd))/alim)))
        vmax=max(vmax,vr); amax=max(amax,ar)
        fk=forward_kinematics(q,ctx.arm).pose
        samples.append(TrajectorySample(start_time_s+u*duration,waypoint.phase,tuple(map(float,q)),
            tuple(map(float,dq)),tuple(map(float,ddq)),(fk.x_m,fk.y_m,fk.z_m,fk.yaw_rad),
            waypoint.branch_id,g,waypoint.payload_mode,gd,gdd,waypoint.event if i==n else None))
    return SegmentMetric(duration,tuple(samples),vmax,amax,swept,swept*1.875/n)


def _pchip_slopes(s: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Shape-preserving first derivatives for a C1 piecewise cubic path."""
    n,d=x.shape
    slopes=np.zeros_like(x)
    if n==2:
        slopes[0]=slopes[1]=(x[1]-x[0])/(s[1]-s[0])
        return slopes
    h=np.diff(s)
    sec=np.diff(x,axis=0)/h[:,None]
    slopes[0]=sec[0]; slopes[-1]=sec[-1]
    for i in range(1,n-1):
        for j in range(d):
            left,right=sec[i-1,j],sec[i,j]
            if left==0.0 or right==0.0 or left*right<=0.0:
                slopes[i,j]=0.0
            else:
                w1=2*h[i]+h[i-1]; w2=h[i]+2*h[i-1]
                slopes[i,j]=(w1+w2)/(w1/left+w2/right)
    return slopes


def _cubic_coefficients(s: np.ndarray, x: np.ndarray, slopes: np.ndarray):
    h=np.diff(s)
    a=2*x[:-1]-2*x[1:]+h[:,None]*(slopes[:-1]+slopes[1:])
    b=-3*x[:-1]+3*x[1:]-h[:,None]*(2*slopes[:-1]+slopes[1:])
    c=h[:,None]*slopes[:-1]
    d=x[:-1]
    return a,b,c,d,h


def _eval_spline(value: float, s: np.ndarray, coeff):
    a,b,c,d,h=coeff
    i=min(len(h)-1,max(0,int(np.searchsorted(s,value,side="right")-1)))
    t=min(1.,max(0.,(value-s[i])/h[i]))
    x=((a[i]*t+b[i])*t+c[i])*t+d[i]
    dx=(3*a[i]*t*t+2*b[i]*t+c[i])/h[i]
    ddx=(6*a[i]*t+2*b[i])/h[i]**2
    return x,dx,ddx


def _spline_derivative_bounds(s: np.ndarray, coeff):
    a,b,c,_,h=coeff
    d1max=np.zeros(a.shape[1]); d2max=np.zeros(a.shape[1])
    for i,hi in enumerate(h):
        for j in range(a.shape[1]):
            candidates=[c[i,j],3*a[i,j]+2*b[i,j]+c[i,j]]
            if abs(a[i,j])>1e-15:
                vertex=-b[i,j]/(3*a[i,j])
                if 0<vertex<1: candidates.append(3*a[i,j]*vertex**2+2*b[i,j]*vertex+c[i,j])
            d1max[j]=max(d1max[j],max(abs(v) for v in candidates)/hi)
            d2max[j]=max(d2max[j],max(abs(2*b[i,j]),abs(6*a[i,j]+2*b[i,j]))/hi**2)
    return d1max,d2max


def _time_at_path_fraction(s: float) -> float:
    lo,hi=0.,1.
    for _ in range(48):
        mid=(lo+hi)/2
        if _quintic(mid)[0]<s: lo=mid
        else: hi=mid
    return (lo+hi)/2


def _path_segment(run: Sequence[CartesianWaypoint], ctx: PlanningContext, start_time: float,
                  step: float, speed_scale: float):
    qspan=ctx.arm.characteristic_spans
    gspan=float(ctx.p("robot.finger_range_m")[1]-ctx.p("robot.finger_range_m")[0])
    spans=np.asarray((*qspan,gspan),dtype=float)
    raw=np.asarray([(*p.q,float(p.gripper_m)) for p in run],dtype=float)
    # Merge coincident states while preserving the final event/state for zero-motion phases.
    nodes=[raw[0]]
    for row in raw[1:]:
        if np.linalg.norm((row-nodes[-1])/spans)>1e-13: nodes.append(row)
    end=run[-1]
    if len(nodes)==1:
        q=raw[-1,:4]; fk=forward_kinematics(q,ctx.arm).pose
        sample=TrajectorySample(start_time,end.phase,tuple(map(float,q)),(0.,)*4,(0.,)*4,
            (fk.x_m,fk.y_m,fk.z_m,fk.yaw_rad),end.branch_id,float(raw[-1,4]),end.payload_mode,0.,0.,end.event)
        return (sample,),0.,0.,0.,0.,0.
    x=np.asarray(nodes,dtype=float)
    dx=np.diff(x,axis=0)
    ds=np.linalg.norm(dx/spans,axis=1)
    s=np.concatenate(([0.],np.cumsum(ds))); s/=s[-1]
    slopes=_pchip_slopes(s,x)
    coeff=_cubic_coefficients(s,x,slopes)
    d1,d2=_spline_derivative_bounds(s,coeff)
    # The fourth-order body-motion bound includes fingers and the carried box.
    r1=float(ctx.p("robot.link1_length_m"))+float(ctx.p("robot.link2_length_m"))+0.07
    r2=float(ctx.p("robot.link2_length_m"))+0.035
    body_d1=r1*d1[0]+r2*d1[1]+d1[2]+0.07*d1[3]+d1[4]
    qvmax=np.asarray((*ctx.motion_velocity_limits,float(ctx.p("robot.finger_velocity_m_s"))),dtype=float)
    qamax=np.asarray((*ctx.motion_acceleration_limits,float(ctx.p("robot.finger_acceleration_m_s2"))),dtype=float)
    max_s=1.875; max_ss=10/math.sqrt(3)
    unit_v=d1*max_s
    unit_a=d2*max_s**2+d1*max_ss
    safety=float(ctx.p("planning.parameterization_safety_scale"))
    duration=max(float(np.max(unit_v/qvmax)),
                 math.sqrt(float(np.max(unit_a/qamax))))*safety/speed_scale
    if duration<=0 or not math.isfinite(duration):
        raise ValueError("Spline phase has invalid temporal duration")
    intervals=max(1,math.ceil(max_s*body_d1/(step*speed_scale)))
    uvalues=set(np.linspace(0.,1.,intervals+1).tolist())
    uvalues.update(_time_at_path_fraction(float(knot)) for knot in s[1:-1])
    uvalues=sorted(uvalues)
    samples=[]; vmax=amax=sample_bound=0.
    for index,u in enumerate(uvalues):
        path_s,dsdu,d2sdu2=_quintic(u)
        state,dxds,d2xds2=_eval_spline(path_s,s,coeff)
        q=state[:4]; g=float(state[4])
        dq=dxds[:4]*dsdu/duration
        ddq=(d2xds2[:4]*dsdu**2+dxds[:4]*d2sdu2)/duration**2
        gd=float(dxds[4]*dsdu/duration)
        gdd=float((d2xds2[4]*dsdu**2+dxds[4]*d2sdu2)/duration**2)
        if not check_joint_limits(q,ctx.arm): raise ValueError("Cubic path exceeds an M3 joint limit")
        glo,ghi=map(float,ctx.p("robot.finger_range_m"))
        if not glo-1e-12<=g<=ghi+1e-12: raise ValueError("Cubic path exceeds gripper limits")
        vmax=max(vmax,float(np.max(np.abs(np.asarray((*dq,gd))/qvmax))))
        amax=max(amax,float(np.max(np.abs(np.asarray((*ddq,gdd))/qamax))))
        fk=forward_kinematics(q,ctx.arm).pose
        event=end.event if index==len(uvalues)-1 else None
        samples.append(TrajectorySample(start_time+u*duration,end.phase,tuple(map(float,q)),
            tuple(map(float,dq)),tuple(map(float,ddq)),(fk.x_m,fk.y_m,fk.z_m,fk.yaw_rad),
            end.branch_id,g,end.payload_mode,gd,gdd,event))
    for a,b in zip(samples,samples[1:]):
        sample_bound=max(sample_bound,_swept_bound(np.asarray(b.q)-np.asarray(a.q),
                                                    b.gripper_m-a.gripper_m,ctx))
    # Analytical derivative bound controls both motion limits and swept sampling.
    return tuple(samples),duration,vmax,amax,max(sample_bound,body_d1*max_s/intervals),body_d1


def build_trajectory(
    waypoints: Iterable[CartesianWaypoint], ctx: PlanningContext, *,
    contact_phases: set[str] | None = None,
    sample_step_override_m: float | None = None,
):
    """C1 shape-preserving joint spline, stopped only at semantic phase/event boundaries."""
    points=tuple(waypoints)
    if len(points)<2: raise ValueError("At least two waypoints are required")
    if ctx.p("planning.motion_profile")!="quintic_rest_to_rest":
        raise ValueError("Unsupported SSOT motion profile")
    step=float(sample_step_override_m if sample_step_override_m is not None
               else ctx.p("planning.path_sweep_sample_step_m"))
    if not math.isfinite(step) or step<=0: raise ValueError("Path sampling step must be positive")
    low_speed=contact_phases or {
        Phase.DESCENT.value,Phase.GRASP_CLOSE.value,Phase.LIFT.value,
        Phase.PLACE_DESCENT.value,Phase.RELEASE.value,Phase.RETREAT.value,
    }
    output=[]; t=0.; vmax=amax=bound=0.
    i=0
    while i<len(points)-1:
        phase=points[i+1].phase
        payload=points[i+1].payload_mode
        j=i+1
        while j<len(points)-1:
            endpoint,next_point=points[j],points[j+1]
            if endpoint.event is not None or next_point.phase!=phase or next_point.payload_mode!=payload:
                break
            j+=1
        run=points[i:j+1]
        scale=float(ctx.p("planning.contact_speed_scale")) if phase in low_speed else 1.
        samples,duration,vr,ar,sb,_=_path_segment(run,ctx,t,step,scale)
        if output and samples:
            first=samples[0]
            last=output[-1]
            if (abs(first.time_s-last.time_s)<1e-12 and first.q==last.q and
                    first.gripper_m==last.gripper_m and first.payload_mode==last.payload_mode and
                    first.event is None and last.event is None and first.phase==last.phase):
                samples=samples[1:]
        output.extend(samples)
        t+=duration; vmax=max(vmax,vr); amax=max(amax,ar); bound=max(bound,sb)
        i=j
    return tuple(output),{
        "duration_s":t,
        "max_velocity_ratio":vmax,
        "max_acceleration_ratio":amax,
        "max_swept_sample_bound_m":bound,
    }
