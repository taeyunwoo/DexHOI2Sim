"""Evaluation metric for HOI replication (MuJoCo, deterministic).

The question that matters: **when the hand executes the generated motion in
physics, does the object follow its intended (reference) trajectory?**

We roll the sequence out under gravity + contact — the hand PD-tracks the
reference joint trajectory, the object is a free rigid body — and record the
object's simulated path. Comparing it to the given (recorded) object trajectory
tells us whether the interaction is physically reproducible: a good grasp carries
the object along its reference path (small error); a failed grasp lets it stay /
slip / fall (large error).

  object_traj_error_mm : mean per-frame object position error (sim vs reference)
  final_error_mm       : object position error at the last frame
  grasp_success        : True if final_error_mm < threshold (object ended up where
                         the reference says it should)
"""
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mano2urdf_mujoco import (load_frames, load_dexycb_object_tag,
                              transform_frames_to_tag, build_hand_model,
                              object_xml, build_qpos_indexer, hide_collision_geoms)

SUCCESS_MM = 50.0        # object ends within 5 cm of its reference position


def trajectory_error(urdf_dir, meshes, obj_pose_tag, frames, fps=30):
    """Physics rollout (hand PD-tracks the reference, object dynamic); return the
    simulated object positions sampled at each reference frame."""
    ctr = obj_pose_tag[:, :3].mean(0)
    oa, ob = object_xml(meshes)
    model, _ = build_hand_model(
        Path(urdf_dir), gravity=-9.81, floor_z=0.0,
        cam_pos=f"{ctr[0]:.3f} {ctr[1]-0.5:.3f} {ctr[2]+0.3:.3f}",
        lookat=f"{ctr[0]:.3f} {ctr[1]:.3f} {ctr[2]:.3f}",
        object_body=ob, object_asset=oa, actuated=True)
    hide_collision_geoms(model)
    data = mujoco.MjData(model)
    to_qpos = build_qpos_indexer(model)
    base_q = data.qpos.copy()
    obj_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj_free")]
    act_qadr = np.array([model.jnt_qposadr[model.actuator_trnid[a, 0]]
                         for a in range(model.nu)])

    ref_q = np.stack([to_qpos(fr, base_q.copy()) for fr in frames])   # (T, nq)
    ref_q[:, obj_qadr:obj_qadr + 7] = obj_pose_tag
    T = len(frames)

    data.qpos[:] = ref_q[0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    n_steps = int((T / float(fps)) / dt)
    stride = max(1, int(round(1.0 / (fps * dt))))
    sim_obj = []
    for s in range(n_steps):
        tt = s * dt * fps
        i0 = min(int(tt), T - 1); i1 = min(i0 + 1, T - 1); a = tt - i0
        ref = (1 - a) * ref_q[i0] + a * ref_q[i1]
        data.ctrl[:] = ref[act_qadr]
        mujoco.mj_step(model, data)
        if s % stride == 0:
            sim_obj.append(data.qpos[obj_qadr:obj_qadr + 3].copy())
    return np.array(sim_obj)                                        # (~T, 3)


def _score(urdf_dir, meshes, obj_pose_tag, frames, obj_name):
    sim_obj = trajectory_error(urdf_dir, meshes, obj_pose_tag, frames)
    n = min(len(sim_obj), len(obj_pose_tag))
    err = np.linalg.norm(sim_obj[:n] - obj_pose_tag[:n, :3], axis=1) * 1000.0
    return {
        "object": obj_name,
        "n_frames": int(n),
        "object_traj_error_mm": round(float(err.mean()), 2),
        "final_error_mm": round(float(err[-1]), 2),
        "grasp_success": bool(err[-1] < SUCCESS_MM),
    }


def evaluate(dexycb_root, subject, session, urdf_dir, frames_dir, root):
    """DexYCB: load object + master→tag frame, then score."""
    frames = load_frames(Path(frames_dir) / "frames")
    obj_name, meshes, obj_pose_tag, tag = load_dexycb_object_tag(
        dexycb_root, subject, session)
    frames = transform_frames_to_tag(frames, tag)
    return _score(urdf_dir, meshes, obj_pose_tag, frames, obj_name)


def evaluate_custom(urdf_dir, frames_dir, object_cad, object_poses_npy):
    """Custom: frames + object poses already in a Z-up world frame; solid object."""
    frames = load_frames(Path(frames_dir) / "frames")
    obj_pose_tag = np.load(object_poses_npy).astype(np.float32)
    cad = str(Path(object_cad).resolve())
    meshes = {"vis": cad, "col": cad}
    return _score(urdf_dir, meshes, obj_pose_tag, frames, Path(object_cad).name)
