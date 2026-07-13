"""Verify the MuJoCo-posed MANO2URDF hand against MANO ground truth.

For each frame we (a) set the 51 MuJoCo qpos from the frame json, run mj_forward,
and read the 16 hand-link body world positions; (b) run MANO (smplx) forward for
the same axis-angle pose to get the 21 GT joints; then compare per-joint.

If the MuJoCo articulation were wrong (e.g. wrong euler/axis composition) the
error would be centimetres. Correct FK should be ~millimetres (URDF is a faithful
kinematic reparam of MANO; residual is the same argmax/blendshape gap as the
2.1 mm vertex check).
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
import sys
import re
import json
import glob
import argparse
from pathlib import Path

import numpy as np
import torch
import smplx
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mano2urdf_mujoco import mesh_collision_urdf, MJ_COMPILER, build_qpos_indexer

# MANO2URDF link name -> smplx MANO joint index (out.joints: 0 wrist, then
# index/middle/pinky/ring/thumb each 1..3, tips appended after 15)
LINK_TO_MANO = {
    "right_wrist_rz": 0,   # palm/wrist body (after the 6 wrist DOF); "right_wrist"
                           # is an intermediate prismatic link, not the joint
    "right_index1": 1, "right_index2": 2, "right_index3": 3,
    "right_middle1": 4, "right_middle2": 5, "right_middle3": 6,
    "right_pinky1": 7, "right_pinky2": 8, "right_pinky3": 9,
    "right_ring1": 10, "right_ring2": 11, "right_ring3": 12,
    "right_thumb1": 13, "right_thumb2": 14, "right_thumb3": 15,
}
_UNUSED = {
    "right_index1": 1, "right_index2": 2, "right_index3": 3,
    "right_middle1": 4, "right_middle2": 5, "right_middle3": 6,
    "right_pinky1": 7, "right_pinky2": 8, "right_pinky3": 9,
    "right_ring1": 10, "right_ring2": 11, "right_ring3": 12,
    "right_thumb1": 13, "right_thumb2": 14, "right_thumb3": 15,
}


def mano_joints(pose48, trans, betas, model_dir):
    T = pose48.shape[0]
    m = smplx.create(model_path=model_dir, model_type="mano", is_rhand=True,
                     use_pca=False, flat_hand_mean=True, batch_size=T)
    out = m(global_orient=torch.from_numpy(pose48[:, :3]).float(),
            hand_pose=torch.from_numpy(pose48[:, 3:]).float(),
            betas=torch.from_numpy(betas).unsqueeze(0).expand(T, -1).contiguous(),
            transl=torch.from_numpy(trans).float())
    return out.joints.detach().cpu().numpy()   # (T, 21, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf-dir", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--poses-npy", required=True)
    ap.add_argument("--trans-npy", required=True)
    ap.add_argument("--betas-npy", required=True)
    ap.add_argument("--model-dir", required=True, help="smplx root (has mano/)")
    args = ap.parse_args()

    urdf_dir = Path(args.urdf_dir).resolve()
    # compile hand-only model (no scene needed for FK)
    urdf_path = next(urdf_dir.glob("*.urdf"))
    txt = mesh_collision_urdf(urdf_path.read_text())
    txt = re.sub(r"(<robot[^>]*>)", r"\1\n" + MJ_COMPILER, txt, count=1)
    cwd = os.getcwd(); os.chdir(urdf_dir)
    try:
        model = mujoco.MjModel.from_xml_string(txt)
    finally:
        os.chdir(cwd)
    data = mujoco.MjData(model)
    to_qpos = build_qpos_indexer(model)
    base_q = data.qpos.copy()

    body_id = {ln: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ln)
               for ln in LINK_TO_MANO}
    missing = [ln for ln, i in body_id.items() if i < 0]
    if missing:
        print("[warn] MuJoCo bodies not found:", missing)

    frames = [json.load(open(f)) for f in sorted(glob.glob(str(Path(args.frames_dir) / "*.json")))]
    pose48 = np.load(args.poses_npy)
    trans = np.load(args.trans_npy)
    betas = np.load(args.betas_npy).astype(np.float32)
    gt = mano_joints(pose48, trans, betas, args.model_dir)   # (T,21,3)

    per_joint_err = {ln: [] for ln in LINK_TO_MANO}
    all_err = []
    for t, fr in enumerate(frames):
        data.qpos[:] = to_qpos(fr, base_q)
        mujoco.mj_forward(model, data)
        for ln, mano_idx in LINK_TO_MANO.items():
            bid = body_id[ln]
            if bid < 0:
                continue
            mj_pos = data.xpos[bid]
            g = gt[t, mano_idx]
            e = np.linalg.norm(mj_pos - g)
            per_joint_err[ln].append(e)
            all_err.append(e)

    all_err = np.array(all_err) * 1000.0   # mm
    print(f"\n[MuJoCo-FK vs MANO joints]  T={len(frames)}")
    print(f"  mean = {all_err.mean():.2f} mm   median = {np.median(all_err):.2f} mm   "
          f"max = {all_err.max():.2f} mm")
    print("  worst links:")
    means = {ln: np.mean(v) * 1000 for ln, v in per_joint_err.items() if v}
    for ln, e in sorted(means.items(), key=lambda x: -x[1])[:6]:
        print(f"    {ln:16s} {e:6.2f} mm")


if __name__ == "__main__":
    main()
