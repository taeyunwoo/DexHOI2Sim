"""
Vertex-level verification of URDF rigid skinning vs MANO LBS.

For each vertex v (778), assign it to argmax LBS bone b(v). At a given pose:
  - MANO position : v_lbs (from smplx forward, full LBS)
  - URDF position : T_{b(v)} @ v_rest_in_world

Compute per-vertex error, report mean / max / per-link stats.

Inputs:
  --urdf-dir   : dir from generate_urdf.py (has rest_verts.npy, lbs_argmax.npy, betas.npy)
  --poses-npy  : (T,48) full-pose axis-angle from pose_to_joint_angles.py
  --trans-npy  : (T,3) trans
  --is_rhand
  --out        : report dir (writes verify_report.json + per_frame_mean_err.npy)
"""
import argparse, os, sys, json
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "mano2urdf"))

import smplx
from mano_helpers import get_kinematic_order_mano, get_mano_data


def setup_smplx_root():
    MODEL_DIR = os.path.join(ROOT, "outputs", "_smplx_root")
    os.makedirs(os.path.join(MODEL_DIR, "mano"), exist_ok=True)
    for p in ["MANO_RIGHT.pkl", "MANO_LEFT.pkl"]:
        dst = os.path.join(MODEL_DIR, "mano", p)
        if not os.path.exists(dst):
            os.symlink(os.path.join(ROOT, "assets", p), dst)
    return MODEL_DIR


def urdf_link_T_world(rest_joints, parent_dict, finger_euler_xyz, wrist_R, wrist_t):
    """
    Returns dict name -> 4x4 world T of link's *origin frame*.
    finger_euler_xyz: dict name -> (θx,θy,θz) intrinsic XYZ
    wrist_R: 3x3 rotation, wrist_t: 3-vec translation.
    URDF link frame at joint = parent_world · trans(rest_child - rest_parent) · Rx Ry Rz
    The link's STL was exported with vertices in world coords minus joints_dict[child]
    (i.e. rest_verts_in_link_frame = v_rest_world - rest_joints[child]).
    """
    order = []
    visited = set()
    def visit(n):
        if n in visited: return
        p = parent_dict[n]
        if p is not None and p not in visited:
            visit(p)
        visited.add(n); order.append(n)
    for n in parent_dict: visit(n)

    world_T = {}
    for name in order:
        parent = parent_dict[name]
        if parent is None:
            # MANO LBS: joint_pos_wrist = rest_j_wrist + transl, rotation = R_global
            # NOTE: rotation is NOT applied to rest_j_wrist; the wrist joint stays at rest pos in MANO frame.
            T = np.eye(4); T[:3, :3] = wrist_R; T[:3, 3] = wrist_t + rest_joints[name]
            world_T[name] = T
        else:
            offset = rest_joints[name] - rest_joints[parent]
            T_parent = world_T[parent]
            T_off = np.eye(4); T_off[:3, 3] = offset
            tx, ty, tz = finger_euler_xyz[name]
            Rmat = (R.from_euler('x', tx) * R.from_euler('y', ty) * R.from_euler('z', tz)).as_matrix()
            T_rot = np.eye(4); T_rot[:3, :3] = Rmat
            world_T[name] = T_parent @ T_off @ T_rot
    return world_T


def run_mano_verts(pose48, trans, betas, is_rhand, model_dir):
    Tn = pose48.shape[0]
    m = smplx.create(model_path=model_dir, model_type="mano",
                     is_rhand=is_rhand, use_pca=False, flat_hand_mean=True,
                     batch_size=Tn)
    out = m(global_orient=torch.from_numpy(pose48[:, :3]).float(),
            hand_pose=torch.from_numpy(pose48[:, 3:]).float(),
            betas=torch.from_numpy(betas).unsqueeze(0).expand(Tn, -1).contiguous(),
            transl=torch.from_numpy(trans).float())
    return out.vertices.detach().cpu().numpy(), out.joints.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf-dir", required=True)
    ap.add_argument("--poses-npy", required=True)
    ap.add_argument("--trans-npy", required=True)
    ap.add_argument("--is_rhand", action="store_true")
    ap.add_argument("--lhand", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()
    is_rhand = (not args.lhand) if not args.is_rhand else True
    os.makedirs(args.out, exist_ok=True)

    betas = np.load(os.path.join(args.urdf_dir, "betas.npy")).astype(np.float32)
    lbs_argmax = np.load(os.path.join(args.urdf_dir, "lbs_argmax.npy"))   # (778,)
    rest_verts_world = np.load(os.path.join(args.urdf_dir, "rest_verts.npy"))  # (778,3) in MANO rest frame (= world at θ=0,trans=0)

    pose48 = np.load(args.poses_npy).astype(np.float32)
    trans = np.load(args.trans_npy).astype(np.float32)
    if args.max_frames:
        pose48 = pose48[:args.max_frames]; trans = trans[:args.max_frames]
    Tn = pose48.shape[0]

    model_dir = setup_smplx_root()
    name_order = list(get_kinematic_order_mano(is_rhand).keys())
    parent_dict = get_kinematic_order_mano(is_rhand)

    # rest joints (β-dependent), zero pose, zero trans  → rest_joints[name]
    rest_joints_arr, _, joints_dict = get_mano_data(model_path=model_dir, is_rhand=is_rhand,
                                                    v_template=torch.from_numpy(rest_verts_world).float())
    # joints_dict only has 21 (16 + 5 tips); take first 16 by name_order
    rest_joints = {n: joints_dict[n] for n in name_order}

    # rest verts per link in link frame = rest_verts_world - rest_joints[link]
    link_idx_map = {name: i for i, name in enumerate(name_order)}
    v_link_frame = np.zeros_like(rest_verts_world)  # (778,3)
    for v in range(rest_verts_world.shape[0]):
        b = lbs_argmax[v]
        v_link_frame[v] = rest_verts_world[v] - rest_joints[name_order[b]]

    # MANO ground truth verts
    mano_verts, mano_joints = run_mano_verts(pose48, trans, betas, is_rhand, model_dir)  # (T,778,3)

    # axis-angle → euler per frame
    finger_aa = pose48[:, 3:].reshape(Tn, 15, 3)
    global_aa = pose48[:, :3]
    finger_eulers_per_frame = R.from_rotvec(finger_aa.reshape(-1, 3)).as_euler('XYZ').reshape(Tn, 15, 3)
    wrist_R_per_frame = R.from_rotvec(global_aa).as_matrix()  # (T,3,3)
    finger_names = name_order[1:]

    per_frame_mean = np.zeros(Tn)
    per_frame_max = np.zeros(Tn)
    per_link_sum = np.zeros(len(name_order))
    per_link_cnt = np.zeros(len(name_order))

    for t in range(Tn):
        fe = {n: finger_eulers_per_frame[t, i] for i, n in enumerate(finger_names)}
        Tworld = urdf_link_T_world(rest_joints, parent_dict, fe,
                                   wrist_R_per_frame[t], trans[t])
        # compute URDF verts
        urdf_verts = np.zeros_like(rest_verts_world)
        for v in range(rest_verts_world.shape[0]):
            b = lbs_argmax[v]
            T = Tworld[name_order[b]]
            urdf_verts[v] = T[:3, :3] @ v_link_frame[v] + T[:3, 3]
        err = np.linalg.norm(urdf_verts - mano_verts[t], axis=1) * 1000  # mm
        per_frame_mean[t] = err.mean()
        per_frame_max[t] = err.max()
        for v in range(rest_verts_world.shape[0]):
            b = lbs_argmax[v]
            per_link_sum[b] += err[v]; per_link_cnt[b] += 1

    per_link_mean = per_link_sum / np.maximum(per_link_cnt, 1)
    report = {
        "num_frames": int(Tn),
        "mean_err_mm": float(per_frame_mean.mean()),
        "max_err_mm": float(per_frame_max.max()),
        "p95_err_mm": float(np.percentile(per_frame_mean, 95)),
        "per_link_mean_err_mm": {name_order[i]: float(per_link_mean[i]) for i in range(len(name_order))},
        "per_link_vertex_count": {name_order[i]: int(per_link_cnt[i]) for i in range(len(name_order))},
    }
    with open(os.path.join(args.out, "verify_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    np.save(os.path.join(args.out, "per_frame_mean_err.npy"), per_frame_mean)
    np.save(os.path.join(args.out, "per_frame_max_err.npy"), per_frame_max)

    print(f"[verify] T={Tn}  mean={report['mean_err_mm']:.2f}mm  max={report['max_err_mm']:.2f}mm  p95={report['p95_err_mm']:.2f}mm")
    print(f"[verify] worst links:")
    for n, e in sorted(report["per_link_mean_err_mm"].items(), key=lambda x: -x[1])[:6]:
        print(f"   {n:<22} {e:.2f} mm")


if __name__ == "__main__":
    main()
