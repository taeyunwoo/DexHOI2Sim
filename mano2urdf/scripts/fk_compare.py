"""
MANO θ → URDF joint angles 의 naive feed-forward 가 안 맞는다는 걸 보여주는 스크립트.

URDF chain per finger joint (artigrasp mano2urdf 구조):
    parent_link -- joint_x([1,0,0]) -> link -- joint_y([0,1,0]) -> link_y -- joint_z([0,0,1]) -> link_z
즉 link_z 의 회전 = Rx(θx) · Ry(θy) · Rz(θz)

MANO 의 finger joint 회전 = exp([θ]_×) where θ ∈ R^3 (axis-angle)

(A) Naive : θx,θy,θz = MANO axis-angle 의 components
(B) Correct : (θx,θy,θz) = axis_angle_to_euler_xyz(MANO axis-angle)
"""
import os, sys, json
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "mano2urdf"))

import smplx
from mano_helpers import get_mano_joint_names, get_kinematic_order_mano

ASSETS = os.path.join(ROOT, "assets")
# smplx expects model_path that contains mano/MANO_RIGHT.pkl OR has model files directly
# We'll set up a small symlink tree.
MODEL_DIR = os.path.join(ROOT, "outputs", "_smplx_root")
os.makedirs(os.path.join(MODEL_DIR, "mano"), exist_ok=True)
for p in ["MANO_RIGHT.pkl", "MANO_LEFT.pkl"]:
    dst = os.path.join(MODEL_DIR, "mano", p)
    if not os.path.exists(dst):
        os.symlink(os.path.join(ASSETS, p), dst)


def run_mano(pose48, betas10, is_rhand=True):
    """pose48: [global(3), finger(45)] axis-angle, betas10: shape.
    Return joints (21,3) in world (mano default = local; we ignore global trans).
    """
    m = smplx.create(model_path=MODEL_DIR, model_type="mano",
                     is_rhand=is_rhand, use_pca=False,
                     flat_hand_mean=True, batch_size=1)
    out = m(global_orient=pose48[:, :3],
            hand_pose=pose48[:, 3:],
            betas=betas10,
            return_full_pose=True)
    return out.joints.detach().numpy().squeeze()  # (16,3) or (21,3) with tips


def urdf_fk_joint_positions(rest_joints, parent_dict, finger_angles, is_rhand=True):
    """
    rest_joints: dict name->(3,) at rest pose (β only, θ=0). Wrist at origin assumed.
    parent_dict: from get_kinematic_order_mano
    finger_angles: dict name->(θx, θy, θz) for each 15 finger joints
    Returns: dict name->global (3,) position after applying URDF chain.

    Wrist global rotation/translation = identity for this test.
    Per-joint: world_T_joint = world_T_parent · trans(rest_child - rest_parent) · Rx · Ry · Rz
    Child link origin in joint frame = 0, so joint position = world_T_joint[:3,3].
    """
    # Build kinematic chain order (root first)
    # parent_dict maps child->parent; wrist's parent is None
    order = []
    visited = set()
    def visit(name):
        if name in visited: return
        p = parent_dict[name]
        if p is not None and p not in visited:
            visit(p)
        visited.add(name); order.append(name)
    for n in parent_dict:
        visit(n)

    world_T = {}  # name -> 4x4
    for name in order:
        parent = parent_dict[name]
        if parent is None:
            # wrist: identity (rest_joints[wrist] could be nonzero in world; treat as origin)
            T = np.eye(4)
            T[:3, 3] = rest_joints[name]
            world_T[name] = T
        else:
            # translate by rest offset, then apply finger rotations
            offset = rest_joints[name] - rest_joints[parent]
            T_parent = world_T[parent]
            T_offset = np.eye(4); T_offset[:3, 3] = offset
            tx, ty, tz = finger_angles[name]
            Rmat = (R.from_euler('x', tx) * R.from_euler('y', ty) * R.from_euler('z', tz)).as_matrix()
            T_rot = np.eye(4); T_rot[:3, :3] = Rmat
            world_T[name] = T_parent @ T_offset @ T_rot

    return {n: T[:3, 3] for n, T in world_T.items()}


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    is_rhand = True

    # ---- choose a non-trivial pose ----
    pose48 = torch.zeros(1, 48)
    # finger rotations: 15 joints * 3 = 45
    pose48[0, 3:] = 0.3 * torch.randn(45)  # mild random pose
    betas = torch.zeros(1, 10)

    # ---- MANO reference ----
    joints_mano = run_mano(pose48, betas, is_rhand=is_rhand)  # (21,3) in MANO frame
    # MANO joint order: wrist + 15 finger joints + 5 tips (index_tip, middle_tip, pinky_tip, ring_tip, thumb_tip)

    # ---- rest pose for offsets (θ=0) ----
    rest = run_mano(torch.zeros(1, 48), betas, is_rhand=is_rhand)

    # Name->index mapping (smplx MANO order)
    # MANO 16 joints follow this order (matches get_kinematic_order_mano dict insertion):
    name_order = list(get_kinematic_order_mano(is_rhand).keys())
    # smplx MANO joints[0..15] = wrist + 15 fingers in SMPL-H order which matches name_order
    rest_dict = {n: rest[i] for i, n in enumerate(name_order)}
    parent_dict = get_kinematic_order_mano(is_rhand)

    # finger axis-angle: pose48[0,3:].reshape(15,3) in MANO order (right_index1..3, middle1..3, pinky1..3, ring1..3, thumb1..3)
    finger_aa = pose48[0, 3:].numpy().reshape(15, 3)
    finger_names = name_order[1:]  # skip wrist

    # ---- (A) Naive: θx,θy,θz = axis-angle components ----
    naive_angles = {n: tuple(finger_aa[i]) for i, n in enumerate(finger_names)}
    # ---- (B) Euler XYZ conversion ----
    rotvecs = R.from_rotvec(finger_aa)
    eulers = rotvecs.as_euler('XYZ', degrees=False)  # intrinsic = matches Rx·Ry·Rz chain
    euler_angles = {n: tuple(eulers[i]) for i, n in enumerate(finger_names)}

    pos_naive = urdf_fk_joint_positions(rest_dict, parent_dict, naive_angles, is_rhand)
    pos_euler = urdf_fk_joint_positions(rest_dict, parent_dict, euler_angles, is_rhand)

    # ---- compare to MANO ----
    print(f"{'joint':<18} {'MANO ref [mm]':<30} {'naive err [mm]':<18} {'euler err [mm]':<18}")
    print('-' * 90)
    total_err_n, total_err_e = 0.0, 0.0
    for i, n in enumerate(name_order):
        ref = joints_mano[i] * 1000
        en = np.linalg.norm((pos_naive[n] - joints_mano[i]) * 1000)
        ee = np.linalg.norm((pos_euler[n] - joints_mano[i]) * 1000)
        total_err_n += en; total_err_e += ee
        print(f"{n:<18} {np.round(ref,2)!s:<30} {en:<18.3f} {ee:<18.3f}")
    print('-' * 90)
    print(f"{'mean per-joint err':<18} {'':30} {total_err_n/len(name_order):<18.3f} {total_err_e/len(name_order):<18.3f}")


if __name__ == "__main__":
    main()
