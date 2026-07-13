"""
DexYCB sequence → URDF joint angle frames.

Input:
  - sequence dir (containing pose.npz, meta.yml)
  - subject mano.yml (β)
  - generated URDF dir (for joint_names + gen_info)

Output:
  - {out}/frames/{idx:05d}.json with
      {wrist_xyz, wrist_rpy, joint_angles{...45 finger angles}}
  - {out}/poses_axisangle.npy   (T, 48) full-pose axis-angle (for downstream)
  - {out}/trans.npy             (T, 3)

DexYCB pose_m schema: (T,1,51) = [global_orient(3), hand_pose_PCA(45), trans(3)]
"""
import argparse, os, sys, json, yaml, pickle
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "mano2urdf"))

import smplx
from mano_helpers import get_kinematic_order_mano


def setup_smplx_root():
    MODEL_DIR = os.path.join(ROOT, "outputs", "_smplx_root")
    os.makedirs(os.path.join(MODEL_DIR, "mano"), exist_ok=True)
    for p in ["MANO_RIGHT.pkl", "MANO_LEFT.pkl"]:
        dst = os.path.join(MODEL_DIR, "mano", p)
        if not os.path.exists(dst):
            os.symlink(os.path.join(ROOT, "assets", p), dst)
    return MODEL_DIR


def mano_rest_wrist(betas, is_rhand, model_dir):
    """β-dependent rest position of the wrist joint (zero pose, zero trans).

    MANO places joint 0 at `rest_j_wrist + transl` (the global rotation is NOT
    applied to the wrist itself). The MANO2URDF hand's root/palm link sits at the
    origin, so the 6-DoF wrist must be translated by `trans + rest_j_wrist` to put
    the palm origin on the true wrist joint — otherwise the whole hand is offset
    by ~rest_j_wrist (≈9 cm) AND every finger rotation pivots about the wrong
    point. Returns (3,) numpy."""
    m = smplx.create(model_path=model_dir, model_type="mano", is_rhand=is_rhand,
                     use_pca=False, flat_hand_mean=True, batch_size=1)
    out = m(betas=torch.from_numpy(betas).unsqueeze(0))
    return out.joints.detach().cpu().numpy()[0, 0]   # joint 0 = wrist


def dexycb_pose_to_axis_angle(pose_m, betas, is_rhand, model_dir):
    """
    pose_m: (T, 51) numpy = [glo(3), pca(45), trans(3)]
    Returns:
      full_pose48 (T, 48) axis-angle, trans (T, 3)

    CRITICAL: DexYCB's PCA coefficients are expressed in the MANO model's own PCA
    basis (`hands_components` / `hands_mean` stored inside MANO_{RIGHT,LEFT}.pkl —
    the same basis the DexYCB toolkit / manopth use). smplx's `use_pca=True`
    expansion applies a DIFFERENT basis, so `smplx(...).full_pose` yields a
    TWISTED hand (~0.4 rad/joint, ~22 mm mesh error).

    We expand with the pkl's own basis using pure numpy — NO manopth dependency:
        full_hand = hands_mean + pca @ hands_components
    (Verified 0.0 rad identical to manopth; and once expanded, smplx's forward
    model reproduces the hand to 0.0 mm.) So the only runtime dep is smplx + the
    MANO pkl the user already downloads.
    """
    T = pose_m.shape[0]
    glo = pose_m[:, :3].astype(np.float32)
    pca = pose_m[:, 3:48].astype(np.float32)          # (T,45) PCA coefficients
    trans = pose_m[:, 48:51].astype(np.float32)

    pkl = os.path.join(ROOT, "assets",
                       "MANO_RIGHT.pkl" if is_rhand else "MANO_LEFT.pkl")
    with open(pkl, "rb") as f:
        mano = pickle.load(f, encoding="latin1")
    hands_components = np.asarray(mano["hands_components"], dtype=np.float32)  # (45,45)
    hands_mean = np.asarray(mano["hands_mean"], dtype=np.float32)              # (45,)

    full_hand = hands_mean[None, :] + pca @ hands_components      # (T,45) axis-angle
    full_pose = np.concatenate([glo, full_hand], axis=1).astype(np.float32)  # (T,48)
    return full_pose, trans, None
    full_pose = out.full_pose.detach().cpu().numpy()
    return full_pose, trans, out


def axis_angle_to_urdf_angles(pose48, is_rhand):
    """
    pose48: (T, 48)
    Returns:
      wrist_rpy (T, 3) — intrinsic XYZ euler of global_orient
      finger_angles dict: name -> (T, 3) where each row is (θx, θy, θz) intrinsic XYZ euler
    """
    global_aa = pose48[:, :3]              # (T,3)
    finger_aa = pose48[:, 3:].reshape(-1, 15, 3)  # (T,15,3)

    wrist_rpy = R.from_rotvec(global_aa).as_euler('XYZ', degrees=False)

    name_order = list(get_kinematic_order_mano(is_rhand).keys())
    finger_names = name_order[1:]  # 15 finger joints

    finger_angles = {}
    for i, n in enumerate(finger_names):
        finger_angles[n] = R.from_rotvec(finger_aa[:, i]).as_euler('XYZ', degrees=False)
    return wrist_rpy, finger_angles, finger_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=None, help="DexYCB seq dir (pose.npz, meta.yml)")
    ap.add_argument("--betas-yml", required=True)
    ap.add_argument("--urdf-dir", required=True, help="dir produced by generate_urdf.py")
    ap.add_argument("--out", required=True)
    # custom (non-DexYCB): full-axis-angle hand pose, no PCA/DexYCB parsing
    ap.add_argument("--poses-npy", default=None,
                    help="(T,48) full MANO axis-angle [global(3)+finger(45)]")
    ap.add_argument("--trans-npy", default=None, help="(T,3) wrist translation")
    ap.add_argument("--left", action="store_true", help="left hand (default right)")
    args = ap.parse_args()

    with open(args.betas_yml) as f:
        betas = np.asarray(yaml.safe_load(f)["betas"], dtype=np.float32)

    model_dir = setup_smplx_root()
    if args.poses_npy:                                   # custom: axis-angle given
        is_rhand = not args.left
        pose48 = np.load(args.poses_npy).astype(np.float32)          # (T,48)
        trans = np.load(args.trans_npy).astype(np.float32)           # (T,3)
        print(f"[custom] T={pose48.shape[0]}  rhand={is_rhand}")
    else:                                                # DexYCB
        with open(os.path.join(args.seq, "meta.yml")) as f:
            meta = yaml.safe_load(f)
        is_rhand = meta["mano_sides"][0] == "right"
        pose_m = np.load(os.path.join(args.seq, "pose.npz"))["pose_m"][:, 0, :]
        print(f"[seq] {os.path.basename(args.seq)}  T={pose_m.shape[0]}  rhand={is_rhand}")
        pose48, trans, _ = dexycb_pose_to_axis_angle(pose_m, betas, is_rhand, model_dir)
    wrist_rpy, finger_angles, finger_names = axis_angle_to_urdf_angles(pose48, is_rhand)

    # world wrist position = trans + rest_j_wrist (palm link origin sits at 0)
    rest_j_wrist = mano_rest_wrist(betas, is_rhand, model_dir)
    wrist_world = trans + rest_j_wrist[None, :]

    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
    for t in range(pose48.shape[0]):
        frame = {
            "wrist_xyz": wrist_world[t].tolist(),
            "wrist_rpy": wrist_rpy[t].tolist(),
            "joint_angles": {}
        }
        for n in finger_names:
            tx, ty, tz = finger_angles[n][t]
            frame["joint_angles"][f"{n}_x"] = float(tx)
            frame["joint_angles"][f"{n}_y"] = float(ty)
            frame["joint_angles"][f"{n}_z"] = float(tz)
        with open(os.path.join(args.out, "frames", f"{t:05d}.json"), "w") as f:
            json.dump(frame, f, indent=2)

    np.save(os.path.join(args.out, "poses_axisangle.npy"), pose48)
    np.save(os.path.join(args.out, "trans.npy"), trans)
    print(f"[seq] saved {pose48.shape[0]} frames -> {args.out}")


if __name__ == "__main__":
    main()
