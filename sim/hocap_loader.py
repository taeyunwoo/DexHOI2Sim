"""Load an HO-Cap (irvlutd.github.io/HOCap) session into the DexHOI2Sim pipeline.

HO-Cap format (per session <root>/<subject>/<session>/):
  meta.yaml    : num_frames, mano_sides (['right'] / ['left'] / ['right','left']),
                 object_ids (e.g. ['G10_1',...]), task_id (1=pick&place, 2=handover,
                 3=affordance use), subject_id
  poses_m.npy  : (2, T, 51)  hand poses; index 0=right, 1=left; per hand
                 [global_orient(3), hand_pose_PCA(45), translation(3)]  (manopth PCA)
  poses_o.npy  : (N_obj, T, 7)  object poses  [qx,qy,qz,qw, tx,ty,tz]  (xyzw quat)
  <root>/calibration/mano/<subject>.yaml : per-subject MANO betas
  <root>/models/<id>/textured_mesh.obj   : object CAD (textured)

Frame is Z-up world (objects lift in +z). MANO config matches DexYCB
(manopth use_pca=True, ncomps=45, flat_hand_mean=False) so the PCA basis is read
straight from the MANO .pkl — no manopth dependency.
"""
import pickle
from pathlib import Path

import numpy as np
import yaml

_ASSETS = Path(__file__).resolve().parent.parent / "mano2urdf" / "assets"


def _expand_pca(pca, side):
    """PCA coeffs (T,45) -> full finger axis-angle (T,45), via the MANO pkl basis."""
    pkl = _ASSETS / ("MANO_RIGHT.pkl" if side == "right" else "MANO_LEFT.pkl")
    with open(pkl, "rb") as f:
        m = pickle.load(f, encoding="latin1")
    comps = np.asarray(m["hands_components"], np.float32)   # (45,45)
    mean = np.asarray(m["hands_mean"], np.float32)          # (45,)
    return mean[None, :] + pca.astype(np.float32) @ comps


def load_hocap_session(hocap_root, subject, session):
    """Return dict:
        hands   : [{side, betas(10), pose48(T,48 axis-angle), trans(T,3)}]  (active hands)
        objects : [{id, mesh(path), poses(T,7 [x,y,z,qw,qx,qy,qz])}]        (Z-up world)
        task_id, task_name, num_frames
    """
    root = Path(hocap_root)
    sdir = root / subject / session
    meta = yaml.safe_load(open(sdir / "meta.yaml"))
    betas = np.asarray(yaml.safe_load(
        open(root / "calibration" / "mano" / f"{subject}.yaml"))["betas"], np.float32)
    poses_m = np.load(sdir / "poses_m.npy").astype(np.float32)   # (2,T,51)
    poses_o = np.load(sdir / "poses_o.npy").astype(np.float32)   # (N,T,7) xyzw+trans

    hands = []
    for side in meta["mano_sides"]:
        pm = poses_m[0 if side == "right" else 1]               # (T,51)
        glo, pca, trans = pm[:, :3], pm[:, 3:48], pm[:, 48:51]
        pose48 = np.concatenate([glo, _expand_pca(pca, side)], axis=1)
        hands.append({"side": side, "betas": betas,
                      "pose48": pose48.astype(np.float32), "trans": trans})

    objects = []
    for i, oid in enumerate(meta["object_ids"]):
        po = poses_o[i]                                         # (T,7) xyzw+trans
        pos = po[:, 4:7]
        quat_wxyz = po[:, [3, 0, 1, 2]]                         # xyzw -> wxyz (MuJoCo)
        mesh = root / "models" / oid / "textured_mesh.obj"
        objects.append({"id": oid, "mesh": str(mesh),
                        "poses": np.concatenate([pos, quat_wxyz], axis=1).astype(np.float32)})

    task_names = {1: "pick_and_place", 2: "handover", 3: "affordance_use"}
    return {"hands": hands, "objects": objects,
            "task_id": meta["task_id"],
            "task_name": task_names.get(meta["task_id"], str(meta["task_id"])),
            "num_frames": int(meta["num_frames"]), "subject": subject, "session": session}
