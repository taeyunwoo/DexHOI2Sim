"""
DexYCB sequence loader for simulation replay.

Conventions (from dex-ycb-toolkit):
- pose_m[f, 0, :]  = [axis_angle_root(3), pose(45), trans(3)]   # MANO, axis-angle
- pose_y[f, i, :]  = [qx, qy, qz, qw, tx, ty, tz]               # object 6D, xyzw quat
- ycb_ids are 1-indexed YCB class ids; map to model folders via _YCB_CLASSES.
- 30 FPS. A frame with all-zero pose_m or pose_y row means "no annotation that frame".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

_YCB_CLASSES = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}

FPS = 30


@dataclass
class DexYCBSequence:
    """One DexYCB capture sequence (one (subject, session) pair)."""

    root: Path                  # dataset root (.../dexycb)
    subject: str                # e.g. "20200709-subject-01"
    session: str                # e.g. "20200709_141754"

    # Resolved from meta.yml / pose.npz / calibration:
    num_frames: int
    ycb_ids: list[int]          # 1-indexed YCB class ids, in pose_y column order
    grasp_obj_id: int           # the ycb_id of the grasped object
    grasp_obj_col: int          # column index in pose_y for the grasped object
    mano_side: str              # "right" | "left"
    mano_betas: np.ndarray      # (10,) subject-specific shape

    pose_m: np.ndarray          # (F, 51): [root_aa(3), pose_aa(45), trans(3)]
    pose_y: np.ndarray          # (F, N_obj, 7): [qx, qy, qz, qw, tx, ty, tz]

    # --- derived views ---
    @property
    def mano_global_orient(self) -> np.ndarray:  # (F, 3)
        return self.pose_m[:, 0:3]

    @property
    def mano_pose_pca(self) -> np.ndarray:       # (F, 45) — DexYCB stores PCA coeffs, NOT axis-angle
        return self.pose_m[:, 3:48]

    @property
    def mano_trans(self) -> np.ndarray:          # (F, 3)
        return self.pose_m[:, 48:51]

    def annotated_frames(self) -> np.ndarray:
        """Bool mask: True where MANO annotation is non-zero."""
        return np.any(self.pose_m != 0, axis=-1)

    def object_pose(self, obj_col: int) -> np.ndarray:  # (F, 7) xyzw quat + xyz
        return self.pose_y[:, obj_col, :]

    def grasp_object_pose(self) -> np.ndarray:
        return self.object_pose(self.grasp_obj_col)

    def object_name(self, obj_col: int) -> str:
        return _YCB_CLASSES[self.ycb_ids[obj_col]]

    def object_mesh_dir(self, obj_col: int) -> Path:
        return self.root / "models" / self.object_name(obj_col)

    def object_mesh_path(self, obj_col: int, kind: str = "textured_simple.obj") -> Path:
        return self.object_mesh_dir(obj_col) / kind


def list_subjects(root: Path | str) -> list[str]:
    root = Path(root)
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and d.name.startswith("20") and "-subject-" in d.name
    )


def list_sessions(root: Path | str, subject: str) -> list[str]:
    root = Path(root)
    return sorted(d.name for d in (root / subject).iterdir() if d.is_dir())


def list_all_sequences(root: Path | str) -> list[tuple[str, str]]:
    """Return [(subject, session), ...] for every annotated sequence under root."""
    out = []
    for subj in list_subjects(root):
        for sess in list_sessions(root, subj):
            if (Path(root) / subj / sess / "pose.npz").exists():
                out.append((subj, sess))
    return out


def _load_betas(root: Path, mano_calib_name: str) -> np.ndarray:
    # meta.yml stores the bare tag; the actual dir is prefixed with "mano_".
    calib_dir = root / "calibration" / f"mano_{mano_calib_name}"
    with open(calib_dir / "mano.yml") as f:
        d = yaml.safe_load(f)
    return np.asarray(d["betas"], dtype=np.float32)


def load_sequence(root: Path | str, subject: str, session: str) -> DexYCBSequence:
    root = Path(root)
    seq_dir = root / subject / session

    with open(seq_dir / "meta.yml") as f:
        meta = yaml.safe_load(f)

    pose = np.load(seq_dir / "pose.npz")
    pose_m = pose["pose_m"].squeeze(1).astype(np.float32)   # (F, 51)
    pose_y = pose["pose_y"].astype(np.float32)              # (F, N_obj, 7)

    grasp_col = int(meta["ycb_grasp_ind"])
    ycb_ids = [int(i) for i in meta["ycb_ids"]]

    return DexYCBSequence(
        root=root,
        subject=subject,
        session=session,
        num_frames=int(meta["num_frames"]),
        ycb_ids=ycb_ids,
        grasp_obj_id=ycb_ids[grasp_col],
        grasp_obj_col=grasp_col,
        mano_side=str(meta["mano_sides"][0]),
        mano_betas=_load_betas(root, meta["mano_calib"][0]),
        pose_m=pose_m,
        pose_y=pose_y,
    )


def _summary(seq: DexYCBSequence) -> str:
    mask = seq.annotated_frames()
    return (
        f"[{seq.subject}/{seq.session}] frames={seq.num_frames} "
        f"annotated={int(mask.sum())}/{seq.num_frames} "
        f"side={seq.mano_side} "
        f"objects={[_YCB_CLASSES[i] for i in seq.ycb_ids]} "
        f"grasp={seq.object_name(seq.grasp_obj_col)} "
        f"betas[:3]={seq.mano_betas[:3].tolist()}"
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/root/data/dexycb")
    p.add_argument("--subject", default=None)
    p.add_argument("--session", default=None)
    p.add_argument("--n", type=int, default=3, help="how many sequences to summarise")
    args = p.parse_args()

    root = Path(args.root)
    if args.subject and args.session:
        seq = load_sequence(root, args.subject, args.session)
        print(_summary(seq))
        print("\nFirst annotated frame, MANO global_orient / trans:")
        f0 = int(np.argmax(seq.annotated_frames()))
        print(f"  frame {f0}  orient={seq.mano_global_orient[f0]}  trans={seq.mano_trans[f0]}")
        print(f"  grasp obj pose (xyzw+xyz): {seq.grasp_object_pose()[f0]}")
    else:
        seqs = list_all_sequences(root)
        print(f"Found {len(seqs)} sequences under {root}\n")
        for s in seqs[: args.n]:
            print(_summary(load_sequence(root, *s)))
