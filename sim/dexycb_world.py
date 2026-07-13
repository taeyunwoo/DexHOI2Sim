"""
DexYCB master-camera frame → AprilTag (world, Z-up) frame conversions.

DexYCB stores all poses (pose_y for objects, pose_m for MANO) in the *master*
camera's frame — the master camera is the one whose `extrinsics` entry is the
identity matrix; its serial is in `extrinsics.yml`'s top-level `master:` key.
That frame has +Z pointing into the scene (camera-forward), +Y down — not
useful for an IsaacGym Z-up simulation.

The `apriltag:` entry in `extrinsics.yml` is the 3×4 matrix `T_master_tag`
giving the AprilTag origin's pose in the master frame. The tag sits flat on
the table, so its frame is gravity-aligned (Z up). We invert to get
`T_tag_master` and apply it to every pose.

This module is a focused 80-line helper — no IsaacGym, no smplx dependencies.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True)
class TagFrame:
    R_tag_master: np.ndarray   # (3, 3)
    t_tag_master: np.ndarray   # (3,)
    master_serial: str

    def transform_points(self, p_master: np.ndarray) -> np.ndarray:
        """p_master: (..., 3) master-frame points → tag-frame points."""
        return p_master @ self.R_tag_master.T + self.t_tag_master

    def transform_quat_xyzw(self, q_master_xyzw: np.ndarray) -> np.ndarray:
        """q_master: (..., 4) xyzw quaternions in master frame → tag frame."""
        shape = q_master_xyzw.shape
        flat = q_master_xyzw.reshape(-1, 4)
        Rm = R.from_quat(flat).as_matrix()                     # (N, 3, 3)
        # einsum to broadcast left-multiply by R_tag_master
        Rt = np.einsum("ij,njk->nik", self.R_tag_master, Rm)
        out = R.from_matrix(Rt).as_quat()                      # (N, 4) xyzw
        return out.reshape(shape)

    def transform_axisangle(self, aa_master: np.ndarray) -> np.ndarray:
        """aa_master: (..., 3) axis-angle in master frame → tag-frame axis-angle."""
        Rm = R.from_rotvec(aa_master).as_matrix()
        Rt = self.R_tag_master @ Rm
        return R.from_matrix(Rt).as_rotvec()


def load_tag_frame(dexycb_root: Path | str, extrinsics_tag: str) -> TagFrame:
    """Read calibration/extrinsics_<tag>/extrinsics.yml and build TagFrame."""
    path = Path(dexycb_root) / "calibration" / f"extrinsics_{extrinsics_tag}" / "extrinsics.yml"
    with open(path) as f:
        ext = yaml.load(f, Loader=yaml.FullLoader)
    # `apriltag` lives inside the `extrinsics:` block, not at top level.
    tag = np.asarray(ext["extrinsics"]["apriltag"], dtype=np.float64).reshape(3, 4)
    R_master_tag = tag[:, :3]
    t_master_tag = tag[:, 3]
    R_tag_master = R_master_tag.T                              # orthonormal inverse
    t_tag_master = -R_tag_master @ t_master_tag
    return TagFrame(R_tag_master=R_tag_master.astype(np.float32),
                    t_tag_master=t_tag_master.astype(np.float32),
                    master_serial=str(ext["master"]))


def transform_pose_m(pose_m: np.ndarray, frame: TagFrame) -> np.ndarray:
    """pose_m: (T, 51) = [global_orient(3), hand_pose_PCA(45), trans(3)].

    Transforms global_orient + trans into tag frame. PCA finger pose is local
    (relative to parent joint) so it is invariant under world-frame change."""
    out = pose_m.copy()
    out[:, 0:3] = frame.transform_axisangle(pose_m[:, 0:3])
    out[:, 48:51] = frame.transform_points(pose_m[:, 48:51])
    return out


def transform_pose_y(pose_y: np.ndarray, frame: TagFrame) -> np.ndarray:
    """pose_y: (T, N_obj, 7) = [qx, qy, qz, qw, tx, ty, tz] per object."""
    out = pose_y.copy()
    out[..., 0:4] = frame.transform_quat_xyzw(pose_y[..., 0:4])
    out[..., 4:7] = frame.transform_points(pose_y[..., 4:7])
    return out
