"""Build a synthetic, license-free example HOI bundle (examples/sample.npz).

A zero-shape MANO right hand descends onto a small box and curls its fingers,
while the box sits on the table. Purely procedural — no dataset assets — just to
show the bundle format and let `replicate.py --bundle` run out of the box.

Bundle keys:
  betas(10,), hand_pose(T,48 axis-angle), trans(T,3), side,
  object_verts(V,3)+object_faces(F,3), object_poses(T,7 [x,y,z,qw,qx,qy,qz]),
  object_color(3,)
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

T = 40

# --- object: an 8 cm box centered at the origin, resting on the table (z=0) ---
hx, hy, hz = 0.04, 0.04, 0.05
v = np.array([[sx*hx, sy*hy, sz*hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
             dtype=np.float32)
v[:, 2] += hz                                   # bottom on the table
F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
              [2, 3, 7], [2, 7, 6], [1, 5, 7], [1, 7, 3], [0, 2, 6], [0, 6, 4]])
object_poses = np.tile([0, 0, 0, 1, 0, 0, 0], (T, 1)).astype(np.float32)  # static

# --- hand: descend onto the box, palm down, fingers curl 0 -> ~1 rad ---
# MANO rest right hand: fingers point -X, palm faces -Y. R_x(+90deg) turns the
# palm to -Z (down) while fingers keep pointing -X, so the hand descends onto the
# box from above with fingers reaching across its top.
betas = np.zeros(10, np.float32)
global_orient = np.tile(R.from_euler("X", np.pi/2).as_rotvec(), (T, 1))
s = np.linspace(0, 1, T)[:, None]
# wrist = trans + rest_j_wrist(~[0.096,0,0]); box spans x[-0.04,0.04] at origin, so
# put the wrist just +X of the box and descend onto it.
trans = (1 - s) * np.array([0.0, 0.0, 0.22]) + s * np.array([0.0, 0.0, 0.085])

curl = np.linspace(0.0, 1.1, T)                  # flexion ramp (rot about local z)
finger = np.zeros((T, 15, 3), np.float32)
finger[:, :, 2] = curl[:, None]                  # curl every finger joint
hand_pose = np.concatenate([global_orient, finger.reshape(T, 45)], axis=1).astype(np.float32)

bundle = dict(betas=betas, hand_pose=hand_pose, trans=trans.astype(np.float32),
              side="right", object_verts=v, object_faces=F,
              object_poses=object_poses,
              object_color=np.array([0.2, 0.5, 0.85], np.float32))
np.savez("examples/sample.npz", **bundle)

# also a human-readable JSON copy (viewable on GitHub without downloading)
import json
json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
           for k, v in bundle.items()}, open("examples/sample.json", "w"), indent=1)
print("wrote examples/sample.npz + examples/sample.json  (T=%d frames, synthetic)" % T)
