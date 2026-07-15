"""Build a synthetic, license-free **two-hand** example bundle
(examples/sample_twohands.npz).

Two zero-shape MANO hands descend onto a small box from opposite sides and curl
their fingers — a minimal bimanual grasp. Purely procedural (no dataset assets),
just to show the two-hand bundle format and let `replicate.py --bundle` run a
bimanual scene out of the box.

Two-hand bundle keys:
  hands = [ {betas(10,), hand_pose(T,48), trans(T,3), side}, ... ]   # list of 2
  object_verts(V,3)+object_faces(F,3), object_poses(T,7 [x,y,z,qw,qx,qy,qz]),
  object_color(3,)
(A single `hands` list also works for one hand; the flat betas/hand_pose/trans/side
keys remain the one-hand form used by examples/sample.npz.)
"""
import json
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

# --- both hands: palm down (R_x 90°), descend z 0.22 -> 0.085, fingers curl ---
# MANO's right rest fingers point -X (wrist ends ~+0.096 X of trans); the left hand
# is the mirror (fingers +X, wrist ~-0.096 X). With trans x=0 the two hands sit on
# opposite sides of the origin box and reach across its top toward each other.
betas = np.zeros(10, np.float32)
global_orient = np.tile(R.from_euler("X", np.pi/2).as_rotvec(), (T, 1))
s = np.linspace(0, 1, T)[:, None]
trans = (1 - s) * np.array([0.0, 0.0, 0.22]) + s * np.array([0.0, 0.0, 0.085])
curl = np.linspace(0.0, 1.1, T)
finger = np.zeros((T, 15, 3), np.float32)
finger[:, :, 2] = curl[:, None]
hand_pose = np.concatenate([global_orient, finger.reshape(T, 45)], axis=1).astype(np.float32)

hand = dict(betas=betas, hand_pose=hand_pose, trans=trans.astype(np.float32))
hands = [dict(side="right", **hand), dict(side="left", **hand)]

bundle = dict(hands=hands, object_verts=v, object_faces=F,
              object_poses=object_poses,
              object_color=np.array([0.2, 0.5, 0.85], np.float32))
# npz can't hold a list-of-dicts directly → store the hands list as an object array;
# load_bundle reads d["hands"] back as a list.
np.savez("examples/sample_twohands.npz",
         hands=np.array(hands, dtype=object), object_verts=v, object_faces=F,
         object_poses=object_poses, object_color=bundle["object_color"])

json.dump({"hands": [{k: (val.tolist() if isinstance(val, np.ndarray) else val)
                      for k, val in h.items()} for h in hands],
           "object_verts": v.tolist(), "object_faces": F.tolist(),
           "object_poses": object_poses.tolist(),
           "object_color": bundle["object_color"].tolist()},
          open("examples/sample_twohands.json", "w"), indent=1)
print("wrote examples/sample_twohands.npz + .json  (T=%d, 2 hands, synthetic)" % T)
