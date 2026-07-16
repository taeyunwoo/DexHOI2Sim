"""Reproduce a DexYCB (MANO) hand sequence in MuJoCo using the MANO2URDF
**45-DOF** hand (the analytic, no-IK hand — this is the repo's PRIMARY hand,
not the 22-DOF ArtiMANO).

Pipeline (all analytic — no optimization):
    betas.yml            --(mano2urdf/generate_urdf.py)-->  hand.urdf + meshes
    seq (pose_m)         --(pose_to_joint_angles.py)---->  frames/*.json (51 qpos)
    hand.urdf            --(this file: inject <mujoco>, →MJCF)
    frames + object CAD  --(this file: set qpos/step)---->  kinematic | physics mp4

The 45 finger DOF map 1:1 by joint name to the frame json's `joint_angles`;
the 6 wrist DOF come from `wrist_xyz` (translate 0x/0y/0z) + `wrist_rpy`
(rotate 0rx/0ry/0rz).

Usage:
    MUJOCO_GL=egl python mano2urdf_mujoco.py \
        --urdf-dir <generate_urdf out>  --frames-dir <pose_to_joint_angles out/frames> \
        --mode kinematic  --out out.mp4
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
import re
import json
import glob
import argparse
from pathlib import Path

import xml.etree.ElementTree as ET

import numpy as np
import mujoco
import imageio.v2 as imageio


def mesh_collision_urdf(urdf_text: str) -> str:
    """Rewrite every link's <collision> to use the SAME MANO mesh (+origin) as
    its <visual>, replacing the box/capsule collision primitives. MuJoCo uses the
    mesh's convex hull for collision, so both visual and collision share the true
    MANO hand geometry per link."""
    root = ET.fromstring(urdf_text)
    for link in root.findall("link"):
        vis = link.find("visual")
        if vis is None:
            continue
        mesh = vis.find("geometry/mesh")
        if mesh is None:
            continue
        vorigin = vis.find("origin")
        for col in link.findall("collision"):   # drop primitive collisions
            link.remove(col)
        col = ET.SubElement(link, "collision")
        if vorigin is not None:
            o = ET.SubElement(col, "origin")
            o.set("rpy", vorigin.get("rpy", "0 0 0"))
            o.set("xyz", vorigin.get("xyz", "0 0 0"))
        geo = ET.SubElement(col, "geometry")
        cm = ET.SubElement(geo, "mesh")
        cm.set("filename", mesh.get("filename"))
        cm.set("scale", mesh.get("scale", "1 1 1"))
    return ET.tostring(root, encoding="unicode")


def widen_base_limits(urdf_text: str, lim: float = 3.0) -> str:
    """Widen the 6-DoF wrist prismatic joints' limits (default ±0.8 m in the
    generated URDF). In the tag/world frame the hand sits ~0.5–0.8 m from the
    origin, so ±0.8 m gets clamped by strict simulators (IsaacGym) → a global
    per-frame offset. Widen to ±`lim` m so the base can reach anywhere."""
    root = ET.fromstring(urdf_text)
    for j in root.findall("joint"):
        nm = j.get("name", "")
        if j.get("type") == "prismatic" and "wrist_0" in nm:
            lo = j.find("limit")
            if lo is not None:
                lo.set("lower", f"{-lim}")
                lo.set("upper", f"{lim}")
    return ET.tostring(root, encoding="unicode")

# inertia bounds so MuJoCo accepts the massless x→y→z joint-decomposition links.
# discardvisual="false" is REQUIRED: MANO2URDF puts the real MANO-partitioned
# hand meshes in <visual> and only simple primitives (box palm, capsule fingers)
# in <collision>. MuJoCo's URDF default discards visuals → the palm would render
# as a box. Keeping visuals lets us show the true hand mesh (collisions hidden).
MJ_COMPILER = ('<mujoco><compiler meshdir="meshes" balanceinertia="true" '
               'boundmass="0.001" boundinertia="1e-6" discardvisual="false"/></mujoco>')


def hide_collision_geoms(model):
    """Make the hand's collision meshes invisible so only the real MANO visual
    meshes render (each hand link has a visual mesh contype=0 + a duplicate
    collision mesh contype!=0). The floor plane and the object (name 'obj', which
    has no separate visual) stay visible."""
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if nm.startswith("obj"):                 # keep the CAD object visible
            continue
        if model.geom_contype[i] != 0 or model.geom_conaffinity[i] != 0:
            model.geom_rgba[i, 3] = 0.0

# scene wraps the hand via a top-level <include> (MuJoCo merges asset+worldbody)
SCENE_TPL = """<mujoco model="mano2urdf_scene">
  <option gravity="0 0 {gravity}" timestep="0.002" integrator="implicitfast"
          cone="elliptic" impratio="3"/>
  <default>
    <geom friction="1.5 0.05 0.001" solref="0.02 1" solimp="0.9 0.95 0.001"/>
  </default>
  <visual><global offwidth="640" offheight="480"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".7 .8 .9" rgb2=".2 .3 .4"
             width="256" height="256"/>
    <texture type="2d" name="grid" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .2 .3"
             width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.0"/>
{object_asset}
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse="1 1 1"/>
    <light pos="0.5 -0.5 1.5" dir="-0.3 0.3 -1" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="2 2 0.1" material="grid" pos="0 0 {floor_z}"/>
    <body name="lookat" pos="{lookat}"/>
    <camera name="view" mode="targetbody" target="lookat" pos="{cam_pos}"/>
{object_body}
  </worldbody>
{hand_includes}
{actuators}
</mujoco>"""


def make_actuators(joint_names, kp_wrist=400.0, kp_finger=6.0,
                   kv_wrist=25.0, kv_finger=0.6):
    """Position (PD) actuators for every hand joint. Wrist (0x/0y/0z/0r*) gets a
    stiff gain so the palm holds pose against gravity + object reaction; fingers
    get a softer gain so they can PD-track the reference grasp and exert grip."""
    rows = ["  <actuator>"]
    for jn in joint_names:
        is_wrist = "wrist_0" in jn
        kp = kp_wrist if is_wrist else kp_finger
        kv = kv_wrist if is_wrist else kv_finger
        rows.append(f'    <position name="a_{jn}" joint="{jn}" kp="{kp}" kv="{kv}"/>')
    rows.append("  </actuator>")
    return "\n".join(rows)


MJ_COMPILER_ABS = ('<mujoco><compiler strippath="false" balanceinertia="true" '
                   'boundmass="0.001" boundinertia="1e-6" discardvisual="false"/></mujoco>')


def build_hand_model(urdf_dir, gravity: float, floor_z: float,
                     cam_pos: str, lookat: str, object_body: str = "",
                     object_asset: str = "", actuated: bool = False):
    """Build a scene with ONE OR MORE MANO2URDF hands (bimanual) + floor/light/
    camera/object, and compile. `urdf_dir` may be a single dir or a list (one per
    hand, e.g. right + left — their joint/body names are side-prefixed so they
    don't collide). Mesh paths are made absolute so hands from different dirs all
    resolve. If actuated, add PD position actuators for every hand joint.
    Returns (model, scene_path)."""
    urdf_dirs = [urdf_dir] if not isinstance(urdf_dir, (list, tuple)) else list(urdf_dir)
    work = Path(urdf_dirs[0]).resolve()          # scene + _hand_i.xml live here
    includes, all_joints = [], []
    cwd = os.getcwd()
    os.chdir(work)
    try:
        for i, ud in enumerate(urdf_dirs):
            ud = Path(ud).resolve()
            # pick the generated hand URDF, NOT the intermediate files the backends
            # drop into this same dir (_isaac.urdf, _obj*.urdf, _scene.xml, ...) —
            # those sort before hand.urdf and would silently replace the hand.
            hand_urdf = next(f for f in sorted(ud.glob("*.urdf"))
                             if not f.name.startswith("_"))
            txt = mesh_collision_urdf(hand_urdf.read_text())
            txt = txt.replace('filename="./meshes/', f'filename="{ud}/meshes/')  # abs
            txt = re.sub(r"(<robot[^>]*>)", r"\1\n" + MJ_COMPILER_ABS, txt, count=1)
            hand = mujoco.MjModel.from_xml_string(txt)
            mujoco.mj_saveLastXML(f"_hand_{i}.xml", hand)
            includes.append(f'  <include file="_hand_{i}.xml"/>')
            all_joints += [mujoco.mj_id2name(hand, mujoco.mjtObj.mjOBJ_JOINT, j)
                           for j in range(hand.njnt)]
        actuators = make_actuators(all_joints) if actuated else ""
        scene = SCENE_TPL.format(gravity=gravity, floor_z=f"{floor_z:.3f}",
                                 cam_pos=cam_pos, lookat=lookat,
                                 object_body=object_body, object_asset=object_asset,
                                 actuators=actuators,
                                 hand_includes="\n".join(includes))
        Path("_scene.xml").write_text(scene)
        model = mujoco.MjModel.from_xml_path("_scene.xml")
    finally:
        os.chdir(cwd)
    return model, work / "_scene.xml"


# ---------- DexYCB object + master→tag (Z-up) frame ----------

def load_dexycb_object_tag(dexycb_root, subject, session):
    """Return (obj_name, obj_mesh_path, obj_pose_tag (F,7 [x,y,z, qw,qx,qy,qz]),
    tag_frame). Poses are transformed master-cam → AprilTag (Z-up) so the object
    rests on the table and the hand is upright (fixes the flipped render)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))   # dexycb_loader/world in sim/
    import yaml
    from dexycb_loader import load_sequence
    from dexycb_world import load_tag_frame
    from scipy.spatial.transform import Rotation as Rsc

    seq = load_sequence(dexycb_root, subject, session)
    meta = yaml.safe_load(open(Path(dexycb_root) / subject / session / "meta.yml"))
    tag = load_tag_frame(dexycb_root, meta["extrinsics"])

    pose = seq.grasp_object_pose()               # (F,7) [qx,qy,qz,qw, tx,ty,tz] master
    quat_xyzw = pose[:, :4]
    pos_master = pose[:, 4:7]
    pos_tag = pos_master @ tag.R_tag_master.T + tag.t_tag_master
    R_tag = np.einsum("ij,njk->nik", tag.R_tag_master,
                      Rsc.from_quat(quat_xyzw).as_matrix())
    quat_wxyz = Rsc.from_matrix(R_tag).as_quat()[:, [3, 0, 1, 2]]   # xyzw→wxyz
    obj_pose_tag = np.concatenate([pos_tag, quat_wxyz], axis=1).astype(np.float32)  # (F,7)

    obj_name = seq.object_name(seq.grasp_obj_col)
    mdir = Path(dexycb_root) / "models" / obj_name
    meshes = {"vis": mdir / "textured.obj",           # UV mesh for textured render
              "col": mdir / "textured_simple.obj",    # simpler convex collision
              "tex": mdir / "texture_map.png"}         # CAD texture
    return obj_name, meshes, obj_pose_tag, tag


def dexycb_camera_in_tag(dexycb_root, subject, session, tag, serial="932122062010"):
    """DexYCB reference camera pose expressed in the tag (Z-up) frame + its fovy,
    so the sim view matches the original DexYCB recording (same as render_de_compare).
    Returns (cam_pos_tag (3,), cam_target_tag (3,), fovy_deg)."""
    import yaml
    root = Path(dexycb_root)
    meta = yaml.safe_load(open(root / subject / session / "meta.yml"))
    ext = yaml.load(open(root / "calibration" / f"extrinsics_{meta['extrinsics']}"
                         / "extrinsics.yml"), Loader=yaml.FullLoader)
    Tmc = np.asarray(ext["extrinsics"][serial], dtype=np.float64).reshape(3, 4)
    R_mc, t_mc = Tmc[:, :3], Tmc[:, 3]
    cam_pos_tag = tag.R_tag_master @ t_mc + tag.t_tag_master
    R_tag_cam = tag.R_tag_master @ R_mc
    cam_target_tag = cam_pos_tag + R_tag_cam @ np.array([0., 0., 1.])   # +Z looks fwd
    intr = yaml.load(open(root / "calibration" / "intrinsics" / f"{serial}_640x480.yml"),
                     Loader=yaml.FullLoader)
    fy = float(intr["color"]["fy"])
    fovy = float(np.degrees(2 * np.arctan(480.0 / (2 * fy))))
    return cam_pos_tag.astype(np.float32), cam_target_tag.astype(np.float32), fovy


def transform_frames_to_tag(frames, tag):
    """Rewrite each frame's wrist_xyz/wrist_rpy from master-cam into the tag
    (Z-up) frame. Finger joint angles are local → unchanged."""
    from scipy.spatial.transform import Rotation as Rsc
    out = []
    for fr in frames:
        p = np.asarray(fr["wrist_xyz"], dtype=np.float64)
        p_tag = tag.R_tag_master @ p + tag.t_tag_master
        R_m = Rsc.from_euler("XYZ", fr["wrist_rpy"]).as_matrix()
        R_t = tag.R_tag_master @ R_m
        rpy_tag = Rsc.from_matrix(R_t).as_euler("XYZ")
        g = dict(fr)
        g["wrist_xyz"] = p_tag.tolist()
        g["wrist_rpy"] = rpy_tag.tolist()
        out.append(g)
    return out


def object_xml(meshes, mass=0.3, color=None, sfx=""):
    """(asset_str, body_str) for a free-floating object. If `meshes` has a "tex"
    (DexYCB), the visual is textured; otherwise (custom CAD) it's a solid `color`
    (RGB, default tan). Collision uses the (simpler) col mesh, invisible. `sfx`
    disambiguates names when several objects share one scene (obj_free{sfx}, …)."""
    if meshes.get("tex") and color is None:
        vis_asset = (f'    <texture type="2d" name="obj_tex{sfx}" file="{meshes["tex"]}"/>\n'
                     f'    <material name="obj_mat{sfx}" texture="obj_tex{sfx}" specular="0.2"'
                     f' shininess="0.3" reflectance="0.0"/>\n')
    else:
        c = color or (0.75, 0.72, 0.62)
        vis_asset = (f'    <material name="obj_mat{sfx}" rgba="{c[0]} {c[1]} {c[2]} 1"'
                     f' specular="0.2" shininess="0.3"/>\n')
    vis_geom = f'material="obj_mat{sfx}"'
    asset = (f'    <mesh name="obj_vis{sfx}" file="{meshes["vis"]}"/>\n'
             f'    <mesh name="obj_col{sfx}" file="{meshes["col"]}"/>\n'
             + vis_asset)
    body = (f'    <body name="object{sfx}" pos="0 0 0">\n'
            f'      <freejoint name="obj_free{sfx}"/>\n'
            f'      <geom name="obj_col{sfx}" type="mesh" mesh="obj_col{sfx}" mass="{mass}"'
            f' contype="1" conaffinity="1" group="3" rgba="1 1 1 0"/>\n'
            f'      <geom name="obj_vis{sfx}" type="mesh" mesh="obj_vis{sfx}" {vis_geom}'
            f' contype="0" conaffinity="0" group="1" mass="0"/>\n'
            f'    </body>\n')
    return asset, body


def estimate_table_z(mesh_path, obj_pose_tag):
    """Infer the desk/table-top height from the object at rest. The object never
    penetrates the table, so its lowest vertex touches the surface when resting on
    it (start/end of the sequence). Transform the object's lowest point into the
    world for every frame and take a low percentile of that height → the table top."""
    import trimesh
    from scipy.spatial.transform import Rotation as Rsc
    V = np.asarray(trimesh.load(mesh_path, process=False, force="mesh").vertices, float)
    pos = obj_pose_tag[:, :3]
    quat_xyzw = obj_pose_tag[:, [4, 5, 6, 3]]                # wxyz stored → xyzw
    step = max(1, len(obj_pose_tag) // 200)                 # sample ≤200 frames
    bottom = []
    for t in range(0, len(obj_pose_tag), step):
        R = Rsc.from_quat(quat_xyzw[t]).as_matrix()
        bottom.append((V @ R.T)[:, 2].min() + pos[t, 2])
    return float(np.percentile(bottom, 3))                  # resting height (robust)


def scene_camera(points, back=1.6, up_frac=0.55, zoom=1.5):
    """Frame the whole scene: look at the bbox center of `points` (N,3) and place
    the camera back (−Y) + up (+Z) so the full motion fits. `zoom` pulls the camera
    in (distance ÷ zoom) — zoom=1.5 gives a 1.5× closer, tighter view.
    Returns (cam_pos_str, lookat_str)."""
    lo, hi = points.min(0), points.max(0)
    ctr = (lo + hi) / 2.0
    diag = float(np.linalg.norm(hi - lo))
    dist = (diag * back + 0.3) / zoom
    cam = ctr + np.array([0.0, -dist, dist * up_frac])
    return (f"{cam[0]:.3f} {cam[1]:.3f} {cam[2]:.3f}",
            f"{ctr[0]:.3f} {ctr[1]:.3f} {ctr[2]:.3f}")


def load_frames(frames_dir: Path):
    files = sorted(glob.glob(str(frames_dir / "*.json")))
    frames = [json.load(open(f)) for f in files]
    return frames


def build_qpos_indexer(model):
    """Return function(frame_json, qpos) that writes ONE hand's DOFs into qpos.
    The side (left/right) is inferred from the frame's joint_angles prefix, so the
    generic wrist_xyz/wrist_rpy map to that hand's <side>_wrist_0* joints. For
    bimanual, call it once per hand's frame on the same qpos vector."""
    name2adr = {}
    for j in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        name2adr[nm] = model.jnt_qposadr[j]

    def to_qpos(fr, q):
        side = next(iter(fr["joint_angles"])).split("_")[0]   # 'left' or 'right'
        for ax, v in zip(["x", "y", "z"], fr["wrist_xyz"]):
            nm = f"{side}_wrist_0{ax}"
            if nm in name2adr: q[name2adr[nm]] = v
        for ax, v in zip(["rx", "ry", "rz"], fr["wrist_rpy"]):
            nm = f"{side}_wrist_0{ax}"
            if nm in name2adr: q[name2adr[nm]] = v
        for nm, v in fr["joint_angles"].items():
            if nm in name2adr: q[name2adr[nm]] = v
        return q
    return to_qpos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf-dir", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--urdf-dir2", default=None, help="second hand URDF (bimanual)")
    ap.add_argument("--frames-dir2", default=None, help="second hand frames (bimanual)")
    ap.add_argument("--mode", default="kinematic", choices=["kinematic", "physics"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    # DexYCB object + tag(Z-up) frame — if given, add the grasped object CAD and
    # transform the (master-cam) hand+object poses into the upright tag frame.
    ap.add_argument("--dexycb-root", default="/root/data/dexycb")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--session", default=None)
    # custom (non-DexYCB) object: a CAD mesh + per-frame object poses already in a
    # Z-up world frame. No texture → solid color. Frames are used as-is.
    ap.add_argument("--object-cad", default=None, help="object mesh (obj/stl/ply)")
    ap.add_argument("--object-poses", default=None,
                    help=".npy (T,7) [x,y,z, qw,qx,qy,qz], Z-up world frame")
    ap.add_argument("--object-color", default="0.75 0.72 0.62",
                    help="solid RGB for custom object")
    ap.add_argument("--objects-json", default=None,
                    help='JSON list [{"mesh","poses","color"}] to load MANY objects')
    args = ap.parse_args()

    urdf_dirs = [Path(args.urdf_dir).resolve()]
    hand_frames = [load_frames(Path(args.frames_dir))]           # list of frame-sets
    if args.urdf_dir2:                                           # bimanual second hand
        urdf_dirs.append(Path(args.urdf_dir2).resolve())
        hand_frames.append(load_frames(Path(args.frames_dir2)))
    work = Path(args.out).resolve().parent
    work.mkdir(parents=True, exist_ok=True)
    print(f"[seq] {len(hand_frames)} hand(s), {len(hand_frames[0])} frames")

    # --- assemble the object list (0..N). Each: {meshes, color, poses(T,7), sfx} ---
    objects = []
    dexycb_cam = None
    tag = None
    if args.objects_json:                                      # many objects (HO-Cap)
        specs = json.load(open(args.objects_json))
        for i, o in enumerate(specs):
            m = str(Path(o["mesh"]).resolve())
            col = tuple(float(x) for x in str(o.get("color", "0.75 0.72 0.62")).split())
            objects.append({"meshes": {"vis": m, "col": m}, "color": col,
                            "poses": np.load(o["poses"]).astype(np.float32), "sfx": str(i)})
        print(f"[obj] {len(objects)} objects (multi, solid color, world frame)")
    elif args.object_cad:                                      # single custom object
        m = str(Path(args.object_cad).resolve())
        col = tuple(float(x) for x in args.object_color.split())
        objects.append({"meshes": {"vis": m, "col": m}, "color": col,
                        "poses": np.load(args.object_poses).astype(np.float32), "sfx": ""})
        print(f"[obj] custom {Path(args.object_cad).name}  (solid color, world frame)")
    elif args.subject and args.session:                        # DexYCB (textured)
        obj_name, meshes, obj_pose_tag, tag = load_dexycb_object_tag(
            args.dexycb_root, args.subject, args.session)
        hand_frames = [transform_frames_to_tag(f, tag) for f in hand_frames]   # → Z-up
        objects.append({"meshes": meshes, "color": None, "poses": obj_pose_tag, "sfx": ""})
        dexycb_cam = dexycb_camera_in_tag(args.dexycb_root, args.subject, args.session, tag)
        print(f"[obj] {obj_name}  (textured, tag frame)   [cam] DexYCB extrinsics")

    xmls = [object_xml(o["meshes"], color=o["color"], sfx=o["sfx"]) for o in objects]
    object_asset = "".join(a for a, _ in xmls)
    object_body = "".join(b for _, b in xmls)

    wrists = np.array([f["wrist_xyz"] for fs in hand_frames for f in fs])
    if dexycb_cam is not None:
        # use the original DexYCB camera pose (tag frame); table plane at z=0
        cam_pos_t, cam_tgt_t, fovy = dexycb_cam
        cam_pos = f"{cam_pos_t[0]:.4f} {cam_pos_t[1]:.4f} {cam_pos_t[2]:.4f}"
        lookat = f"{cam_tgt_t[0]:.4f} {cam_tgt_t[1]:.4f} {cam_tgt_t[2]:.4f}"
        floor_z = 0.0                                          # AprilTag table plane
    else:
        # frame on the hands + the most-moving (manipulated) object so both hands
        # stay large/centred even when many idle objects are spread on the table
        pts = wrists
        if objects:
            mvo = max(objects, key=lambda o: float(
                np.linalg.norm(o["poses"][:, :3] - o["poses"][0, :3], axis=1).max()))
            pts = np.concatenate([wrists, mvo["poses"][:, :3]], axis=0)
        cam_pos, lookat = scene_camera(pts)
        if objects:                                            # desk = lowest object rest
            floor_z = min(estimate_table_z(o["meshes"]["vis"], o["poses"]) for o in objects)
            print(f"[desk] table top inferred at z={floor_z:.3f} m (object rest)")
        else:
            floor_z = float(wrists[:, 2].min() - 0.20)

    physics = args.mode == "physics"
    model, scene_path = build_hand_model(
        urdf_dirs,
        gravity=(-9.81 if physics else 0.0),
        floor_z=floor_z,
        cam_pos=cam_pos, lookat=lookat,
        object_body=object_body, object_asset=object_asset,
        actuated=physics,
    )
    if dexycb_cam is not None:                                 # match DexYCB fovy
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "view")
        model.cam_fovy[cid] = dexycb_cam[2]
    print(f"[hand] compiled  nq={model.nq}  nu={model.nu}  mode={args.mode}  -> {scene_path}")
    hide_collision_geoms(model)            # show real MANO meshes, hide box/capsule
    data = mujoco.MjData(model)
    to_qpos = build_qpos_indexer(model)
    base_q = data.qpos.copy()

    # qpos address of each object's free joint (obj_free{sfx})
    obj_qadr = []
    for o in objects:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj_free{o['sfx']}")
        obj_qadr.append(model.jnt_qposadr[jid])

    # per-frame reference full qpos — combine all hands into one qpos per frame
    T = min(len(fs) for fs in hand_frames)
    if objects:
        T = min(T, min(len(o["poses"]) for o in objects))
    ref_q = np.zeros((T, model.nq), dtype=np.float64)
    for t in range(T):
        q = base_q.copy()
        for fs in hand_frames:
            to_qpos(fs[t], q)                 # writes this hand's DOFs into q
        ref_q[t] = q
    for o, qadr in zip(objects, obj_qadr):    # each object's freejoint pose per frame
        ref_q[:, qadr:qadr + 7] = o["poses"][:T]

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    out_frames = []

    if not physics:
        # kinematic: set full state each frame, FK only
        for t in range(T):
            data.qpos[:] = ref_q[t]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="view")
            out_frames.append(renderer.render())
    else:
        # physics PD: actuators track the reference hand qpos; object obeys
        # gravity + contact. Map each actuator → the qpos adr it drives.
        act_qadr = []
        for a in range(model.nu):
            jid = model.actuator_trnid[a, 0]
            act_qadr.append(model.jnt_qposadr[jid])
        act_qadr = np.array(act_qadr)

        data.qpos[:] = ref_q[0]                # start posed at frame 0
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

        dt = model.opt.timestep
        dur = T / float(args.fps)
        n_steps = int(dur / dt)
        render_stride = max(1, int(round(1.0 / (args.fps * dt))))

        # settle the object onto the table (hand held at frame 0, NOT recorded) so
        # any initial ground/hand penetration jitter happens off-camera
        data.ctrl[:] = ref_q[0][act_qadr]
        for _ in range(40):
            mujoco.mj_step(model, data)
        for s in range(n_steps):
            tt = s * dt * args.fps             # continuous frame index
            i0 = min(int(tt), T - 1)
            i1 = min(i0 + 1, T - 1)
            a = tt - i0
            ref = (1 - a) * ref_q[i0] + a * ref_q[i1]
            data.ctrl[:] = ref[act_qadr]       # position targets for hand joints
            mujoco.mj_step(model, data)
            if s % render_stride == 0:
                renderer.update_scene(data, camera="view")
                out_frames.append(renderer.render())
    renderer.close()

    imageio.mimsave(args.out, out_frames, fps=args.fps, quality=8, macro_block_size=1)
    print(f"[done] {len(out_frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
