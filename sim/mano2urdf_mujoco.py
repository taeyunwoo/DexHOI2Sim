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
  <include file="_hand.xml"/>
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


def build_hand_model(urdf_dir: Path, gravity: float, floor_z: float,
                     cam_pos: str, lookat: str, object_body: str = "",
                     object_asset: str = "", actuated: bool = False):
    """Convert the MANO2URDF hand URDF to MJCF, wrap it in a scene (floor/light/
    camera/object) via a top-level <include>, and compile. The scene + _hand.xml
    are written INTO urdf_dir so the meshdir="meshes" reference resolves.
    If actuated, add PD position actuators for the hand joints (physics mode).
    Returns (model, scene_path)."""
    urdf_path = next(urdf_dir.glob("*.urdf"))
    txt = urdf_path.read_text()
    txt = mesh_collision_urdf(txt)          # collision meshes = visual MANO meshes
    txt = re.sub(r"(<robot[^>]*>)", r"\1\n" + MJ_COMPILER, txt, count=1)
    cwd = os.getcwd()
    os.chdir(urdf_dir)
    try:
        hand = mujoco.MjModel.from_xml_string(txt)   # hand-only: get joint names
        mujoco.mj_saveLastXML("_hand.xml", hand)
        actuators = ""
        if actuated:
            jnames = [mujoco.mj_id2name(hand, mujoco.mjtObj.mjOBJ_JOINT, j)
                      for j in range(hand.njnt)]
            actuators = make_actuators(jnames)
        scene = SCENE_TPL.format(gravity=gravity, floor_z=f"{floor_z:.3f}",
                                 cam_pos=cam_pos, lookat=lookat,
                                 object_body=object_body, object_asset=object_asset,
                                 actuators=actuators)
        Path("_scene.xml").write_text(scene)
        model = mujoco.MjModel.from_xml_path("_scene.xml")
    finally:
        os.chdir(cwd)
    return model, urdf_dir / "_scene.xml"


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


def object_xml(meshes, mass=0.3, color=None):
    """(asset_str, body_str) for a free-floating object. If `meshes` has a "tex"
    (DexYCB), the visual is textured; otherwise (custom CAD) it's a solid `color`
    (RGB, default tan). Collision uses the (simpler) col mesh, invisible."""
    if meshes.get("tex") and color is None:
        vis_asset = (f'    <texture type="2d" name="obj_tex" file="{meshes["tex"]}"/>\n'
                     f'    <material name="obj_mat" texture="obj_tex" specular="0.2"'
                     f' shininess="0.3" reflectance="0.0"/>\n')
        vis_geom = ('material="obj_mat"')
    else:
        c = color or (0.75, 0.72, 0.62)
        vis_asset = (f'    <material name="obj_mat" rgba="{c[0]} {c[1]} {c[2]} 1"'
                     f' specular="0.2" shininess="0.3"/>\n')
        vis_geom = ('material="obj_mat"')
    asset = (f'    <mesh name="obj_vis" file="{meshes["vis"]}"/>\n'
             f'    <mesh name="obj_col" file="{meshes["col"]}"/>\n'
             + vis_asset)
    body = (f'    <body name="object" pos="0 0 0">\n'
            f'      <freejoint name="obj_free"/>\n'
            f'      <geom name="obj_col" type="mesh" mesh="obj_col" mass="{mass}"'
            f' contype="1" conaffinity="1" group="3" rgba="1 1 1 0"/>\n'
            f'      <geom name="obj_vis" type="mesh" mesh="obj_vis" {vis_geom}'
            f' contype="0" conaffinity="0" group="1" mass="0"/>\n'
            f'    </body>\n')
    return asset, body


def load_frames(frames_dir: Path):
    files = sorted(glob.glob(str(frames_dir / "*.json")))
    frames = [json.load(open(f)) for f in files]
    return frames


def build_qpos_indexer(model):
    """Return function frame_json -> full qpos vector (nq)."""
    name2adr = {}
    for j in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        name2adr[nm] = model.jnt_qposadr[j]

    def to_qpos(fr, base_qpos):
        q = base_qpos.copy()
        wx = fr["wrist_xyz"]; wr = fr["wrist_rpy"]
        for nm, v in zip(["right_wrist_0x", "right_wrist_0y", "right_wrist_0z"], wx):
            if nm in name2adr: q[name2adr[nm]] = v
        for nm, v in zip(["right_wrist_0rx", "right_wrist_0ry", "right_wrist_0rz"], wr):
            if nm in name2adr: q[name2adr[nm]] = v
        for nm, v in fr["joint_angles"].items():
            if nm in name2adr: q[name2adr[nm]] = v
        return q
    return to_qpos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf-dir", required=True)
    ap.add_argument("--frames-dir", required=True)
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
    args = ap.parse_args()

    urdf_dir = Path(args.urdf_dir).resolve()
    work = Path(args.out).resolve().parent
    work.mkdir(parents=True, exist_ok=True)

    frames = load_frames(Path(args.frames_dir))
    print(f"[seq] {len(frames)} frames")

    object_asset = object_body = ""
    obj_pose_tag = None
    dexycb_cam = None
    tag = None
    if args.object_cad:                                        # custom (non-DexYCB)
        obj_pose_tag = np.load(args.object_poses).astype(np.float32)   # (T,7) Z-up world
        meshes = {"vis": str(Path(args.object_cad).resolve()),
                  "col": str(Path(args.object_cad).resolve())}          # same mesh
        color = tuple(float(x) for x in args.object_color.split())
        object_asset, object_body = object_xml(meshes, color=color)
        print(f"[obj] custom {Path(args.object_cad).name}  (solid color, world frame)")
    elif args.subject and args.session:
        obj_name, meshes, obj_pose_tag, tag = load_dexycb_object_tag(
            args.dexycb_root, args.subject, args.session)
        frames = transform_frames_to_tag(frames, tag)          # fix flip → Z-up
        object_asset, object_body = object_xml(meshes)
        dexycb_cam = dexycb_camera_in_tag(args.dexycb_root, args.subject, args.session, tag)
        print(f"[obj] {obj_name}  (textured, tag frame)   [cam] DexYCB extrinsics")

    wrists = np.array([f["wrist_xyz"] for f in frames])
    if dexycb_cam is not None:
        # use the original DexYCB camera pose (tag frame); table plane at z=0
        cam_pos_t, cam_tgt_t, fovy = dexycb_cam
        cam_pos = f"{cam_pos_t[0]:.4f} {cam_pos_t[1]:.4f} {cam_pos_t[2]:.4f}"
        lookat = f"{cam_tgt_t[0]:.4f} {cam_tgt_t[1]:.4f} {cam_tgt_t[2]:.4f}"
        floor_z = 0.0                                          # AprilTag table plane
    else:
        ctr = wrists.mean(axis=0)
        lookat = f"{ctr[0]:.3f} {ctr[1]:.3f} {ctr[2]:.3f}"
        cam_pos = f"{ctr[0]:.3f} {ctr[1]-0.55:.3f} {ctr[2]+0.25:.3f}"
        floor_z = float(wrists[:, 2].min() - 0.20)

    physics = args.mode == "physics"
    model, scene_path = build_hand_model(
        urdf_dir,
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

    obj_qadr = None
    if obj_pose_tag is not None:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj_free")
        obj_qadr = model.jnt_qposadr[jid]

    # per-frame reference full qpos
    ref_q = np.stack([to_qpos(fr, base_q) for fr in frames])       # (T, nq)
    if obj_qadr is not None:
        ref_q[:, obj_qadr:obj_qadr + 7] = obj_pose_tag

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    out_frames = []

    if not physics:
        # kinematic: set full state each frame, FK only
        for t in range(len(frames)):
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
        T = len(frames)
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
