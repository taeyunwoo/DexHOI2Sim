"""Replicate the MANO2URDF 45-DOF hand (+object) in **IsaacGym**, and validate
that the Isaac-posed hand matches MANO ground truth (independent of the MuJoCo
backend). Same URDF, same per-frame qpos as the MuJoCo path.

isaacgym MUST be imported before torch. Run with:
    LD_LIBRARY_PATH=/root/miniconda3/envs/isaacgym/lib \
    /root/miniconda3/envs/isaacgym/bin/python mano2urdf_isaac.py \
        --urdf-dir <gen> --frames-dir <frames> \
        --subject 20200709-subject-01 --session 20200709_141754 \
        --out out_isaac.mp4 [--validate]
"""
from isaacgym import gymapi, gymtorch
import os
import sys
import json
import glob
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

# reuse the frame / tag / object helpers from the MuJoCo backend
from mano2urdf_mujoco import (load_frames, load_dexycb_object_tag,
                              transform_frames_to_tag, dexycb_camera_in_tag,
                              mesh_collision_urdf, widen_base_limits,
                              estimate_table_z, scene_camera)

# MANO2URDF hand body → smplx MANO joint index (palm = right_wrist_rz)
LINK_TO_MANO = {
    "right_wrist_rz": 0,
    "right_index1": 1, "right_index2": 2, "right_index3": 3,
    "right_middle1": 4, "right_middle2": 5, "right_middle3": 6,
    "right_pinky1": 7, "right_pinky2": 8, "right_pinky3": 9,
    "right_ring1": 10, "right_ring2": 11, "right_ring3": 12,
    "right_thumb1": 13, "right_thumb2": 14, "right_thumb3": 15,
}


def frames_to_dof_traj(frames, dof_names):
    """(T, ndof) in the asset DOF order. Side (left/right) is inferred from the
    frame's joint_angles prefix so wrist_xyz/wrist_rpy map to <side>_wrist_0*."""
    T = len(frames)
    traj = np.zeros((T, len(dof_names)), dtype=np.float32)
    for t, fr in enumerate(frames):
        side = next(iter(fr["joint_angles"])).split("_")[0]
        vals = dict(fr["joint_angles"])
        for ax, v in zip(["x", "y", "z"], fr["wrist_xyz"]):
            vals[f"{side}_wrist_0{ax}"] = v
        for ax, v in zip(["rx", "ry", "rz"], fr["wrist_rpy"]):
            vals[f"{side}_wrist_0{ax}"] = v
        for i, dn in enumerate(dof_names):
            if dn in vals:
                traj[t, i] = vals[dn]
    return traj


def _prep_hand_urdf(urdf_dir, physics):
    """Write an IsaacGym-ready URDF for one hand (visual-only for kinematic;
    mesh-collision for physics; absolute mesh paths). Returns the filename."""
    import xml.etree.ElementTree as ET
    urdf_dir = Path(urdf_dir)
    src = (list(urdf_dir.glob("subject*.urdf")) or list(urdf_dir.glob("*.urdf")))[0]
    if physics:
        txt = widen_base_limits(mesh_collision_urdf(src.read_text()), lim=20.0)
        root = ET.fromstring(txt)
    else:
        root = ET.fromstring(src.read_text())
        for link in root.findall("link"):
            for col in link.findall("collision"):
                link.remove(col)
    for m in root.iter("mesh"):
        fn = m.get("filename", "")
        if fn.startswith("./meshes/"):
            m.set("filename", f"{urdf_dir}/meshes/" + fn[len("./meshes/"):])
    (urdf_dir / "_isaac.urdf").write_text(ET.tostring(root, encoding="unicode"))
    return "_isaac.urdf"


def mano_joints(frames_dir, betas_npy, model_dir):
    import smplx
    pose48 = np.load(Path(frames_dir).parent / "poses_axisangle.npy")
    trans = np.load(Path(frames_dir).parent / "trans.npy")
    betas = np.load(betas_npy).astype(np.float32)
    T = pose48.shape[0]
    m = smplx.create(model_path=model_dir, model_type="mano", is_rhand=True,
                     use_pca=False, flat_hand_mean=True, batch_size=T)
    out = m(global_orient=torch.from_numpy(pose48[:, :3]).float(),
            hand_pose=torch.from_numpy(pose48[:, 3:]).float(),
            betas=torch.from_numpy(betas).unsqueeze(0).expand(T, -1).contiguous(),
            transl=torch.from_numpy(trans).float())
    return out.joints.detach().cpu().numpy()   # (T,21,3) MANO(master) frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf-dir", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--subject", default=None)
    ap.add_argument("--session", default=None)
    ap.add_argument("--dexycb-root", default="/root/data/dexycb")
    ap.add_argument("--mode", default="kinematic", choices=["kinematic", "physics"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--betas-npy", default=None)
    ap.add_argument("--model-dir", default=None)
    # custom (non-DexYCB): object mesh + Z-up world poses, solid color
    ap.add_argument("--object-cad", default=None)
    ap.add_argument("--object-poses", default=None)
    ap.add_argument("--object-color", default="0.75 0.72 0.62")
    ap.add_argument("--objects-json", default=None,
                    help='JSON list [{"mesh","poses","color"}] to load MANY objects')
    ap.add_argument("--urdf-dir2", default=None, help="second hand URDF (bimanual)")
    ap.add_argument("--frames-dir2", default=None, help="second hand frames (bimanual)")
    args = ap.parse_args()

    physics = args.mode == "physics"
    hand_dirs = [(Path(args.urdf_dir).resolve(), args.frames_dir)]
    if args.urdf_dir2:                                          # bimanual second hand
        hand_dirs.append((Path(args.urdf_dir2).resolve(), args.frames_dir2))
    # prepare each hand's IsaacGym URDF (visual-only for kinematic; mesh-collision
    # for physics; absolute mesh paths) and load its frames
    hand_urdf_files = [_prep_hand_urdf(ud, physics) for ud, _ in hand_dirs]
    frames = load_frames(Path(args.frames_dir))                # first hand (for camera/DexYCB)
    custom = (args.object_cad is not None) or (args.objects_json is not None)
    table_z = 0.0
    tag = None
    objects = []          # each: {meshes, color(list|None for texture), poses(T,7), name}
    if args.objects_json:                                  # many objects (HO-Cap)
        for i, ob in enumerate(json.load(open(args.objects_json))):
            m = str(Path(ob["mesh"]).resolve())
            objects.append({"meshes": {"vis": m, "col": m},
                            "color": [float(x) for x in str(ob.get("color", "0.75 0.72 0.62")).split()],
                            "poses": np.load(ob["poses"]).astype(np.float32), "name": f"object{i}"})
        obj_name = f"{len(objects)} objects"
    elif args.object_cad:                                  # single custom object
        m = str(Path(args.object_cad).resolve())
        objects.append({"meshes": {"vis": m, "col": m},
                        "color": [float(x) for x in args.object_color.split()],
                        "poses": np.load(args.object_poses).astype(np.float32), "name": "object"})
        obj_name = Path(args.object_cad).name
    else:                                                  # DexYCB (textured)
        obj_name, meshes, opose, tag = load_dexycb_object_tag(
            args.dexycb_root, args.subject, args.session)
        frames = transform_frames_to_tag(frames, tag)      # Z-up tag frame
        objects.append({"meshes": meshes, "color": None, "poses": opose, "name": "object"})

    if custom:                                             # frame scene from hands + all objects
        wr = np.array([f["wrist_xyz"] for _, fd in hand_dirs for f in load_frames(Path(fd))])
        pts = np.concatenate([wr] + [o["poses"][:, :3] for o in objects], axis=0)
        lo, hi = pts.min(0), pts.max(0); ctr = (lo + hi) / 2.0
        diag = float(np.linalg.norm(hi - lo)); dist = (diag * 1.6 + 0.3) / 1.5  # 1.5× closer
        cam_pos_t = (ctr + np.array([0.0, -dist, dist * 0.55])).astype(np.float32)
        cam_tgt_t = ctr.astype(np.float32); fovy = 45.0
        table_z = min(estimate_table_z(o["meshes"]["vis"], o["poses"]) for o in objects)
        print(f"[desk] table top inferred at z={table_z:.3f} m (object rest)")
    else:                                                  # DexYCB recording camera
        cam_pos_t, cam_tgt_t, fovy = dexycb_camera_in_tag(
            args.dexycb_root, args.subject, args.session, tag)
    T = len(frames)
    print(f"[isaac] {T} frames, object={obj_name}")

    # ---- sim (Z-up to match tag frame) ----
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, -9.81 if physics else 0.0)
    sp.dt = 1.0 / 60.0
    sp.substeps = 2
    sp.physx.solver_type = 1
    sp.physx.num_position_iterations = 8
    sp.physx.use_gpu = False
    sp.use_gpu_pipeline = False
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)

    # ground plane (Z-up) + lights
    pp = gymapi.PlaneParams()
    pp.normal = gymapi.Vec3(0, 0, 1)
    pp.distance = -float(table_z)              # ground plane at z=table_z (desk top)
    gym.add_ground(sim, pp)
    gym.set_light_parameters(sim, 0, gymapi.Vec3(0.9, 0.9, 0.9),
                             gymapi.Vec3(0.4, 0.4, 0.4), gymapi.Vec3(1, 1, -2))
    gym.set_light_parameters(sim, 1, gymapi.Vec3(0.6, 0.6, 0.6),
                             gymapi.Vec3(0.2, 0.2, 0.2), gymapi.Vec3(-1, -1, -2))

    o = gymapi.AssetOptions()
    o.fix_base_link = True                     # wrist 6-DOF provides base motion
    o.disable_gravity = not physics
    o.collapse_fixed_joints = False            # keep fixed-joint links (visual meshes)
    o.armature = 0.001
    o.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX   # recompute normals (avoid cull)
    o.override_com = True
    o.override_inertia = True
    o.default_dof_drive_mode = int(gymapi.DOF_MODE_POS if physics
                                   else gymapi.DOF_MODE_NONE)
    env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)
    urdf_dir = hand_dirs[0][0]                  # scratch dir for _obj.urdf
    # filter bits: hand shapes share filter 1 (no self-collision), object filter 2.
    dof_trajs, dof_names_all, hand_actors, body_names = [], [], [], []
    for i, ((ud, fdir), uf) in enumerate(zip(hand_dirs, hand_urdf_files)):
        asset = gym.load_asset(sim, str(ud), uf, o)
        dn = [gym.get_asset_dof_name(asset, j)
              for j in range(gym.get_asset_dof_count(asset))]
        dof_trajs.append(frames_to_dof_traj(load_frames(Path(fdir)), dn))
        dof_names_all += dn
        actor = gym.create_actor(env, asset, gymapi.Transform(), f"hand{i}", 0, 1)
        dp = gym.get_actor_dof_properties(env, actor)
        dp["lower"][:] = -20; dp["upper"][:] = 20      # widen limits (avoid clamp)
        if physics:
            dp["driveMode"][:] = gymapi.DOF_MODE_POS
            is_wrist = np.array(["wrist_0" in n for n in dn])
            dp["stiffness"][:] = np.where(is_wrist, 800.0, 40.0)
            dp["damping"][:] = np.where(is_wrist, 40.0, 1.5)
        else:
            dp["driveMode"][:] = gymapi.DOF_MODE_NONE
        gym.set_actor_dof_properties(env, actor, dp)
        for bi in range(gym.get_actor_rigid_body_count(env, actor)):   # skin color
            gym.set_rigid_body_color(env, actor, bi, gymapi.MESH_VISUAL_AND_COLLISION,
                                     gymapi.Vec3(0.85, 0.72, 0.55))
        hand_actors.append(actor)
        if i == 0:
            body_names = [gym.get_asset_rigid_body_name(asset, j)
                          for j in range(gym.get_asset_rigid_body_count(asset))]
    T_dof = min(t.shape[0] for t in dof_trajs)
    dof_traj = np.concatenate([t[:T_dof] for t in dof_trajs], axis=1)   # (T, sum ndof)
    ndof = dof_traj.shape[1]
    T = min([T_dof] + [len(o["poses"]) for o in objects])   # frames to actually play
    dof_traj = dof_traj[:T]

    # one actor per object (kinematic root set per frame; dynamic under physics).
    # DexYCB: textured.obj (UVs) visual maps the CAD texture; else solid color.
    oo = gymapi.AssetOptions()
    oo.fix_base_link = not physics             # physics: dynamic (gravity + contact)
    oo.disable_gravity = not physics
    oo.override_inertia = True
    obj_actors = []
    for oi, ob in enumerate(objects):
        ou = f"_obj{oi}.urdf"
        (urdf_dir / ou).write_text(
            f'<robot name="obj{oi}"><link name="obj{oi}">'
            f'<visual><geometry><mesh filename="{ob["meshes"]["vis"]}"/></geometry></visual>'
            f'<collision><geometry><mesh filename="{ob["meshes"]["col"]}"/></geometry></collision>'
            f'<inertial><mass value="0.3"/>'
            f'<inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>'
            f'</inertial></link></robot>')
        asset = gym.load_asset(sim, str(urdf_dir), ou, oo)
        actor = gym.create_actor(env, asset, gymapi.Transform(), ob["name"], 0, 2)
        if ob["color"] is not None:            # solid color (custom / multi)
            gym.set_rigid_body_color(env, actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                     gymapi.Vec3(*ob["color"]))
        else:                                  # DexYCB CAD texture
            tex = gym.create_texture_from_file(sim, str(ob["meshes"]["tex"]))
            gym.set_rigid_body_texture(env, actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, tex)
        obj_actors.append(actor)
    if physics:                              # match MuJoCo: high friction + generous
        for act in (*hand_actors, *obj_actors):   # contact_offset so near-miss fingers touch
            sh = gym.get_actor_rigid_shape_properties(env, act)
            for p in sh:
                p.friction = 1.5
                p.rolling_friction = 0.05
                p.torsion_friction = 0.05
                p.contact_offset = 0.02
            gym.set_actor_rigid_shape_properties(env, act, sh)

    # offscreen camera at the DexYCB extrinsic pose (tag frame)
    do_render = args.out is not None
    if do_render:
        cp = gymapi.CameraProperties()
        cp.width = args.width; cp.height = args.height
        cp.horizontal_fov = float(np.degrees(2 * np.arctan(
            (args.width / args.height) * np.tan(np.radians(fovy) / 2))))
        cam = gym.create_camera_sensor(env, cp)
        gym.set_camera_location(cam, env,
                                gymapi.Vec3(*cam_pos_t.tolist()),
                                gymapi.Vec3(*cam_tgt_t.tolist()))

    gym.prepare_sim(sim)
    dof_state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(ndof, 2)
    rb_state = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
    root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))  # (2,13)
    cf_state = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))    # (nb,3)
    dof_t = torch.from_numpy(dof_traj).float()
    # each object's root pose per frame: [x,y,z, qx,qy,qz,qw] (IsaacGym xyzw)
    obj_root_t = []
    for ob in objects:
        r = np.zeros((T, 13), dtype=np.float32)
        r[:, 0:3] = ob["poses"][:T, 0:3]
        r[:, 3:7] = ob["poses"][:T, [4, 5, 6, 3]]        # wxyz → xyzw
        obj_root_t.append(torch.from_numpy(r).float())

    if do_render:
        import imageio.v2 as imageio
        out_frames = []
        # each hand base fixed at identity (set_actor_root_state writes ALL actors,
        # so an unset hand root with zero quat would teleport it off-screen); object
        # is the last actor (after the hand actors)
        obj_idx = len(hand_actors)               # objects follow the hand actors
        for hi in range(len(hand_actors)):
            root_state[hi, :] = 0.0; root_state[hi, 6] = 1.0

        def grab():
            gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
            img = gym.get_camera_image(sim, env, cam, gymapi.IMAGE_COLOR)
            out_frames.append(img.reshape(args.height, args.width, 4)[:, :, :3].copy())

        if not physics:
            for t in range(T):
                dof_state[:, 0] = dof_t[t]; dof_state[:, 1] = 0
                gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state.view(-1, 2)))
                for k, rt in enumerate(obj_root_t):
                    root_state[obj_idx + k, :] = rt[t]
                gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state))
                gym.simulate(sim); gym.fetch_results(sim, True)
                grab()
        else:
            # initial state: hand posed at frame 0, every object at its frame-0 pose
            dof_state[:, 0] = dof_t[0]; dof_state[:, 1] = 0
            gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state.view(-1, 2)))
            for k, rt in enumerate(obj_root_t):
                root_state[obj_idx + k, :] = rt[0]
            gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state))
            # settle the object onto the table (hand held at frame 0, NOT recorded)
            # so any initial ground/hand penetration jitter happens off-camera
            gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(dof_t[0].contiguous()))
            for _ in range(40):
                gym.simulate(sim); gym.fetch_results(sim, True)
            # PD-track the reference; object is dynamic (gravity + contact)
            dt = sp.dt
            n_steps = int((T / float(args.fps)) / dt)
            stride = max(1, int(round(1.0 / (args.fps * dt))))
            for s in range(n_steps):
                tt = s * dt * args.fps
                i0 = min(int(tt), T - 1); i1 = min(i0 + 1, T - 1); a = tt - i0
                tgt = (1 - a) * dof_t[i0] + a * dof_t[i1]
                gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(tgt.contiguous()))
                gym.simulate(sim); gym.fetch_results(sim, True)
                if s % stride == 0:
                    grab()
        imageio.mimsave(args.out, out_frames, fps=args.fps, quality=8, macro_block_size=1)
        print(f"[render] {len(out_frames)} frames ({args.mode}) -> {args.out}")

    # ---- validation: Isaac body xpos vs MANO joints ----
    if args.validate:
        gt = mano_joints(args.frames_dir, args.betas_npy, args.model_dir)  # (T,21,3) master
        # transform MANO joints master→tag to compare in the same frame
        gt_tag = np.einsum("ij,tnj->tni", tag.R_tag_master, gt) + tag.t_tag_master
        bidx = {bn: body_names.index(bn) for bn in LINK_TO_MANO if bn in body_names}
        errs = []
        per_body = {bn: [] for bn in bidx}
        for t in range(T):
            dof_state[:, 0] = dof_t[t]; dof_state[:, 1] = 0
            gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state.view(-1, 2)))
            gym.simulate(sim); gym.fetch_results(sim, True)
            gym.refresh_rigid_body_state_tensor(sim)
            rb = rb_state.view(-1, 13)
            for bn, mj in LINK_TO_MANO.items():
                if bn not in bidx:
                    continue
                pos = rb[bidx[bn], 0:3].cpu().numpy()
                e = np.linalg.norm(pos - gt_tag[t, mj])
                errs.append(e); per_body[bn].append(e)
        errs = np.array(errs) * 1000
        print(f"[validate] Isaac-FK vs MANO joints (tag frame): "
              f"mean={errs.mean():.2f}mm  median={np.median(errs):.2f}mm  max={errs.max():.2f}mm")
        means = {bn: np.mean(v) * 1000 for bn, v in per_body.items() if v}
        for bn, e in sorted(means.items(), key=lambda x: -x[1])[:6]:
            print(f"    {bn:16s} {e:7.2f} mm")

    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
