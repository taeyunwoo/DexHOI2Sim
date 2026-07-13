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
                              mesh_collision_urdf, widen_base_limits)

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
    """(T, ndof) in the asset DOF order, from the frame json (names match)."""
    T = len(frames)
    traj = np.zeros((T, len(dof_names)), dtype=np.float32)
    for t, fr in enumerate(frames):
        vals = dict(fr["joint_angles"])
        for nm, v in zip(["right_wrist_0x", "right_wrist_0y", "right_wrist_0z"],
                         fr["wrist_xyz"]):
            vals[nm] = v
        for nm, v in zip(["right_wrist_0rx", "right_wrist_0ry", "right_wrist_0rz"],
                         fr["wrist_rpy"]):
            vals[nm] = v
        for i, dn in enumerate(dof_names):
            if dn in vals:
                traj[t, i] = vals[dn]
    return traj


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
    args = ap.parse_args()

    import xml.etree.ElementTree as ET
    physics = args.mode == "physics"
    urdf_dir = Path(args.urdf_dir).resolve()
    src = (list(urdf_dir.glob("subject*.urdf")) or list(urdf_dir.glob("*.urdf")))[0]
    # kinematic: VISUAL-ONLY URDF (hograspnet *_fkonly) — box/capsule <collision>
    # primitives make IsaacGym emit "resolve collision mesh ''" and drop the body.
    # physics: replace <collision> with the named MANO meshes (convex hulls) so the
    # fingers can contact the object. Mesh paths → absolute either way.
    if physics:
        txt = mesh_collision_urdf(src.read_text())
        txt = widen_base_limits(txt, lim=20.0)
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
    urdf_file = "_isaac.urdf"
    (urdf_dir / urdf_file).write_text(ET.tostring(root, encoding="unicode"))

    frames = load_frames(Path(args.frames_dir))
    custom = args.object_cad is not None
    if custom:                                             # non-DexYCB: solid color
        obj_name = Path(args.object_cad).name
        cad = str(Path(args.object_cad).resolve())
        meshes = {"vis": cad, "col": cad}
        obj_pose_tag = np.load(args.object_poses).astype(np.float32)   # (T,7) Z-up
        obj_color = [float(x) for x in args.object_color.split()]
        ctr = obj_pose_tag[:, :3].mean(0)
        d = 0.6
        cam_pos_t = np.array([ctr[0], ctr[1] - d, ctr[2] + 0.3], np.float32)
        cam_tgt_t = ctr.astype(np.float32); fovy = 55.0
    else:                                                  # DexYCB
        obj_name, meshes, obj_pose_tag, tag = load_dexycb_object_tag(
            args.dexycb_root, args.subject, args.session)
        frames = transform_frames_to_tag(frames, tag)      # Z-up tag frame
        obj_color = None
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
    hand_asset = gym.load_asset(sim, str(urdf_dir), urdf_file, o)
    ndof = gym.get_asset_dof_count(hand_asset)
    dof_names = [gym.get_asset_dof_name(hand_asset, i) for i in range(ndof)]
    body_names = [gym.get_asset_rigid_body_name(hand_asset, i)
                  for i in range(gym.get_asset_rigid_body_count(hand_asset))]
    dof_traj = frames_to_dof_traj(frames, dof_names)       # (T, 51)

    env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)
    # filter bits: hand shapes share filter 1 (no self-collision → 1&1≠0), object
    # filter 2 (hand↔object 1&2=0 → collide). Self-collision of the 63-body hand
    # otherwise jams the solver and suppresses hand↔object contacts.
    hand = gym.create_actor(env, hand_asset, gymapi.Transform(), "hand", 0, 1)
    dp = gym.get_actor_dof_properties(env, hand)
    dp["lower"][:] = -20; dp["upper"][:] = 20      # widen limits (avoid clamp)
    if physics:
        dp["driveMode"][:] = gymapi.DOF_MODE_POS
        # stiff wrist (hold base pose), softer fingers (PD-track grasp)
        is_wrist = np.array(["wrist_0" in n for n in dof_names])
        dp["stiffness"][:] = np.where(is_wrist, 800.0, 40.0)
        dp["damping"][:] = np.where(is_wrist, 40.0, 1.5)
    else:
        dp["driveMode"][:] = gymapi.DOF_MODE_NONE
    gym.set_actor_dof_properties(env, hand, dp)
    for bi in range(gym.get_actor_rigid_body_count(env, hand)):   # skin color
        gym.set_rigid_body_color(env, hand, bi, gymapi.MESH_VISUAL_AND_COLLISION,
                                 gymapi.Vec3(0.85, 0.72, 0.55))

    # object actor (kinematic root, set per frame) — textured.obj (UVs) for the
    # visual so the DexYCB texture maps; textured_simple.obj for collision.
    obj_urdf = (f'<robot name="obj"><link name="obj">'
                f'<visual><geometry><mesh filename="{meshes["vis"]}"/></geometry></visual>'
                f'<collision><geometry><mesh filename="{meshes["col"]}"/></geometry></collision>'
                f'<inertial><mass value="0.3"/>'
                f'<inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>'
                f'</inertial></link></robot>')
    (urdf_dir / "_obj.urdf").write_text(obj_urdf)
    oo = gymapi.AssetOptions()
    oo.fix_base_link = not physics             # physics: dynamic (gravity + contact)
    oo.disable_gravity = not physics
    oo.override_inertia = True
    obj_asset = gym.load_asset(sim, str(urdf_dir), "_obj.urdf", oo)
    obj_actor = gym.create_actor(env, obj_asset, gymapi.Transform(), "object", 0, 2)
    if custom:                               # solid color (no CAD texture)
        gym.set_rigid_body_color(env, obj_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                 gymapi.Vec3(*obj_color))
    else:                                    # DexYCB CAD texture
        obj_tex = gym.create_texture_from_file(sim, str(meshes["tex"]))
        gym.set_rigid_body_texture(env, obj_actor, 0,
                                   gymapi.MESH_VISUAL_AND_COLLISION, obj_tex)
    if physics:                              # match MuJoCo: high friction + generous
        for act in (hand, obj_actor):        # contact_offset so near-miss fingers touch
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
    # object root pose per frame: [x,y,z, qx,qy,qz,qw] (IsaacGym xyzw)
    obj_root = np.zeros((T, 13), dtype=np.float32)
    obj_root[:, 0:3] = obj_pose_tag[:, 0:3]
    obj_root[:, 3:7] = obj_pose_tag[:, [4, 5, 6, 3]]      # wxyz → xyzw
    obj_root_t = torch.from_numpy(obj_root).float()

    if do_render:
        import imageio.v2 as imageio
        out_frames = []
        # hand base fixed at identity (set_actor_root_state writes ALL actors, so an
        # unset hand root with zero quat would teleport it off-screen)
        root_state[0, :] = 0.0; root_state[0, 6] = 1.0

        def grab():
            gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
            img = gym.get_camera_image(sim, env, cam, gymapi.IMAGE_COLOR)
            out_frames.append(img.reshape(args.height, args.width, 4)[:, :, :3].copy())

        if not physics:
            for t in range(T):
                dof_state[:, 0] = dof_t[t]; dof_state[:, 1] = 0
                gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state.view(-1, 2)))
                root_state[1, :] = obj_root_t[t]
                gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state))
                gym.simulate(sim); gym.fetch_results(sim, True)
                grab()
        else:
            # initial state: hand posed at frame 0, object at its frame-0 pose
            dof_state[:, 0] = dof_t[0]; dof_state[:, 1] = 0
            gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state.view(-1, 2)))
            root_state[1, :] = obj_root_t[0]
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
