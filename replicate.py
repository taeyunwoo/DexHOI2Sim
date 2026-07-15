#!/usr/bin/env python
"""DexHOI2Sim — replicate a hand-object interaction in simulation and evaluate it.

One command turns a MANO hand-object sequence (DexYCB, HO-Cap bimanual, or any
MANO β/θ + object CAD) into a simulated replay in MuJoCo and/or IsaacGym, renders
it, and reports the trajectory-tracking metric for benchmarking HOI-generation:

  * object_traj_error_mm : object path in physics (hand PD-tracks the reference)
                           vs the given reference path
  * grasp_success        : object ends within threshold of its reference position
                           (a good grasp carries the object; a bad one drops it)

Usage (DexYCB sequence):
  python replicate.py --subject 20200709-subject-01 --session 20200709_142211 \
      --backend mujoco --mode physics --render --eval --out-dir out/142211

  # both backends, evaluate only
  python replicate.py --subject ... --session ... --backend both --eval --no-render

  # HO-Cap bimanual (pick-and-place / handover / affordance-use)
  python replicate.py --hocap --subject subject_2 --session 20231022_200657 \
      --backend both --mode kinematic --out-dir out/handover
"""
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
ISAAC_ENV = {                       # IsaacGym needs its libs + ninja on PATH
    "LD_LIBRARY_PATH": "/root/miniconda3/envs/isaacgym/lib:" + os.environ.get("LD_LIBRARY_PATH", ""),
    "PATH": "/root/miniconda3/envs/isaacgym/bin:" + os.environ.get("PATH", ""),
    "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "0",
}


def run(cmd, extra_env=None, tag=""):
    env = dict(os.environ)
    env.update(extra_env or {})
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run([str(c) for c in cmd], env=env,
                       capture_output=True, text=True)
    # IsaacGym segfaults on sim destroy AFTER writing output (harmless): 139 / -11
    if r.returncode not in (0, 139, -11):
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise RuntimeError(f"{tag} failed (code {r.returncode})")
    return r.stdout


def locate_betas(dexycb_root, subject, session):
    import yaml
    meta = yaml.safe_load(open(Path(dexycb_root) / subject / session / "meta.yml"))
    calib = meta["mano_calib"][0]          # already ends with _right/_left
    return Path(dexycb_root) / "calibration" / f"mano_{calib}" / "mano.yml"


def load_bundle(path, work):
    """Unpack a single-file HOI bundle (.npz / .pkl / .json dict) into the files the
    custom pipeline reads. Returns {hands:[{betas_yml,poses,trans,left}, ...],
    object_cad, object_poses, object_color}.

    One hand — flat keys:  betas(10,), hand_pose(T,48 axis-angle), trans(T,3),
      side ('right'/'left').
    Two hands — either a `hands` list [{betas,hand_pose,trans,side}, ...] (pkl/json),
      or indexed flat keys for the 2nd hand: betas2/hand_pose2/trans2/side2 (npz-safe).
    Object (shared):  object_poses(T,7 [x,y,z,qw,qx,qy,qz]), object_color(3,), and
      either object_mesh (path) OR object_verts(V,3)+object_faces(F,3) (embedded)."""
    import numpy as np, yaml
    p = str(path)
    if p.endswith(".npz"):
        d = dict(np.load(p, allow_pickle=True))
    elif p.endswith(".json"):
        d = json.load(open(p))
    else:
        import pickle
        d = pickle.load(open(p, "rb"))
    work.mkdir(parents=True, exist_ok=True)

    if "hands" in d:                                  # explicit list (pkl/json)
        entries = list(d["hands"])
    else:                                             # flat keys (+ optional 2nd hand)
        entries = [{"betas": d["betas"], "hand_pose": d["hand_pose"],
                    "trans": d["trans"], "side": d.get("side", "right")}]
        if d.get("hand_pose2") is not None:
            entries.append({"betas": d.get("betas2", d["betas"]),
                            "hand_pose": d["hand_pose2"], "trans": d["trans2"],
                            "side": d.get("side2", "left")})
    hands = []
    for i, h in enumerate(entries):
        suf = "" if i == 0 else str(i + 1)
        byml = work / f"betas{suf}.yml"
        yaml.safe_dump({"betas": np.asarray(h["betas"], float).ravel().tolist()},
                       open(byml, "w"))
        np.save(work / f"poses{suf}.npy", np.asarray(h["hand_pose"], np.float32))
        np.save(work / f"trans{suf}.npy", np.asarray(h["trans"], np.float32))
        hands.append({"betas_yml": str(byml), "poses": str(work / f"poses{suf}.npy"),
                      "trans": str(work / f"trans{suf}.npy"),
                      "left": str(h.get("side", "right")).lower().startswith("l")})

    np.save(work / "object_poses.npy", np.asarray(d["object_poses"], dtype=np.float32))
    if "object_mesh" in d and d["object_mesh"] is not None:
        obj_mesh = str(d["object_mesh"])
    else:                                             # embedded verts+faces → .obj
        v = np.asarray(d["object_verts"], float); f = np.asarray(d["object_faces"], int)
        with open(work / "object.obj", "w") as fh:
            for x in v: fh.write(f"v {x[0]} {x[1]} {x[2]}\n")
            for t in f: fh.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
        obj_mesh = str(work / "object.obj")
    color = d.get("object_color", [0.75, 0.72, 0.62])
    color = " ".join(str(float(c)) for c in np.asarray(color).ravel())
    return {"hands": hands, "object_cad": obj_mesh,
            "object_poses": str(work / "object_poses.npy"), "object_color": color}


def _gen_hand(gen, betas_yml, poses, trans, left, urdf_dir, frames_dir):
    """Build one hand's URDF + per-frame analytic qpos (both cached). Returns
    (urdf_dir_str, frames_dir_str)."""
    if not urdf_dir.exists() or not list(urdf_dir.glob("*.urdf")):
        gu = [PY, gen / "generate_urdf.py", "--betas-yml", betas_yml,
              "--out", urdf_dir, "--name", "hand"]
        gu += ["--lhand"] if left else ["--is_rhand"]
        run(gu, tag="generate_urdf")
    if not list((frames_dir / "frames").glob("*.json")):
        p2j = [PY, gen / "pose_to_joint_angles.py", "--betas-yml", betas_yml,
               "--urdf-dir", urdf_dir, "--out", frames_dir,
               "--poses-npy", poses, "--trans-npy", trans]
        if left: p2j += ["--left"]
        run(p2j, tag="pose_to_joint_angles")
    return (str(urdf_dir), str(frames_dir / "frames"))


def prep_custom(hand_dicts, out, gen):
    """Custom (non-DexYCB) hands: build 1-2 hands from a list of param dicts
    ({betas_yml, poses, trans, left}). Returns hand_specs [(urdf, frames), ...]."""
    specs = []
    for i, h in enumerate(hand_dicts):
        ud = out / ("urdf" if i == 0 else f"urdf{i + 1}")
        fr = out / ("frames" if i == 0 else f"frames{i + 1}")
        specs.append(_gen_hand(gen, h["betas_yml"], h["poses"], h["trans"],
                               h["left"], ud, fr))
    return specs


def prep_hocap(args, out, gen):
    """HO-Cap session -> per-hand (URDF, frames) + the manipulated object.

    Returns (hand_specs, obj_cad, obj_poses_npy, obj_color). Bimanual sessions
    yield two hand_specs; the object is the one that moves the most (the one
    being manipulated / handed over)."""
    import numpy as np, yaml
    sys.path.insert(0, str(ROOT / "sim"))
    from hocap_loader import load_hocap_session
    d = load_hocap_session(args.hocap_root, args.subject, args.session)
    print(f"[hocap] {d['task_name']} | hands={[h['side'] for h in d['hands']]} "
          f"| objs={len(d['objects'])} | T={d['num_frames']}")
    work = out / "hocap"; work.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump({"betas": d["hands"][0]["betas"].tolist()},
                   open(work / "betas.yml", "w"))
    betas = str(work / "betas.yml")

    hand_specs = []
    for h in d["hands"]:
        side = h["side"]
        ud = work / f"urdf_{side}"; fr = work / f"frames_{side}"
        if not (ud / "hand.urdf").exists():
            gu = [PY, gen / "generate_urdf.py", "--betas-yml", betas,
                  "--out", ud, "--name", "hand",
                  "--lhand" if side == "left" else "--is_rhand"]
            run(gu, tag=f"generate_urdf[{side}]")
        np.save(work / f"poses_{side}.npy", h["pose48"])
        np.save(work / f"trans_{side}.npy", h["trans"])
        if not list((fr / "frames").glob("*.json")):
            p2j = [PY, gen / "pose_to_joint_angles.py", "--betas-yml", betas,
                   "--urdf-dir", ud, "--out", fr,
                   "--poses-npy", work / f"poses_{side}.npy",
                   "--trans-npy", work / f"trans_{side}.npy"]
            if side == "left": p2j += ["--left"]
            run(p2j, tag=f"pose_to_joint_angles[{side}]")
        hand_specs.append((str(ud), str(fr / "frames")))

    mv = [float(np.linalg.norm(o["poses"][:, :3] - o["poses"][0, :3], axis=1).max())
          for o in d["objects"]]
    obj = d["objects"][int(np.argmax(mv))]
    obj_poses = work / "obj_poses.npy"
    np.save(obj_poses, obj["poses"])
    print(f"[hocap] manipulated object={obj['id']} (moved {max(mv)*1000:.0f}mm)")
    return hand_specs, obj["mesh"], str(obj_poses), "0.85 0.6 0.3"


def main():
    ap = argparse.ArgumentParser(description="DexHOI2Sim replication + evaluation")
    ap.add_argument("--dexycb-root", default="/root/data/dexycb")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--session", default=None)
    # custom (non-DexYCB): a single bundle file, or the individual pieces
    ap.add_argument("--bundle", default=None,
                    help="single .npz/.pkl with betas/hand_pose/trans/object_* keys")
    ap.add_argument("--custom", action="store_true")
    # HO-Cap (irvlutd) bimanual sessions
    ap.add_argument("--hocap", action="store_true",
                    help="replicate an HO-Cap session (uses --subject/--session)")
    ap.add_argument("--hocap-root", default="/root/data/hocap")
    ap.add_argument("--betas-yml", default=None, help="MANO betas (custom)")
    ap.add_argument("--poses", default=None, help="(T,48) MANO axis-angle (custom)")
    ap.add_argument("--trans", default=None, help="(T,3) wrist translation (custom)")
    ap.add_argument("--left", action="store_true", help="left hand (custom)")
    # second hand (custom bimanual) — give the other hand's pieces
    ap.add_argument("--betas-yml2", default=None, help="2nd hand betas (default: same)")
    ap.add_argument("--poses2", default=None, help="2nd hand (T,48) axis-angle")
    ap.add_argument("--trans2", default=None, help="2nd hand (T,3) translation")
    ap.add_argument("--left2", action="store_true", help="2nd hand is left")
    ap.add_argument("--object-cad", default=None, help="object mesh (custom)")
    ap.add_argument("--object-poses", default=None,
                    help="(T,7) [x,y,z,qw,qx,qy,qz] Z-up world (custom)")
    ap.add_argument("--object-color", default="0.75 0.72 0.62")
    ap.add_argument("--backend", default="mujoco", choices=["mujoco", "isaac", "both"])
    ap.add_argument("--mode", default="physics", choices=["kinematic", "physics"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--render", action="store_true", help="write mp4(s)")
    ap.add_argument("--no-render", dest="render", action="store_false")
    ap.add_argument("--eval", action="store_true", help="compute + save metrics.json")
    ap.set_defaults(render=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    custom_hands = None                               # list of {betas_yml,poses,trans,left}
    if args.bundle:                                   # unpack bundle → custom hands + object
        b = load_bundle(args.bundle, out / "bundle")
        args.custom = True
        custom_hands = b["hands"]
        args.object_cad, args.object_poses = b["object_cad"], b["object_poses"]
        args.object_color = b["object_color"]
        print(f"[bundle] {args.bundle} -> {out/'bundle'}  ({len(custom_hands)} hand(s))")
    elif args.custom:                                 # flag-based custom (1 or 2 hands)
        custom_hands = [{"betas_yml": args.betas_yml, "poses": args.poses,
                         "trans": args.trans, "left": args.left}]
        if args.poses2:
            custom_hands.append({"betas_yml": args.betas_yml2 or args.betas_yml,
                                 "poses": args.poses2, "trans": args.trans2,
                                 "left": args.left2})
    gen = ROOT / "mano2urdf" / "scripts"

    if args.hocap:                                    # HO-Cap: 1-2 hands + object
        hand_specs, obj_cad, obj_poses, obj_color = prep_hocap(args, out, gen)
        obj_args = ["--object-cad", obj_cad, "--object-poses", obj_poses,
                    "--object-color", obj_color]
    elif args.custom:                                 # custom / bundle: 1-2 hands
        hand_specs = prep_custom(custom_hands, out, gen)
        obj_args = ["--object-cad", args.object_cad, "--object-poses", args.object_poses,
                    "--object-color", args.object_color]
        print(f"[custom] {len(hand_specs)} hand(s)")
    else:                                             # DexYCB: single hand via --seq
        urdf_dir = out / "urdf"; frames_dir = out / "frames"
        betas = locate_betas(args.dexycb_root, args.subject, args.session)
        if not urdf_dir.exists() or not list(urdf_dir.glob("*.urdf")):
            gu = [PY, gen / "generate_urdf.py", "--betas-yml", betas,
                  "--out", urdf_dir, "--name", "hand"]
            gu += ["--lhand"] if args.left else ["--is_rhand"]
            run(gu, tag="generate_urdf")
        if not frames_dir.exists() or not list((frames_dir / "frames").glob("*.json")):
            p2j = [PY, gen / "pose_to_joint_angles.py", "--betas-yml", betas,
                   "--urdf-dir", urdf_dir, "--out", frames_dir,
                   "--seq", Path(args.dexycb_root) / args.subject / args.session]
            run(p2j, tag="pose_to_joint_angles")
        hand_specs = [(str(urdf_dir), str(frames_dir / "frames"))]
        obj_args = ["--subject", args.subject, "--session", args.session]

    backends = ["mujoco", "isaac"] if args.backend == "both" else [args.backend]
    metrics = {"mode": args.mode}
    (u0, f0), extra = hand_specs[0], []
    for i, (ud, fd) in enumerate(hand_specs[1:], start=2):    # bimanual: hand 2..N
        extra += [f"--urdf-dir{i}", ud, f"--frames-dir{i}", fd]

    for be in (backends if args.render else []):
        common = ["--urdf-dir", u0, "--frames-dir", f0, "--mode", args.mode, *extra, *obj_args]
        mp4 = out / f"{be}_{args.mode}.mp4"
        script = "mano2urdf_mujoco.py" if be == "mujoco" else "mano2urdf_isaac.py"
        env = {"MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "0"} if be == "mujoco" else ISAAC_ENV
        run([PY, ROOT / "sim" / script, *common, "--out", mp4], extra_env=env, tag=be)
        print(f"[render] {mp4}")

    # 3) evaluation (physics object-trajectory tracking) — single-hand only
    if args.eval and len(hand_specs) > 1:
        print("[eval] skipped: object-trajectory metric is single-hand only "
              "(bimanual sequences render but are not scored yet)")
    elif args.eval:
        sys.path.insert(0, str(ROOT / "sim"))
        u_eval = hand_specs[0][0]
        f_eval = str(Path(hand_specs[0][1]).parent)      # dir containing frames/
        if args.custom:
            from metrics import evaluate_custom
            m = evaluate_custom(u_eval, f_eval, args.object_cad, args.object_poses)
        else:
            from metrics import evaluate
            m = evaluate(args.dexycb_root, args.subject, args.session,
                         u_eval, f_eval, ROOT)
        metrics.update(m)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print("\n=== metrics ===")
        print(json.dumps(metrics, indent=2))
        print(f"[eval] -> {out/'metrics.json'}")


if __name__ == "__main__":
    main()
