#!/usr/bin/env python
"""DexHOI2Sim — replicate a hand-object interaction in simulation and evaluate it.

One command turns a MANO hand-object sequence (DexYCB, or any MANO β/θ + object
CAD) into a simulated replay in MuJoCo and/or IsaacGym, renders it, and reports
metrics for benchmarking HOI-generation methods:

  * reconstruction : URDF-vs-MANO vertex error (how faithfully the hand is built)
  * penetration    : max hand-object interpenetration at the grasp (plausibility)
  * grasp_success  : object displacement when the grasp is dropped into physics
                     (a stable grasp barely moves → success)

Usage (DexYCB sequence):
  python replicate.py --subject 20200709-subject-01 --session 20200709_142211 \
      --backend mujoco --mode physics --render --eval --out-dir out/142211

  # both backends, evaluate only
  python replicate.py --subject ... --session ... --backend both --eval --no-render
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
    """Unpack a single-file HOI bundle (.npz or .pkl dict) into the individual
    files the custom pipeline reads, and return the resolved paths/flags.

    Keys: betas(10,), hand_pose(T,48 axis-angle), trans(T,3), side ('right'/'left'),
          object_poses(T,7 [x,y,z,qw,qx,qy,qz]), object_color(3,), and either
          object_mesh (path) OR object_verts(V,3)+object_faces(F,3) (embedded)."""
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
    betas = np.asarray(d["betas"], dtype=float).ravel().tolist()
    yaml.safe_dump({"betas": betas}, open(work / "betas.yml", "w"))
    np.save(work / "poses.npy", np.asarray(d["hand_pose"], dtype=np.float32))
    np.save(work / "trans.npy", np.asarray(d["trans"], dtype=np.float32))
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
    left = str(d.get("side", "right")).lower().startswith("l")
    return {"betas_yml": str(work / "betas.yml"), "poses": str(work / "poses.npy"),
            "trans": str(work / "trans.npy"), "object_cad": obj_mesh,
            "object_poses": str(work / "object_poses.npy"),
            "object_color": color, "left": left}


def main():
    ap = argparse.ArgumentParser(description="DexHOI2Sim replication + evaluation")
    ap.add_argument("--dexycb-root", default="/root/data/dexycb")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--session", default=None)
    # custom (non-DexYCB): a single bundle file, or the individual pieces
    ap.add_argument("--bundle", default=None,
                    help="single .npz/.pkl with betas/hand_pose/trans/object_* keys")
    ap.add_argument("--custom", action="store_true")
    ap.add_argument("--betas-yml", default=None, help="MANO betas (custom)")
    ap.add_argument("--poses", default=None, help="(T,48) MANO axis-angle (custom)")
    ap.add_argument("--trans", default=None, help="(T,3) wrist translation (custom)")
    ap.add_argument("--object-cad", default=None, help="object mesh (custom)")
    ap.add_argument("--object-poses", default=None,
                    help="(T,7) [x,y,z,qw,qx,qy,qz] Z-up world (custom)")
    ap.add_argument("--object-color", default="0.75 0.72 0.62")
    ap.add_argument("--left", action="store_true", help="left hand (custom)")
    ap.add_argument("--backend", default="mujoco", choices=["mujoco", "isaac", "both"])
    ap.add_argument("--mode", default="physics", choices=["kinematic", "physics"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--render", action="store_true", help="write mp4(s)")
    ap.add_argument("--no-render", dest="render", action="store_false")
    ap.add_argument("--eval", action="store_true", help="compute + save metrics.json")
    ap.set_defaults(render=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.bundle:                                   # unpack bundle → custom args
        b = load_bundle(args.bundle, out / "bundle")
        args.custom = True
        args.betas_yml, args.poses, args.trans = b["betas_yml"], b["poses"], b["trans"]
        args.object_cad, args.object_poses = b["object_cad"], b["object_poses"]
        args.object_color, args.left = b["object_color"], b["left"]
        print(f"[bundle] {args.bundle} -> {out/'bundle'}")
    urdf_dir = out / "urdf"
    frames_dir = out / "frames"
    gen = ROOT / "mano2urdf" / "scripts"
    betas = args.betas_yml if args.custom else \
        locate_betas(args.dexycb_root, args.subject, args.session)

    # 1) β → hand URDF (cached)
    if not urdf_dir.exists() or not list(urdf_dir.glob("*.urdf")):
        gu = [PY, gen / "generate_urdf.py", "--betas-yml", betas,
              "--out", urdf_dir, "--name", "hand"]
        gu += ["--lhand"] if args.left else ["--is_rhand"]
        run(gu, tag="generate_urdf")

    # 2) sequence → per-frame analytic qpos (frames/*.json)
    if not frames_dir.exists() or not list((frames_dir / "frames").glob("*.json")):
        p2j = [PY, gen / "pose_to_joint_angles.py", "--betas-yml", betas,
               "--urdf-dir", urdf_dir, "--out", frames_dir]
        if args.custom:
            p2j += ["--poses-npy", args.poses, "--trans-npy", args.trans]
            if args.left: p2j += ["--left"]
        else:
            p2j += ["--seq", Path(args.dexycb_root) / args.subject / args.session]
        run(p2j, tag="pose_to_joint_angles")

    fdir = frames_dir / "frames"
    backends = ["mujoco", "isaac"] if args.backend == "both" else [args.backend]
    metrics = {"mode": args.mode}
    obj_args = (["--object-cad", args.object_cad, "--object-poses", args.object_poses,
                 "--object-color", args.object_color] if args.custom else
                ["--subject", args.subject, "--session", args.session])

    for be in (backends if args.render else []):
        common = ["--urdf-dir", urdf_dir, "--frames-dir", fdir, "--mode", args.mode, *obj_args]
        mp4 = out / f"{be}_{args.mode}.mp4"
        script = "mano2urdf_mujoco.py" if be == "mujoco" else "mano2urdf_isaac.py"
        env = {"MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "0"} if be == "mujoco" else ISAAC_ENV
        run([PY, ROOT / "sim" / script, *common, "--out", mp4], extra_env=env, tag=be)
        print(f"[render] {mp4}")

    # 3) evaluation (physics object-trajectory tracking)
    if args.eval:
        sys.path.insert(0, str(ROOT / "sim"))
        if args.custom:
            from metrics import evaluate_custom
            m = evaluate_custom(urdf_dir, frames_dir, args.object_cad, args.object_poses)
        else:
            from metrics import evaluate
            m = evaluate(args.dexycb_root, args.subject, args.session,
                         urdf_dir, frames_dir, ROOT)
        metrics.update(m)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print("\n=== metrics ===")
        print(json.dumps(metrics, indent=2))
        print(f"[eval] -> {out/'metrics.json'}")


if __name__ == "__main__":
    main()
