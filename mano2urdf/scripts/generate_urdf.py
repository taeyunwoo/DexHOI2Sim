"""
β + (옵션) v_template 으로부터 URDF + per-link meshes 생성.

Usage:
    python scripts/generate_urdf.py --betas 0,0,...,0 --is_rhand --out outputs/myhand --name myhand
    python scripts/generate_urdf.py --betas-yml /path/to/mano.yml --is_rhand --out outputs/seq01 --name seq01_hand
"""
import argparse, os, sys, yaml, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "mano2urdf"))

import smplx
from export_meshes import export_body_part_meshes
from mano_helpers import get_mano_data, get_mano_joint_names
from urdf import export_mano2urdf


def setup_smplx_root():
    MODEL_DIR = os.path.join(ROOT, "outputs", "_smplx_root")
    os.makedirs(os.path.join(MODEL_DIR, "mano"), exist_ok=True)
    for p in ["MANO_RIGHT.pkl", "MANO_LEFT.pkl"]:
        dst = os.path.join(MODEL_DIR, "mano", p)
        if not os.path.exists(dst):
            os.symlink(os.path.join(ROOT, "assets", p), dst)
    return MODEL_DIR


def parse_betas(args):
    if args.betas_yml is not None:
        with open(args.betas_yml) as f:
            d = yaml.safe_load(f)
        return np.asarray(d["betas"], dtype=np.float32)
    if args.betas is not None:
        return np.asarray([float(x) for x in args.betas.split(",")], dtype=np.float32)
    return np.zeros(10, dtype=np.float32)


def compute_v_template(model_dir, betas, is_rhand):
    """shape-only verts (zero pose) to use as v_template for URDF generation."""
    m = smplx.create(model_path=model_dir, model_type="mano",
                     is_rhand=is_rhand, use_pca=False,
                     flat_hand_mean=True, batch_size=1)
    out = m(betas=torch.from_numpy(betas).unsqueeze(0))
    return out.vertices.detach().squeeze(0)  # (778,3) torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", type=str, default=None, help="comma-separated 10 floats")
    ap.add_argument("--betas-yml", type=str, default=None, help="DexYCB-style mano.yml")
    ap.add_argument("--is_rhand", action="store_true")
    ap.add_argument("--lhand", action="store_true")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--name", type=str, default="mano_hand")
    args = ap.parse_args()

    is_rhand = (not args.lhand) if not args.is_rhand else True
    os.makedirs(args.out, exist_ok=True)
    model_dir = setup_smplx_root()

    betas = parse_betas(args)
    print(f"[gen] betas = {betas}")

    v_template = compute_v_template(model_dir, betas, is_rhand)

    lbs_weight_matrix, verts, joints_dict = get_mano_data(
        model_path=model_dir, is_rhand=is_rhand, v_template=v_template)

    joint_names = get_mano_joint_names(is_rhand)
    joint_mesh_data = export_body_part_meshes(
        out_path=args.out, lbs_weight_matrix=lbs_weight_matrix,
        vertices=verts, joint_names=joint_names,
        decimation_factor_obj=0.5, is_rhand=is_rhand)

    export_mano2urdf(hand_name=args.name, out_path=args.out,
                     joints_dict=joints_dict, joint_mesh_data=joint_mesh_data,
                     is_rhand=is_rhand)

    # save beta + lbs argmax for downstream verifier
    np.save(os.path.join(args.out, "betas.npy"), betas)
    np.save(os.path.join(args.out, "lbs_argmax.npy"),
            np.argmax(lbs_weight_matrix, axis=-1))
    np.save(os.path.join(args.out, "rest_verts.npy"), verts)
    with open(os.path.join(args.out, "gen_info.json"), "w") as f:
        json.dump({"is_rhand": is_rhand, "name": args.name,
                   "joint_names": joint_names}, f, indent=2)
    print(f"[gen] done -> {args.out}")


if __name__ == "__main__":
    main()
