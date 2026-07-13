# Third-Party Notices & Attribution

This project builds on several third-party works. Each retains its own license.
You are responsible for complying with all of them before use or redistribution.

---

## 1. MANO2URDF generator — adapted from ArtiGrasp

The URDF/MJCF hand generator under `assets/` (the `mano2urdf/` code:
`mano2urdf.py`, `mano_helpers.py`, `export_meshes.py`, `urdf.py`,
`rotation_helper.py`, joint-limit `.txt` files) is **adapted from the ArtiGrasp
project**, specifically its `rsc/mano2urdf/` directory:

- **Repo:** https://github.com/zdchan/artigrasp  (`rsc/mano2urdf/`)
- **Paper:** Hui Zhang, Sammy Christen, Zicong Fan, Luocheng Zheng, Jemin Hwangbo,
  Jie Song, Otmar Hilliges. *"ArtiGrasp: Physically Plausible Synthesis of
  Bi-Manual Dexterous Grasping and Articulation."* 3DV 2024.

```bibtex
@inproceedings{zhang2024artigrasp,
  title     = {{ArtiGrasp}: Physically Plausible Synthesis of Bi-Manual Dexterous Grasping and Articulation},
  author    = {Zhang, Hui and Christen, Sammy and Fan, Zicong and Zheng, Luocheng and Hwangbo, Jemin and Song, Jie and Hilliges, Otmar},
  booktitle = {International Conference on 3D Vision (3DV)},
  year      = {2024}
}
```

### ⚠️ License status — UNRESOLVED (read before publishing)

ArtiGrasp's `LICENSE.md` grants **MIT** only to code in
`cmake/scripts/examples/raisimGymTorch/raisimMatlab/raisimPy`. The `mano2urdf`
code lives in **`rsc/`**, which `LICENSE.md` describes as *"from other open
source projects, corresponding licenses inside each folder"* — but **no license
file or copyright header exists inside `rsc/mano2urdf/`**. A hardcoded path
(`raisim_grasp_arctic`) suggests an earlier upstream we could not trace.

**The permissive/MIT license therefore does NOT clearly cover this code.**
Before public redistribution, pick one:

- **(A) Get explicit permission** — email the ArtiGrasp authors (H. Zhang /
  @zdchan) for written confirmation of the license on `rsc/mano2urdf/`. Safest.
- **(B) Do not redistribute the code** — ship only a `download_and_patch.sh`
  that clones ArtiGrasp and applies our patches at install time (our patches are
  ours; their code stays on their repo).
- **(C) Clean-reimplement** the generator from the MANO spec so no ArtiGrasp
  code ships. Most work, zero risk.

Our local modifications to this code (headless matplotlib, `np.int64` fix, etc.)
are released under this project's license; they do not change the upstream status.

---

## 2. MANO hand model

- **Source:** https://mano.is.tue.mpg.de/  (MPI)
- **License:** MANO custom **non-commercial research** license; registration required.
- **Do NOT redistribute** `MANO_RIGHT.pkl` / `MANO_LEFT.pkl` or derived meshes.
  Users must register and download themselves. Provide only a fetch script/instructions.

## 3. DexYCB dataset

- **Source:** https://dex-ycb.github.io/  (NVIDIA)
- **License:** DexYCB dataset license (CC BY-NC 4.0, non-commercial).
- **Do NOT redistribute** raw DexYCB data. Ship a `download_dexycb.py` only.
- Rendered result videos in this README are derived visualizations of DexYCB;
  keep them short/low-res and credit DexYCB.

## 4. ArtiMANO / retargeting recipe — ManipTrans & DexTrack

The 22-DOF ArtiMANO hand and MANO→ArtiMANO optimization recipe follow:

- **ManipTrans** (CVPR 2025) — https://github.com/ManipTrans/ManipTrans
- **DexTrack** (ICLR 2025) — https://github.com/Meowuu7/DexTrack

Cite both; check each repo's LICENSE before vendoring any assets (e.g. `rh_mano.urdf`).

## 5. Diffusion Evolution

- **DiffEvo** (Zhang et al., ICLR 2025) — used for the physics-tracking optimizer.
  Installed from PyPI (`diffevo`); cite the paper.
