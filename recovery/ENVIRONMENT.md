# FlowSpec-VLA environment recovery record

This record deliberately separates observed historical evidence from the current installation and from a proposed reconstruction. The archived virtual environment is not portable and is not part of this snapshot.

## A. Historical experiment runtime evidence

The byte-identical environment records in `results/frozen/gate0/environment.json` and `results/frozen/historical_invalid_phase1/environment.json`, plus the final reports, establish:

| Component | Observed value | Confidence |
|---|---|---|
| Python | 3.12.14 | observed, build hash unavailable |
| PyTorch | 2.7.1+cu118 | observed, wheel hash unavailable |
| torchvision | 0.22.1+cu118 | observed, wheel hash unavailable |
| Transformers | 5.5.4 | observed, wheel hash unavailable |
| LeRobot package | 0.6.2 | observed; source commit exactly pinned |
| hf_libero | 0.1.4 | observed |
| robosuite | 1.4.0 plus archived log-path patch | modified file exactly hashed |
| robomimic | 0.2.0 | observed |
| bddl | 1.0.1 | observed |
| Mujoco Python package | 3.8.1 | observed |
| CUDA build reported by PyTorch | 11.8 | observed |
| cuDNN Python package | 9.1.0.70 (9.1.0 family) | observed |
| Hardware | 2 x RTX 3090, 24 GiB each | observed |
| Platform | Linux 5.15.0-181-generic, x86_64, glibc 2.35 | observed |
| Rendering | headless OSMesa | observed |

The complete historical Python package list is retained in the archived environment JSON. Exact Linux distribution package versions, NVIDIA driver version, wheel hashes (except the robosuite wheel used for patch verification), OSMesa/Mesa package versions, compiler versions, and container digest were not frozen. Exact bitwise end-to-end reproduction is therefore not claimed.

## B. Current installed package state at PASS 2A

`recovery/current_installed_packages.txt` is a point-in-time `pip list --format=freeze` export from the still-present local venv. It is supporting metadata, not a portable environment lock. The venv itself contains machine-specific paths and is intentionally omitted.

## C. Proposed portable reconstruction

1. Create a fresh Python 3.12 environment on Linux x86_64.
2. Install a PyTorch 2.7.1 / torchvision 0.22.1 CUDA 11.8-compatible pair from a trusted archive, recording hashes at reconstruction time.
3. Clone LeRobot and check out the exact commit from `UPSTREAMS.yaml`; install that checkout editable or as an immutable local wheel.
4. Install the observed simulator packages: `hf_libero==0.1.4`, `robosuite==1.4.0`, `robomimic==0.2.0`, `bddl==1.0.1`, and `mujoco==3.8.1`.
5. Apply `recovery/patches/robosuite-1.4.0-log-path.patch` from the Python environment's site-packages parent, then verify the patched file SHA256 is `02be2de43b701b02b7da7f5863c733c35fa00945cc91dbb2e54e3b2413235290`.
6. Install a Mesa/OSMesa stack suitable for headless Mujoco rendering. Set `MUJOCO_GL=osmesa`. Package names differ by distribution; historical package versions are unknown.
7. Download all pinned Hugging Face assets with the exact revisions in `UPSTREAMS.yaml`.
8. Copy `recovery/path_map.example.env` to a machine-local file, fill paths, and source it. Do not commit that local file.
9. Run `recovery/verify_archive.sh` before attempting simulator or GPU reproduction.

For scientific analysis from the stored aggregate evidence, Python 3 with the standard library is sufficient; no GPU, simulator, model weights, NAS, NumPy, or raw NPZ file is required.

For full training reproduction, the six final inference weights are insufficient: the base model, recipe/model processor assets, dataset, and deterministic training implementation are required. Exact resume additionally requires the corresponding large `complete_state.pt` files, which PASS 2A inventories but intentionally does not archive.
