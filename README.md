# icub_mjlab: iCub Robot Learning in MuJoCo with mjlab

Reinforcement Learning (RL) environment for the **iCub** humanoid robot based on **[mjlab](https://github.com/mujocolab/mjlab)** and accelerated by **MuJoCo Warp** on NVIDIA GPUs.

This repository implements the cube manipulation and lifting task (`Mjlab-Lift-Cube-iCub`), featuring a **Teacher Fabric-Guided Policy (FGP)** inspired by **DextrAH-RGB** and **FABRIC**.

---

## 🚀 Prerequisites

- **OS:** Linux (Ubuntu 22.04 / 24.04).
- **GPU:** NVIDIA GPU with CUDA (e.g., RTX 30/40/50 series).
- **Git LFS:** [Git Large File Storage](https://git-lfs.com/) (required to pull `.stl` 3D CAD meshes).
- **Package Manager:** [uv](https://docs.astral.sh/uv/) installed.

---

## 📦 Installation

```bash
# 1. Install Git LFS
sudo apt install git-lfs && git lfs install

# 2. Clone repository with submodules
git clone --recurse-submodules https://github.com/mrojasanchez/icub_mjlab.git
cd icub_mjlab

# 3. Install dependencies in editable mode
uv sync
uv pip install -e .
```

> **Tip (Git LFS Mesh Files):**  
> The robot CAD meshes (`.stl`) in `src/icub_mjlab/assets` are tracked via Git LFS. If meshes fail to load (~130-byte files), run:
> ```bash
> git submodule update --init --recursive && git -C src/icub_mjlab/assets lfs pull
> ```

---

## 🎮 Interactive Playback (Zero-Shot & Kinematics)

Visualize the scene, kinematics, and initial joint configurations in the interactive viewer:

```bash
# Rest position (Home Pose)
uv run play Mjlab-Lift-Cube-iCub --agent zero

# Random actions
uv run play Mjlab-Lift-Cube-iCub --agent random

# Single environment
uv run play Mjlab-Lift-Cube-iCub --agent zero --num-envs 1
```

---

## 🏋️ Training the Policy

Training is executed with the `train-icub` CLI entry point (or `python -m icub_mjlab.scripts.train`).

### 1. Hardware Presets (`--preset`)

| Preset | Target Hardware / Purpose | Envs (`num_envs`) | Mini-batches | VRAM Footprint |
| :--- | :--- | :---: | :---: | :---: |
| **`recommended`** *(Default)* | **RTX 5090 32GB** (sweet spot: speed & stability) | **2048** | **8** | **~16 – 18 GB** |
| **`maximum`** | **RTX 5090 32GB** (maximum throughput) | **4096** | **8** | **~26 – 28 GB** |
| **`base`** | Low VRAM / Rapid Debugging | **64** | **4** | **~3 – 4 GB** |

### 2. Launch Training

```bash
# Default run with recommended preset (RTX 5090)
uv run train-icub --preset recommended --run-name my_experiment

# Maximum throughput preset (4096 envs)
uv run train-icub --preset maximum --run-name dextrah_max

# Debug / low VRAM
uv run train-icub --preset base --no-wandb
```

### 3. Checkpoints, W&B Logging & Video Recording

- **Automatic Best Model Tracking:** The runner automatically tracks the highest success rate (`Metrics/lift_height/episode_success`), continuously saving and uploading `model_best.pt` to Weights & Biases.
- **Storage-Friendly Sweeps & Runs:**
  - `--video`: Enables headless rollout recording.
  - `--video-only-final`: Records video only at the end of training (avoids exhausting cloud storage).
  - `--upload-only-final`: Saves and uploads only the final model and `model_best.pt`.
- **Resuming Checkpoints:**
  ```bash
  # Resume latest local checkpoint
  uv run train-icub --preset recommended --resume

  # Resume from Weights & Biases cloud artifact
  uv run train-icub --preset recommended --resume --wandb-run-path "entity/project/run_id" --wandb-checkpoint-name model_best.pt
  ```

---

## 🔬 Hyperparameter Sweeps (Weights & Biases)

Grid and Bayesian hyperparameter sweeps are configured via [`src/icub_mjlab/scripts/sweep.yaml`](src/icub_mjlab/scripts/sweep.yaml).

### 1. Initialize the Sweep
Register the sweep configuration on Weights & Biases:
```bash
uv run wandb sweep src/icub_mjlab/scripts/sweep.yaml --project icub-dextrah
```
This generates a sweep path, for example: `wandb agent <entity>/icub-dextrah/abc123xyz`.

### 2. Run the Sweep Agent
Launch the agent to execute trials on your GPU:
```bash
# Run all combinations in the sweep
uv run wandb agent <entity>/icub-dextrah/<sweep_id>

# Run a limited number of trials on this machine
uv run wandb agent <entity>/icub-dextrah/<sweep_id> --count 3
```

> **Note:** `sweep.yaml` enables `--video-only-final` and `--upload-only-final` by default, ensuring full metric telemetry in W&B while keeping storage overhead minimal across dozens of runs.

---

## 🔍 Policy Evaluation and Playback

Evaluate and visualize trained policies using the `play` CLI:

```bash
# Play from a local checkpoint
uv run play Mjlab-Lift-Cube-iCub --checkpoint-file logs/rsl_rl/dextrah_teacher_fgp/<run>/model_best.pt

# Play directly from Weights & Biases
uv run play Mjlab-Lift-Cube-iCub --wandb-run-path "entity/project/run_id" --wandb-checkpoint-name model_best.pt
```

### Evaluation Options

| Option | Command Flag | Description |
| :--- | :--- | :--- |
| **Viewer Backend** | `--viewer viser` / `--viewer native` | Web-based interactive viewer ([Viser](https://viser.studio)) or native GLFW. |
| **Parallel Envs** | `--num-envs 16` | Render multiple environments simultaneously to test robustness. |
| **Record Video** | `--video --video-length 500` | Render an offscreen evaluation video saved to `<log_dir>/videos/play/`. |

---

## 📂 Project Structure

```text
icub_mjlab/
├── pyproject.toml              # Package definition and CLI entry points (train-icub)
├── README.md                   # Project documentation
├── src/
│   └── icub_mjlab/
│       ├── __init__.py         # Mjlab-Lift-Cube-iCub task registration
│       ├── env_cfg.py          # MDP definition (scene, rewards, observations, events)
│       ├── icub_constants.py   # Kinematic limits, home pose, and action calibration
│       ├── observations.py     # Asymmetric Actor-Critic observation terms
│       ├── rewards.py          # DextrAH-RGB & FABRIC reward specifications
│       ├── rl_cfg.py           # PPO runner configuration and recurrent LSTM architectures
│       ├── rl_runner.py        # Custom RSL-RL runner (model_best.pt tracking & W&B export)
│       ├── table.xml           # Worktable XML model and textures
│       ├── dextrah_rgb/        # Modular Geometric Fabrics & Fabric Action subpackage
│       │   ├── __init__.py     # Re-exports FabricAction, FabricActionCfg, and Fabric terms
│       │   ├── actions/
│       │   │   ├── __init__.py
│       │   │   └── fabric_action.py # Action manager term for Geometric Fabrics
│       │   └── fabrics/
│       │       ├── __init__.py
│       │       ├── core.py         # Geometric fabric formulations
│       │       ├── integrator.py   # Numerical integrators
│       │       └── terms.py        # Attraction, avoidance, and geometry terms
│       ├── scripts/
│       │   ├── __init__.py
│       │   ├── sweep.yaml      # Weights & Biases hyperparameter sweep configuration
│       │   └── train.py        # Training launcher with presets, sweeps, and W&B support
│       └── assets/             # Submodule with iCub MJCF models, meshes, and textures
```

---

## 📜 References and Acknowledgments

- **iCub Robot Model:** Geometries and meshes from [CAMADA-UTFSM/icub_mj_model](https://github.com/CAMADA-UTFSM/icub_mj_model).
- **Framework:** Built on [mjlab](https://github.com/mujocolab/mjlab) and MuJoCo Warp.
- **Reference Papers:**
  - *DextrAH-RGB: Dexterous Manipulation with RGB Vision and Asymmetric RL*
  - *FABRIC: Geometric Fabrics for Fast and Reactive Robot Motion Planning*
