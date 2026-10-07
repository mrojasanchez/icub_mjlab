# icub_mjlab: iCub Robot Learning in MuJoCo with mjlab

Reinforcement Learning (RL) environment for the **iCub** humanoid robot based on **[mjlab](https://github.com/mujocolab/mjlab)** and accelerated by **MuJoCo Warp** on NVIDIA GPUs.

This repository implements the cube manipulation and lifting task (`Mjlab-Lift-Cube-iCub`), featuring a **Teacher Fabric-Guided Policy (FGP)** inspired by **DextrAH-RGB** and **FABRIC**, built upon Isaac Lab's manager-based design pattern adapted to MuJoCo.

---

## 🚀 Prerequisites

- **Operating System:** Linux (tested on Ubuntu 22.04 / 24.04).
- **GPU:** NVIDIA GPU with CUDA support (e.g., RTX 30/40/50 series).
- **Package Manager:** [uv](https://docs.astral.sh/uv/) installed.
- **Framework:** `mjlab` workspace configured.

---

## 📦 Installation

1. **Clone this repository with submodules:**
   ```bash
   git clone --recurse-submodules https://github.com/mrojasanchez/icub_mjlab.git
   cd icub_mjlab
   ```

   > **Note:** If you already cloned without submodules, initialize them with:
   > ```bash
   > git submodule update --init --recursive
   > ```

2. **Install environment and dependencies using `uv`:**
   ```bash
   uv sync
   uv pip install -e .
   ```

---

## 🎮 Simulation and Visualization (Play Mode)

Inspect the scene, kinematics, and initial joint configurations in interactive viewer:

- **Robot in rest position (Home Pose):**
  ```bash
  uv run play Mjlab-Lift-Cube-iCub --agent zero
  ```

- **Robot with random actions:**
  ```bash
  uv run play Mjlab-Lift-Cube-iCub --agent random
  ```

- **Render a single environment:**
  ```bash
  uv run play Mjlab-Lift-Cube-iCub --agent zero --num-envs 1
  ```

---

## 🏋️ Training the Policy

Training is executed via the custom training launcher script in [`src/icub_mjlab/scripts/train.py`](src/icub_mjlab/scripts/train.py) or via the registered CLI command `train-icub`.

### 1. Hardware Presets (`--preset`)

The script provides optimized configurations for different GPU memory tiers:

| Preset | Target Hardware / Purpose | Envs (`num_envs`) | Batch / Iter | Mini-batches | VRAM Footprint |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`recommended`** *(Default)* | **RTX 5090 32GB** (optimal stability & speed) | **2048** | **49,152** | **4** (12,288 ea.) | **~16 – 18 GB** |
| **`maximum`** | **RTX 5090 32GB** (maximum throughput) | **4096** | **98,304** | **8** (12,288 ea.) | **~26 – 28 GB** |
| **`base`** | Debugging / low-memory GPUs | **64** | **1,536** | **4** (384 ea.) | **~3 – 4 GB** |

> **Note:** The `maximum` preset scales mini-batches to 8 to keep the recurrent BPTT memory buffer of the 2048-dim Critic LSTM safe within 32 GB VRAM while doubling parallel physics simulation.

### 2. Training Commands

#### A. Launch with the Recommended Preset (RTX 5090)
```bash
uv run train-icub --preset recommended
```
*Or using standard python execution:*
```bash
uv run python -m icub_mjlab.scripts.train --preset recommended
```

#### B. Launch with the Maximum Preset (4096 Parallel Envs)
```bash
uv run train-icub --preset maximum --run-name dextrah_5090_max
```

#### C. Launch with the Base Preset (Quick Debugging)
```bash
uv run train-icub --preset base
```

---

### 3. Weights & Biases (W&B) Logging & Model Upload

By default, training metrics and neural network checkpoints (`.pt`) are automatically logged and uploaded to Weights & Biases:

- **Custom project and run name:**
  ```bash
  uv run train-icub --preset recommended --wandb-project icub-dextrah --run-name dextrah_exp_01
  ```

- **Run offline (sync logs to W&B later):**
  ```bash
  uv run train-icub --preset recommended --wandb-mode offline
  ```

- **Disable W&B (fallback to local TensorBoard):**
  ```bash
  uv run train-icub --preset base --no-wandb
  ```

---

### 4. Custom Overrides and Resuming Checkpoints

You can override any hyperparameter from the CLI:

- **Override iterations, save interval, or target GPU:**
  ```bash
  uv run train-icub --preset recommended --max-iterations 20000 --save-interval 100 --gpu-id 0
  ```

- **Resume training from a local checkpoint:**
  ```bash
  uv run train-icub --preset recommended --resume --load-checkpoint model_2000.pt
  ```

- **Standard mjlab training command (alternative):**
  ```bash
  uv run train Mjlab-Lift-Cube-iCub
  ```

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
│       ├── table.xml           # Worktable XML model and textures
│       ├── actions/
│       │   └── fabric_action.py # Action manager term for Geometric Fabrics
│       ├── fabrics/
│       │   ├── core.py         # Geometric fabric formulations
│       │   ├── integrator.py   # Numerical integrators
│       │   └── terms.py        # Attraction, avoidance, and geometry terms
│       ├── scripts/
│       │   ├── __init__.py
│       │   └── train.py        # Training launcher with presets and W&B support
│       └── assets/             # Submodule with iCub MJCF models, meshes, and textures
```

---

## 📜 References and Acknowledgments

- **iCub Robot Model:** Geometries and meshes provided by [CAMADA-UTFSM/icub_mj_model](https://github.com/CAMADA-UTFSM/icub_mj_model).
- **Framework:** Developed using [mjlab](https://github.com/mujocolab/mjlab) and MuJoCo Warp.
- **Reference Papers:**
  - *DextrAH-RGB: Dexterous Manipulation with RGB Vision and Asymmetric RL*
  - *FABRIC: Geometric Fabrics for Fast and Reactive Robot Motion Planning*
