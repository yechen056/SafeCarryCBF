<div align="right">
  🌐 <strong>English</strong></a>
  |
  <a href="./README_zh.md">简体中文</a>
</div>

<h1 align="center">FlowCarryCBF</h1>
<h2 align="center">Flow Matching with Predictive Whole-Body CBF for Safe Mobile Bimanual Manipulation</h2>

## 📖 Introduction

**FlowCarryCBF** is a vision-based framework for safe mobile bimanual manipulation in dynamic environments. It combines Conditional Flow Matching for coordinated mobile manipulation with a predictive whole-body CBF-QP for safe, obstacle-aware execution.

## 🖥️ Demos

<p align="center">
  <img src="docs/demos.gif" alt="demos">
</p>

## 🛠️ Installation

### 1. Quick installation

```bash
cd FlowCarryCBF
conda env create -f environment.yml

conda activate flowcarrycbf
pip install -e . --no-deps
```

### 2. Manual installation (Optional)

```bash
conda create -n flowcarrycbf python=3.11 pinocchio=2.7.0 -c conda-forge
conda activate flowcarrycbf

pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0.0" \
  --extra-index-url https://pypi.nvidia.com
pip install -r requirements.txt
pip install -e . --no-deps

pip install "git+https://github.com/iROSA-lab/mushroom-rl.git@20e6ca13f367c9f751dbb0c593a05b80d00b5fe7"
pip install "git+https://github.com/mbreyer/robot_helpers.git@ba6e7afeda5e74afdf509384d4f3aad895396b19"

#Accept the NVIDIA EULA before the first Isaac Sim:
export OMNI_KIT_ACCEPT_EULA=YES
```

## 🚀 Quick Start

### 1. 🕹️ Data Collection

The collector first prescreens reproducible Oracle plans and then records only trajectories that succeed during PhysX replay:

```bash
bash scripts/collect_data.sh \
  --episodes 100 \
  --template-ids dual_arm_easy,dual_joint_easy,cube_arm_easy,cube_joint_easy \
  --output data/raw_4tasks
```

To resume an interrupted collection, rerun the same command with `--resume`.

### 2. 🗂️ Data Preprocessing

```bash
bash scripts/convert_data.sh \
  --source data/raw_4tasks \
  --output data/processed_4tasks.zarr \
  --episodes 100
```

### 3. 🚆 Training

```bash
bash scripts/train_policy.sh \
  --data data/processed_4tasks.zarr \
  --output checkpoints/flow_4tasks.pt
```

### 4. 🦾 Evaluation

```bash
bash scripts/eval_policy.sh \
  --model checkpoints/flow_4tasks.pt \
  --category all \
  --runs 1 \
  --seed 0 \
  --methods flow,flow_cbf \
  --workers 4 \
  --output-dir outputs/eval_all_tasks
```

- `--category all` evaluates all 12 templates: four `easy`, four `stress`, and four `hard`.
- `--runs 1` runs one deterministic seed for each selected template.
- `--workers 4` runs four independent Isaac Sim processes in parallel (default: `1`; reduce if resources are limited).
- Add `--record-video` to save first-person and overview videos for every episode.

We provide the [model (download)](https://huggingface.co/yechen056/flow_cbf/resolve/main/flow_4tasks.pt?download=true) on Hugging Face:
> 💡 Extract the model to `checkpoints/`

The project contains four task families across three levels: `Easy`, `Stress`, and `Hard`.

| Category | Dynamic Tasks | Mixed Dynamic–Static Tasks |
| :--- | :--- | :--- |
| Easy | `dual_arm_easy`, `dual_joint_easy` | `cube_arm_easy`, `cube_joint_easy` |
| Stress | `dual_arm_stress`, `dual_joint_stress` | `cube_arm_stress`, `cube_joint_stress` |
| Hard | `dual_arm_hard`, `dual_joint_hard` | `cube_arm_hard`, `cube_joint_hard` |


## 📦 Data Format

### TIAGo raw HDF5

| Field | dtype | Shape |
| :--- | :--- | :--- |
| `rgb` | `uint8` | `(T, 2, 480, 640, 3)` |
| `proprio` | `float32` | `(T, 17)` |
| `actions` | `float32` | `(T, 17)` |

### Training Zarr

| Field | dtype | Shape |
| :--- | :--- | :--- |
| `rgb` | `uint8` | `(N, 2, 240, 320, 3)` |
| `proprio` | `float32` | `(N, 17)` |
| `actions` | `float32` | `(N, 17)` |
| `episode_ends` | `int64` | `(E,)` |

## 🗂️ Repository Structure

```text
FlowCarryCBF/
├── flowcarrycbf/
│   ├── cli/                         # Collection, conversion, training, evaluation
│   ├── config/                      # Simulation runtime configuration
│   ├── envs/tasks/                  # Isaac Sim TIAGo carrying tasks
│   ├── policies/flowcarry_cbf/      # Flow policy, perception, kinematics, CBF-QP
│   └── robots/                      # Robot handlers and assets
├── scripts/                         # Unified shell entry points
├── environment.yml
├── requirements.txt
└── pyproject.toml
```

# 📜 License

This project is released under the [MIT License](LICENSE.txt).

# 🙏 Acknowledgements

This work builds upon excellent open-source projects including [Flow Matching Policy](https://hri-eu.github.io/flow-matching-policy/), [SafeFlowMPC](https://github.com/TU-Wien-ACIN-CDS/SafeFlowMPC), [SafeFlowMatcher](https://github.com/takahashi-seiryu/SafeFlowMatcher), [UR5e-DP-Family](https://github.com/yechen056/UR5e-DP-Family), and [ActPerMoMa](https://github.com/pearl-robot-lab/ActPerMoMa). We thank the authors and maintainers for their contributions.
