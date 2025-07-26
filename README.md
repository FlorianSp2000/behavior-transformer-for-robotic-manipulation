# Behavior Transformers BeT on PushT Manipulation Task

This repository contains an implementation of the Behavior Transformer (BeT) for the PushT robotic manipulation task, built using the LeRobot framework as part of a coding challenge.

## Results

### Qualitative Results: Agent Rollouts
Below are sample rollouts after 25,000 training steps comparing our BeT implementation to VQ-BeT and Diffusion Policy baselines, which were already integrated in LeRobot.

| BeT | VQ-BeT | Diffusion Policy |
|:---:|:---:|:---:|
| ![BeT Rollout](./media/eval_bet.gif) | ![VQ-BeT Rollout](./media/eval_vqbet.gif) | ![Diffusion Policy Rollout](./media/eval_diffusion.gif) |

---

### Performance Metrics
The primary metric is the **Success Rate**, where an episode is a success if the block achieves at least 95% overlap with the target area. All policies were trained for 25k steps.

| Policy | Success Rate |
| :--- | :---: |
| **BeT (Our Implementation)** | **XX.X%** |
| VQ-BeT (Baseline) | XX.X% |
| Diffusion Policy (Baseline)| XX.X% |

*(Note: Please fill in the success rates from your evaluation runs.)*

---

## Setup & Usage

### 1. Installation
The code was tested on Ubuntu 22.04 with Python 3.10.

First clone repository with submodules

```
git clone --recursive https://github.com/FlorianSp2000/behavior-transformer-for-robotic-manipulation
```

```
# Create a virtual environment with Python 3.10 and activate it
cd lerobot
conda create -y -n lerobot python=3.10
conda activate lerobot
```

```
# Install lerobot and dependencies for PushT
pip install -e .
pip install -e ".[pusht]"
pip install -e ./lerobot
```

```
# (Optional) To use W&B:
wandb login
```

### Implementation Details
Our BeT policy is implemented in lerobot/policies/bet/ following the LeRobot framework's conventions.

- Architecture: The model uses a ResNet-18 vision backbone and a minGPT transformer, matching the approach in the VQ-BeT baseline. The transformer has one MLP prediction head for action bin classification and residual offset prediction.

- Action Discretization: We use k-means clustering to discretize the continuous action space. The k-means fitting process runs automatically for the first kmeans_fit_steps of training, collecting actions from the dataset to build the clusters.

- Normalization & Loss: The implementation uses min-max normalization for observations and actions. The total loss is a weighted sum of Focal Loss for the classification task and MSE Loss for the offset regression, as described in the original paper. (TODO: add link https://arxiv.org/pdf/2206.11251)


### Design choices and Challenges 


### Future Improvements


### Dataset: PushT Environment

**Task**: Push T-shaped object onto target area using robot end-effector
- 206 episodes, ~124 frames each at 10 FPS
- 96×96 RGB images + robot joint positions/actions
- Success criterion: ≥95% block-target overlap

 
## 🐳 Cluster Usage

For Slurm clusters:
```bash
# Build container
singularity build --fakeroot lerobot.sif lerobot.def

# Run Jupyter with GPU
singularity exec --nv --bind external/bet:/bet lerobot.sif jupyter notebook \
  --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=/bet
```

## 📈 Reproducing Results
Shell scripts are provided to run training and evaluation. Edit the scripts to change default parameters like output paths if desired. The final model configurations (config.json) are included in the repository.

1. Run baseline training: `sbatch train_baselines.sbatch`
2. Train BeT model: `sbatch bet_train.sbatch` 
3. Evaluate models: `sbatch bet_eval.sbatch`

Results saved to `outputs/train/` and `outputs/eval/` by default. Modify output paths in scripts as needed.

---

*Implementation completed as part of AI Imitation Learning coding challenge*