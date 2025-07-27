#!/bin/bash
#SBATCH --job-name=bet-train
#SBATCH --output=logs/bet/%x_%j.out
#SBATCH --error=logs/bet/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --partition=day

# Set up environment variables
export SINGULARITY_IMAGE="lerobot.sif"

# Create output directories
mkdir -p external/bet/outputs/eval

set -o allexport
source .env
set +o allexport

echo "Starting trainings for PushT..."
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Function to run training
run_train() {
    local model_name=$1
    local policy_path=$2
    local output_dir=$3
    local steps=25000
    local output_dir="${base_output_dir}_st${steps}"

    echo "================================"
    echo "Evaluating: $model_name"
    echo "Policy path: $policy_path"
    echo "Output dir: $output_dir"
    echo "================================"
    
    singularity exec --nv \
        --bind external/bet:/bet \
        --bind ~/.cache/huggingface:/root/.cache/huggingface \
        --pwd /bet/lerobot \
        ${SINGULARITY_IMAGE} \
        python -m lerobot.scripts.train \
        --policy.type="$policy_path" \
        --steps $steps \
        --batch_size 64 \
        --policy.device=cuda \
        --policy.use_amp=false \
        --policy.push_to_hub=False \
        --output_dir="$output_dir" \
        --env.type=pusht \
        --dataset.repo_id=lerobot/pusht \
        --wandb.enable=True \
        --wandb.project="bet" \
        --log_freq=1000 \
        --seed=42 \
        --save_freq=5000
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ $model_name training completed successfully"
    else
        echo "✗ $model_name training failed with exit code $exit_code"
    fi
    echo ""
}

# train DiffusionPolicy
run_train "DiffusionPolicy" \
    "diffusion" \
    "/bet/outputs/train/diffusion_pusht_baseline"

# train VQ-BeT
run_train "VQ-BeT" \
    "vqbet" \
    "/bet/outputs/train/vqbet_pusht_baseline"

echo "Job completed: $(date)"
