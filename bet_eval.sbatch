#!/bin/bash
#SBATCH --job-name=lerobot_eval
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

echo "Starting baseline evaluations for PushT..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Function to run evaluation
run_eval() {
    local model_name=$1
    local policy_path=$2
    local output_dir=$3
    
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
        python -m lerobot.scripts.eval \
        --policy.path="$policy_path" \
        --policy.device=cuda \
        --policy.use_amp=false \
        --output_dir="$output_dir" \
        --env.type=pusht \
        --eval.n_episodes=500 \
        --eval.batch_size=50 
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ $model_name evaluation completed successfully"
    else
        echo "✗ $model_name evaluation failed with exit code $exit_code"
    fi
    echo ""
}

# Evaluate DiffusionPolicy
# run_eval "DiffusionPolicy" \
#     "/bet/outputs/train/diffusion_pusht_baseline/checkpoints/last/pretrained_model" \
#     "/bet/outputs/eval/diffusion_pusht_st25k"

# Evaluate VQ-BeT  
# run_eval "VQ-BeT" \
#     "/bet/outputs/train/vqbet_pusht_baseline/checkpoints/last/pretrained_model" \
#     "/bet/outputs/eval/vqbet_pusht_st25k"

# Evaluate BeT
# "/bet/outputs/train/test_bet_30k_off1000" \
run_eval "BeT" \
    "/bet/lerobot/_st25000/checkpoints/last/pretrained_model" \
    "/bet/outputs/eval/bet_pusht_st25k"

echo "Job completed: $(date)"
