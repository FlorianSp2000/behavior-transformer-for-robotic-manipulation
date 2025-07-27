import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Optional

def load_eval_results(base_path: str = "outputs/eval") -> Dict[str, Dict]:
    """
    Load evaluation results from all model directories.
    
    Args:
        base_path: Base directory containing model evaluation folders
        
    Returns:
        Dictionary mapping model names to their evaluation results
    """
    base_path = Path(base_path)
    results = {}
    
    # Default model names to look for
    model_names = ["bet_pusht_st25k", "bet_pusht_st30k", "vqbet_pusht_st25k", "diffusion_pusht_st25k"]
    
    for model_name in model_names:
        eval_file = base_path / model_name / "eval_info.json"
        if eval_file.exists():
            try:
                with open(eval_file, 'r') as f:
                    data = json.load(f)
                    results[model_name] = data.get("aggregated", {})
                print(f"✓ Loaded results for {model_name}")
            except Exception as e:
                print(f"✗ Error loading {model_name}: {e}")
        else:
            print(f"✗ No eval_info.json found for {model_name} at {eval_file}")
    
    return results

def create_comparison_plots(results: Dict[str, Dict], save_path: Optional[str] = None):
    """
    Create clean research-style comparison plots.
    
    Args:
        results: Dictionary mapping model names to evaluation results
        save_path: Optional path to save the plot
    """
    if not results:
        print("No results to plot!")
        return
    
    # Extract metrics to plot (removed num_params_M)
    metrics = [
        ("pc_success", "Success Rate (%)", "C0"),
        ("avg_sum_reward", "Average Sum Reward", "C1"), 
        ("avg_max_reward", "Average Max Reward", "C2"),
    ]
    
    # Get model names and ensure consistent ordering
    model_names = list(results.keys())
    model_display_names = {
        "diffusion_pusht_st25k": "Diffusion Policy",
        "vqbet_pusht_st25k": "VQ-BeT", 
        "bet_pusht_st25k": "BeT (25k steps)",
        "bet_pusht_st30k": "BeT (30k steps)"
    }
    
    # Set up the plot
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))  # Reduced width since 3 plots
    if len(metrics) == 1:
        axes = [axes]
    
    # Color scheme - 4 colors for 4 models
    colors = ['#2E8B57', '#4682B4', '#DC143C', '#FF8C00']  # Green, Blue, Red, Orange
    
    for idx, (metric_key, metric_label, _) in enumerate(metrics):
        ax = axes[idx]
        
        # Extract values for this metric
        values = []
        labels = []
        colors_used = []
        
        for i, model_name in enumerate(model_names):
            if metric_key in results[model_name]:
                values.append(results[model_name][metric_key])
                labels.append(model_display_names.get(model_name, model_name))
                colors_used.append(colors[i % len(colors)])
            else:
                print(f"Warning: {metric_key} not found for {model_name}")
        
        if not values:
            continue
            
        # Create bar plot
        bars = ax.bar(labels, values, color=colors_used, 
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Customize appearance
        ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
        ax.set_title(metric_label, fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            label = f'{value:.1f}'
            
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                   label, ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Style the axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Rotate x-axis labels for better readability with 4 models
        ax.tick_params(axis='x', rotation=20)
    
    # Overall plot styling
    plt.tight_layout()
    plt.suptitle('Model Performance Comparison on PushT', 
                fontsize=16, fontweight='bold', y=1.02)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot model evaluation comparison')
    parser.add_argument('--base_path', type=str, default='outputs/eval',
                       help='Base directory containing model evaluation folders')
    parser.add_argument('--save_path', type=str, default=None,
                       help='Path to save the plot (optional)')
    parser.add_argument('--show_table', action='store_true',
                       help='Also print results table')
    
    args = parser.parse_args()
    
    # Load results
    results = load_eval_results(args.base_path)
    
    if not results:
        print("No evaluation results found!")
        return
            
    # Create main comparison plots
    save_path = args.save_path if args.save_path else "model_comparison.png"
    create_comparison_plots(results, save_path)
    
if __name__ == "__main__":
    main()