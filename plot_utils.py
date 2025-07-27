""" Plotting scripts for visualizing policy trajectories and action clusters in the BeT framework."""
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
import numpy as np
import torch

def create_trajectory_comparison_plot(policies, env, num_rollouts=3, max_steps=200, device="cpu"):
    """
    Takes a dictionary of policy_name: policy instance and PushT gym environment,
    runs multiple rollouts for each policy, and plots the agent trajectories in the environment.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for policy_idx, (policy_name, policy) in enumerate(policies.items()):
        ax = axes[policy_idx]
        policy.reset()
        
        # Get initial frame for background
        obs, info = env.reset(seed=42)
        initial_frame = env.render()
        ax.imshow(initial_frame, extent=[0, 680, 0, 680], alpha=0.6, origin='lower')
        
        # Run multiple trajectories for this policy
        for rollout in range(num_rollouts):
            obs, info = env.reset(seed=42)
            trajectory_x, trajectory_y = [], []
            
            for step in range(max_steps):
                state = torch.from_numpy(obs["agent_pos"])
                # rendered image is 680x680 pixels
                agent_x_pixel = state[0] * (680 / 512)
                agent_y_pixel = state[1] * (680 / 512)
                
                trajectory_x.append(agent_x_pixel)
                trajectory_y.append(agent_y_pixel)
                
                image = torch.from_numpy(obs["pixels"])    
                state = state.to(torch.float32)
                image = image.to(torch.float32) / 255
                image = image.permute(2, 0, 1)
                state = state.to(device, non_blocking=True)
                image = image.to(device, non_blocking=True)
                state = state.unsqueeze(0)
                image = image.unsqueeze(0)
                
                observation = {
                    "observation.state": state,
                    "observation.image": image,
                }
                
                with torch.inference_mode():
                    action = policy.select_action(observation)
                
                numpy_action = action.squeeze(0).to("cpu").numpy()
                obs, reward, done, _, _ = env.step(numpy_action)
                
                if done:
                    print(f"Success for: {policy_name}")
                    break
            
            # Create gradient trajectory using LineCollection
            if len(trajectory_x) > 1:
                points = np.array([trajectory_x, trajectory_y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Create purple-to-yellow gradient for all trajectories
                n_segments = len(segments)
                colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_segments))  # Purple to yellow
                
                # Create LineCollection with gradient colors
                lc = LineCollection(segments, colors=colors, linewidth=3, alpha=0.8)
                ax.add_collection(lc)
            
            # Mark start position (dark purple) and end position (yellow)
            if len(trajectory_x) > 0:
                ax.plot(trajectory_x[0], trajectory_y[0], 'o', markersize=10, 
                       color='darkviolet', alpha=0.9, label='Start' if rollout == 0 else "")
                ax.plot(trajectory_x[-1], trajectory_y[-1], 's', markersize=8, 
                       color='yellow', alpha=0.9, label='End' if rollout == 0 else "")
                
        ax.set_title(policy_name, fontsize=16, pad=20)  # Increased font size and padding
        ax.set_aspect('equal')
        ax.set_xlim(0, 680)
        ax.set_ylim(0, 680)
        ax.axis('off')  # Remove axes
        if policy_idx == 0:  # Add legend to first subplot only
            ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

"""
Example Usage
import gym_pusht  # noqa: F401
import gymnasium as gym
from pathlib import Path

env = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="pixels_agent_pos",
    max_episode_steps=300,
)
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
from lerobot.policies.bet.modeling_bet import BeTPolicy
diffusion_pretrained_policy_path = Path("outputs/train/diffusion_pusht_baseline/checkpoints/last/pretrained_model")
vqbet_pretrained_policy_path = Path("outputs/train/vqbet_pusht_baseline/checkpoints/last/pretrained_model")
bet_pretrained_policy_path = Path("lerobot/_st25000/checkpoints/last/pretrained_model")

diffusion_policy = DiffusionPolicy.from_pretrained(diffusion_pretrained_policy_path)
vqbet_policy = VQBeTPolicy.from_pretrained(vqbet_pretrained_policy_path)
bet_policy = BeTPolicy.from_pretrained(bet_pretrained_policy_path)

create_trajectory_comparison_plot({'BeT': bet_policy, "VQ-BeT": vqbet_policy, 'Diffusion Policy': diffusion_policy}, 
                                  env, num_rollouts=3, max_steps=300,
                                  device=device
                                 )
"""

def visualize_action_clusters(policy, dataloader=None, device=None, max_batches=5, figsize=(10, 4)):
    """
    Visualize k-means clusters of trained policy in action space.
    
    Args:
        policy: Trained policy with fitted k-means clusters
        dataloader: Optional dataloader for cluster utilization analysis
        device: Device for tensor operations (required if dataloader provided)
        max_batches: Number of batches to sample for utilization analysis
        figsize: Figure size tuple
    """
    
    action_head = policy.bet.action_head
    
    # Check if k-means is fitted
    if not action_head._have_fit_kmeans:
        print("❌ K-means not fitted yet! Train a few steps first.")
        return
    
    # Get unnormalized cluster centers
    cluster_centers = action_head._cluster_centers
    cluster_centers_dict = {'action': cluster_centers}
    cluster_centers = policy.unnormalize_outputs(cluster_centers_dict)['action'].cpu().numpy()
    
    # Determine plot layout
    if dataloader is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        plot_utilization = True
    else:
        fig, axes = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
        axes = [axes]  # Make it a list for consistent indexing
        plot_utilization = False
    
    # Plot 1: Cluster centers in action space
    ax = axes[0]
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
              c='red', s=60, marker='x', linewidth=2, label='Cluster Centers')
    ax.set_xlabel('Action Dim 0')
    ax.set_ylabel('Action Dim 1')
    ax.set_title('K-means Cluster Centers')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Cluster utilization (if dataloader provided)
    if plot_utilization and device is not None:
        policy.eval()
        
        # Collect actions from dataloader
        all_actions = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                actions = batch['action']
                all_actions.append(actions.reshape(-1, actions.shape[-1]))
        
        all_actions = torch.cat(all_actions, dim=0).cpu().numpy()
        
        # Add action scatter to first plot
        axes[0].scatter(all_actions[:, 0], all_actions[:, 1], 
                       alpha=0.4, s=0.5, c='blue', label='Actions')
        axes[0].legend()
        axes[0].set_title('Actions vs Cluster Centers')
        
        # Compute cluster utilization
        from scipy.spatial.distance import cdist
        distances = cdist(all_actions, cluster_centers)
        closest_clusters = distances.argmin(axis=1)
        cluster_counts = np.bincount(closest_clusters, minlength=len(cluster_centers))
        
        # Plot utilization
        ax = axes[1]
        bars = ax.bar(range(len(cluster_counts)), cluster_counts, alpha=0.7)
        ax.set_xlabel('Cluster Index')
        ax.set_ylabel('Number of Actions')
        ax.set_title('Cluster Utilization')
        ax.grid(True, alpha=0.3)
        
        # Highlight unused clusters
        unused_indices = np.where(cluster_counts == 0)[0]
        for idx in unused_indices:
            bars[idx].set_color('red')
            bars[idx].set_alpha(0.8)
        
        # Print summary statistics
        unused_clusters = len(unused_indices)
        print(f"Clusters: {len(cluster_centers)} total, {unused_clusters} unused ({unused_clusters/len(cluster_centers)*100:.1f}%)")
        print(f"Action range: [{all_actions.min():.1f}, {all_actions.max():.1f}]")
        print(f"Cluster range: [{cluster_centers.min():.1f}, {cluster_centers.max():.1f}]")
        
        if unused_clusters > len(cluster_centers) * 0.2:
            print("⚠️  Warning: >20% clusters unused - consider reducing n_clusters")
    
    else:
        # Just print cluster info
        print(f"Cluster centers: {len(cluster_centers)} total")
        print(f"Cluster range: [{cluster_centers.min():.1f}, {cluster_centers.max():.1f}]")
    
    plt.tight_layout()
    plt.show()

"""
Example usage:
# Just visualize cluster centers (lightweight):
visualize_action_clusters(policy)

# with dataloader for utilization analysis:
visualize_action_clusters(policy, dataloader, device, max_batches=10, figsize=(12, 5))
"""

def plot_training_losses(classification_losses, offset_losses, policy, 
                        window_size=100, show_original=True, figsize=(15, 5), skip_initial=1000):
    """
    Plot training loss components in separate subplots. Called in exploration notebook to visualize training progress.
    
    Args:
        classification_losses: List of classification loss values
        offset_losses: List of offset loss values  
        policy: Policy object to get offset multiplier
        window_size: Window size for moving average smoothing
        show_original: Whether to show original (noisy) loss curves
        figsize: Figure size tuple
        skip_initial: Number of initial steps to skip (default 1000, when losses are 0)
    """
    
    # Convert tensors to floats if needed
    classification_losses = [x.item() if hasattr(x, 'item') else x for x in classification_losses]
    offset_losses = [x.item() if hasattr(x, 'item') else x for x in offset_losses]
    
    # Skip initial steps where losses are 0 (before k-means fitting)
    if len(classification_losses) > skip_initial:
        classification_losses = classification_losses[skip_initial:]
        offset_losses = offset_losses[skip_initial:]
        print(f"Skipped first {skip_initial} steps (k-means fitting period)")
    else:
        print(f"Warning: Only {len(classification_losses)} steps available, not skipping any")
    
    # Scale offset losses by multiplier for total loss comparison
    scaled_offset_losses = [l * policy.bet.action_head._offset_loss_multiplier for l in offset_losses]
    total_losses = [c + o for c, o in zip(classification_losses, scaled_offset_losses)]
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f'Training Loss Analysis (Moving Average Window = {window_size})', fontsize=14)
    
    def plot_loss_component(ax, data, title, color, ylabel="Loss"):
        """Plot a single loss component with optional smoothing"""
        # Create x-axis starting from skip_initial
        steps = range(skip_initial, skip_initial + len(data))
        
        # Plot original data if requested
        if show_original:
            ax.plot(steps, data, color=color, alpha=0.3, linewidth=0.5, label='Original')
        
        # Calculate and plot smoothed data
        smoothed_data = pd.Series(data).rolling(window=window_size, min_periods=1).mean()
        ax.plot(steps, smoothed_data, color=color, linewidth=2, label='Smoothed')
        
        ax.set_title(title)
        ax.set_xlabel('Training Step')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set reasonable y-limits (exclude initial spikes)
        if len(data) > 100:
            y_max = np.percentile(data[50:], 95)  # 95th percentile after warmup
            ax.set_ylim(0, y_max * 1.1)
    
    # Plot each component
    plot_loss_component(
        axes[0], total_losses, 
        'Total Loss', 'blue'
    )
    
    plot_loss_component(
        axes[1], classification_losses, 
        'Classification Loss (Focal)', 'green'
    )
    
    plot_loss_component(
        axes[2], offset_losses, 
        f'Offset Loss (×{policy.bet.action_head._offset_loss_multiplier})', 'red'
    )
    
    plt.tight_layout()
    plt.show()

"""
Example usage:

plot_training_losses(classification_losses, offset_losses, policy, 
                    window_size=50, figsize=(18, 6), skip_initial=1000)
"""