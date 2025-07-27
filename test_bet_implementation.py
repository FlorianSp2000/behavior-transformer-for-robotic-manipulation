"""
Test BeT Implementation
For debugging. Validates that the BeT policy can overfit a single batch.
"""

from pathlib import Path
import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.bet.configuration_bet import BeTConfig
from lerobot.policies.bet.modeling_bet import BeTPolicy


def setup_bet_model(device="cuda"):
    """Initialize BeT model and dataset for testing"""
    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    cfg = BeTConfig(
        input_features=input_features, 
        output_features=output_features,
        n_clusters=12,
        kmeans_fit_steps=300
    )
    policy = BeTPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.to(device)
    
    delta_timestamps = {
        "observation.image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }
    dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, drop_last=True
    )
    
    return policy, dataloader


def overfit_single_batch(policy, dataloader, device, num_steps=200):
    """Test if model can overfit a single batch"""
    
    batch = next(iter(dataloader))
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    
    print("=== Overfitting Single Batch ===")
    print(f"Batch size: {batch['action'].shape[0]}")
    print(f"Action shape: {batch['action'].shape}")
    print(f"K-means fitting steps: {policy.config.kmeans_fit_steps}")
    print(f"Total training steps: {policy.config.kmeans_fit_steps + num_steps}")
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    policy.train()
    losses = []
    
    total_steps = policy.config.kmeans_fit_steps + num_steps
    for step in range(total_steps):
        optimizer.zero_grad()
        loss, loss_dict = policy.forward(batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if step % 200 == 0 or step == policy.config.kmeans_fit_steps - 1:
            if step < policy.config.kmeans_fit_steps:
                print(f"K-means fitting step {step}: Loss = {loss.item():.6f}")
            else:
                print(f"Training step {step - policy.config.kmeans_fit_steps}: Loss = {loss.item():.6f}")
            
            if 'classification_loss' in loss_dict:
                print(f"  Classification: {loss_dict['classification_loss']:.6f}")
            if 'offset_loss' in loss_dict:
                print(f"  Offset (unscaled): {loss_dict['offset_loss']:.6f}")
    
    # Find where k-means fitting ended (when real training started)
    kmeans_end = policy.config.kmeans_fit_steps
    training_losses = losses[kmeans_end:]
    
    # Test final predictions
    policy.eval()
    policy.reset()
    with torch.no_grad():
        batch = policy.normalize_inputs(batch)
        batch = dict(batch)
        batch['observation.images'] = torch.stack([batch[key] for key in policy.config.image_features], dim=-4)
        batch = policy.normalize_targets(batch)
        
        action_head_output, _ = policy.bet(batch, rollout=False)
        predictions = action_head_output['predicted_action']
        targets = batch['action']
                
        # Calculate prediction accuracy
        mse = torch.nn.functional.mse_loss(predictions, targets)
        print(f"  Final prediction MSE: {mse.item():.6f}")
    
    # Report loss progression for training phase only
    print(f"\nLoss progression (training phase only):")
    print(f"  K-means fitting completed at step {kmeans_end}")
    print(f"  Training start loss: {training_losses[0]:.6f}")
    if len(training_losses) > 10:
        print(f"  After 10 training steps: {training_losses[9]:.6f}")
    if len(training_losses) > 100:
        print(f"  After 100 training steps: {training_losses[99]:.6f}")
    print(f"  Final training loss: {training_losses[-1]:.6f}")
    
    loss_reduction = (training_losses[0] - training_losses[-1]) / training_losses[0]
    print(f"  Training loss reduction: {loss_reduction:.1%}")
    
    # Determine success based on training phase only
    success = training_losses[-1] < training_losses[0] * 0.1
    
    return success


def test_bet_implementation():
    """Main test function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup model and data
    policy, dataloader = setup_bet_model(device)
    
    # Run overfitting test
    success = overfit_single_batch(policy, dataloader, device, num_steps=1000)

if __name__ == "__main__":
    test_bet_implementation()