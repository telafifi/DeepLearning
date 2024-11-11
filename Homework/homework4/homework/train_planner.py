import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .metrics import PlannerMetric
from .datasets.road_dataset import load_data

def masked_mse_loss(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked MSE loss for waypoint prediction
    
    Args:
        preds: Predicted waypoints (batch_size, n_waypoints, 2)
        targets: Target waypoints (batch_size, n_waypoints, 2) 
        mask: Boolean mask (batch_size, n_waypoints)
    
    Returns:
        Masked MSE loss averaged over valid points
    """
    # Convert mask to float and add coordinate dimension
    mask = mask.float().unsqueeze(-1)  # Add dimension for coordinates
    
    # Only compute loss for valid waypoints
    squared_diff = (preds - targets) ** 2
    masked_diff = squared_diff * mask  # Zero out invalid waypoints
    
    # Average only over valid points
    valid_points = mask.sum() + 1e-6  # Add small epsilon to avoid division by zero
    loss = masked_diff.sum() / valid_points
    
    return loss

def masked_l1_loss(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked L1 loss for waypoint prediction
    
    Args:
        preds: Predicted waypoints (batch_size, n_waypoints, 2)
        targets: Target waypoints (batch_size, n_waypoints, 2) 
        mask: Boolean mask (batch_size, n_waypoints)
    
    Returns:
        Masked L1 loss averaged over valid points, with longitudinal error weighted double
    """
    # Convert mask to float and add coordinate dimension
    mask = mask.float().unsqueeze(-1)  # (batch_size, n_waypoints, 1)
    
    # Compute L1 (absolute) differences
    l1_diff = torch.abs(preds - targets)  # (batch_size, n_waypoints, 2)
    
    # Apply mask
    masked_diff = l1_diff * mask
    
    # Compute separate longitudinal (x) and lateral (y) errors
    longitudinal_loss = masked_diff[..., 0].sum() / (mask[..., 0].sum() + 1e-6)
    lateral_loss = masked_diff[..., 1].sum() / (mask[..., 0].sum() + 1e-6)
    
    # Weight longitudinal error double
    total_loss = longitudinal_loss + lateral_loss
    
    return total_loss

def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")  # For ARM Macs
    else:
        print("GPU not available, using CPU")
        device = torch.device("cpu")
    
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    #Setup TensorBoard logging
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)
    
    # Load the model
    model = load_model(model_name, **kwargs)
    model.to(device)
    model.train()
    
    # Load the training and validation data
    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2, transform_pipeline="default")
    val_data = load_data("drive_data/val", shuffle=False)
    
    # Create loss function and optimizer
    loss_func = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Need to initialize metrics
    global_step = 0
    metrics = PlannerMetric()
    
    # Training loop
    for epoch in range(num_epoch):
        # Need to reset metrics
        # clear metrics at beginning of epoch
        metrics.reset()
        
        model.train()
        total_loss = 0
        
        for batch in train_data:
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)
            optimizer.zero_grad()
            
            preds = model(track_left, track_right, **kwargs)
            loss = masked_l1_loss(preds, waypoints, waypoints_mask)
            
            loss.backward()
            optimizer.step()

            global_step += 1
            total_loss += loss.item()
            
        total_loss /= len(train_data)
            
        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                # Load the image and label to the device
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                # Forward pass through the model and calculate the accuracy
                # No need to calculate the loss since we are only interested in the accuracy
                # as this is the validation set
                preds = model(track_left, track_right, **kwargs)
                # Update metrics
                metrics.add(preds, waypoints, waypoints_mask)

        val_metrics = metrics.compute()
        logger.add_scalar("L1_Error", val_metrics['l1_error'], epoch)
        logger.add_scalar("Longitudinal_Error", val_metrics['longitudinal_error'], epoch)
        logger.add_scalar("Lateral_Error", val_metrics['lateral_error'], epoch)
        logger.add_scalar("Num_Samples", val_metrics['num_samples'], epoch)

        # print on first, last, every 10th epoch
        # if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"Train Loss: {total_loss:.4f} "
            f"val_L1_err={val_metrics['l1_error']:.4f} "
            f"val_Long_err={val_metrics['longitudinal_error']:.4f} "
            f"val_Lat_err={val_metrics['lateral_error']:.4f} "
            f"val_Num_Samples={val_metrics['num_samples']:.4f} "
        )
        
    # save and overwrite the model in the root directory with the final model
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=70)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=2024)

    train(**vars(parser.parse_args()))