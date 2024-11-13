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

def masked_l1_loss(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
   """
   Compute masked L1 loss for waypoint prediction, weighting longitudinal errors more heavily.
   Longitudinal refers to forward/backward error while lateral refers to left/right error.
   
   Args:
       preds: Predicted waypoints with shape (batch_size, n_waypoints, 2)
              Where the last dimension contains (x,y) coordinates
       targets: Ground truth waypoints with shape (batch_size, n_waypoints, 2)
               Same format as predictions
       mask: Boolean mask indicating valid waypoints with shape (batch_size, n_waypoints)
             Used to ignore invalid/missing waypoints in loss calculation
   
   Returns:
       torch.Tensor: Scalar loss value combining longitudinal and lateral errors,
                    averaged over all valid waypoints
   """
   # Convert boolean mask to float and add dimension for coordinates
   # Shape goes from (batch_size, n_waypoints) -> (batch_size, n_waypoints, 1)
   # This allows broadcasting when multiplying with coordinate-wise differences
   mask = mask.float().unsqueeze(-1)  
   
   # Compute absolute differences between predictions and targets
   # Result shape: (batch_size, n_waypoints, 2)
   # Where [..., 0] is longitudinal (x) error and [..., 1] is lateral (y) error
   l1_diff = torch.abs(preds - targets)
   
   # Apply mask to ignore invalid waypoints
   # Only include errors for waypoints where mask is True
   masked_diff = l1_diff * mask
   
   # Compute average longitudinal (x) and lateral (y) errors separately
   # Sum errors and divide by number of valid points
   # Add small epsilon (1e-6) to avoid division by zero
   longitudinal_loss = masked_diff[..., 0].sum() / (mask[..., 0].sum() + 1e-6)  # x-axis error
   lateral_loss = masked_diff[..., 1].sum() / (mask[..., 0].sum() + 1e-6)      # y-axis error
   
   # Combine losses, double-weighting the longitudinal error
   # This puts more emphasis on predicting the forward/backward position correctly
   # compared to the left/right position
   total_loss = longitudinal_loss + lateral_loss
   
   return total_loss

def train_step(model, train_data, optimizer, device, **kwargs):
    """
    Performs one training epoch over the provided data.
    
    Args:
        model: The neural network model to train
        train_data: DataLoader containing training batches
        optimizer: The optimizer used for updating model weights
        device: Device to run computations on (cuda/cpu)
        **kwargs: Additional arguments to pass to the model's forward pass
    
    Returns:
        float: Average loss value across all batches in the epoch
    """
    # Set model to training mode - enables dropout, batch norm updates etc.
    model.train()
    
    # Initialize total loss accumulator for the epoch
    total_loss = 0
    
    # Iterate through batches in the training data
    for batch in train_data:
        # Move batch data to appropriate device (GPU/CPU)
        track_left = batch["track_left"].to(device)      # Left lane boundaries
        track_right = batch["track_right"].to(device)    # Right lane boundaries
        waypoints = batch["waypoints"].to(device)        # Target waypoints
        waypoints_mask = batch["waypoints_mask"].to(device)  # Mask for valid waypoints
        
        # Zero out gradients from previous batch
        # This is necessary because PyTorch accumulates gradients
        optimizer.zero_grad()
        
        # Forward pass: compute model predictions
        # **kwargs allows passing additional arguments to model.forward()
        preds = model(track_left, track_right, **kwargs)
        
        # Compute loss using masked L1 loss
        # This ignores invalid waypoints during loss calculation
        loss = masked_l1_loss(preds, waypoints, waypoints_mask)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update model weights using the optimizer
        optimizer.step()
        
        # Accumulate batch loss
        total_loss += loss.item()  # .item() gets the scalar value from the loss tensor
    
    # Compute average loss across all batches
    total_loss /= len(train_data)
    
    return total_loss
    
def validation_step(model, val_data, metrics, device, **kwargs):
    """
   Performs validation on the model using validation data. 
   Calculates performance metrics without updating model weights.
   
   Args:
       model: The neural network model to evaluate
       val_data: DataLoader containing validation batches 
       metrics: Object that tracks and computes evaluation metrics
       device: Device to run computations on (cuda/cpu)
       **kwargs: Additional arguments to pass to model's forward pass
   """
   # Context manager that disables gradient computation
   # This saves memory and speeds up validation since we don't need gradients
    with torch.inference_mode():
       # Set model to evaluation mode - disables dropout, uses running stats for batch norm
       model.eval()

       # Iterate through validation batches
       for batch in val_data:
           # Move batch data to appropriate device (GPU/CPU)
           track_left = batch["track_left"].to(device)      # Left lane boundaries
           track_right = batch["track_right"].to(device)    # Right lane boundaries
           waypoints = batch["waypoints"].to(device)        # Ground truth waypoints 
           waypoints_mask = batch["waypoints_mask"].to(device)  # Mask for valid waypoints

           # Forward pass only - no loss calculation or backprop needed
           # We only want to evaluate model predictions during validation
           preds = model(track_left, track_right, **kwargs)
           
           # Update running metrics (e.g., longitudinal/lateral errors)
           # metrics.add() accumulates statistics across all validation batches
           # These will be used to compute final validation metrics after all batches
           metrics.add(preds, waypoints, waypoints_mask)

       # Note: Final metric computation typically happens outside this function
       # by calling metrics.compute() after all validation batches are processed

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Need to initialize metrics
    metrics = PlannerMetric()
    
    # Training loop
    for epoch in range(num_epoch):
        # Need to reset metrics
        # clear metrics at beginning of epoch
        metrics.reset()
        
        total_loss = train_step(model, train_data, optimizer, device, **kwargs)
            
        validation_step(model, val_data, metrics, device, **kwargs)

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
    # MLP Number of epochs
    # parser.add_argument("--num_epoch", type=int, default=70)
    #Transformer Number of epochs
    parser.add_argument("--num_epoch", type=int, default=30)
    # MLP Learning rate
    # parser.add_argument("--lr", type=float, default=5e-3)
    # Transformer Learning rate
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    train(**vars(parser.parse_args()))