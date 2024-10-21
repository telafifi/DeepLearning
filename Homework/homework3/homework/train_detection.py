import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .metrics import AccuracyMetric, DetectionMetric
from .datasets.road_dataset import load_data

def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    num_classes: int = 3,
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

    # Setup TensorBoard logging
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Load the model
    model = load_model(model_name, **kwargs)
    model.to(device)
    model.train()

    # Load the training and validation data
    train_data = load_data("road_data/train", shuffle=True, batch_size=batch_size, num_workers=2, transform_pipeline="aug")
    val_data = load_data("road_data/val", shuffle=False)

    # Loss functions and optimizer
    seg_loss_func = nn.CrossEntropyLoss()
    depth_loss_func = nn.functional.smooth_l1_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize metrics
    train_metric = DetectionMetric(num_classes=num_classes)
    val_metric = DetectionMetric(num_classes=num_classes)

    # Training loop
    for epoch in range(num_epoch):
        train_metric.reset()
        val_metric.reset()
        
        model.train()
        total_loss = 0

        # Training phase
        for batch in train_data:
            images = batch["image"].to(device)  # (B, 3, 96, 128)
            depth_gt = batch["depth"].to(device)  # (B, 96, 128)
            labels = batch["track"].to(device)  # (B, 96, 128)

            # Backward pass and optimization
            # Forward pass
            logits, raw_depth = model(images)

            # Compute losses
            seg_loss = seg_loss_func(logits, labels)
            depth_loss = depth_loss_func(raw_depth, depth_gt)
            loss = seg_loss + depth_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
        
        total_loss /= len(train_data)


        # Validation phase
        model.eval()
        with torch.no_grad():
            for batch in val_data:
                images = batch["image"].to(device)
                depth_gt = batch["depth"].to(device)
                labels = batch["track"].to(device)

                pred_labels, pred_depth = model.predict(images)

                # Update validation metrics
                val_metric.add(pred_labels, labels, pred_depth, depth_gt)

        # Compute and log metrics
        val_metrics = val_metric.compute()

        # Log metrics to TensorBoard
        logger.add_scalar("Val/IoU", val_metrics["iou"], epoch)
        logger.add_scalar("Val/Abs_Depth_Error", val_metrics["abs_depth_error"], epoch)
        logger.add_scalar("Val/TP_Depth_Error", val_metrics["tp_depth_error"], epoch)

        print(f"Epoch {epoch+1}/{num_epoch}")
        print(f"Train Loss: {total_loss:.4f}")
        print(f"Val IoU: {val_metrics['iou']:.4f}, Val Depth Error: {val_metrics['abs_depth_error']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        # Print metrics at the end of each epoch
        # print(
        #     f"Epoch {epoch + 1}/{num_epoch} "
        #     f"Val IoU: {val_metrics['iou']:.4f}, "
        #     f"Val Abs Depth Error: {val_metrics['abs_depth_error']:.4f}, "
        #     f"Val TP Depth Error: {val_metrics['tp_depth_error']:.4f}"
        # )

    # Save the model after training
    save_model(model)
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    train(**vars(parser.parse_args()))
