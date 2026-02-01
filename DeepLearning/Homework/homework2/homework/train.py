import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import ClassificationLoss, load_model, save_model
from .utils import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 60,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps") # for Arm Macs
    else:
        print("GPU not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Load in a model and move it to the device to utilize GPU
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    # Load in training and validation data
    train_data = load_data("classification_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("classification_data/val", shuffle=False)

    # Create loss function and optimizer
    loss_func = ClassificationLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        # Iterate through each element in the training data
        for img, label in train_data:
            # Load the image and label to the device
            img, label = img.to(device), label.to(device)

            # Forward pass through the model and calculate the loss
            preds = model(img)
            loss = loss_func(preds, label)

            # Backward pass to calculate the gradients
            optimizer.zero_grad()
            loss.backward()
            
            # Update the model weights using the optimizer
            optimizer.step()

            # Calculate the accuracy and append to the metrics
            # This is done to calculate the average accuracy at the end of the epoch
            acc = (preds.argmax(dim=1) == label).float().mean().item()
            metrics["train_acc"].append(acc)

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for img, label in val_data:
                # Load the image and label to the device
                img, label = img.to(device), label.to(device)

                # Forward pass through the model and calculate the accuracy
                # No need to calculate the loss since we are only interested in the accuracy
                # as this is the validation set
                preds = model(img)
                acc = (preds.argmax(dim=1) == label).float().mean().item()
                metrics["val_acc"].append(acc)

        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        logger.add_scalar("Train/Accuracy", epoch_train_acc, epoch)
        logger.add_scalar("Val/Accuracy", epoch_val_acc, epoch)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
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
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=4)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
