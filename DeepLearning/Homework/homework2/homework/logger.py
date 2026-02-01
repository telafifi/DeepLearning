from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb


def test_logging(logger: tb.SummaryWriter):
    """
    Here we will show how to log training and validation metrics to tensorboard
    to visualize the output and allow for easier debugging and comparison of different runs
    
    For training, log the training loss every iteration and the average accuracy every epoch

    For validation, log only the average accuracy every epoch

    Make sure the logging is in the correct spot so the global_step is set correctly,
    for epoch=0, iteration=0: global_step=0
    """
    # strongly simplified training loop
    global_step = 0
    for epoch in range(10):
        # Hold the metrics to render the accuracies
        metrics = {"train_acc": [], "val_acc": []}

        # example training loop
        torch.manual_seed(epoch)
        for iteration in range(20):
            dummy_train_loss = 0.9 ** (epoch + iteration / 20.0)
            dummy_train_accuracy = epoch / 10.0 + torch.randn(10)

            # log train_loss for each iteration
            logger.add_scalar('train_loss', dummy_train_loss, global_step)

            # save additional metrics to be averaged
            metrics["train_acc"].append(sum(dummy_train_accuracy) / len(dummy_train_accuracy))

            global_step += 1

        # log average train_accuracy for each epoch
        avg_train_accuracy = sum(metrics["train_acc"]) / len(metrics["train_acc"])
        logger.add_scalar('train_accuracy', avg_train_accuracy, global_step)

        # example validation loop
        torch.manual_seed(epoch)
        for _ in range(10):
            dummy_validation_accuracy = epoch / 10.0 + torch.randn(10)

            # save additional metrics to be averaged
            metrics["val_acc"].append(sum(dummy_validation_accuracy) / len(dummy_validation_accuracy))

        # log average val_accuracy for each epoch
        avg_val_accuracy = sum(metrics["val_acc"]) / len(metrics["val_acc"])
        logger.add_scalar('val_accuracy', avg_val_accuracy, global_step)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    args = parser.parse_args()

    log_dir = Path(args.exp_dir) / f"logger_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    test_logging(logger)
