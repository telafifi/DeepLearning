import inspect
import math
from datetime import datetime
from pathlib import Path

import torch

from . import ae, autoregressive, bsq

patch_models = {
    n: m for M in [ae, bsq] for n, m in inspect.getmembers(M) if inspect.isclass(m) and issubclass(m, torch.nn.Module)
}

ar_models = {
    n: m
    for M in [autoregressive]
    for n, m in inspect.getmembers(M)
    if inspect.isclass(m) and issubclass(m, torch.nn.Module)
}


def train(model_name_or_path: str, epochs: int = 5, batch_size: int = 64):
    import lightning as L
    from lightning.pytorch.loggers import TensorBoardLogger

    from .data import ImageDataset, TokenDataset

    class PatchTrainer(L.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def training_step(self, x, batch_idx):
            x = x.float() / 255.0 - 0.5

            x_hat, additional_losses = self.model(x)
            loss = torch.nn.functional.mse_loss(x_hat, x)
            self.log("train/loss", loss, prog_bar=True)
            for k, v in additional_losses.items():
                self.log(f"train/{k}", v)
            return loss + sum(additional_losses.values())

        def validation_step(self, x, batch_idx):
            x = x.float() / 255.0 - 0.5

            with torch.no_grad():
                x_hat, additional_losses = self.model(x)
                loss = torch.nn.functional.mse_loss(x_hat, x)
            self.log("validation/loss", loss, prog_bar=True)
            for k, v in additional_losses.items():
                self.log(f"validation/{k}", v)
            if batch_idx == 0:
                self.logger.experiment.add_images(
                    "input", (x[:64] + 0.5).clamp(min=0, max=1).permute(0, 3, 1, 2), self.global_step
                )
                self.logger.experiment.add_images(
                    "prediction", (x_hat[:64] + 0.5).clamp(min=0, max=1).permute(0, 3, 1, 2), self.global_step
                )
            return loss

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=1e-3)

        def train_dataloader(self):
            dataset = ImageDataset("train")
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

        def val_dataloader(self):
            dataset = ImageDataset("valid")
            return torch.utils.data.DataLoader(dataset, batch_size=4096, num_workers=4, shuffle=True)

    class AutoregressiveTrainer(L.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def training_step(self, x, batch_idx):
            x_hat, additional_losses = self.model(x)
            loss = (
                torch.nn.functional.cross_entropy(x_hat.view(-1, x_hat.shape[-1]), x.view(-1), reduction="sum")
                / math.log(2)
                / x.shape[0]
            )
            self.log("train/loss", loss, prog_bar=True)
            for k, v in additional_losses.items():
                self.log(f"train/{k}", v)
            return loss + sum(additional_losses.values())

        def validation_step(self, x, batch_idx):
            with torch.no_grad():
                x_hat, additional_losses = self.model(x)
                loss = (
                    torch.nn.functional.cross_entropy(x_hat.view(-1, x_hat.shape[-1]), x.view(-1), reduction="sum")
                    / math.log(2)
                    / x.shape[0]
                )
            self.log("validation/loss", loss, prog_bar=True)
            for k, v in additional_losses.items():
                self.log(f"validation/{k}", v)
            return loss

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=1e-3)

        def train_dataloader(self):
            dataset = TokenDataset("train")
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

        def val_dataloader(self):
            dataset = TokenDataset("valid")
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    class CheckPointer(L.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            fn = Path(f"checkpoints/{timestamp}_{model_name}.pth")
            fn.parent.mkdir(exist_ok=True, parents=True)
            torch.save(model, fn)
            torch.save(model, Path(__file__).parent / f"{model_name}.pth")

    # Load or create the model
    if Path(model_name_or_path).exists():
        model = torch.load(model_name_or_path, weights_only=False)
        model_name = model.__class__.__name__
    else:
        model_name = model_name_or_path
        if model_name in patch_models:
            model = patch_models[model_name]()
        elif model_name in ar_models:
            model = ar_models[model_name]()
        else:
            raise ValueError(f"Unknown model: {model_name}")

    # Create the lightning model
    if isinstance(model, (autoregressive.Autoregressive)):
        l_model = AutoregressiveTrainer(model)
    else:
        l_model = PatchTrainer(model)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = TensorBoardLogger("logs", name=f"{timestamp}_{model_name}")
    trainer = L.Trainer(max_epochs=epochs, logger=logger, callbacks=[CheckPointer()])
    trainer.fit(
        model=l_model,
    )


if __name__ == "__main__":
    from fire import Fire

    Fire(train)
