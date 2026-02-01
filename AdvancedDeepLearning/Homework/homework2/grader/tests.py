import math
from pathlib import Path

import numpy as np
import torch

from .grader import Case, Grader

CKPT_TEMPLATE = "*_{}.pth"


class PatchAutoEncoderGrader(Grader):
    """Patch AutoEncoder"""

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    VALIDATION_LOSS_BOUND = 0.01, 0.015
    BOTTLENECK_DIM_BOUND = 256

    def load_model(self) -> torch.nn.Module:
        """
        Load a model from the checkpoint
        """
        model = self.module.ae.load().to(self.device)
        model.eval()
        return model

    def validation_step(self, model, x):
        x = x.float() / 255.0 - 0.5
        with torch.no_grad():
            z = model.encode(x)
            assert z.shape[-1] <= self.BOTTLENECK_DIM_BOUND, f"Bottleneck dimension is too large: {z.shape[-1]}"
            x_hat = model.decode(z)
            loss = torch.nn.functional.mse_loss(x_hat, x)
        return loss

    def normalize_score(self, loss, min_loss, max_loss):
        """
        Returns a score based on model's loss normalized to [0, 1]

        If the loss is less than or equal to min_loss, you get 1.0 (full score)
        If the loss is greater than or equal to max_loss, you get 0.0 (no points)
        Otherwise, score is linearly interpolated between these extremes
        """
        # Normalize so that lower loss gives higher score
        score_normalized = 1.0 - (loss - min_loss) / (max_loss - min_loss)
        return np.clip(score_normalized, 0.0, 1.0)

    @Case(score=30, timeout=50000)
    def test_validation_loss(self):
        """Image Reconstruction MSE Loss"""
        # return 1.0
        dataset = self.module.ImageDataset("valid")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4096, num_workers=4, shuffle=False)
        model = self.load_model()
        losses = []
        for x in dataloader:
            x = x.to(self.device)
            loss = self.validation_step(model, x)
            losses.append(loss.item())
        mean_loss = sum(losses) / len(losses)
        print("Validation loss:", mean_loss)
        return self.normalize_score(mean_loss, *self.VALIDATION_LOSS_BOUND)


class BSQPatchAutoEncoderGrader(PatchAutoEncoderGrader):
    """BSQ Patch AutoEncoder"""

    VALIDATION_LOSS_BOUND = 0.005, 0.01
    BOTTLENECK_SIZE_BOUND = 1200

    def load_model(self) -> torch.nn.Module:
        """
        Load a model from the checkpoint
        """
        model = self.module.bsq.load().to(self.device)
        model.eval()
        return model

    def validation_step(self, model, x):
        x = x.float() / 255.0 - 0.5
        with torch.no_grad():
            z = model.encode_index(x)
            bsq_bottleneck_size = z.shape[1] * z.shape[2]
            assert (
                bsq_bottleneck_size <= self.BOTTLENECK_SIZE_BOUND
            ), f"Bottleneck size is too large: {bsq_bottleneck_size}"
            x_hat = model.decode_index(z)
            loss = torch.nn.functional.mse_loss(x_hat, x)
        return loss


class AutoregressiveGrader(PatchAutoEncoderGrader):
    """Autoregressive Model"""

    KIND = "AutoregressiveModel"
    TOKENIZER_KIND = "BSQPatchAutoEncoder"

    COMPRESSION_RATE_BOUND = 4500, 4800
    REGRESSIVENESS_SAMPLES = 100
    REGRESSIVENESS_CHANGE_RATIO_BOUND = 0.20

    def load_models(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        """
        Load a model from the checkpoint
        """
        model = self.module.autoregressive.load().to(self.device)
        model.eval()
        tokenizer = self.module.bsq.load().to(self.device)
        tokenizer.eval()
        return model, tokenizer

    def validation_step(self, model, x):
        with torch.no_grad():
            x_hat, _ = model(x)
            loss = (
                torch.nn.functional.cross_entropy(x_hat.view(-1, x_hat.shape[-1]), x.view(-1), reduction="sum")
                / math.log(2)
                / x.shape[0]
            )
        return loss

    @Case(score=15, timeout=50000)
    def test_validation_loss(self):
        """Autoregressive prediction loss"""
        from tqdm import tqdm

        # return 1.0
        dataset = self.module.ImageDataset("valid")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)

        model, tokenizer = self.load_models()

        total_loss = 0
        for x in tqdm(dataloader, desc="Compute validation autoregressive loss"):
            x = x.float().to(self.device) / 255.0 - 0.5
            tokenized_x = tokenizer.encode_index(x)
            loss = self.validation_step(model, tokenized_x)
            total_loss += loss.item()
            print("loss:", loss.item())
        mean_loss = total_loss / len(dataloader)
        print("Validation loss:", mean_loss)
        return self.normalize_score(mean_loss, *self.COMPRESSION_RATE_BOUND)

    @Case(score=15, timeout=50000)
    def test_autoregressiveness(self):
        """Check autoregressiveness of the model"""
        # return 1.0
        import random

        model, tokenizer = self.load_models()

        dataset = self.module.ImageDataset("valid")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)

        x = next(iter(dataloader)).to(self.device)
        x = x.float().to(self.device) / 255.0 - 0.5
        x = tokenizer.encode_index(x)

        with torch.no_grad():
            pred, _ = model(x)

        cls_dim = pred.shape[-1]
        pred_flat = pred.view(x.shape[0], -1, cls_dim)

        sample_num = self.REGRESSIVENESS_SAMPLES
        x_flat = x.view(x.shape[0], -1).clone()
        x_flat_samples = x_flat.repeat(sample_num, 1)

        for bidx in range(sample_num):
            tidx = random.randint(1, pred_flat.shape[1] - 2)
            x_flat_samples[bidx, tidx] = (x_flat_samples[bidx, tidx] + 1) % max(x_flat.max() + 1, 2)

        with torch.no_grad():
            x_modified = x_flat_samples.view(sample_num, *x.shape[1:])
            pred_modified, _ = model(x_modified)
            pred_modified_flat = pred_modified.view(sample_num, -1, cls_dim)

        threshold = 1e-5
        mean_num_tokens_same = ((pred_flat - pred_modified_flat).abs() < threshold).all(-1).sum(1).float().mean()

        token_change_ratio = mean_num_tokens_same / pred_flat.shape[1]
        print(f"token change ratio: {token_change_ratio:.2f}")
        assert (
            abs(token_change_ratio - 0.5) <= self.REGRESSIVENESS_CHANGE_RATIO_BOUND
        ), f"token change ratio is too large: {token_change_ratio:.2f}"


class GenerationGrader(AutoregressiveGrader):
    """Image Generation from Autoregressive Model"""

    N_IMAGES = 8
    NLL_BOUND = 6.50

    def test_validation_loss(self):
        pass

    def test_autoregressiveness(self):
        pass

    @Case(score=10, timeout=100000)
    def test_generation(self):
        """Check image generation from the model"""
        # return 1.0
        ar_model, tk_model = self.load_models()

        dummy_index = tk_model.encode_index(torch.zeros(1, 100, 150, 3, device=self.device))
        _, h, w = dummy_index.shape

        print(f"generating {self.N_IMAGES} images from the autoregressive model")
        generations = ar_model.generate(self.N_IMAGES, h, w, device=self.device)

        # run the model on the generations to get the logits
        logits, _ = ar_model(generations)

        # compute the NLL of the generations
        nll = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]), generations.view(-1), reduction="mean"
        )
        print("Generation NLL:", nll.item())
        assert nll.item() < self.NLL_BOUND, f"Generation NLL is too high: {nll.item():.4f}"

        images = tk_model.decode_index(generations).cpu()
        # flatten the images and convert to a tensor for efficient comparison
        flattened_images = images.reshape(images.size(0), -1)
        _, counts = torch.unique(flattened_images, dim=0, return_counts=True)
        duplicate_found = (counts > 1).any().item()

        assert not duplicate_found, "Duplicate images found among the generated images."


class CompressionGrader(AutoregressiveGrader):
    """Image Compression"""

    SOURCE_IMG_DIR = "data/valid"

    COMPRESSION_RATIO_BOUND = 7.5, 14.0
    MSE_BOUND = 0.1
    NUM_SAMPLES = 5

    def test_validation_loss(self):
        pass

    def test_autoregressiveness(self):
        pass

    def normalize_score(self, ratio, min_ratio, max_ratio):
        """
        Returns a score based on model's compression ratio normalized to [0, 1]

        If the ratio is greater than or equal to max_ratio, you get 1.0 (full score)
        If the ratio is less than or equal to min_ratio, you get 0.0 (no points)
        Otherwise, score is linearly interpolated between these extremes
        """
        # Normalize so that lower loss gives higher score
        score_normalized = (ratio - min_ratio) / (max_ratio - min_ratio)
        return np.clip(score_normalized, 0.0, 1.0)

    @Case(score=5, timeout=100000)
    def test_compression(self):
        """Check image compression ratio and reconstruction quality"""
        # return 1.0
        import os
        import random

        import numpy as np
        from PIL import Image
        from tqdm import tqdm

        valid_images = [f for f in os.listdir(self.SOURCE_IMG_DIR) if f.endswith(".jpg")]
        if not valid_images:
            raise ValueError(f"No images found in {self.SOURCE_IMG_DIR}")

        compression_ratios = []
        mses = []

        # Create compressor
        ar_model, tk_model = self.load_models()
        cmp = self.module.Compressor(tk_model, ar_model)

        for _ in tqdm(range(self.NUM_SAMPLES), desc="Checking compression"):
            random_image_file = random.choice(valid_images)
            image_path = Path(self.SOURCE_IMG_DIR) / random_image_file

            # Load image and convert to tensor
            original_image = Image.open(image_path)
            x = torch.tensor(np.array(original_image), dtype=torch.uint8, device=self.device)

            # Calculate original size in KB
            original_size_kb = os.path.getsize(image_path) / 1024

            # Calculate compressed size
            compressed_bytes = cmp.compress(x.float() / 255.0 - 0.5)
            compressed_size_kb = len(compressed_bytes) / 1024

            # Decompress and calculate MSE
            decompressed_image = cmp.decompress(compressed_bytes)
            mse = torch.nn.functional.mse_loss(decompressed_image, x.float() / 255.0 - 0.5)
            compression_ratio = original_size_kb / compressed_size_kb

            compression_ratios.append(compression_ratio)
            mses.append(mse.item())

        mean_compression_ratio = sum(compression_ratios) / len(compression_ratios)
        mean_mse = sum(mses) / len(mses)

        print(f"MSE: {mean_mse:.6f}")
        assert mean_mse <= self.MSE_BOUND, f"MSE is too high: {mean_mse:.6f}"

        # Calculate compression ratio
        print(f"Compression ratio: {mean_compression_ratio:.2f}x")
        return self.normalize_score(mean_compression_ratio, *self.COMPRESSION_RATIO_BOUND)
