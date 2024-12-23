from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    class Block(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
            super().__init__()
            padding = (kernel_size - 1) // 2
            
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
            
            self.model = nn.Sequential(
                self.conv,
                self.bn,
                self.relu,
            )
            
            # Skip connection to allow for residual learning
            # If the input and output dimensions are different, use a linear layer to match the dimensions
            # Otherwise, use an identity function
            if in_channels != out_channels:
                self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            else:
                self.skip = nn.Identity()

        def forward(self, x):
            # Only apply pooling if spatial dimensions are large enough
            x = self.skip(x) + self.model(x)
                
            return x

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))
        
        n_blocks = 4
        
        # Convolutional layers, starting with a large kernel size to capture global features
        # Followed by a series of blocks with increasing number of channels
        cnn_layers = [
            torch.nn.Conv2d(3, in_channels, kernel_size=11, stride=2, padding=5),
            torch.nn.ReLU(),
        ]
        
        # Add blocks with increasing number of channels
        c1 = in_channels
        for _ in range(n_blocks):
            c2 = c1 * 2
            cnn_layers.append(self.Block(c1, c2, kernel_size=3, stride=2))
            c1 = c2
            
        # Perform convolution with a 1x1 kernel and then global average pooling
        # to allow for classification
        cnn_layers.append(torch.nn.Conv2d(c1, c1, kernel_size=1))
        cnn_layers.append(torch.nn.AdaptiveAvgPool2d(1))
        self.network = torch.nn.Sequential(*cnn_layers)
        
        # Fully connected layers
        self.fc1 = nn.Linear(c1, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Regularization - Dropout is applied to the fully connected layers to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Normalize the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        
        z = self.network(z)
        
        # Flatten the tensor for fully connected layers
        z = z.view(z.size(0), -1)  # Flatten -> (B, 128 * 8 * 8)
        
        # Fully connected layers with dropout
        z = F.relu(self.fc1(z))  # -> (B, 256)
        z = self.dropout(z)
        
        logits = self.fc2(z)  # -> (B, num_classes)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class ConvBlock(nn.Module):
    """
    A Convolutional Block with Convolution, Batch Normalization, and ReLU Activation
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UpConvBlock(nn.Module):
    """
    A convolution block that performs up-sampling using ConvTranspose2d, thereby increasing the spatial dimensions
    of the input tensor.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.upconv(x)))

class Encoder(nn.Module):
    """
    An encoder that downsamples the input image to extract features
    using three separate layers with increasing number of channels
    """
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.down1 = ConvBlock(in_channels, 16, stride=2)  # (B, 16, 48, 64)
        self.down2 = ConvBlock(16, 32, stride=2)  # (B, 32, 24, 32)
        self.down3 = ConvBlock(32, 64, stride=2)  # (B, 64, 24, 32)
        
    def forward(self, x):
        d1 = self.down1(x)  # (B, 16, 48, 64)
        d2 = self.down2(d1)  # (B, 32, 24, 32)
        d3 = self.down3(d2)  # (B, 64, 24, 32)
        
        return d1, d2, d3
    
class Decoder(nn.Module):
    """
    A decoder that upsamples the features extracted by the encoder to generate
    segmentation masks and depth maps
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.up1 = UpConvBlock(64, 32)  # (B, 32, 48, 64)
        self.skip1 = ConvBlock(64, 32)  # Combine up1 and down2
        self.up2 = UpConvBlock(32, 16)  # (B, 16, 96, 128)
        self.skip2 = ConvBlock(32, 16)  # Combine up2 and down1
        self.up3 = UpConvBlock(16, 16)  # (B, 16, 96, 128)
        
    def forward(self, d1, d2, d3):
        u1 = self.up1(d3)  # (B, 32, 48, 64)
        skip1 = self.skip1(torch.cat([u1, d2], dim=1))  # Combine up1 and down1
        u2 = self.up2(skip1)  # (B, 16, 96, 128) 
        skip2 = self.skip2(torch.cat([u2, d1], dim=1))  # Combine up1 and down1
        u3 = self.up3(skip2)  # (B, 16, 96, 128) 
        
        return u3

class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))
        
        # Encoder: Down-sampling layers
        self.encoder = Encoder(in_channels)

        # Decoder: Up-sampling layers
        self.decoder = Decoder()

        # Segmentation Head: Predict 3 class logits
        self.seg_norm = nn.BatchNorm2d(16)
        self.seg_head = nn.Conv2d(16, num_classes, kernel_size=1)  # (B, 3, 96, 128)

        # Depth Head: Predict 1-channel depth map
        self.depth_head = nn.Conv2d(16, 1, kernel_size=1)  # (B, 1, 96, 128)
        
        # Initialize segmentation head
        nn.init.xavier_uniform_(self.seg_head.weight)
        nn.init.constant_(self.seg_head.bias, 0)
        nn.init.xavier_uniform_(self.depth_head.weight)
        nn.init.constant_(self.depth_head.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Encoder
        d1, d2, d3 = self.encoder(z)

        
        # Decoder Segmentation
        seg_dec = self.decoder(d1, d2, d3)
        
        # Output Heads
        logits = self.seg_head(self.seg_norm(seg_dec))  # (B, 3, 96, 128)
        
        # Decoder Depth
        depth_dec = self.decoder(d1, d2, d3)
        
        # Output Heads
        depth = self.depth_head(depth_dec)  # (B, 1, 96, 128)

        return logits, torch.sigmoid(depth).squeeze(1)  # Constrain depth to [0, 1]

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        
        pred = logits.argmax(dim=1)

        return pred, raw_depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
