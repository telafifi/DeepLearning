"""
Implement the following models for classification.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss function

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return -torch.nn.functional.log_softmax(logits, dim=1)[range(logits.shape[0]), target].mean()


class LinearClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
            
        Define a simple linear model for image classification
        convert the input image to a 1D tensor and pass it through a linear layer
        """
        super().__init__()
        
        input_size = h * w * 3
        self.model = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Reshape the input tensor from (batch_size, 3, H, W) to (batch_size, H * W * 3)
        # this flattens the image into a long vector that can be used in a linear layer
        flattened = x.view(x.size(0), -1)  # Flatten the input
        
        # Run the model
        return self.model(flattened)


class MLPClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        An MLP (Multi-layer perceptron) with a single hidden layer.
        This will incorporate a single hidden layer between the input and output layers
        adding a non-linear transformation to the model.

        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()
        
        input_size = h * w * 3
        hidden_dim = 128
        
        # List the layers in the network to convert the input size
        # to the hidden layer size and then to the output size
        # The ReLU activation function is used to introduce non-linearity
        layers = []
        layers.append(nn.Linear(input_size, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        flattened = x.view(x.size(0), -1)  # Flatten the input
        return self.model(flattened)


class MLPClassifierDeep(nn.Module):
    class Block(nn.Module):
        """
        A simple block that includes a linear layer, layer normalization, and ReLU activation function.
        This acts as a building block for the deep MLP model by acting as a single hidden layer
        with the deep network having multiple hidden layers.
        """
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.LayerNorm(out_channels),
                nn.ReLU(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        num_layers: int = 4,
    ):
        """
        An MLP with multiple hidden layers.

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int, number of classes
            num_layers: int, number of hidden layers
        """
        super().__init__()

        input_size = 3 * h * w  # Flattened input size (3 channels for RGB)

        layers = []
        hidden_dim = 128

        # First layer (input to first hidden layer), perform a simple linear transformation
        # followed by a ReLU activation function
        layers.append(nn.Linear(input_size, hidden_dim))
        layers.append(nn.ReLU())

        # Once the initial transformation is done, add in hidden layers
        # that include a linear transformation, layer normalization, and ReLU activation function
        # via the block transformation
        for _ in range(num_layers - 1):
            layers.append(self.Block(hidden_dim, hidden_dim))

        # Output layer - finally convert the hidden layer to the output layer
        # via a linear transformation
        layers.append(nn.Linear(hidden_dim, num_classes))

        # Sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        flattened = x.view(x.size(0), -1)  # Flatten the input
        return self.model(flattened)


class MLPClassifierDeepResidual(nn.Module):
    class Block(nn.Module):
        """
        A simple block that includes a linear layer, layer normalization, and ReLU activation function.
        This block includes a skip connection to allow for residual learning.
        This acts as a building block for the deep MLP model by acting as a single hidden layer
        with the deep network having multiple hidden layers.
        """
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.LayerNorm(out_channels),
                nn.ReLU(),
            )
            
            # Skip connection to allow for residual learning
            # If the input and output dimensions are different, use a linear layer to match the dimensions
            # Otherwise, use an identity function
            if in_channels != out_channels:
                self.skip = nn.Linear(in_channels, out_channels)
            else:
                self.skip = nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.skip(x) + self.model(x)

    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 128,  # Size of hidden layers
        num_layers: int = 4,  # Number of hidden layers
    ):
        """
        An MLP with multiple hidden layers using residual connections.
        
        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()

        input_size = 3 * h * w  # Flattened input size (3 channels for RGB)

        layers = []
        hidden_dim = 128

        # First layer (input to first hidden layer), perform a simple linear transformation
        layers.append(nn.Linear(input_size, hidden_dim))

        # Once the initial transformation is done, add in hidden layers
        # that include a linear transformation, layer normalization, and ReLU activation function
        # via the block transformation with a skip connection
        for _ in range(num_layers - 1):
            layers.append(self.Block(hidden_dim, hidden_dim))

        # Output layer - finally convert the hidden layer to the output layer
        # via a linear transformation
        layers.append(nn.Linear(hidden_dim, num_classes))

        # Sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        flattened = x.view(x.size(0), -1)  # Flatten the input
        return self.model(flattened)


model_factory = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r
