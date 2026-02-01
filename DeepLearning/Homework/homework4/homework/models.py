from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
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
                nn.Dropout(0.1),
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
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()
        # Register input normalization parameters as buffers
        self.register_buffer('input_mean', torch.tensor(INPUT_MEAN[:2], dtype=torch.float32))
        self.register_buffer('input_std', torch.tensor(INPUT_STD[:2], dtype=torch.float32))

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        
        hidden_dim = 128
        num_layers = 5
        
        # Input dimension is n_track * 4 because:
        # - n_track points on each side
        # - 2 sides (left and right)
        # - 2 coordinates (x, y) per point
        input_dim = n_track * 4
        
        # Output dimension is n_waypoints * 2 because:
        # - n_waypoints to predict
        # - 2 coordinates (x, y) per waypoint
        output_dim = n_waypoints * 2

        layers = []
        # First layer (input to first hidden layer), perform a simple linear transformation
        layers.append(nn.Linear(input_dim, hidden_dim))

        # Once the initial transformation is done, add in hidden layers
        # that include a linear transformation, layer normalization, and ReLU activation function
        # via the block transformation with a skip connection
        for _ in range(num_layers - 1):
            layers.append(self.Block(hidden_dim, hidden_dim))

        # Output layer - finally convert the hidden layer to the output layer
        # via a linear transformation
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Sequential model
        self.model = nn.Sequential(*layers)
        

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input using predefined mean and std
        """
        return (x - self.input_mean.to(x.device)[None, None, :]) / self.input_std.to(x.device)[None, None, :]

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Get batch size and number of track points
        batch_size, n_track, _ = track_left.shape
        
        # Normalize inputs
        track_left = self.normalize_input(track_left)
        track_right = self.normalize_input(track_right)

        # Concatenate left and right track boundaries along the feature dimension
        # New shape: (b, n_track, 4)
        track_features = torch.cat([track_left, track_right], dim=-1)
        
        # Flatten the track features to prepare for the model
        # New shape: (b, n_track * 4)
        track_features = track_features.reshape(batch_size, -1)
        
        # Pass through the model to get predictions
        # Assuming self.model outputs a tensor of shape (b, n_waypoints * 2)
        waypoints_flat = self.model(track_features)
        
        # Reshape the output to get the required waypoints format
        # Final shape: (b, n_waypoints, 2)
        n_waypoints = waypoints_flat.shape[1] // 2
        waypoints = waypoints_flat.reshape(batch_size, n_waypoints, 2)
        
        return waypoints


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        # Learned query embeddings for waypoints
        self.query_embed = nn.Embedding(n_waypoints, d_model)
        
        # Linear layers to project track points to d_model dimension
        self.track_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )
        
        # Transformer decoder layer with cross-attention
        self.transformer_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=4,  # Using 4 heads since d_model = 64
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        
        # Stack multiple decoder layers
        self.transformer = nn.TransformerDecoder(
            self.transformer_layer,
            num_layers=3
        )
        
        # Output projection to predict waypoint coordinates
        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.shape[0]
        
        # Combine and encode track points
        # Shape: (b, 2*n_track, 2) -> (b, 2*n_track, d_model)
        track_points = torch.cat([track_left, track_right], dim=1)
        memory = self.track_encoder(track_points)
        
        # Get learned query embeddings
        # Shape: (n_waypoints, d_model) -> (b, n_waypoints, d_model)
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply transformer decoder
        # Shape: (b, n_waypoints, d_model)
        decoded = self.transformer(queries, memory)
        
        # Project to waypoint coordinates
        # Shape: (b, n_waypoints, 2)
        waypoints = self.output_proj(decoded)
        
        return waypoints

class CNNPlanner(torch.nn.Module):
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
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)
        
        n_blocks = 4
        in_channels = 32
        
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
        self.fcc = nn.Sequential(
            nn.Linear(c1, 256),
            nn.ReLU(),
            # Regularization - Dropout is applied to the fully connected layers to prevent overfitting
            nn.Dropout(0.3),
            nn.Linear(256, n_waypoints * 2),
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        
        z = self.network(x)
        
        # Flatten the tensor for fully connected layers
        z = z.view(z.size(0), -1)  # Flatten -> (B, 128 * 8 * 8)
        
        logits = self.fcc(z)  # -> (B, n_waypoints)
        
        # Reshape to (B, n_waypoints, 2) for final waypoint coordinates
        waypoints = logits.view(-1, self.n_waypoints, 2)

        return waypoints


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
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
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
