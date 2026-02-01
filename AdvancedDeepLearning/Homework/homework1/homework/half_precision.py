# Half precision BigNet - store weights in float16 to cut memory in half
# Based on the README: use HalfLinear that inherits from nn.Linear so load_state_dict works

from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


class HalfLinear(torch.nn.Linear):
    """
    Linear layer with weights in float16. Inheriting from nn.Linear keeps param
    names (weight, bias) so we can load the bignet checkpoint without any conversion.
    We don't backprop through this layer (numerically unstable in fp16).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        # Call parent with float16 dtype so weight/bias are stored in half precision
        super().__init__(in_features, out_features, bias, device=None, dtype=torch.float16)
        # Don't train the base weights - we're doing inference-only half precision
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is float32, weights are float16 - cast for the matmul then cast output back
        out_dtype = x.dtype
        x_half = x.to(torch.float16)
        # Use no_grad so we don't backprop through the fp16 computation (unstable)
        with torch.no_grad():
            out = torch.nn.functional.linear(x_half, self.weight, self.bias)
        return out.to(out_dtype)


class HalfBigNet(torch.nn.Module):
    """
    BigNet with all linear layers in half precision. LayerNorm stays in float32
    to avoid numerical issues (as suggested in the README).
    """

    class Block(torch.nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.model = torch.nn.Sequential(
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
            )

        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> HalfBigNet:
    # You should not need to change anything here
    # PyTorch can load float32 states into float16 models
    net = HalfBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
