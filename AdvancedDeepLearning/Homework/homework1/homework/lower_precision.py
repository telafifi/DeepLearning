# Extra credit: <9MB model using hybrid 4-bit (first 4 layers) + 3-bit (rest) for accuracy

from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .low_precision import Linear4Bit


def block_quantize_3bit(x: torch.Tensor, group_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """
    3-bit block quantization: 8 levels (0..7), map to [-norm, norm].
    Pack 8 values (8*3=24 bits) into 3 bytes.
    """
    assert x.dim() == 1
    assert x.size(0) % group_size == 0
    assert group_size % 8 == 0

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values
    normalization = torch.clamp(normalization, min=1e-8)
    x_norm = (x + normalization) / (2 * normalization)
    x_quant = (x_norm * 7).round().clamp(0, 7).to(torch.int32)  # 8 levels

    # Pack 8 values of 3 bits into 3 bytes: bits 0-2 v0, 3-5 v1, 6-8 v2 (split), ...
    n_triplets = group_size // 8
    packed = torch.zeros(x_quant.size(0), n_triplets * 3, dtype=torch.uint8, device=x.device)
    for t in range(n_triplets):
        base = t * 8
        v = x_quant[:, base : base + 8]  # (n_groups, 8)
        b0 = (v[:, 0] & 7) | ((v[:, 1] & 7) << 3) | ((v[:, 2] & 3) << 6)
        b1 = ((v[:, 2] >> 2) & 1) | ((v[:, 3] & 7) << 1) | ((v[:, 4] & 7) << 4) | ((v[:, 5] & 1) << 7)
        b2 = ((v[:, 5] >> 1) & 3) | ((v[:, 6] & 7) << 2) | ((v[:, 7] & 7) << 5)
        packed[:, t * 3] = b0.to(torch.uint8)
        packed[:, t * 3 + 1] = b1.to(torch.uint8)
        packed[:, t * 3 + 2] = b2.to(torch.uint8)
    return packed, normalization.to(torch.float16)


def block_dequantize_3bit(x_quant_3: torch.Tensor, normalization: torch.Tensor) -> torch.Tensor:
    """Unpack 3 bytes to 8 values of 3 bits and dequantize."""
    assert x_quant_3.dim() == 2
    num_groups, n_bytes = x_quant_3.shape
    assert n_bytes % 3 == 0
    n_triplets = n_bytes // 3
    group_size = n_triplets * 8

    normalization = normalization.to(torch.float32)
    x_quant = x_quant_3.new_empty(num_groups, group_size, dtype=torch.int32)
    for t in range(n_triplets):
        b0 = x_quant_3[:, t * 3].to(torch.int32)
        b1 = x_quant_3[:, t * 3 + 1].to(torch.int32)
        b2 = x_quant_3[:, t * 3 + 2].to(torch.int32)
        x_quant[:, t * 8] = b0 & 7
        x_quant[:, t * 8 + 1] = (b0 >> 3) & 7
        x_quant[:, t * 8 + 2] = ((b0 >> 6) & 3) | ((b1 & 1) << 2)
        x_quant[:, t * 8 + 3] = (b1 >> 1) & 7
        x_quant[:, t * 8 + 4] = (b1 >> 4) & 7
        x_quant[:, t * 8 + 5] = ((b1 >> 7) & 1) | ((b2 & 3) << 1)
        x_quant[:, t * 8 + 6] = (b2 >> 2) & 7
        x_quant[:, t * 8 + 7] = (b2 >> 5) & 7
    x_norm = x_quant.to(torch.float32) / 7
    x = (x_norm * 2 * normalization) - normalization
    return x.view(-1)


class Linear3Bit(torch.nn.Module):
    """Linear layer with 3-bit quantized weights (<9MB total for BigNet)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 32) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size
        n_groups = (out_features * in_features) // group_size
        n_bytes = (group_size // 8) * 3
        self.register_buffer(
            "weight_q3",
            torch.zeros(n_groups, n_bytes, dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(n_groups, 1, dtype=torch.float16),
            persistent=False,
        )
        self._register_load_state_dict_pre_hook(Linear3Bit._load_state_dict_pre_hook, with_module=True)
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]
            weight_flat = weight.flatten()
            x_q3, norm = block_quantize_3bit(weight_flat, self._group_size)
            self.weight_q3.copy_(x_q3.to(self.weight_q3.dtype))
            self.weight_norm.copy_(norm.to(self.weight_norm.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            weight_flat = block_dequantize_3bit(self.weight_q3, self.weight_norm)
            weight = weight_flat.view(self._shape)
            return torch.nn.functional.linear(x, weight, self.bias)


def _linear_layer(use_4bit: bool, channels: int, group_size: int):
    """Return Linear4Bit or Linear3Bit so state_dict keys (weight) match BigNet."""
    if use_4bit:
        return Linear4Bit(channels, channels, group_size=16)
    return Linear3Bit(channels, channels, group_size=group_size)


class LowerBigNet(torch.nn.Module):
    """BigNet with hybrid 4-bit (first 4 linear layers) + 3-bit (rest) to stay <9MB with better accuracy."""

    class Block(torch.nn.Module):
        def __init__(self, channels: int, layer_types: tuple[bool, bool, bool], group_size: int = 32):
            super().__init__()
            self.model = torch.nn.Sequential(
                _linear_layer(layer_types[0], channels, group_size),
                torch.nn.ReLU(),
                _linear_layer(layer_types[1], channels, group_size),
                torch.nn.ReLU(),
                _linear_layer(layer_types[2], channels, group_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, group_size: int = 32):
        super().__init__()
        # First 4 linear layers in 4-bit (blocks 0–1 first layer), rest 3-bit; keeps memory <9MB
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, (True, True, True), group_size),   # 3 × 4-bit
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, (True, False, False), group_size), # 1 × 4-bit, 2 × 3-bit
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, (False, False, False), group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, (False, False, False), group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, (False, False, False), group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, (False, False, False), group_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None):
    net = LowerBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
