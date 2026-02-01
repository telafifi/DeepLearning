import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10, n_layers: int = 4):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_latent = d_latent
        self.embed = torch.nn.Embedding(n_tokens, d_latent)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=4,
            dim_feedforward=d_latent * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=False,
            norm_first=False,
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # x: (B, h, w) integer tokens
        B, h, w = x.shape
        L = h * w
        # Flatten to (B, L)
        x_flat = x.reshape(B, L)
        # Prepend start token (0): (B, L+1)
        start = torch.zeros(B, 1, dtype=torch.long, device=x.device)
        x_shifted = torch.cat([start, x_flat], dim=1)  # (B, L+1)
        # Embed: (B, L+1, d_latent)
        emb = self.embed(x_shifted)
        # Transformer expects (seq_len, batch, d_model)
        emb = emb.permute(1, 0, 2)  # (L+1, B, d_latent)
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            emb.size(0), device=x.device
        )
        out = self.transformer(emb, mask=causal_mask)  # (L+1, B, d_latent)
        # Take positions 0..L-1 to predict positions 1..L (i.e. x)
        out = out[:-1]  # (L, B, d_latent)
        out = out.permute(1, 0, 2)  # (B, L, d_latent)
        logits = self.head(out)  # (B, L, n_tokens)
        logits = logits.reshape(B, h, w, self.n_tokens)
        return logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        if device is None:
            device = next(self.parameters()).device
        L = h * w
        tokens = torch.zeros(B, 1, dtype=torch.long, device=device)  # start token
        for t in range(L - 1):
            # Build input grid: first t+1 positions filled, rest zeros
            x_grid = torch.zeros(B, h, w, dtype=torch.long, device=device)
            for i in range(t + 1):
                r, c = i // w, i % w
                x_grid[:, r, c] = tokens[:, i]
            logits, _ = self.forward(x_grid)
            # Logits at position t predict next token (position t in our 0-indexed content)
            r, c = t // w, t % w
            next_logits = logits[:, r, c, :]  # (B, n_tokens)
            next_token = torch.multinomial(
                torch.softmax(next_logits, dim=-1), num_samples=1
            ).squeeze(-1)  # (B,)
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
        # tokens is (B, L+1) with start + L tokens; drop start
        tokens = tokens[:, 1:]  # (B, L)
        return tokens.reshape(B, h, w)
