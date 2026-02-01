from pathlib import Path
from typing import cast

import torch
from PIL import Image

from .autoregressive import Autoregressive
from .bsq import Tokenizer


def generation(tokenizer: Path, autoregressive: Path, n_images: int, output: Path):
    """
    Tokenize images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    n_images: Number of image to generate
    output: Path to save the images
    """
    output = Path(output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))

    dummy_index = tk_model.encode_index(torch.zeros(1, 100, 150, 3, device=device))
    _, h, w = dummy_index.shape

    generations = ar_model.generate(n_images, h, w, device=device)
    images = tk_model.decode_index(generations).cpu()
    np_images = (255 * (images + 0.5).clip(0, 1)).to(torch.uint8).numpy()
    for idx, im in enumerate(np_images):
        Image.fromarray(im).save(output / f"generation_{idx}.png")


if __name__ == "__main__":
    from fire import Fire

    Fire(generation)
