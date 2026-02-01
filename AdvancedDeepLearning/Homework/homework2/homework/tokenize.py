from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .bsq import Tokenizer


def tokenize(tokenizer: Path, output: Path, *images_or_dirs: Path):
    """
    Tokenize images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    output: Path to save the tokenize image tensor.
    images: Path to the image / images to compress.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))

    # Expand directories to individual image paths
    image_paths = []
    for path in images_or_dirs:
        path = Path(path)
        if path.is_dir():
            image_paths.extend(list(path.glob("*.jpg")))
        else:
            image_paths.append(path)

    # Load and compress all images
    compressed_tensors = []
    for image_path in tqdm(image_paths):
        image = Image.open(image_path)
        x = torch.tensor(np.array(image), dtype=torch.uint8, device=device)
        with torch.inference_mode():
            x = x.float() / 255.0 - 0.5
            cmp_image = tk_model.encode_index(x)
            compressed_tensors.append(cmp_image.cpu())

    # Store the tensor in the lowest number of bits possible
    compressed_tensor = torch.stack(compressed_tensors)
    # We rely on numpy here for uint support and faster loading (not that this really matters at this size)
    np_compressed_tensor = compressed_tensor.numpy()
    if np_compressed_tensor.max() < 2**8:
        np_compressed_tensor = np_compressed_tensor.astype(np.uint8)
    elif np_compressed_tensor.max() < 2**16:
        np_compressed_tensor = np_compressed_tensor.astype(np.uint16)
    else:
        np_compressed_tensor = np_compressed_tensor.astype(np.uint32)

    torch.save(np_compressed_tensor, output)


if __name__ == "__main__":
    from fire import Fire

    Fire(tokenize)
