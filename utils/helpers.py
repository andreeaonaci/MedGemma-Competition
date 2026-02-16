import logging
from typing import Tuple

import numpy as np
import torch
from PIL import Image


logger = logging.getLogger(__name__)


def load_image(uploaded_file):
    """
    Load uploaded image file into PIL.Image format.

    No device placement here.
    """
    return Image.open(uploaded_file).convert("RGB")


def convert_to_tensor(image):
    """
    Convert PIL or numpy image to torch.FloatTensor.
    Automatically move to CUDA if available.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    tensor = torch.from_numpy(image).float()

    # Convert HWC → CHW
    if tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1)

    tensor = tensor.unsqueeze(0)  # Add batch dimension

    if torch.cuda.is_available():
        return tensor.to("cuda")
    else:
        logger.warning("CUDA not available. Tensor kept on CPU.")
        return tensor


def ensure_same_dimensions(image_a, image_b) -> Tuple:
    """
    Ensure both tensors have identical spatial dimensions.
    Moves both to same device safely.
    """
    if image_a.shape != image_b.shape:
        raise ValueError("Images must have same dimensions.")

    device = image_a.device
    image_b = image_b.to(device)

    return image_a, image_b
