import numpy as np
import torch
import torch.nn.functional as F


def _to_numpy(x):
    """
    Convert torch tensor or numpy array to numpy array.
    Moves tensor to CPU if necessary.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _normalize_similarity(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize value to 0–1 range.
    """
    if max_val - min_val == 0:
        return 0.0
    return float((value - min_val) / (max_val - min_val))


def ssim_similarity(a, b):
    """
    Compute Structural Similarity Index (SSIM).
    Returns normalized score in range [0, 1].
    """

    a = _to_numpy(a).astype(np.float64)
    b = _to_numpy(b).astype(np.float64)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_a = a.mean()
    mu_b = b.mean()

    sigma_a = a.var()
    sigma_b = b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()

    numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2)

    ssim = numerator / denominator if denominator != 0 else 0.0

    # SSIM theoretical range [-1, 1] → normalize to [0, 1]
    return _normalize_similarity(ssim, -1, 1)


def cosine_similarity(a, b):
    """
    Compute cosine similarity between flattened tensors.
    Returns normalized score [0, 1].
    """

    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)

    a = a.flatten().float()
    b = b.flatten().float()

    sim = F.cosine_similarity(a, b, dim=0).item()

    # cosine range [-1, 1] → normalize
    return _normalize_similarity(sim, -1, 1)


def l2_distance(a, b):
    """
    Compute L2 distance.
    Returns similarity score normalized to [0, 1].
    """

    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)

    a = a.flatten().float()
    b = b.flatten().float()

    distance = torch.norm(a - b, p=2).item()

    # Convert distance to similarity
    similarity = 1.0 / (1.0 + distance)

    return float(similarity)


def mse_similarity(a, b):
    """
    Compute Mean Squared Error.
    Returns similarity score normalized to [0, 1].
    """

    a = _to_numpy(a).astype(np.float64)
    b = _to_numpy(b).astype(np.float64)

    mse = np.mean((a - b) ** 2)

    similarity = 1.0 / (1.0 + mse)

    return float(similarity)


def pearson_correlation(a, b):
    """
    Compute Pearson correlation coefficient.
    Returns normalized score [0, 1].
    """

    a = _to_numpy(a).flatten()
    b = _to_numpy(b).flatten()

    if a.std() == 0 or b.std() == 0:
        return 0.0

    corr = np.corrcoef(a, b)[0, 1]

    # range [-1, 1] → normalize
    return _normalize_similarity(corr, -1, 1)


def histogram_intersection(a, b, bins: int = 256):
    """
    Compute histogram intersection similarity.
    Returns normalized score [0, 1].
    """

    a = _to_numpy(a).flatten()
    b = _to_numpy(b).flatten()

    hist_a, _ = np.histogram(a, bins=bins, range=(a.min(), a.max()), density=True)
    hist_b, _ = np.histogram(b, bins=bins, range=(b.min(), b.max()), density=True)

    intersection = np.minimum(hist_a, hist_b).sum()
    total = hist_a.sum()

    if total == 0:
        return 0.0

    return float(intersection / total)


# Clean dispatch dictionary
_METRIC_REGISTRY = {
    "ssim": ssim_similarity,
    "cosine": cosine_similarity,
    "l2": l2_distance,
    "mse": mse_similarity,
    "pearson": pearson_correlation,
    "histogram": histogram_intersection,
}


def compute_similarity(tensor_a, tensor_b, method: str):
    """
    Dispatch similarity computation using registry mapping.

    Easily extensible by adding new entries to _METRIC_REGISTRY.
    """

    if method not in _METRIC_REGISTRY:
        raise ValueError(f"Unsupported similarity method: {method}")

    return _METRIC_REGISTRY[method](tensor_a, tensor_b)
