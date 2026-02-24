import cv2
import numpy as np


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale if it is not already.
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    """
    Apply Otsu thresholding.
    Returns binary image.
    """
    gray = _to_grayscale(image)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh


def multi_threshold(image: np.ndarray, thresholds: list) -> np.ndarray:
    """
    Apply multi-level threshold segmentation.
    Each threshold splits intensity ranges into segments.
    """
    gray = _to_grayscale(image)
    output = np.zeros_like(gray)

    thresholds = sorted(thresholds)

    for i, t in enumerate(thresholds):
        if i == 0:
            output[gray <= t] = int(255 / (len(thresholds) + 1))
        else:
            output[(gray > thresholds[i - 1]) & (gray <= t)] = int(
                (i + 1) * 255 / (len(thresholds) + 1)
            )

    output[gray > thresholds[-1]] = 255

    return output.astype(np.uint8)


def canny_edges(image: np.ndarray, low: int, high: int) -> np.ndarray:
    """
    Perform Canny edge detection.
    Returns edge map.
    """
    gray = _to_grayscale(image)
    edges = cv2.Canny(gray, low, high)
    return edges


def detect_contours(image: np.ndarray) -> np.ndarray:
    """
    Detect contours and draw them on a copy of the image.
    Returns image with contours drawn.
    """
    gray = _to_grayscale(image)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    output = image.copy()
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

    return output


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply global histogram equalization.
    """
    if len(image.shape) == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        return cv2.equalizeHist(image)


def clahe_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (adaptive histogram equalization).
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(image)


def adjust_brightness(image: np.ndarray, value: float) -> np.ndarray:
    """Adjust brightness by adding scalar value."""
    image = image.astype(np.float32)
    adjusted = image + value
    adjusted = np.clip(adjusted, 0, 255)
    return adjusted.astype(np.uint8)


def adjust_contrast(image: np.ndarray, value: float) -> np.ndarray:
    """Adjust contrast. Value >1 increases, 0–1 decreases."""
    image = image.astype(np.float32)
    mean = np.mean(image)
    adjusted = (image - mean) * value + mean
    adjusted = np.clip(adjusted, 0, 255)
    return adjusted.astype(np.uint8)

