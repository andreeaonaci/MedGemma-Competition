import cv2
import numpy as np


def _ensure_numpy(image):
    """
    Ensure image is a numpy array.
    Converts torch tensors to numpy if necessary.
    """
    try:
        import torch
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
    except ImportError:
        pass

    return image


def _resize_to_match(image_a, image_b):
    """
    Resize image_b to match dimensions of image_a.
    """
    h, w = image_a.shape[:2]
    return cv2.resize(image_b, (w, h))


def _to_grayscale(image):
    """
    Convert image to grayscale if needed.
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def align_images(image_a, image_b):
    """
    Align two images using ORB feature detection and homography estimation.

    Steps:
    - Resize to same dimensions
    - Convert to grayscale
    - Detect ORB keypoints
    - Match features
    - Estimate homography
    - Warp image_b to align with image_a

    Raises:
        ValueError if insufficient keypoints or homography fails.
    """

    image_a = _ensure_numpy(image_a)
    image_b = _ensure_numpy(image_b)

    image_b = _resize_to_match(image_a, image_b)

    gray_a = _to_grayscale(image_a)
    gray_b = _to_grayscale(image_b)

    orb = cv2.ORB_create(5000)

    kp1, des1 = orb.detectAndCompute(gray_a, None)
    kp2, des2 = orb.detectAndCompute(gray_b, None)

    if des1 is None or des2 is None:
        raise ValueError("Insufficient keypoints detected.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 10:
        raise ValueError("Insufficient feature matches for homography.")

    matches = sorted(matches, key=lambda x: x.distance)

    pts_a = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_b = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 5.0)

    if H is None:
        raise ValueError("Homography estimation failed.")

    height, width = image_a.shape[:2]
    aligned_b = cv2.warpPerspective(image_b, H, (width, height))

    return image_a, aligned_b


def compute_difference_map(image_a, image_b):
    """
    Compute absolute difference map between aligned images.
    Returns normalized difference map.
    """

    image_a = _ensure_numpy(image_a)
    image_b = _ensure_numpy(image_b)

    if image_a.shape != image_b.shape:
        raise ValueError("Images must have same dimensions for difference computation.")

    diff = cv2.absdiff(image_a, image_b)

    if len(diff.shape) == 3:
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    diff = diff.astype(np.float32)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    return diff


def generate_heatmap_overlay(image, diff_map):
    """
    Generate heatmap overlay highlighting differences.

    Steps:
    - Normalize difference map
    - Apply colormap
    - Blend with original image
    """

    image = _ensure_numpy(image)
    diff_map = _ensure_numpy(diff_map)

    if diff_map.dtype != np.uint8:
        diff_map = diff_map.astype(np.uint8)

    heatmap = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)

    return overlay
