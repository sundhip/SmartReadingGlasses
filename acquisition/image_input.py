"""
Image Input Module (Method 1: Saved Image Input).
Handles robust loading, validation, and error reporting for image files.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import os
import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

class ImageLoadError(Exception):
    """Raised when an image cannot be located, opened, or decoded."""
    pass

def load_saved_image(image_path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Safely loads an image from the filesystem.

    Args:
        image_path: File path to the image.

    Returns:
        Tuple containing:
            - np.ndarray: BGR image matrix.
            - dict: Metadata about the loaded image.

    Raises:
        ImageLoadError: If file does not exist, has invalid format, or is corrupted.
    """
    path = Path(image_path).resolve()

    if not path.exists():
        raise ImageLoadError(f"Image file does not exist: {path}")

    if not path.is_file():
        raise ImageLoadError(f"Specified path is not a file: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported image extension '{ext}'. Supported formats: {sorted(list(SUPPORTED_EXTENSIONS))}"
        )

    file_size = path.stat().st_size
    if file_size == 0:
        raise ImageLoadError(f"Image file is empty (0 bytes): {path}")

    # Use cv2.imdecode with numpy fromfile to prevent Windows Unicode path silent failures
    try:
        raw_bytes = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        raise ImageLoadError(f"Failed to read image data from {path}: {e}")

    if image is None or image.size == 0:
        raise ImageLoadError(f"Corrupted or invalid image data at: {path}")

    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1

    metadata = {
        "file_path": str(path),
        "filename": path.name,
        "width": width,
        "height": height,
        "channels": channels,
        "file_size_bytes": file_size,
    }

    return image, metadata
