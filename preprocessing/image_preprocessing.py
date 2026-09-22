"""
Image Preprocessing Module (Phase 4).
Modular OpenCV pipeline for text image conditioning prior to OCR.
Supports "auto", "otsu", "adaptive_gaussian", and "none" binarization.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import cv2
import numpy as np

from config import PreprocessConfig, PROCESSED_DIR

@dataclass
class PreprocessReport:
    """Records the sequence and parameters of applied preprocessing operations."""
    original_shape: Tuple[int, ...]
    final_shape: Tuple[int, ...]
    applied_steps: List[str] = field(default_factory=list)
    saved_debug_paths: Dict[str, str] = field(default_factory=dict)
    enhanced_gray: Optional[np.ndarray] = None

def resize_for_ocr(image: np.ndarray, max_dim: int = PreprocessConfig.MAX_IMAGE_DIM) -> Tuple[np.ndarray, bool]:
    """Downscales large images to max_dim while preserving aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image, False

    scale = max_dim / float(max(h, w))
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, True

def deskew_image(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    """Detects angle of text lines and rotates the image straight."""
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:
        return gray, 0.0

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle

    if abs(angle) < 0.5 or abs(angle) > 40.0:
        return gray, 0.0

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        gray,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return deskewed, angle

def preprocess_for_ocr(
    image: np.ndarray,
    method: str = "auto",
    enhance_contrast: bool = True,
    reduce_noise: bool = True,
    deskew: bool = PreprocessConfig.ENABLE_DESKEW,
    save_debug: bool = True,
    prefix: str = "page"
) -> Tuple[np.ndarray, PreprocessReport]:
    """
    Executes the modular OpenCV preprocessing pipeline.

    Args:
        image: Input BGR or Grayscale image.
        method: "auto", "otsu", "adaptive_gaussian", or "none".
        enhance_contrast: Whether to apply CLAHE.
        reduce_noise: Whether to apply edge-preserving bilateral filtering.
        deskew: Whether to auto-straighten text lines.
        save_debug: Save intermediate images into data/processed/.
        prefix: Filename prefix for saved debug steps.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot preprocess empty image.")

    orig_shape = image.shape
    report = PreprocessReport(original_shape=orig_shape, final_shape=orig_shape)

    # 1. Safe resizing
    working_img, resized = resize_for_ocr(image)
    if resized:
        report.applied_steps.append(f"Resized down to {working_img.shape[1]}x{working_img.shape[0]}")

    # 2. Grayscale conversion
    if len(working_img.shape) == 3:
        gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
        report.applied_steps.append("Grayscale conversion (BGR2GRAY)")
    else:
        gray = working_img.copy()

    # 3. Deskewing
    if deskew:
        gray, angle = deskew_image(gray)
        if abs(angle) >= 0.5:
            report.applied_steps.append(f"Deskewed by {angle:.2f} degrees")

    # 4. Contrast Enhancement (CLAHE)
    if enhance_contrast:
        clahe = cv2.createCLAHE(
            clipLimit=PreprocessConfig.CLAHE_CLIP_LIMIT,
            tileGridSize=PreprocessConfig.CLAHE_TILE_GRID_SIZE
        )
        gray = clahe.apply(gray)
        report.applied_steps.append("Contrast enhancement (CLAHE)")

    # 5. Noise Reduction
    if reduce_noise:
        gray = cv2.bilateralFilter(
            gray,
            d=PreprocessConfig.BILATERAL_D,
            sigmaColor=PreprocessConfig.BILATERAL_SIGMA_COLOR,
            sigmaSpace=PreprocessConfig.BILATERAL_SIGMA_SPACE
        )
        report.applied_steps.append("Edge-preserving bilateral noise reduction")

    # Store high-contrast enhanced grayscale image for OCR fallback
    report.enhanced_gray = gray.copy()

    # 6. Binarization Strategy
    chosen_method = method
    if method == "auto":
        # Check standard deviation of quadrant brightnesses to detect shadow gradients
        h, w = gray.shape[:2]
        quads = [
            np.mean(gray[:h//2, :w//2]),
            np.mean(gray[:h//2, w//2:]),
            np.mean(gray[h//2:, :w//2]),
            np.mean(gray[h//2:, w//2:]),
        ]
        quad_spread = max(quads) - min(quads)
        # If lighting variation across quadrants is significant (> 25 intensity levels),
        # adaptive thresholding is vastly superior to global Otsu.
        if quad_spread > 25.0:
            chosen_method = "adaptive_gaussian"
            report.applied_steps.append(f"Auto-selected Adaptive Gaussian (quadrant lighting spread: {quad_spread:.1f})")
        else:
            chosen_method = "otsu"
            report.applied_steps.append("Auto-selected Otsu binarization (uniform lighting)")

    if chosen_method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        final_img = binary
        report.applied_steps.append("Otsu binarization")
    elif chosen_method == "adaptive_gaussian":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )
        final_img = binary
        report.applied_steps.append("Adaptive Gaussian binarization")
    else:
        final_img = gray
        report.applied_steps.append("Enhanced grayscale output (no manual binarization)")

    # 7. Morphological cleanup
    if chosen_method in ("otsu", "adaptive_gaussian"):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, PreprocessConfig.MORPH_KERNEL_SIZE)
        final_img = cv2.morphologyEx(final_img, cv2.MORPH_OPEN, kernel)
        report.applied_steps.append("Morphological opening cleanup")

    report.final_shape = final_img.shape

    if save_debug:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_file = PROCESSED_DIR / f"{prefix}_preprocessed.png"
        cv2.imwrite(str(out_file), final_img)
        report.saved_debug_paths["preprocessed"] = str(out_file)

    return final_img, report
