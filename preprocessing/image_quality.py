"""
Image Quality Analysis Module (Phase 3).
Performs objective checks on resolution, brightness, contrast, blur/sharpness,
and skew before running OCR. Emits actionable warnings without fabricating percentages.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import cv2
import numpy as np

from config import QualityConfig

@dataclass
class QualityReport:
    """Structured report containing image quality metrics and diagnostic warnings."""
    width: int
    height: int
    mean_brightness: float
    contrast_std: float
    sharpness_score: float
    estimated_skew_degrees: float
    warnings: List[str] = field(default_factory=list)
    is_acceptable: bool = True

    def summary(self) -> str:
        """Returns a human-readable text summary of the quality analysis."""
        lines = [
            f"Image Dimensions: {self.width}x{self.height}",
            f"Brightness: {self.mean_brightness:.1f} / 255.0",
            f"Contrast (std dev): {self.contrast_std:.1f}",
            f"Sharpness Score (Laplacian variance): {self.sharpness_score:.1f}",
            f"Estimated Skew: {self.estimated_skew_degrees:.2f} degrees",
            f"Status: {'ACCEPTABLE FOR OCR' if self.is_acceptable else 'POTENTIALLY PROBLEMATIC'}"
        ]
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - [WARNING] {w}")
        else:
            lines.append("Warnings: None (image quality is good for OCR)")
        return "\n".join(lines)

def estimate_skew_angle(gray_image: np.ndarray) -> float:
    """
    Estimates the text skew angle using minimum bounding area of text contours.

    Args:
        gray_image: Grayscale image.

    Returns:
        float: Estimated skew angle in degrees (-45.0 to 45.0).
    """
    # Invert binary image so text is white (foreground)
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find non-zero coordinates
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return 0.0

    # Find minimum bounding box around text pixels
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # Normalize angle from minAreaRect convention
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle

    return float(angle)

def analyze_image_quality(image: np.ndarray) -> QualityReport:
    """
    Analyzes an input BGR or Grayscale image for OCR suitability.

    Args:
        image: np.ndarray image matrix.

    Returns:
        QualityReport with objective metrics and human-readable warnings.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot analyze empty image.")

    if len(image.shape) == 3:
        height, width, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        height, width = image.shape
        gray = image.copy()

    warnings: List[str] = []

    # 1. Resolution Check
    if width < QualityConfig.MIN_WIDTH or height < QualityConfig.MIN_HEIGHT:
        warnings.append(
            f"Low resolution ({width}x{height}). Recommended minimum is "
            f"{QualityConfig.MIN_WIDTH}x{QualityConfig.MIN_HEIGHT} for reliable text OCR."
        )

    # 2. Brightness (Mean Grayscale Intensity)
    mean_brightness = float(np.mean(gray))
    if mean_brightness < QualityConfig.TOO_DARK_THRESHOLD:
        warnings.append(
            f"Image is too dark (mean brightness {mean_brightness:.1f} < {QualityConfig.TOO_DARK_THRESHOLD}). "
            "Characters may be obscured by shadows."
        )
    elif mean_brightness > QualityConfig.TOO_BRIGHT_THRESHOLD:
        warnings.append(
            f"Image is overexposed/washed out (mean brightness {mean_brightness:.1f} > {QualityConfig.TOO_BRIGHT_THRESHOLD}). "
            "Text strokes may be faded."
        )

    # 3. Contrast (Standard Deviation)
    contrast_std = float(np.std(gray))
    if contrast_std < QualityConfig.LOW_CONTRAST_THRESHOLD:
        warnings.append(
            f"Low contrast (intensity std dev {contrast_std:.1f} < {QualityConfig.LOW_CONTRAST_THRESHOLD}). "
            "Text may blend into the paper background."
        )

    # 4. Blur / Sharpness (Laplacian Variance)
    # Higher variance indicates sharp text edges; lower indicates blur / defocus.
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness_score = float(laplacian.var())
    if sharpness_score < QualityConfig.BLUR_THRESHOLD:
        warnings.append(
            f"Image appears blurry or out of focus (sharpness score {sharpness_score:.1f} < {QualityConfig.BLUR_THRESHOLD}). "
            "Camera autofocus or steady positioning is recommended."
        )

    # 5. Skew Angle
    skew_angle = estimate_skew_angle(gray)
    if abs(skew_angle) > QualityConfig.MAX_RECOMMENDED_SKEW_DEG:
        warnings.append(
            f"Significant text tilt detected ({skew_angle:.1f} degrees). "
            "Automatic deskewing is recommended for better OCR alignment."
        )

    # Acceptable if no critical flaws (severe blur or extreme darkness)
    is_acceptable = (
        sharpness_score >= (QualityConfig.BLUR_THRESHOLD * 0.5)
        and mean_brightness >= (QualityConfig.TOO_DARK_THRESHOLD * 0.7)
    )

    return QualityReport(
        width=width,
        height=height,
        mean_brightness=mean_brightness,
        contrast_std=contrast_std,
        sharpness_score=sharpness_score,
        estimated_skew_degrees=skew_angle,
        warnings=warnings,
        is_acceptable=is_acceptable,
    )
