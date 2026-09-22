"""
Tesseract OCR Engine Module (Phase 5).
Wraps Tesseract via pytesseract with multi-platform binary discovery,
word-level confidence extraction, and defensive error handling.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import os
import shutil
import cv2
import numpy as np
import pytesseract

from config import OCRConfig

class OCRError(Exception):
    """Raised when OCR execution fails or binary cannot be found."""
    pass

@dataclass
class OCRResult:
    """Encapsulates raw text and extraction confidence metrics."""
    raw_text: str
    mean_confidence: float
    word_count: int
    word_confidences: List[Tuple[str, float]] = field(default_factory=list)
    is_empty: bool = True

    def summary(self) -> str:
        """Returns a concise summary of the OCR extraction."""
        status = "EMPTY RESULT" if self.is_empty else f"{self.word_count} words recognized"
        conf_str = f"{self.mean_confidence:.1f}%" if self.word_count > 0 else "N/A"
        return f"OCR Status: {status} | Mean Word Confidence: {conf_str}"

class TesseractOCREngine:
    """
    Manages communication with Tesseract OCR engine.
    Auto-discovers tesseract on Windows, Linux, and Raspberry Pi OS.
    """

    def __init__(self, lang: str = OCRConfig.DEFAULT_LANG, psm: int = OCRConfig.DEFAULT_PSM):
        self.lang = lang
        self.psm = psm
        self.tesseract_path = self._discover_and_configure_binary()

    def _discover_and_configure_binary(self) -> Optional[str]:
        """
        Locates the tesseract executable and configures pytesseract and TESSDATA_PREFIX.
        """
        # 1. Check if already configured or in PATH
        which_path = shutil.which("tesseract")
        if which_path:
            pytesseract.pytesseract.tesseract_cmd = which_path
            self._configure_tessdata(Path(which_path).parent)
            return which_path

        # 2. Check known candidates across Windows and Linux / Raspberry Pi
        for candidate in OCRConfig.POSSIBLE_TESSERACT_PATHS:
            p = Path(candidate)
            if p.exists() and p.is_file():
                pytesseract.pytesseract.tesseract_cmd = str(p)
                self._configure_tessdata(p.parent)
                return str(p)

        return None

    def _configure_tessdata(self, tesseract_dir: Path) -> None:
        """Ensures TESSDATA_PREFIX is set to locate language models."""
        if "TESSDATA_PREFIX" not in os.environ:
            tessdata = tesseract_dir / "tessdata"
            if tessdata.exists():
                os.environ["TESSDATA_PREFIX"] = str(tessdata)

    def is_available(self) -> bool:
        """Checks if Tesseract binary can be executed successfully."""
        if not self.tesseract_path:
            self.tesseract_path = self._discover_and_configure_binary()
        if not self.tesseract_path:
            return False
        try:
            ver = pytesseract.get_tesseract_version()
            return ver is not None
        except Exception:
            return False

    def get_version(self) -> str:
        """Returns installed Tesseract version or raises OCRError."""
        if not self.is_available():
            raise OCRError(
                "Tesseract executable not found.\n"
                "Please install Tesseract OCR:\n"
                "- Windows: winget install UB-Mannheim.TesseractOCR OR scoop install tesseract\n"
                "- Raspberry Pi / Debian: sudo apt install tesseract-ocr tesseract-ocr-eng"
            )
        return str(pytesseract.get_tesseract_version())

    def extract_text(
        self,
        image: np.ndarray,
        psm: Optional[int] = None,
        lang: Optional[str] = None,
        fallback_image: Optional[np.ndarray] = None
    ) -> OCRResult:
        """
        Runs OCR on a conditioned image and extracts text and word confidence.
        Includes automatic resolution scaling and dual-image fallback for book reading reliability.

        Args:
            image: Preprocessed Grayscale or Binarized image.
            psm: Page segmentation mode override.
            lang: Language override.
            fallback_image: Optional secondary image representation (e.g. enhanced grayscale).

        Returns:
            OCRResult object with raw text and confidence statistics.

        Raises:
            OCRError: If tesseract binary is missing or execution errors out.
        """
        if not self.is_available():
            raise OCRError("Tesseract OCR is not installed or not discoverable on this system.")

        if image is None or image.size == 0:
            raise ValueError("Cannot perform OCR on empty image.")

        target_psm = psm if psm is not None else self.psm
        target_lang = lang if lang is not None else self.lang

        def _prepare_resolution(img: np.ndarray) -> np.ndarray:
            """Ensures characters have sufficient pixel height for Tesseract LSTM neural net."""
            h, w = img.shape[:2]
            if w < 1200:
                scale = max(1.5, 1200.0 / float(w))
                return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            return img

        def _run_tesseract(img_to_ocr: np.ndarray, active_psm: int) -> OCRResult:
            scaled = _prepare_resolution(img_to_ocr)
            custom_config = f"--oem {OCRConfig.DEFAULT_OEM} --psm {active_psm} -c preserve_interword_spaces=1"
            data = pytesseract.image_to_data(
                scaled,
                lang=target_lang,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            raw_text = pytesseract.image_to_string(
                scaled,
                lang=target_lang,
                config=custom_config
            )

            word_confidences: List[Tuple[str, float]] = []
            n_boxes = len(data["text"])
            for i in range(n_boxes):
                word = data["text"][i].strip()
                conf = float(data["conf"][i])
                if word and conf >= 0:
                    word_confidences.append((word, conf))

            if word_confidences:
                mean_conf = sum(c for _, c in word_confidences) / len(word_confidences)
                is_empty = False
            else:
                mean_conf = 0.0
                is_empty = len(raw_text.strip()) == 0

            return OCRResult(
                raw_text=raw_text,
                mean_confidence=mean_conf,
                word_count=len(word_confidences),
                word_confidences=word_confidences,
                is_empty=is_empty
            )

        try:
            # Pass 1: Run OCR on primary preprocessed image with target PSM
            result = _run_tesseract(image, target_psm)

            # Pass 2: If default PSM yielded low confidence or few words, try PSM 6 (uniform block of text)
            if (result.mean_confidence < 70.0 or result.word_count < 12) and target_psm != 6:
                alt_result = _run_tesseract(image, 6)
                score_curr = result.word_count * (result.mean_confidence / 100.0)
                score_alt = alt_result.word_count * (alt_result.mean_confidence / 100.0)
                if score_alt > score_curr:
                    result = alt_result

            # Pass 3: Evaluate CLAHE-enhanced grayscale if available
            # Modern Tesseract LSTM often excels on grayscale by avoiding binary clipping
            if fallback_image is not None:
                score_curr = result.word_count * (result.mean_confidence / 100.0)
                fb_result = _run_tesseract(fallback_image, target_psm)
                fb_score = fb_result.word_count * (fb_result.mean_confidence / 100.0)

                if fb_score > score_curr:
                    result = fb_result
                    score_curr = fb_score

                # Also try PSM 6 on grayscale if still few words or low confidence
                if (result.word_count < 12 or result.mean_confidence < 70.0):
                    fb_result6 = _run_tesseract(fallback_image, 6)
                    fb6_score = fb_result6.word_count * (fb_result6.mean_confidence / 100.0)
                    if fb6_score > score_curr:
                        result = fb_result6

            return result
        except Exception as e:
            raise OCRError(f"Tesseract OCR processing failed: {e}")
