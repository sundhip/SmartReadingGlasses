"""OCR package for text recognition and conservative post-processing."""
from .engine import TesseractOCREngine, OCRResult, OCRError
from .postprocess import clean_ocr_text, CleanedTextResult

__all__ = [
    "TesseractOCREngine",
    "OCRResult",
    "OCRError",
    "clean_ocr_text",
    "CleanedTextResult",
]
