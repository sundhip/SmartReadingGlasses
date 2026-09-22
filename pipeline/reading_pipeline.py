"""
Complete Reading Pipeline Orchestrator (Phase 8).
Executes the full pipeline:
IMAGE (File or Camera)
  -> QUALITY ANALYSIS
  -> PREPROCESSING
  -> OCR
  -> TEXT POST-PROCESSING
  -> TEXT-TO-SPEECH
Logs metrics, timings, and saves intermediate outputs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import time
import cv2
import numpy as np

from acquisition.image_input import load_saved_image
from acquisition.pi_camera import RaspberryPiCamera
from preprocessing.image_quality import analyze_image_quality, QualityReport
from preprocessing.image_preprocessing import preprocess_for_ocr, PreprocessReport
from ocr.engine import TesseractOCREngine, OCRResult
from ocr.postprocess import clean_ocr_text, CleanedTextResult
from speech.tts import TextToSpeechEngine
from config import OUTPUT_DIR, PROCESSED_DIR

@dataclass
class PipelineResult:
    """Comprehensive artifact bundle produced by a complete pipeline run."""
    source_type: str
    source_path: str
    quality: QualityReport
    preprocessing: PreprocessReport
    ocr: OCRResult
    cleaned_text: CleanedTextResult
    audio_path: Optional[str]
    timings: Dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0

    @property
    def audio_file_path(self) -> Optional[str]:
        """Backward-compatible alias for audio_path."""
        return self.audio_path

    def summary(self) -> str:
        """Formats an executive summary of the pipeline execution."""
        lines = [
            "==================================================",
            "SMART READING GLASSES — PIPELINE EXECUTION REPORT",
            "==================================================",
            f"Input Source:       {self.source_type} ({self.source_path})",
            f"Original Image:     {self.quality.width}x{self.quality.height}",
            f"Quality Status:     {'Acceptable' if self.quality.is_acceptable else 'Warnings flagged'}",
            f"Preprocessed Image: {self.preprocessing.final_shape[1]}x{self.preprocessing.final_shape[0]}",
            f"OCR Result:         {self.ocr.word_count} words (Mean Confidence: {self.ocr.mean_confidence:.1f}%)",
            f"Cleaned Text:       {self.cleaned_text.word_count} words, {self.cleaned_text.line_count} lines",
            f"Audio Output:       {self.audio_path if self.audio_path else 'None / Muted'}",
            "--------------------------------------------------",
            "STAGE TIMINGS (Seconds):",
        ]
        for stage, duration in self.timings.items():
            lines.append(f"  - {stage.ljust(18)}: {duration:.3f}s")
        lines.append(f"  * TOTAL TIME        : {self.total_time:.3f}s")
        lines.append("--------------------------------------------------")
        lines.append("RAW OCR TEXT:")
        lines.append(self.ocr.raw_text.strip() if self.ocr.raw_text.strip() else "[NO TEXT RECOGNIZED]")
        lines.append("--------------------------------------------------")
        lines.append("CLEANED OCR TEXT (SENT TO TTS):")
        lines.append(self.cleaned_text.cleaned_text if self.cleaned_text.cleaned_text else "[NO TEXT]")
        lines.append("==================================================")
        return "\n".join(lines)

class ReadingPipeline:
    """
    Main pipeline controller. Links image acquisition, quality verification,
    OpenCV conditioning, Tesseract OCR, text cleaning, and Text-to-Speech.
    """

    def __init__(
        self,
        tesseract_lang: str = "eng",
        tts_rate: int = 160,
        enable_tts: bool = True
    ):
        self.enable_tts = enable_tts
        self.ocr_engine = TesseractOCREngine(lang=tesseract_lang)
        self.tts_engine = TextToSpeechEngine(rate=tts_rate) if enable_tts else None
        self.camera = RaspberryPiCamera()

    def run_on_image(
        self,
        image_path: str | Path,
        play_audio: bool = True,
        save_audio: bool = True,
        binarization_method: str = "otsu"
    ) -> PipelineResult:
        """Executes full pipeline using a saved image file (Method 1)."""
        t_start = time.time()
        timings: Dict[str, float] = {}

        # 1. Image Acquisition
        t0 = time.time()
        image, meta = load_saved_image(image_path)
        timings["acquisition"] = time.time() - t0

        return self._execute_downstream(
            image=image,
            source_type="Saved File",
            source_path=str(meta["file_path"]),
            prefix=Path(image_path).stem,
            timings=timings,
            t_start=t_start,
            play_audio=play_audio,
            save_audio=save_audio,
            binarization_method=binarization_method
        )

    def run_on_camera(
        self,
        play_audio: bool = True,
        save_audio: bool = True,
        binarization_method: str = "otsu"
    ) -> PipelineResult:
        """Executes full pipeline using Raspberry Pi Camera Module 3 (Method 2)."""
        t_start = time.time()
        timings: Dict[str, float] = {}

        # 1. Camera Acquisition
        t0 = time.time()
        image, meta = self.camera.capture()
        timings["acquisition"] = time.time() - t0

        return self._execute_downstream(
            image=image,
            source_type="Pi Camera Module 3",
            source_path=str(meta["file_path"]),
            prefix="camera_capture",
            timings=timings,
            t_start=t_start,
            play_audio=play_audio,
            save_audio=save_audio,
            binarization_method=binarization_method
        )

    def _execute_downstream(
        self,
        image: np.ndarray,
        source_type: str,
        source_path: str,
        prefix: str,
        timings: Dict[str, float],
        t_start: float,
        play_audio: bool,
        save_audio: bool,
        binarization_method: str
    ) -> PipelineResult:
        """Shared downstream processing for both input methods."""

        # 2. Quality Analysis
        t0 = time.time()
        quality = analyze_image_quality(image)
        timings["quality_check"] = time.time() - t0

        # 3. OpenCV Preprocessing
        t0 = time.time()
        preprocessed_img, prep_report = preprocess_for_ocr(
            image=image,
            method=binarization_method,
            prefix=prefix,
            save_debug=True
        )
        timings["preprocessing"] = time.time() - t0

        # 4. OCR
        t0 = time.time()
        ocr_result = self.ocr_engine.extract_text(
            preprocessed_img,
            fallback_image=prep_report.enhanced_gray
        )
        timings["ocr"] = time.time() - t0

        # 5. Text Post-Processing
        t0 = time.time()
        cleaned_text = clean_ocr_text(ocr_result.raw_text)
        timings["postprocessing"] = time.time() - t0

        # Save text artifacts
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        raw_text_path = OUTPUT_DIR / f"{prefix}_raw_ocr.txt"
        raw_text_path.write_text(ocr_result.raw_text, encoding="utf-8")

        cleaned_text_path = OUTPUT_DIR / f"{prefix}_cleaned.txt"
        cleaned_text_path.write_text(cleaned_text.cleaned_text, encoding="utf-8")

        # 6. Text-To-Speech
        audio_path = None
        if self.enable_tts and self.tts_engine:
            t0 = time.time()
            target_wav = OUTPUT_DIR / f"{prefix}_speech.wav" if save_audio else None
            text_for_speech = getattr(cleaned_text, "speech_text", cleaned_text.cleaned_text)
            audio_path = self.tts_engine.speak(
                text_for_speech if text_for_speech else cleaned_text.cleaned_text,
                play_audio=play_audio,
                save_path=target_wav
            )
            timings["text_to_speech"] = time.time() - t0
        else:
            timings["text_to_speech"] = 0.0

        total_time = time.time() - t_start

        return PipelineResult(
            source_type=source_type,
            source_path=source_path,
            quality=quality,
            preprocessing=prep_report,
            ocr=ocr_result,
            cleaned_text=cleaned_text,
            audio_path=audio_path,
            timings=timings,
            total_time=total_time
        )
