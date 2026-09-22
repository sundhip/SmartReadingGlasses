"""
Comprehensive Automated Test Suite for Smart Reading Glasses.
Tests modules: Acquisition, Quality, Preprocessing, OCR, Postprocessing, TTS, and Full Pipeline.
"""

from pathlib import Path
import os
import sys
import pytest
import numpy as np
import cv2

# Add root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acquisition.image_input import load_saved_image, ImageLoadError
from preprocessing.image_quality import analyze_image_quality, estimate_skew_angle
from preprocessing.image_preprocessing import preprocess_for_ocr, resize_for_ocr, deskew_image
from ocr.engine import TesseractOCREngine, OCRResult
from ocr.postprocess import clean_ocr_text
from speech.tts import TextToSpeechEngine
from pipeline.reading_pipeline import ReadingPipeline
from tests.test_samples import generate_sample_book_page, generate_all_test_pages


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Generates test book pages in a temporary directory."""
    temp_dir = tmp_path_factory.mktemp("test_pages")
    generate_all_test_pages(temp_dir)
    return temp_dir


class TestImageAcquisition:
    """Tests for Phase 2: Saved Image Input."""

    def test_missing_file_raises_error(self):
        with pytest.raises(ImageLoadError, match="does not exist"):
            load_saved_image("nonexistent_page_12345.jpg")

    def test_directory_path_raises_error(self, tmp_path):
        with pytest.raises(ImageLoadError, match="not a file"):
            load_saved_image(tmp_path)

    def test_unsupported_format_raises_error(self, tmp_path):
        bad_file = tmp_path / "test.txt"
        bad_file.write_text("not an image")
        with pytest.raises(ImageLoadError, match="Unsupported image extension"):
            load_saved_image(bad_file)

    def test_empty_file_raises_error(self, tmp_path):
        empty_img = tmp_path / "empty.jpg"
        empty_img.touch()
        with pytest.raises(ImageLoadError, match="is empty"):
            load_saved_image(empty_img)

    def test_valid_image_loads_successfully(self, test_data_dir):
        img_path = test_data_dir / "page1_standard_book.png"
        image, meta = load_saved_image(img_path)
        assert isinstance(image, np.ndarray)
        assert image.ndim == 3
        assert meta["width"] == 1000
        assert meta["height"] == 1400
        assert meta["file_size_bytes"] > 0


class TestImageQuality:
    """Tests for Phase 3: Image Quality Analysis."""

    def test_standard_page_quality(self, test_data_dir):
        img_path = test_data_dir / "page1_standard_book.png"
        image, _ = load_saved_image(img_path)
        report = analyze_image_quality(image)

        assert report.width == 1000
        assert report.height == 1400
        assert report.mean_brightness > 150  # Light background
        assert report.contrast_std > 20
        assert report.sharpness_score > 50
        assert report.is_acceptable is True

    def test_blurry_image_triggers_warning(self):
        sharp_img = np.ones((480, 640), dtype=np.uint8) * 255
        cv2.putText(sharp_img, "Test Text", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 3)
        blurry_img = cv2.GaussianBlur(sharp_img, (55, 55), 0)

        report = analyze_image_quality(blurry_img)
        assert any("blurry" in w.lower() for w in report.warnings)


class TestImagePreprocessing:
    """Tests for Phase 4: OpenCV Preprocessing."""

    def test_resize_preserves_bounds(self):
        huge_img = np.ones((3000, 4000, 3), dtype=np.uint8) * 200
        resized, was_resized = resize_for_ocr(huge_img, max_dim=1500)
        assert was_resized is True
        assert max(resized.shape[:2]) == 1500

    def test_preprocessing_pipeline_produces_binary(self, test_data_dir):
        img_path = test_data_dir / "page1_standard_book.png"
        image, _ = load_saved_image(img_path)
        processed, report = preprocess_for_ocr(image, method="otsu", save_debug=False)

        assert isinstance(processed, np.ndarray)
        assert processed.ndim == 2  # Grayscale/Binary
        assert "Grayscale conversion (BGR2GRAY)" in report.applied_steps
        assert "Otsu binarization" in report.applied_steps


class TestOCREngine:
    """Tests for Phase 5: Tesseract OCR."""

    def test_ocr_available_and_version(self):
        engine = TesseractOCREngine()
        assert engine.is_available() is True
        version = engine.get_version()
        assert version.startswith("5.")

    def test_ocr_extracts_text_from_sample_page(self, test_data_dir):
        img_path = test_data_dir / "page1_standard_book.png"
        image, _ = load_saved_image(img_path)
        processed, _ = preprocess_for_ocr(image, method="otsu", save_debug=False)

        engine = TesseractOCREngine()
        result = engine.extract_text(processed)

        assert not result.is_empty
        assert result.word_count >= 25
        assert "reading" in result.raw_text.lower()
        assert result.mean_confidence > 50.0

    def test_ocr_on_blank_image_returns_empty(self):
        blank = np.ones((500, 500), dtype=np.uint8) * 255
        engine = TesseractOCREngine()
        result = engine.extract_text(blank)
        assert result.is_empty is True
        assert result.word_count == 0


class TestTextPostProcessing:
    """Tests for Phase 6: Text Post-Processing."""

    def test_hyphen_reconnection(self):
        raw = "This is a demon-\nstration of hyphen repair."
        res = clean_ocr_text(raw)
        assert "demonstration" in res.cleaned_text
        assert "demon-" not in res.cleaned_text

    def test_whitespace_and_newline_normalization(self):
        raw = "First   sentence.\n\n\nSecond     sentence with   spaces."
        res = clean_ocr_text(raw)
        assert "First sentence." in res.cleaned_text
        assert "Second sentence with spaces." in res.cleaned_text

    def test_noise_lines_removed(self):
        raw = "~\nReal Title\n|\nReal body sentence.\n_"
        res = clean_ocr_text(raw)
        assert "Real Title" in res.cleaned_text
        assert "Real body sentence." in res.cleaned_text
        assert "~" not in res.cleaned_text
        assert "|" not in res.cleaned_text


class TestTextToSpeech:
    """Tests for Phase 7: Text-to-Speech."""

    def test_tts_initialization(self):
        tts = TextToSpeechEngine()
        assert tts.is_available() is True

    def test_tts_saves_audio_file(self, tmp_path):
        tts = TextToSpeechEngine()
        out_wav = tmp_path / "test_voice.wav"
        saved = tts.speak("Testing Smart Reading Glasses text to speech.", play_audio=False, save_path=out_wav)
        assert saved == str(out_wav)
        assert out_wav.exists()
        assert out_wav.stat().st_size > 1000


class TestCompletePipeline:
    """Tests for Phase 8 & 9: Full End-to-End Pipeline."""

    def test_pipeline_on_book_page(self, test_data_dir, tmp_path):
        pipeline = ReadingPipeline(enable_tts=True)
        img_path = test_data_dir / "page1_standard_book.png"

        result = pipeline.run_on_image(
            img_path,
            play_audio=False,  # Audio generation tested, muted for fast test runs
            save_audio=True
        )

        assert result.source_type == "Saved File"
        assert result.ocr.word_count >= 25
        assert result.cleaned_text.word_count >= 25
        assert "reading" in result.cleaned_text.cleaned_text.lower()
        assert result.audio_path is not None
        assert Path(result.audio_path).exists()
        assert "ocr" in result.timings
        assert "preprocessing" in result.timings
        assert result.total_time > 0.0
