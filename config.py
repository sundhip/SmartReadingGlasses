"""
Global Configuration for Smart Reading Glasses.
Contains default parameters for camera acquisition, quality checks,
preprocessing, OCR engine, and Text-to-Speech (TTS).
Designed to be lightweight and compatible with Raspberry Pi Zero 2 W.
"""

from pathlib import Path
import os
import sys

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Directory layout
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

# Ensure runtime directories exist
for directory in [INPUT_DIR, PROCESSED_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Acquisition Configuration
# ---------------------------------------------------------
class CameraConfig:
    # Sensor resolution for Raspberry Pi Camera Module 3 Standard is 4608x2592
    # For Pi Zero 2 W memory conservation (512MB RAM total),
    # 1920x1080 or 1280x720 is ideal for fast OCR without OOM crashes.
    DEFAULT_CAPTURE_WIDTH = 1920
    DEFAULT_CAPTURE_HEIGHT = 1080
    AUTOFOCUS_MODE = "auto"  # Camera Module 3 supports continuous or auto focus
    WARMUP_SECONDS = 1.5
    DEFAULT_SAVE_NAME = "camera_capture.jpg"

# ---------------------------------------------------------
# Quality Analysis Thresholds
# ---------------------------------------------------------
class QualityConfig:
    # Minimum image dimensions for reliable OCR
    MIN_WIDTH = 640
    MIN_HEIGHT = 480
    
    # Brightness (mean grayscale intensity [0, 255])
    TOO_DARK_THRESHOLD = 50.0
    TOO_BRIGHT_THRESHOLD = 220.0
    
    # Contrast (standard deviation of grayscale intensity)
    LOW_CONTRAST_THRESHOLD = 30.0
    
    # Sharpness / Blur metric (Laplacian variance)
    # Variance < 80 generally indicates motion or out-of-focus blur on text
    BLUR_THRESHOLD = 80.0
    
    # Maximum recommended skew angle before warning (degrees)
    MAX_RECOMMENDED_SKEW_DEG = 15.0

# ---------------------------------------------------------
# Preprocessing Configuration
# ---------------------------------------------------------
class PreprocessConfig:
    # Maximum dimension to constrain processing footprint for Pi Zero 2 W
    MAX_IMAGE_DIM = 2000
    
    # Bilateral filter parameters for text (preserves edges while removing paper grain)
    BILATERAL_D = 9
    BILATERAL_SIGMA_COLOR = 75
    BILATERAL_SIGMA_SPACE = 75
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_GRID_SIZE = (8, 8)
    
    # Binarization method: "otsu", "adaptive_gaussian", "adaptive_mean", "none"
    BINARIZATION_METHOD = "otsu"
    
    # Morphological kernel size for noise cleanup if needed
    MORPH_KERNEL_SIZE = (2, 2)
    
    # Enable deskewing
    ENABLE_DESKEW = True

# ---------------------------------------------------------
# OCR Configuration
# ---------------------------------------------------------
class OCRConfig:
    # Tesseract PSM (Page Segmentation Modes):
    # 3 = Fully automatic page segmentation, but no OSD (Default)
    # 6 = Assume a single uniform block of text (Good for book paragraphs)
    DEFAULT_PSM = 3
    DEFAULT_OEM = 3  # Default engine mode (LSTM + Legacy where available)
    DEFAULT_LANG = "eng"
    
    # Candidate binary paths across operating systems
    POSSIBLE_TESSERACT_PATHS = [
        # Windows standard locations
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path.home() / r"AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        str(Path.home() / r"scoop\apps\tesseract\current\tesseract.exe"),
        str(Path.home() / r"scoop\shims\tesseract.exe"),
        r"C:\ProgramData\chocolatey\bin\tesseract.exe",
        # Linux / Raspberry Pi OS locations
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]

# ---------------------------------------------------------
# Text-to-Speech (TTS) Configuration
# ---------------------------------------------------------
class TTSConfig:
    # Speech rate (words per minute). 120-130 provides calm, natural, clear book reading.
    RATE = 125  # Natural, comfortable listening pace (not rushed)
    VOLUME = 1.0  # Range 0.0 to 1.0
    VOICE_INDEX = 0  # 0 for default system voice
    SAVE_AUDIO_COPY = True
    AUDIO_FILENAME = "spoken_output.wav"
    PICO_SPEED_LEVEL = 82  # SVOX Pico natural speed percentage (80-85 is calm & human-like)

