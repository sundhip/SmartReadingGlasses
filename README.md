# Smart Reading Glasses

An assistive reading device prototype that captures printed book pages and reads them aloud using Optical Character Recognition (OCR) and Text-to-Speech (TTS).

---

## 1. Project Overview
The **Smart Reading Glasses** project is designed to aid individuals with visual impairments or reading difficulties by converting printed physical literature into real-time spoken audio.

For **College Review 2**, this repository contains the **~20% core foundation**:
```
BOOK PAGE
   ↓
IMAGE CAPTURE (Pi Camera Module 3 or Saved Image File)
   ↓
IMAGE QUALITY ANALYSIS (Sharpness, Contrast, Skew, Brightness)
   ↓
OPENCV PREPROCESSING (Grayscale, CLAHE, Noise Filtering, Deskewing, Binarization)
   ↓
TESSERACT OCR (Text extraction with word confidence scoring)
   ↓
TEXT POST-PROCESSING (Hyphen reconnection, whitespace & layout cleanup)
   ↓
TEXT-TO-SPEECH (Lightweight offline audio synthesis)
   ↓
AUDIO OUTPUT (Live speaker playback + WAV file export)
```

> **IMPORTANT — NO AI AT THIS STAGE**: In accordance with Review 2 project scope, heavy cloud LLMs, generative AI, object/face detection, and complex neural networks are intentionally excluded to keep the pipeline lightweight, responsive, and deployable on low-power ARM microcomputers.

---

## 2. System Architecture

```
+-------------------------------------------------------------------------------+
|                             ACQUISITION LAYER                                 |
|  [Method 1: Saved Image] (Laptop/PC)   |  [Method 2: Pi Camera Module 3]      |
|  acquisition/image_input.py            |  acquisition/pi_camera.py            |
+-------------------------------------------------------------------------------+
                                        │
                                        ▼
+-------------------------------------------------------------------------------+
|                         IMAGE QUALITY ANALYSIS                                |
|  preprocessing/image_quality.py                                               |
|  - Laplacian Variance (Blur/Sharpness Check)                                 |
|  - Mean Intensity & Standard Deviation (Brightness & Contrast Check)         |
|  - Minimum Area Bounding Rect (Skew Angle Estimation)                         |
+-------------------------------------------------------------------------------+
                                        │
                                        ▼
+-------------------------------------------------------------------------------+
|                         OPENCV PREPROCESSING                                  |
|  preprocessing/image_preprocessing.py                                         |
|  - Aspect-Ratio Constrained Resizing (RAM safety for Pi Zero 2 W)             |
|  - Affine Deskewing                                                           |
|  - CLAHE (Contrast Limited Adaptive Histogram Equalization for Shadows)       |
|  - Edge-Preserving Bilateral Filtering (removes paper grain)                  |
|  - Adaptive Gaussian / Otsu Dynamic Binarization                              |
|  - Morphological Opening (salt-and-pepper speckle cleanup)                    |
+-------------------------------------------------------------------------------+
                                        │
                                        ▼
+-------------------------------------------------------------------------------+
|                            OCR ENGINE LAYER                                   |
|  ocr/engine.py                                                                |
|  - Tesseract OCR engine (LSTM neural network engine mode)                     |
|  - Extraction of raw text and per-word confidence metrics                     |
+-------------------------------------------------------------------------------+
                                        │
                                        ▼
+-------------------------------------------------------------------------------+
|                         TEXT POST-PROCESSING                                  |
|  ocr/postprocess.py                                                           |
|  - Line-wrap hyphen repair ("para-\ngraph" -> "paragraph")                    |
|  - Whitespace & newline normalization                                         |
|  - Spurious OCR artifact filtering (~, |, _)                                 |
|  - Sentence & paragraph flow reconstruction for natural listening             |
+-------------------------------------------------------------------------------+
                                        │
                                        ▼
+-------------------------------------------------------------------------------+
|                          TEXT-TO-SPEECH (TTS)                                 |
|  speech/tts.py                                                                |
|  - pyttsx3 offline synthesis (SAPI5 on Windows / espeak-ng on Linux)          |
|  - Configurable reading cadence (default: 160 WPM)                            |
|  - Audio file caching in data/output/spoken_output.wav                        |
|  - Live playback via 3.5mm jack / USB audio / HDMI / I2S                      |
+-------------------------------------------------------------------------------+
```

---

## 3. Folder Structure

```
SmartReadingGlasses/
├── acquisition/
│   ├── __init__.py
│   ├── image_input.py          # Validates & loads saved image files
│   └── pi_camera.py            # Picamera2 & rpicam-still hardware capture
├── preprocessing/
│   ├── __init__.py
│   ├── image_quality.py        # Objective blur, lighting, skew analysis
│   └── image_preprocessing.py  # Modular OpenCV enhancement pipeline
├── ocr/
│   ├── __init__.py
│   ├── engine.py               # Tesseract OCR wrapper & confidence parsing
│   └── postprocess.py          # Conservative text sanitation & hyphen repair
├── speech/
│   ├── __init__.py
│   └── tts.py                  # pyttsx3 lightweight offline speech engine
├── pipeline/
│   ├── __init__.py
│   └── reading_pipeline.py     # End-to-end orchestrator & benchmarking
├── data/
│   ├── input/                  # Input book pages
│   ├── processed/              # Saved OpenCV preprocessed stages
│   └── output/                 # OCR text logs and generated speech .wav files
├── tests/
│   ├── __init__.py
│   ├── test_samples.py         # Realistic book page generator (8 page types)
│   ├── test_pipeline.py        # 18 automated pytest test cases
│   └── run_benchmark.py        # Benchmark runner evaluating accuracy & time
├── config.py                   # Centralized hardware & algorithm configuration
├── main.py                     # Command-line user interface
├── requirements.txt            # Lightweight Python dependencies
├── .gitignore
└── README.md                   # Comprehensive documentation
```

---

## 4. Laptop Setup
For local development, algorithms can be tested without hardware attached using saved book page images.

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

---

## 5. Virtual Environment Setup

### Windows (PowerShell):
```powershell
cd SmartReadingGlasses
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS:
```bash
cd SmartReadingGlasses
python3 -m venv .venv
source .venv/bin/activate
```

---

## 6. Dependencies
Install the required packages into your virtual environment:
```bash
pip install -r requirements.txt
```

Packages included:
- `opencv-python`: Image filtering, deskewing, and binarization
- `numpy`: Matrix and numerical operations
- `Pillow`: Image encoding and test sample generation
- `pytesseract`: Python interface to Tesseract OCR engine
- `pyttsx3`: Lightweight offline text-to-speech engine
- `pytest`: Automated test framework

---

## 7. Tesseract Installation

### Windows:
Tesseract must be installed as a system executable:
- **Using Scoop (Recommended - No Admin Needed)**:
  ```powershell
  scoop install tesseract
  ```
- **Using Winget**:
  ```powershell
  winget install UB-Mannheim.TesseractOCR
  ```
- **Manual Installer**: Download 64-bit installer from [UB-Mannheim GitHub](https://github.com/UB-Mannheim/tesseract/wiki). Default location `C:\Program Files\Tesseract-OCR` or `%LOCALAPPDATA%\Programs\Tesseract-OCR`.

### Raspberry Pi OS (Debian Linux):
```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-eng
```

---

## 8. TTS Installation

- **Windows**: Built-in Windows SAPI5 voice drivers are utilized automatically by `pyttsx3`.
- **Raspberry Pi OS**: Install the lightweight `espeak-ng` synthesizer and ALSA audio drivers:
  ```bash
  sudo apt install -y espeak-ng alsa-utils libasound2-dev
  ```

---

## 9. Saved-Image Usage (Method 1)

Place any photo or scan of a book page in `data/input/`:

1. **Run Full Pipeline with Audio Playback**:
   ```bash
   python main.py --input data/input/page1_standard_book.png
   ```

2. **Run with Verbose Quality Analysis & Preprocessing Steps**:
   ```bash
   python main.py --input data/input/page1_standard_book.png --verbose
   ```

3. **Run Muted (Generates WAV audio file without live speaker output)**:
   ```bash
   python main.py --input data/input/page1_standard_book.png --no-audio
   ```

4. **Select Specific Binarization Algorithm**:
   ```bash
   # Options: auto (default), otsu, adaptive_gaussian, none
   python main.py --input data/input/page5_shadow_gradient.png --method adaptive_gaussian
   ```

5. **Run Built-In Interactive Demo**:
   ```bash
   python main.py --demo
   ```

---

## 10. Raspberry Pi 5 Setup
The Raspberry Pi 5 is used for laboratory development and camera testing.

1. Install **Raspberry Pi OS (64-bit Bookworm)** on a high-speed microSD card (A2 / U3 recommended).
2. Connect Raspberry Pi 5 to official 27W USB-C power supply.
3. Update system packages:
   ```bash
   sudo apt update && sudo apt full-upgrade -y
   ```
4. Install system dependencies:
   ```bash
   sudo apt install -y python3-pip python3-venv tesseract-ocr tesseract-ocr-eng espeak-ng libcamera-apps
   ```

---

## 11. Camera Setup
Hardware: **Raspberry Pi Camera Module 3 Standard** (Sony IMX708 sensor with autofocus).

> **CRITICAL CABLE NOTICE FOR PI 5 & PI ZERO 2 W**:
> - The Raspberry Pi 5 and Pi Zero 2 W use a **mini 22-pin 0.5mm pitch CSI connector**, whereas older Pi 4 boards used a 15-pin 1.0mm pitch connector.
> - Ensure you are using the correct **Raspberry Pi Camera Cable for Pi 5 / Zero** (22-pin to 15-pin or 22-pin to 22-pin depending on camera board revision).
> - Insert the cable with silver contact pins facing the board contacts and secure the plastic latch firmly.

---

## 12. Camera Testing

1. Test camera detection on Raspberry Pi terminal:
   ```bash
   rpicam-hello -t 3000
   ```
   *Expected output*: Camera preview window appears and logs: `[INFO] Camera: imx708 [4608x2592 10-bit]`.

2. Test autofocus capture:
   ```bash
   rpicam-still -o test_page.jpg --autofocus-mode auto
   ```

3. If camera is not detected:
   - Check ribbon cable seating on both camera board and Pi connector.
   - Run `rpicam-hello --list-cameras`.
   - Ensure `/boot/firmware/config.txt` has `camera_auto_detect=1`.

---

## 13. Full Pipeline Execution on Raspberry Pi (Method 2)

Run the end-to-end wearable pipeline:
```bash
python main.py --camera
```

What happens automatically:
1. Camera Module 3 activates and engages autofocus on the book page.
2. High-resolution frame is acquired into `data/input/camera_capture.jpg`.
3. Quality check verifies illumination, focus, and tilt.
4. Preprocessing enhances contrast and cleans character edges into `data/processed/`.
5. Tesseract extracts raw text and calculates confidence scores.
6. Post-processing joins broken sentences and repairs word wraps.
7. Text-to-Speech speaks the book content aloud through connected headphones or USB/3.5mm speaker.

---

## 14. Raspberry Pi Zero 2 W Deployment

The Raspberry Pi Zero 2 W is the **final wearable target**.

### Technical Compatibility Review:
| Parameter | Raspberry Pi 5 (Dev) | Raspberry Pi Zero 2 W (Target) | Design Accommodation in Code |
| :--- | :--- | :--- | :--- |
| **CPU** | Quad-core Cortex-A76 @ 2.4GHz | Quad-core Cortex-A53 @ 1.0GHz | Pure C-optimized OpenCV & Tesseract LSTM; no heavy frameworks |
| **RAM** | 4GB / 8GB | 512MB LPDDR2 | `resize_for_ocr` constrains max dimension; strict memory release |
| **Storage** | microSD / NVMe | microSD | Intermediate artifacts saved efficiently in PNG/WAV |
| **Camera Port**| 2x 4-lane MIPI CSI | 1x 2-lane MIPI CSI (22-pin mini) | Supported by `rpicam-still` CLI and `picam2` |
| **Audio** | HDMI / Bluetooth / USB | Bluetooth / I2S DAC / USB Audio | Configurable TTS destination; fallback to WAV cache |

### Optimization Guidelines for Pi Zero 2 W:
1. **Enable ZRAM or 1GB Swap**:
   ```bash
   sudo dphys-swapfile swapoff
   # Set CONF_SWAPSIZE=1024 in /etc/dphys-swapfile
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```
2. **Use Headless Mode**: Disable desktop GUI (`sudo raspi-config` -> System Options -> Boot / Auto Login -> Console).
3. **Capture at 1080p rather than 12MP**: Set in `config.py` (`DEFAULT_CAPTURE_WIDTH = 1920`, `DEFAULT_CAPTURE_HEIGHT = 1080`) to prevent memory spikes.

---

## 15. Troubleshooting

- **"Tesseract OCR: NOT DETECTED"**:
  Ensure Tesseract executable is located in PATH or at `%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe`. Run `main.py --check-system` to diagnose.
- **"Raspberry Pi Camera is not available on this system"**:
  Expected when running on laptop/Windows. Will automatically activate when code is executed on Raspberry Pi OS with Camera Module 3 attached.
- **OCR Text has scrambled symbols**:
  Inspect `data/processed/` images. If the page had harsh shadows, run with `--method adaptive_gaussian`. If blurry, reposition camera to engage autofocus.
- **Audio does not play on Raspberry Pi**:
  Verify audio output device with `aplay -l`. Test audio playback with `speaker-test -t wav -c 2`.

---

## 16. Known Limitations
1. **Curved Book Spines**: Extreme 3D page warping near thick book bindings can degrade OCR character alignment without 3D dewarping algorithms.
2. **Complex Multi-Column Newspapers**: Standard segmentation mode (PSM 3) may occasionally read across adjacent columns if spacing is narrow.
3. **Severe Motion Blur**: If the user moves their head abruptly during capture, OCR confidence drops significantly.

---

## 17. Current Implementation Status (~20% Review 2)

- [x] Dual-input architecture (Saved Image & Pi Camera Module 3)
- [x] Safe image loading with format & corruption defense
- [x] Quantitative quality checks (sharpness, brightness, contrast, skew)
- [x] Modular OpenCV preprocessing (resizing, CLAHE, bilateral filtering, deskewing, Otsu & adaptive thresholding)
- [x] Tesseract OCR integration with confidence logging
- [x] Conservative text cleaning & hyphen repair
- [x] Offline Text-to-Speech synthesis & WAV audio generation
- [x] Complete end-to-end CLI orchestrator (`main.py`)
- [x] 18 automated unit and integration tests passing (`pytest`)
- [x] 8-page qualitative benchmark suite
- [x] Pi Zero 2 W memory conservation architecture

---

## 18. Future Development (Intentionally NOT Implemented for Review 2)
The following are reserved for subsequent review milestones:
- Local/Edge LLM summarization and question answering
- Word meaning lookup and vocabulary explanation
- Face detection and obstacle recognition
- Multi-language translation
- Cloud synchronization
- Standalone 3D printed smart glasses chassis and battery integration
