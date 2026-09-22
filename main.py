"""
Smart Reading Glasses — Main CLI & GUI Entry Point.
Provides options for:
1. Live Camera Viewfinder & Interactive GUI (--live / --gui)
2. Saved book image processing (--input <path>)
3. One-shot Raspberry Pi Camera capture (--camera)
4. Demonstration mode (--demo)
5. System diagnostic check (--check-system)
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so package imports resolve cleanly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.reading_pipeline import ReadingPipeline
from acquisition.image_input import ImageLoadError
from acquisition.pi_camera import CameraCaptureError
from ocr.engine import OCRError
from speech.tts import TTSError


def check_system_diagnostics():
    """Runs a system-wide diagnostic check and reports component statuses."""
    print("=" * 60)
    print("SMART READING GLASSES — SYSTEM DIAGNOSTICS")
    print("=" * 60)
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version:    {sys.version.split()[0]}")
    print(f"Operating System:  {sys.platform}")

    # 1. OpenCV
    try:
        import cv2
        print(f"OpenCV Version:    {cv2.__version__} [OK]")
    except ImportError as e:
        print(f"OpenCV:            NOT INSTALLED ({e}) [FAIL]")

    # 2. Tesseract OCR
    pipeline = ReadingPipeline(enable_tts=False)
    if pipeline.ocr_engine.is_available():
        ver = pipeline.ocr_engine.get_version()
        path = pipeline.ocr_engine.tesseract_path
        print(f"Tesseract OCR:     Version {ver} [OK]")
        print(f"Tesseract Path:    {path}")
    else:
        print("Tesseract OCR:     NOT DETECTED [WARNING]")
        print("                   Please install Tesseract OCR or verify PATH.")

    # 3. Text-to-Speech
    try:
        from speech.tts import TextToSpeechEngine
        tts = TextToSpeechEngine()
        if tts.is_available():
            print("Text-to-Speech:    pyttsx3 Initialized [OK]")
        else:
            print("Text-to-Speech:    Driver unavailable / headless [WARNING]")
    except Exception as e:
        print(f"Text-to-Speech:    Error: {e} [WARNING]")

    # 4. Raspberry Pi Camera
    if pipeline.camera.is_available():
        print(f"Camera Backend:    {pipeline.camera._backend} [OK]")
    else:
        print("Raspberry Pi Cam:  Not detected (Normal on laptop/PC development)")
        print("                   Backend will be active when deployed on Raspberry Pi.")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Smart Reading Glasses — Book Page OCR and Speech Pipeline."
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--live", "--gui",
        action="store_true",
        help="Launch always-on Live Camera Viewfinder GUI (see video, aim at book, press SPACE to read)"
    )
    group.add_argument(
        "--input", "-i",
        type=str,
        help="Path to a saved book page image file (e.g. data/input/page1.jpg)"
    )
    group.add_argument(
        "--camera", "-c",
        action="store_true",
        help="One-shot capture using connected Raspberry Pi Camera Module 3"
    )
    group.add_argument(
        "--check-system",
        action="store_true",
        help="Run diagnostic check on OpenCV, Tesseract, TTS, and Camera"
    )
    group.add_argument(
        "--demo",
        action="store_true",
        help="Run pipeline on a sample book page"
    )

    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Do not play audio aloud through speakers (audio file is still saved)"
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable Text-to-Speech entirely"
    )
    parser.add_argument(
        "--method",
        choices=["auto", "otsu", "adaptive_gaussian", "none"],
        default="auto",
        help="Binarization algorithm for OpenCV preprocessing (default: auto)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed quality analysis and debug info"
    )

    args = parser.parse_args()

    # If user ran `python main.py` with no flags, default to --live GUI!
    if not (args.live or args.input or args.camera or args.check_system or args.demo):
        args.live = True

    if args.check_system:
        check_system_diagnostics()
        return

    # Option 1: Interactive Live Viewfinder GUI
    if args.live:
        from ui.live_reader_gui import start_live_reader_gui
        start_live_reader_gui(method=args.method, enable_tts=not args.no_tts)
        return

    # Non-GUI Pipeline Execution
    enable_tts = not args.no_tts
    pipeline = ReadingPipeline(enable_tts=enable_tts)
    play_audio = not args.no_audio

    try:
        if args.camera:
            print("[INFO] Initiating Raspberry Pi Camera capture...")
            result = pipeline.run_on_camera(
                play_audio=play_audio,
                binarization_method=args.method
            )
        elif args.demo:
            sample_path = PROJECT_ROOT / "data" / "input" / "page1_standard_book.png"
            if not sample_path.exists():
                print("[INFO] Creating realistic sample book page for demonstration...")
                from tests.test_samples import generate_sample_book_page
                generate_sample_book_page(sample_path)
            print(f"[INFO] Running demo pipeline on: {sample_path}")
            result = pipeline.run_on_image(
                sample_path,
                play_audio=play_audio,
                binarization_method=args.method
            )
        else:
            input_file = Path(args.input)
            print(f"[INFO] Processing saved image: {input_file}")
            result = pipeline.run_on_image(
                input_file,
                play_audio=play_audio,
                binarization_method=args.method
            )

        if args.verbose:
            print("\n--- QUALITY ANALYSIS DETAIL ---")
            print(result.quality.summary())
            print("\n--- PREPROCESSING STEPS APPLIED ---")
            for step in result.preprocessing.applied_steps:
                print(f"  * {step}")

        print("\n" + result.summary())

    except (ImageLoadError, CameraCaptureError, OCRError, TTSError, ValueError) as e:
        print(f"\n[PIPELINE ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Operation canceled by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
