"""
Benchmark Runner for Smart Reading Glasses (Phase 9).
Executes the full pipeline on 8 distinct test pages and logs qualitative metrics:
1. Standard book page
2. Two-column academic layout
3. Small font citations
4. Large font storybook
5. Uneven shadow gradient
6. Skewed / tilted book page
7. Noisy archival paper
8. Technical instruction manual
"""

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_samples import generate_all_test_pages
from pipeline.reading_pipeline import ReadingPipeline
from config import INPUT_DIR

def run_all_benchmarks():
    print("=" * 70)
    print("SMART READING GLASSES — 8-PAGE COMPREHENSIVE BENCHMARK (PHASE 9)")
    print("=" * 70)

    created_files = generate_all_test_pages(INPUT_DIR)
    print(f"Generated {len(created_files)} test pages in: {INPUT_DIR}\n")

    pipeline = ReadingPipeline(enable_tts=True)

    summary_records = []

    for img_path in created_files:
        print(f"--> Processing: {img_path.name}")
        t0 = time.time()
        res = pipeline.run_on_image(img_path, play_audio=False, save_audio=True)
        elapsed = time.time() - t0

        record = {
            "name": img_path.name,
            "dimensions": f"{res.quality.width}x{res.quality.height}",
            "sharpness": res.quality.sharpness_score,
            "brightness": res.quality.mean_brightness,
            "contrast": res.quality.contrast_std,
            "skew": res.quality.estimated_skew_degrees,
            "warnings_count": len(res.quality.warnings),
            "warnings": res.quality.warnings,
            "words": res.ocr.word_count,
            "confidence": res.ocr.mean_confidence,
            "ocr_time": res.timings.get("ocr", 0.0),
            "total_time": res.total_time,
            "clean_sample": res.cleaned_text.cleaned_text[:70].replace("\n", " ")
        }
        summary_records.append(record)

        print(f"    Words: {record['words']} | Conf: {record['confidence']:.1f}% | Total Time: {record['total_time']:.2f}s")
        if record["warnings"]:
            print(f"    Warnings ({len(record['warnings'])}):")
            for w in record["warnings"]:
                print(f"      - {w}")
        print(f"    Sample: \"{record['clean_sample']}...\"\n")

    print("=" * 70)
    print("BENCHMARK SUMMARY TABLE")
    print("=" * 70)
    header = f"{'Page':<28} | {'Words':<6} | {'Conf%':<6} | {'Time(s)':<8} | {'Quality':<12}"
    print(header)
    print("-" * len(header))
    for r in summary_records:
        status = "Pass" if r["warnings_count"] == 0 else f"{r['warnings_count']} Warn"
        print(f"{r['name']:<28} | {r['words']:<6} | {r['confidence']:<6.1f} | {r['total_time']:<8.2f} | {status:<12}")
    print("=" * 70)

if __name__ == "__main__":
    run_all_benchmarks()
