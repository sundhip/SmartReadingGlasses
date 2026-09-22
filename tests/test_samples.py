"""
Sample Generator for Smart Reading Glasses (Phase 9 Testing).
Generates 8 realistic book/page images with distinct typographical properties:
1. Standard novel page (headings, regular paragraphs)
2. Two-column academic layout
3. Small font index / footnotes
4. Large font storybook
5. Low contrast / uneven lighting shadow gradient
6. Skewed / tilted book page (rotational misalignment)
7. Grainy paper noise
8. Technical manual with bullet points and numerals
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

def get_font(size: int):
    """Attempts to load a clean truetype font, or falls back to PIL default font."""
    try:
        # Standard clean sans-serif on Windows
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()

def generate_sample_book_page(output_path: Path):
    """Generates a standard novel page."""
    w, h = 1000, 1400
    img = Image.new("RGB", (w, h), color=(252, 250, 245))
    draw = ImageDraw.Draw(img)

    title_font = get_font(36)
    heading_font = get_font(26)
    body_font = get_font(20)

    # Book Title
    draw.text((w // 2, 80), "THE ART OF READING", fill=(20, 20, 20), font=title_font, anchor="ms")
    draw.text((w // 2, 130), "Chapter 1: The Modern Printed Word", fill=(60, 60, 60), font=heading_font, anchor="ms")
    draw.line([(150, 160), (850, 160)], fill=(180, 180, 180), width=2)

    paragraphs = [
        "Reading is an essential cognitive human endeavor that enables individuals to absorb "
        "knowledge across centuries. Throughout history, books have preserved philosophical ideas, "
        "scientific discoveries, and literary achievements.",

        "In modern society, printed books remain a vital bridge for continuous education and culture. "
        "Assistive reading technologies empower people with visual impairments or reading difficulties "
        "to interact with physical literature independently.",

        "A reading system combining optical character recognition with natural text-to-speech provides "
        "immediate voice feedback, transforming ink on physical paper into accessible spoken audio.",

        "The engineering objective of wearable reading glasses is to deliver accurate text capture, "
        "lightweight processing on edge devices, and responsive speech playback without requiring "
        "constant cloud connectivity."
    ]

    y = 210
    line_spacing = 32
    for para in paragraphs:
        # Simple word wrap
        words = para.split()
        current_line = []
        for word in words:
            test_line = " ".join(current_line + [word])
            # Draw approximate width
            if len(test_line) > 60:
                draw.text((120, y), " ".join(current_line), fill=(30, 30, 30), font=body_font)
                y += line_spacing
                current_line = [word]
            else:
                current_line.append(word)
        if current_line:
            draw.text((120, y), " ".join(current_line), fill=(30, 30, 30), font=body_font)
            y += line_spacing
        y += 24  # Paragraph break

    # Page footer
    draw.text((w // 2, 1330), "- Page 1 -", fill=(100, 100, 100), font=body_font, anchor="ms")

    img.save(str(output_path))
    return output_path

def generate_two_column_page(output_path: Path):
    """Generates a 2-column textbook layout."""
    w, h = 1200, 1500
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_font = get_font(34)
    col_font = get_font(18)

    draw.text((w // 2, 70), "ADVANCED EMBEDDED COMPUTING", fill=(0, 0, 0), font=title_font, anchor="ms")
    draw.line([(80, 100), (w - 80, 100)], fill=(120, 120, 120), width=2)

    col1_text = [
        "1.1 Microcontroller Architectures",
        "Embedded edge processors require",
        "careful power and memory budgets.",
        "When designing wearable hardware,",
        "engineers evaluate battery life,",
        "thermal limits, and real-time audio",
        "throughput. Low latency pipelines",
        "ensure smooth speech playback."
    ]

    col2_text = [
        "1.2 Optical Sensor Interfaces",
        "Modern camera modules provide",
        "high resolution image capture with",
        "fast autofocus mechanisms.",
        "Image preprocessing filters reduce",
        "background noise and correct uneven",
        "illumination before feeding data",
        "into the optical recognition core."
    ]

    # Draw Column 1
    y = 140
    for line in col1_text:
        draw.text((100, y), line, fill=(20, 20, 20), font=col_font)
        y += 28

    # Draw Column 2
    y = 140
    for line in col2_text:
        draw.text((650, y), line, fill=(20, 20, 20), font=col_font)
        y += 28

    draw.line([(600, 130), (600, 420)], fill=(200, 200, 200), width=1)
    img.save(str(output_path))
    return output_path

def generate_small_font_page(output_path: Path):
    """Generates a small print index/notes page."""
    w, h = 900, 1200
    img = Image.new("RGB", (w, h), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)

    font_title = get_font(28)
    font_small = get_font(14)

    draw.text((80, 60), "REFERENCE NOTES AND CITATIONS", fill=(10, 10, 10), font=font_title)
    draw.line([(80, 95), (820, 95)], fill=(150, 150, 150), width=1)

    notes = [
        "[1] Smith, J. Principles of Optical Character Recognition, Academic Press, 2021.",
        "[2] Raspberry Pi Foundation. Raspberry Pi Camera Module 3 Specifications, 2023.",
        "[3] Davies, E. R. Computer and Machine Vision: Theory, Algorithms, Practicalities, 2018.",
        "[4] Tesseract Open Source OCR Engine Documentation and Usage Guidelines, 2024.",
        "[5] World Health Organization. World Report on Vision and Assistive Devices, 2020.",
        "[6] IEEE Standard for Wearable Electronics and Embedded Sensor Interfaces, 2022."
    ]

    y = 130
    for note in notes:
        draw.text((80, y), note, fill=(30, 30, 30), font=font_small)
        y += 32

    img.save(str(output_path))
    return output_path

def generate_large_font_page(output_path: Path):
    """Generates large font text."""
    w, h = 1000, 800
    img = Image.new("RGB", (w, h), color=(255, 253, 240))
    draw = ImageDraw.Draw(img)

    font_large = get_font(40)
    lines = [
        "CHAPTER 5",
        "THE JOURNEY BEGINS",
        "Every grand adventure starts",
        "with a single printed sentence."
    ]

    y = 160
    for line in lines:
        draw.text((w // 2, y), line, fill=(20, 20, 20), font=font_large, anchor="ms")
        y += 80

    img.save(str(output_path))
    return output_path

def generate_shadow_page(output_path: Path):
    """Generates a page with an illumination shadow gradient across it."""
    w, h = 900, 1100
    img = Image.new("L", (w, h), color=255)
    draw = ImageDraw.Draw(img)

    body_font = get_font(22)
    lines = [
        "LIGHTING EXPERIMENT SAMPLE",
        "This test page evaluates how well the CLAHE",
        "contrast enhancement and Otsu binarization",
        "can recover text obscured by uneven ambient shadows.",
        "Real-world reading glasses frequently encounter",
        "directional shadows from room overhead lamps."
    ]

    y = 120
    for line in lines:
        draw.text((80, y), line, fill=20, font=body_font)
        y += 48

    # Apply synthetic shadow gradient from top-left to bottom-right
    arr = np.array(img, dtype=np.float32)
    grad_x = np.linspace(0.4, 1.0, w)
    grad_y = np.linspace(0.4, 1.0, h)
    grid_x, grid_y = np.meshgrid(grad_x, grad_y)
    shadow_mask = grid_x * grid_y
    shadowed = (arr * shadow_mask).clip(0, 255).astype(np.uint8)

    cv2.imwrite(str(output_path), shadowed)
    return output_path

def generate_skewed_page(output_path: Path):
    """Generates a page with 7 degrees rotational skew."""
    base_path = output_path.parent / "temp_unskewed.png"
    generate_sample_book_page(base_path)

    img = cv2.imread(str(base_path))
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, 6.5, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, (w, h), borderValue=(252, 250, 245))

    cv2.imwrite(str(output_path), rotated)
    if base_path.exists():
        base_path.unlink()
    return output_path

def generate_noisy_page(output_path: Path):
    """Generates a page with paper grain and noise."""
    w, h = 900, 1000
    img = Image.new("RGB", (w, h), color=(245, 242, 235))
    draw = ImageDraw.Draw(img)

    font_text = get_font(22)
    lines = [
        "ARCHIVAL PAPER SAMPLE",
        "Simulating antique or low-grade recycled paper.",
        "The bilateral filter preserves the sharp character",
        "strokes while removing paper grain and speckles."
    ]

    y = 120
    for line in lines:
        draw.text((100, y), line, fill=(35, 30, 25), font=font_text)
        y += 50

    arr = np.array(img, dtype=np.int16)
    noise = np.random.normal(0, 15, arr.shape).astype(np.int16)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_path), noisy_arr)
    return output_path

def generate_technical_manual_page(output_path: Path):
    """Generates a technical page with bullets and numerals."""
    w, h = 950, 1100
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    font_header = get_font(26)
    font_body = get_font(20)

    draw.text((80, 60), "TECHNICAL INSTRUCTION MANUAL: REV 2.0", fill=(0, 0, 0), font=font_header)
    draw.line([(80, 95), (870, 95)], fill=(150, 150, 150), width=2)

    lines = [
        "Step 1: Check power supply voltage at 5.0V +/- 5%.",
        "Step 2: Connect ribbon cable between camera and CSI port.",
        "Step 3: Run pipeline with parameters: --method otsu --no-audio.",
        "Step 4: Verify OCR extracted confidence exceeds 80%.",
        "Step 5: Ensure speech synthesizer audio latency is under 200ms."
    ]

    y = 130
    for line in lines:
        draw.text((80, y), line, fill=(20, 20, 20), font=font_body)
        y += 45

    img.save(str(output_path))
    return output_path

def generate_all_test_pages(data_input_dir: Path):
    """Generates full suite of 8 diverse test pages."""
    data_input_dir.mkdir(parents=True, exist_ok=True)
    pages = {
        "page1_standard_book.png": generate_sample_book_page,
        "page2_two_column.png": generate_two_column_page,
        "page3_small_font.png": generate_small_font_page,
        "page4_large_font.png": generate_large_font_page,
        "page5_shadow_gradient.png": generate_shadow_page,
        "page6_tilted_skew.png": generate_skewed_page,
        "page7_noisy_paper.png": generate_noisy_page,
        "page8_technical_manual.png": generate_technical_manual_page,
    }
    created_paths = []
    for filename, generator in pages.items():
        out_p = data_input_dir / filename
        generator(out_p)
        created_paths.append(out_p)
    return created_paths

if __name__ == "__main__":
    from config import INPUT_DIR
    created = generate_all_test_pages(INPUT_DIR)
    print(f"Generated {len(created)} test pages in {INPUT_DIR}")
