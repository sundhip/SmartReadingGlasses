"""
Text Post-Processing Module (Phase 6).
Conservatively cleans raw OCR text while preserving genuine punctuation and words.
Strictly separates RAW OCR TEXT from CLEANED OCR TEXT, and provides speech-ready formatting.
"""

from dataclasses import dataclass
import re
from typing import List

@dataclass
class CleanedTextResult:
    """Holds both original raw text and sanitized text with transformation metrics."""
    raw_text: str
    cleaned_text: str
    speech_text: str
    character_count: int
    word_count: int
    line_count: int

def prepare_text_for_speech(text: str) -> str:
    """
    Transforms cleaned text into natural, human-like speech format:
    - Expands common abbreviations (Mr., Dr., etc.) so TTS pronounces them properly.
    - Removes unpronounceable OCR artifacts and symbols that cause robotic glitching.
    - Normalizes punctuation pauses so the reader breathes and pauses naturally.
    """
    if not text or not text.strip():
        return ""

    s = text

    # 1. Expand common abbreviations
    abbreviations = [
        (r"\bMr\.", "Mister"),
        (r"\bMrs\.", "Missus"),
        (r"\bMs\.", "Miz"),
        (r"\bDr\.", "Doctor"),
        (r"\bProf\.", "Professor"),
        (r"\betc\.", "etcetera"),
        (r"\be\.g\.", "for example"),
        (r"\bi\.e\.", "that is"),
        (r"\bvs\.", "versus"),
        (r"\bNo\.", "Number"),
        (r"\bFig\.", "Figure"),
        (r"\bapprox\.", "approximately"),
        (r"\bdept\.", "department"),
    ]
    for pattern, replacement in abbreviations:
        s = re.sub(pattern, replacement, s, flags=re.IGNORECASE)

    # 2. Replace symbols with spoken words
    s = re.sub(r"&", " and ", s)
    s = re.sub(r"%", " percent ", s)
    s = re.sub(r"\+", " plus ", s)
    s = re.sub(r"=", " equals ", s)

    # 3. Strip symbols that cause TTS stutter or literal symbol reading
    # e.g., pipes, underscores, tildes, slashes, brackets, asterisks
    s = re.sub(r"[|_~^\\/`*#<>{}\[\]()\"'“”‘’]+", " ", s)

    # 4. Clean up multiple dashes and ellipsis to a gentle pause
    s = re.sub(r"[-—–]{2,}", " - ", s)
    s = re.sub(r"\.{2,}", "...", s)

    # 5. Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # 6. Ensure speech ends with a clean terminal pause
    if s and s[-1] not in {".", "!", "?"}:
        s += "."

    return s

def clean_ocr_text(raw_text: str) -> CleanedTextResult:
    """
    Applies book-friendly post-processing to raw OCR text:
    1. Normalizes line endings (\\r\\n -> \\n).
    2. Repairs hyphenated word splits across line wraps (e.g. 'auto-\\n matic' -> 'automatic').
    3. Cleans isolated OCR border noise lines (stray '|', '~', '_', margin specks).
    4. Reconnects wrapped lines inside book paragraphs into smooth, fluid sentences.
    5. Preserves legitimate paragraph breaks and section headings.
    6. Generates speech-optimized text.

    Args:
        raw_text: Raw string directly from Tesseract.

    Returns:
        CleanedTextResult with raw, cleaned, and speech-ready text versions.
    """
    if not raw_text or not raw_text.strip():
        return CleanedTextResult(
            raw_text=raw_text,
            cleaned_text="",
            speech_text="",
            character_count=0,
            word_count=0,
            line_count=0,
        )

    # 1. Normalize line breaks and quotes
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # 2. Repair end-of-line hyphenation across wrapped lines
    # Handles: 'inter-\n national' -> 'international'
    text = re.sub(r"(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)", r"\1\2", text)

    # 3. Process line by line to filter edge noise and header/footer artifacts
    raw_lines = text.split("\n")
    cleaned_lines = []

    for line in raw_lines:
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append("")
            continue

        # Filter out standalone OCR noise lines (e.g., '|', '---', '~', '_ _', '1')
        alnum_chars = [c for c in stripped if c.isalnum()]
        if len(alnum_chars) == 0:
            continue
        # Filter single stray characters that aren't valid words ('a', 'I')
        if len(stripped) == 1 and stripped not in {"a", "A", "I", "1", "2", "3"}:
            continue

        # Normalize internal spacing
        normalized = re.sub(r"[ \t]+", " ", stripped)
        cleaned_lines.append(normalized)

    # 4. Fluid Book Paragraph Reconstruction
    # In printed books, sentences wrap across multiple lines within a paragraph.
    # Lines should be joined with spaces unless there is an empty line or a clear heading.
    paragraphs = []
    current_para = []

    for line in cleaned_lines:
        if not line:
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
        else:
            current_para.append(line)

    if current_para:
        paragraphs.append(" ".join(current_para))

    final_text = "\n\n".join(paragraphs).strip()

    # Generate speech-ready text with expanded abbreviations and natural pauses
    speech_ready = prepare_text_for_speech(final_text)

    words = [w for w in re.split(r"\s+", final_text) if w]
    lines = [line for line in final_text.split("\n") if line.strip()]

    return CleanedTextResult(
        raw_text=raw_text,
        cleaned_text=final_text,
        speech_text=speech_ready,
        character_count=len(final_text),
        word_count=len(words),
        line_count=len(lines),
    )

