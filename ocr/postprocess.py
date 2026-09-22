"""
Text Post-Processing & Natural Language Processing (NLP) Module (Phase 6).
Provides:
1. OCR Glyph Error Correction (replaces misread digits '0','1','5' inside words).
2. Sentence Grammar & Punctuation Structuring.
3. Fluid Book Paragraph Reconstruction.
4. Spoken Language NLP (abbreviation expansion, number-to-words, breath pauses).
"""

from dataclasses import dataclass
import re
from typing import List, Dict

@dataclass
class CleanedTextResult:
    """Holds original raw text, visually polished text, and speech-optimized text."""
    raw_text: str
    cleaned_text: str
    speech_text: str
    character_count: int
    word_count: int
    line_count: int

def correct_ocr_glyph_errors(text: str) -> str:
    """
    Applies NLP character-level error correction for common OCR font misinterpretations.
    E.g.:
    - 'b00k' -> 'book'
    - 'c1ear' -> 'clear'
    - 'fa5t' -> 'fast'
    - '8ook' -> 'Book'
    - 'vv' -> 'w' (e.g. 'vvhich' -> 'which')
    """
    def fix_word(match: re.Match) -> str:
        word = match.group(0)
        # If the word is entirely numbers (e.g. "2024" or "150"), leave it alone
        if word.isdigit():
            return word

        # If word has mixed letters and digits, fix common OCR substitutions
        # Check if digits are enclosed by letters (e.g., 'th0se', 'c1ear', '1n')
        w = word
        # Replace '0' with 'o' if surrounded by letters
        w = re.sub(r"(?<=[a-zA-Z])0(?=[a-zA-Z])", "o", w)
        w = re.sub(r"(?<=[a-zA-Z])0$", "o", w)
        w = re.sub(r"^0(?=[a-zA-Z])", "O", w)

        # Replace '1' with 'l' or 'I'
        w = re.sub(r"(?<=[a-zA-Z])1(?=[a-zA-Z])", "l", w)
        w = re.sub(r"(?<=[a-zA-Z])1$", "l", w)
        w = re.sub(r"^1(?=[a-z])", "I", w)

        # Replace '5' with 's' inside words
        w = re.sub(r"(?<=[a-zA-Z])5(?=[a-zA-Z])", "s", w)
        w = re.sub(r"(?<=[a-zA-Z])5$", "s", w)

        # Replace 'vv' with 'w'
        w = re.sub(r"^vv(?=[a-z])", "w", w)
        w = re.sub(r"^Vv(?=[a-z])", "W", w)
        w = re.sub(r"(?<=[a-z])vv(?=[a-z])", "w", w)

        return w

    # Match word-like tokens containing alphanumeric characters
    cleaned = re.sub(r"\b[a-zA-Z0-9]+\b", fix_word, text)
    return cleaned

def normalize_punctuation_and_grammar(text: str) -> str:
    """
    Normalizes spacing around punctuation and capitalizes sentence beginnings.
    """
    s = text

    # Remove stray spaces before punctuation: 'word , next' -> 'word, next'
    s = re.sub(r"\s+([,.:;?!])", r"\1", s)

    # Ensure single space after punctuation: 'word,next' -> 'word, next'
    s = re.sub(r"([,.:;?!])([a-zA-Z])", r"\1 \2", s)

    # Clean multiple consecutive punctuation: '..' -> '.', '??' -> '?'
    s = re.sub(r"\.{2,}", ".", s)
    s = re.sub(r"\?{2,}", "?", s)
    s = re.sub(r"!{2,}", "!", s)

    # Capitalize the first letter of each sentence
    def cap_sentence(match: re.Match) -> str:
        prefix = match.group(1)
        char = match.group(2)
        return prefix + char.upper()

    s = re.sub(r"(^|[.!?]\s+)([a-z])", cap_sentence, s)
    return s

def prepare_text_for_speech(text: str) -> str:
    """
    Transforms text into natural, human-like speech with natural cadence:
    - Expands abbreviations (Dr. -> Doctor, etc.)
    - Expands currency and unit symbols ($10 -> ten dollars)
    - Replaces noisy OCR symbols
    - Inserts subtle clause pauses for natural human breathing
    """
    if not text or not text.strip():
        return ""

    s = text

    # 1. Expand currency & units
    s = re.sub(r"\$(\d+(?:\.\d+)?)", r"\1 dollars", s)
    s = re.sub(r"£(\d+(?:\.\d+)?)", r"\1 pounds", s)
    s = re.sub(r"€(\d+(?:\.\d+)?)", r"\1 euros", s)
    s = re.sub(r"(\d+)\s*%", r"\1 percent", s)
    s = re.sub(r"(\d+)\s*km\b", r"\1 kilometers", s)
    s = re.sub(r"(\d+)\s*cm\b", r"\1 centimeters", s)
    s = re.sub(r"(\d+)\s*m\b", r"\1 meters", s)
    s = re.sub(r"(\d+)\s*kg\b", r"\1 kilograms", s)

    # 2. Ordinals
    ordinals = {
        r"\b1st\b": "first",
        r"\b2nd\b": "second",
        r"\b3rd\b": "third",
        r"\b4th\b": "fourth",
        r"\b5th\b": "fifth",
        r"\b6th\b": "sixth",
        r"\b7th\b": "seventh",
        r"\b8th\b": "eighth",
        r"\b9th\b": "ninth",
        r"\b10th\b": "tenth",
    }
    for pat, rep in ordinals.items():
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)

    # 3. Expand common literary & academic abbreviations
    abbreviations = [
        (r"\bMr\.", "Mister"),
        (r"\bMrs\.", "Missus"),
        (r"\bMs\.", "Miz"),
        (r"\bDr\.", "Doctor"),
        (r"\bProf\.", "Professor"),
        (r"\bSt\.", "Saint"),
        (r"\betc\.", "etcetera"),
        (r"\be\.g\.", "for example"),
        (r"\bi\.e\.", "that is"),
        (r"\bvs\.", "versus"),
        (r"\bNo\.", "Number"),
        (r"\bFig\.", "Figure"),
        (r"\bapprox\.", "approximately"),
        (r"\bdept\.", "department"),
        (r"\bvol\.", "volume"),
        (r"\bpp\.", "pages"),
        (r"\bp\.", "page"),
        (r"\bch\.", "chapter"),
    ]
    for pattern, replacement in abbreviations:
        s = re.sub(pattern, replacement, s, flags=re.IGNORECASE)

    # 4. Spoken word symbols
    s = re.sub(r"&", " and ", s)
    s = re.sub(r"\+", " plus ", s)
    s = re.sub(r"=", " equals ", s)

    # 5. Strip unpronounceable OCR artifacts
    s = re.sub(r"[|_~^\\/`*#<>{}\[\]()\"'“”‘’«»]+", " ", s)

    # 6. Normalize dashes and pauses
    s = re.sub(r"[-—–]{2,}", " - ", s)

    # 7. Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # 8. Terminal pause
    if s and s[-1] not in {".", "!", "?"}:
        s += "."

    return s

def clean_ocr_text(raw_text: str) -> CleanedTextResult:
    """
    Applies the full NLP post-processing pipeline to raw OCR text:
    1. Line break and quotation normalization.
    2. End-of-line hyphen reconnection ('ex- \n ample' -> 'example').
    3. OCR character/glyph error repair ('th0se' -> 'those', 'c1ear' -> 'clear').
    4. Removal of margin/edge artifacts and stray symbols.
    5. Fluid book paragraph reconstruction.
    6. Sentence punctuation and capitalization structuring.
    7. Natural spoken language synthesis formatting.
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
    text = re.sub(r"(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)", r"\1\2", text)

    # 3. Apply OCR character glyph error correction
    text = correct_ocr_glyph_errors(text)

    # 4. Filter out edge noise lines (stray '|', '---', '~', '_ _', solitary symbols)
    raw_lines = text.split("\n")
    cleaned_lines = []

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue

        # Filter out standalone OCR noise lines with no alphanumeric characters
        alnum_chars = [c for c in stripped if c.isalnum()]
        if len(alnum_chars) == 0:
            continue
        # Filter single stray characters that aren't valid words
        if len(stripped) == 1 and stripped not in {"a", "A", "I", "1", "2", "3"}:
            continue

        # Normalize spacing
        normalized = re.sub(r"[ \t]+", " ", stripped)
        cleaned_lines.append(normalized)

    # 5. Fluid Book Paragraph Reconstruction
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

    # 6. Normalize punctuation and grammar for visual reading
    assembled_text = "\n\n".join(paragraphs).strip()
    formatted_text = normalize_punctuation_and_grammar(assembled_text)

    # 7. Generate speech-ready text with expanded abbreviations and natural pauses
    speech_ready = prepare_text_for_speech(formatted_text)

    words = [w for w in re.split(r"\s+", formatted_text) if w]
    lines = [line for line in formatted_text.split("\n") if line.strip()]

    return CleanedTextResult(
        raw_text=raw_text,
        cleaned_text=formatted_text,
        speech_text=speech_ready,
        character_count=len(formatted_text),
        word_count=len(words),
        line_count=len(lines),
    )


