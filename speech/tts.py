"""
Enhanced Text-to-Speech (TTS) Module (Phase 7).
Provides high-clarity, human-sounding offline speech synthesis:
1. SVOX Pico (pico2wave) — Natural, warm human inflection (preferred for embedded Linux).
2. Tuned espeak-ng — Enhanced with natural prosody (-v en-us+f3, adjusted pitch & cadence).
3. pyttsx3 — SAPI5 on Windows / speech-dispatcher fallback.
Guarantees immediate speaker playback via native PipeWire/ALSA/winsound.
"""

from pathlib import Path
from typing import Optional
import os
import sys
import shutil
import subprocess
import pyttsx3

from config import TTSConfig, OUTPUT_DIR

class TTSError(Exception):
    """Raised when TTS engine fails to initialize, speak, or save audio."""
    pass

def play_audio_file(audio_path: str) -> bool:
    """
    Plays a WAV audio file directly through system speakers/headphones.
    Compatible with Raspberry Pi OS Bookworm (PipeWire / ALSA) and Windows.
    """
    path = str(Path(audio_path).resolve())
    if not os.path.exists(path):
        return False

    if sys.platform.startswith("linux"):
        for player in ["pw-play", "paplay", "aplay"]:
            if shutil.which(player):
                try:
                    subprocess.run(
                        [player, path],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    return True
                except Exception:
                    continue
    elif sys.platform == "win32":
        try:
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
            return True
        except Exception:
            pass

    return False

class TextToSpeechEngine:
    """
    High-clarity, lightweight, offline Text-to-Speech engine.
    Calibrated for calm, natural book reading pace (120-130 WPM) with clear pronunciation.
    """

    def __init__(
        self,
        rate: int = TTSConfig.RATE,
        volume: float = TTSConfig.VOLUME,
        voice_variant: str = "en-us+f3",
    ):
        self.rate = rate
        self.volume = volume
        self.voice_variant = voice_variant
        self.has_pico = shutil.which("pico2wave") is not None
        self.has_espeak = shutil.which("espeak-ng") is not None

    def is_available(self) -> bool:
        """Returns True if at least one TTS engine is present."""
        return self.has_pico or self.has_espeak or (pyttsx3 is not None)

    def speak(
        self,
        text: str,
        play_audio: bool = True,
        save_path: Optional[Path | str] = None,
    ) -> Optional[str]:
        """
        Synthesizes text into calm, high-clarity spoken audio and plays it aloud.
        """
        if not text or not text.strip():
            text_to_speak = "No readable text was detected on the page."
        else:
            from ocr.postprocess import prepare_text_for_speech
            text_to_speak = prepare_text_for_speech(text)

        # Destination audio file
        if save_path:
            target_audio_file = str(save_path)
        else:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            target_audio_file = str(OUTPUT_DIR / TTSConfig.AUDIO_FILENAME)

        Path(target_audio_file).parent.mkdir(parents=True, exist_ok=True)
        synthesized = False

        # ---------------------------------------------------------
        # Engine Option 1: SVOX Pico (Natural human inflection on Pi)
        # ---------------------------------------------------------
        if self.has_pico and sys.platform.startswith("linux"):
            try:
                # Use SVOX speed level tag to slow down rushed speech to calm book reading pace
                pico_speed = getattr(TTSConfig, "PICO_SPEED_LEVEL", 82)
                pico_text = f"<speed level='{pico_speed}'>{text_to_speak}</speed>"
                cmd = ["pico2wave", "-l", "en-US", "-w", target_audio_file, pico_text]
                res = subprocess.run(cmd, capture_output=True, timeout=12)
                if res.returncode == 0 and os.path.exists(target_audio_file) and os.path.getsize(target_audio_file) > 100:
                    synthesized = True
                else:
                    # Fallback to plain text without tags if tag was unsupported
                    cmd = ["pico2wave", "-l", "en-US", "-w", target_audio_file, text_to_speak]
                    res = subprocess.run(cmd, capture_output=True, timeout=12)
                    if res.returncode == 0 and os.path.exists(target_audio_file):
                        synthesized = True
            except Exception:
                pass

        # ---------------------------------------------------------
        # Engine Option 2: Tuned espeak-ng (Calm Cadence & Natural Pauses)
        # ---------------------------------------------------------
        if not synthesized and self.has_espeak and sys.platform.startswith("linux"):
            try:
                # -v en-us+f3: Natural female English voice
                # -p 48: Warm, natural pitch
                # -s 125: Calm, comfortable reading pace (not rushed)
                # -g 12: Increased gap between words for distinct enunciation
                cmd = [
                    "espeak-ng",
                    "-v", self.voice_variant,
                    "-p", "48",
                    "-s", str(self.rate),
                    "-g", "12",
                    "-w", target_audio_file,
                    text_to_speak
                ]
                res = subprocess.run(cmd, capture_output=True, timeout=12)
                if res.returncode == 0 and os.path.exists(target_audio_file):
                    synthesized = True
            except Exception:
                pass

        # ---------------------------------------------------------
        # Engine Option 3: pyttsx3 (SAPI5 on Windows / Standard Fallback)
        # ---------------------------------------------------------
        if not synthesized:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)
                voices = engine.getProperty("voices")
                if voices:
                    engine.setProperty("voice", voices[0].id)
                engine.save_to_file(text_to_speak, target_audio_file)
                engine.runAndWait()
                engine.stop()
                del engine
                if os.path.exists(target_audio_file):
                    synthesized = True
            except Exception:
                pass

        # ---------------------------------------------------------
        # Audio Playback Through Speakers
        # ---------------------------------------------------------
        if play_audio and os.path.exists(target_audio_file):
            played = play_audio_file(target_audio_file)
            if not played and self.has_espeak:
                subprocess.run(
                    [
                        "espeak-ng",
                        "-v", self.voice_variant,
                        "-p", "48",
                        "-s", str(self.rate),
                        "-g", "12",
                        text_to_speak
                    ],
                    check=False
                )

        return target_audio_file if os.path.exists(target_audio_file) else None
