"""
Raspberry Pi Camera Acquisition Module (Method 2: Camera Capture).
Supports Raspberry Pi Camera Module 3 Standard with autofocus.
Uses the modern Raspberry Pi camera stack (Picamera2 / rpicam-still / libcamera-still).
Designed for Raspberry Pi 5 and compatible with Raspberry Pi Zero 2 W.
Gracefully detects if camera hardware is unavailable (e.g., during PC development).
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import os
import sys
import shutil
import subprocess
import time
import numpy as np
import cv2

from config import CameraConfig, INPUT_DIR

class CameraCaptureError(Exception):
    """Raised when camera initialization or capture fails."""
    pass

class RaspberryPiCamera:
    """
    Interface for Raspberry Pi Camera Module 3.
    Attempts hardware access via Picamera2 (modern Python API)
    or rpicam-still / libcamera-still CLI tools.
    """

    def __init__(
        self,
        width: int = CameraConfig.DEFAULT_CAPTURE_WIDTH,
        height: int = CameraConfig.DEFAULT_CAPTURE_HEIGHT,
        autofocus: bool = True,
        warmup_seconds: float = CameraConfig.WARMUP_SECONDS,
    ):
        self.width = width
        self.height = height
        self.autofocus = autofocus
        self.warmup_seconds = warmup_seconds
        self._picam2 = None
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        """
        Detects which camera backend is available on the current OS/device.
        Returns: 'picamera2', 'rpicam-still', 'libcamera-still', or 'unavailable'
        """
        # Try native Picamera2 first (official library name is 'picamera2')
        try:
            import picamera2
            return "picamera2"
        except ImportError:
            pass

        # Try rpicam-still CLI (modern Raspberry Pi OS Bookworm utility for Pi 5 & Pi Zero 2 W)
        if shutil.which("rpicam-still"):
            return "rpicam-still"

        # Try libcamera-still (Raspberry Pi OS Bullseye utility)
        if shutil.which("libcamera-still"):
            return "libcamera-still"

        return "unavailable"

    def is_available(self) -> bool:
        """Checks if Raspberry Pi camera hardware stack is present."""
        return self._backend != "unavailable"

    def capture(self, output_path: Optional[Path | str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Captures an image from Raspberry Pi Camera Module 3.
        """
        if output_path is None:
            save_path = INPUT_DIR / CameraConfig.DEFAULT_SAVE_NAME
        else:
            save_path = Path(output_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self._backend == "unavailable":
            raise CameraCaptureError(
                "Raspberry Pi Camera is not available on this system.\n"
                "To use the camera:\n"
                "1. Connect Raspberry Pi Camera Module 3 to the Raspberry Pi 5 or Pi Zero 2 W via ribbon cable.\n"
                "2. Ensure Raspberry Pi OS (Bookworm or later) is running with 'rpicam-still' or 'python3-picamera2' installed.\n"
                "3. Verify detection using terminal command: 'rpicam-hello' or 'rpicam-hello --list-cameras'."
            )

        if self._backend == "picamera2":
            return self._capture_picamera2(save_path)
        elif self._backend in ("rpicam-still", "libcamera-still"):
            return self._capture_cli(save_path, self._backend)
        else:
            raise CameraCaptureError(f"Unsupported camera backend: {self._backend}")

    def _capture_picamera2(self, save_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Captures using the official Picamera2 Python API."""
        try:
            from picamera2 import Picamera2

            picam = Picamera2()
            camera_config = picam.create_still_configuration(
                main={"size": (self.width, self.height), "format": "BGR888"}
            )
            picam.configure(camera_config)
            picam.start()

            # Warmup and autofocus trigger: Macro range (10cm-50cm) prevents hunting to infinity
            if self.autofocus:
                try:
                    from libcamera import controls
                    picam.set_controls({
                        "AfMode": controls.AfModeEnum.Auto,
                        "AfRange": controls.AfRangeEnum.Macro,
                        "AfSpeed": controls.AfSpeedEnum.Fast,
                        "AfTrigger": controls.AfTriggerEnum.Start,
                    })
                except Exception:
                    try:
                        picam.set_controls({
                            "AfMode": 1,       # Auto single-shot focus lock
                            "AfRange": 1,      # Macro range (10cm - 50cm for books)
                            "AfSpeed": 1,      # Fast autofocus response
                            "AfTrigger": 0,    # Trigger AF scan
                        })
                    except Exception:
                        pass
            time.sleep(self.warmup_seconds)

            image = picam.capture_array()
            picam.stop()
            picam.close()

            cv2.imwrite(str(save_path), image)

            metadata = {
                "source": "Raspberry Pi Camera Module 3",
                "backend": "Picamera2",
                "file_path": str(save_path),
                "width": image.shape[1],
                "height": image.shape[0],
                "channels": image.shape[2] if len(image.shape) > 2 else 1,
                "timestamp": time.time(),
            }
            return image, metadata
        except Exception as e:
            # Fallback to CLI if Picamera2 had a driver collision
            if shutil.which("rpicam-still"):
                return self._capture_cli(save_path, "rpicam-still")
            raise CameraCaptureError(f"Picamera2 capture failed: {e}")

    def _capture_cli(self, save_path: Path, binary_name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Captures using rpicam-still or libcamera-still CLI."""
        cmd = [
            binary_name,
            "-o", str(save_path),
            "--width", str(self.width),
            "--height", str(self.height),
            "-t", str(int(self.warmup_seconds * 1000)),
            "-n",
        ]

        if self.autofocus:
            cmd.extend([
                "--autofocus-mode", "auto",
                "--autofocus-range", "macro",
                "--autofocus-speed", "fast"
            ])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode != 0:
                raise CameraCaptureError(
                    f"{binary_name} exited with code {result.returncode}:\n{result.stderr}"
                )
        except subprocess.TimeoutExpired:
            raise CameraCaptureError(f"{binary_name} timed out waiting for camera capture.")
        except Exception as e:
            raise CameraCaptureError(f"Failed to execute {binary_name}: {e}")

        if not save_path.exists() or save_path.stat().st_size == 0:
            raise CameraCaptureError(f"Camera command succeeded but output file was not created at {save_path}")

        image = cv2.imread(str(save_path))
        if image is None:
            raise CameraCaptureError(f"Unable to read captured image from {save_path}")

        metadata = {
            "source": "Raspberry Pi Camera Module 3",
            "backend": binary_name,
            "file_path": str(save_path),
            "width": image.shape[1],
            "height": image.shape[0],
            "channels": image.shape[2] if len(image.shape) > 2 else 1,
            "timestamp": time.time(),
        }
        return image, metadata
