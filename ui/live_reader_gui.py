"""
Professional Desktop GUI for Smart Reading Glasses (Review 2).
Built with native Tkinter and Picamera2 / OpenCV.
Features:
- Live 30 FPS Camera Viewfinder
- Target book framing guide
- Real-time sharpness & autofocus indicator
- One-click [READ BOOK PAGE] button (also triggered by SPACEBAR)
- Hands-Free Auto-Read Mode
- High-contrast, large-font readable text display area
- [🔊 Replay Speech] button
- Stage latency & confidence status bar
"""

import sys
import os
from pathlib import Path
from typing import Optional
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

from pipeline.reading_pipeline import ReadingPipeline
from speech.tts import play_audio_file
from config import QualityConfig, INPUT_DIR


class LiveCameraStream:
    """Streams live frames from Raspberry Pi Camera Module 3 or OpenCV webcam with Fast Macro Autofocus."""

    def __init__(self, preview_width: int = 640, preview_height: int = 480, capture_width: int = 1920, capture_height: int = 1080):
        self.p_width = preview_width
        self.p_height = preview_height
        self.c_width = capture_width
        self.c_height = capture_height
        self.backend = "unknown"
        self._picam2 = None
        self._cap = None
        self._has_lores = False
        self._init_camera()

    def _init_camera(self):
        # 1. Native Picamera2 (Raspberry Pi 5 / Bookworm with Camera Module 3)
        try:
            from picamera2 import Picamera2
            picam = Picamera2()

            # Configure dual stream: High-Res (1080p) for OCR, Lo-Res (640x480) for 30 FPS preview
            try:
                config = picam.create_preview_configuration(
                    main={"size": (self.c_width, self.c_height), "format": "BGR888"},
                    lores={"size": (self.p_width, self.p_height), "format": "BGR888"}
                )
                picam.configure(config)
                self._has_lores = True
            except Exception:
                config = picam.create_preview_configuration(
                    main={"size": (1280, 720), "format": "BGR888"}
                )
                picam.configure(config)
                self._has_lores = False

            # Configure Camera Module 3 Autofocus:
            # AfMode: 2 (Continuous) or 1 (Auto)
            # AfRange: 1 (Macro: 10cm to 50cm - strictly optimizes for book reading, avoids infinite hunting)
            # AfSpeed: 1 (Fast lens response)
            try:
                from libcamera import controls
                picam.set_controls({
                    "AfMode": controls.AfModeEnum.Continuous,
                    "AfRange": controls.AfRangeEnum.Macro,
                    "AfSpeed": controls.AfSpeedEnum.Fast,
                })
            except Exception:
                try:
                    picam.set_controls({
                        "AfMode": 2,
                        "AfRange": 1,
                        "AfSpeed": 1,
                    })
                except Exception:
                    pass

            picam.start()
            self._picam2 = picam
            self.backend = "Picamera2 (Camera Module 3 HD AutoFocus)"
            return
        except Exception:
            pass

        # 2. Universal OpenCV VideoCapture fallback
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.c_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.c_height)
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    self._cap = cap
                    self.backend = "OpenCV VideoCapture"
                    return
                cap.release()
        except Exception:
            pass

        self.backend = "unavailable"

    def is_available(self) -> bool:
        return self.backend != "unavailable"

    def read_preview_frame(self) -> Optional[np.ndarray]:
        """Returns 640x480 frame for smooth UI display."""
        if self._picam2:
            try:
                if self._has_lores:
                    return self._picam2.capture_array("lores")
                else:
                    frame = self._picam2.capture_array("main")
                    return cv2.resize(frame, (self.p_width, self.p_height))
            except Exception:
                return None
        elif self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret and frame is not None:
                if frame.shape[1] != self.p_width or frame.shape[0] != self.p_height:
                    return cv2.resize(frame, (self.p_width, self.p_height))
                return frame
        return None

    def capture_highres_frame(self) -> Optional[np.ndarray]:
        """Returns full high-resolution 1080p frame for accurate book OCR."""
        if self._picam2:
            try:
                return self._picam2.capture_array("main")
            except Exception:
                return self.read_preview_frame()
        elif self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            return frame if ret else None
        return None

    def trigger_autofocus(self):
        """Forces an immediate fast macro autofocus lock on Camera Module 3."""
        if self._picam2:
            try:
                from libcamera import controls
                # Trigger single-shot PDAF sweep in Macro range (10cm - 50cm)
                self._picam2.set_controls({
                    "AfMode": controls.AfModeEnum.Auto,
                    "AfRange": controls.AfRangeEnum.Macro,
                    "AfSpeed": controls.AfSpeedEnum.Fast,
                    "AfTrigger": controls.AfTriggerEnum.Start
                })
                # Re-arm continuous tracking after 400ms
                def _restore():
                    time.sleep(0.4)
                    try:
                        self._picam2.set_controls({
                            "AfMode": controls.AfModeEnum.Continuous,
                            "AfRange": controls.AfRangeEnum.Macro,
                            "AfSpeed": controls.AfSpeedEnum.Fast
                        })
                    except Exception:
                        pass
                threading.Thread(target=_restore, daemon=True).start()
            except Exception:
                try:
                    self._picam2.set_controls({
                        "AfMode": 1,
                        "AfRange": 1,
                        "AfSpeed": 1,
                        "AfTrigger": 0
                    })
                    def _restore():
                        time.sleep(0.4)
                        try:
                            self._picam2.set_controls({
                                "AfMode": 2,
                                "AfRange": 1,
                                "AfSpeed": 1
                            })
                        except Exception:
                            pass
                    threading.Thread(target=_restore, daemon=True).start()
                except Exception:
                    pass

    def close(self):
        if self._picam2:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception:
                pass
            self._picam2 = None
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None


class SmartReadingGlassesApp:
    """Native Desktop GUI application for Smart Reading Glasses."""

    def __init__(self, root: tk.Tk, method: str = "auto", enable_tts: bool = True):
        self.root = root
        self.method = method
        self.enable_tts = enable_tts
        self.pipeline = ReadingPipeline(enable_tts=enable_tts)
        self.stream = LiveCameraStream(preview_width=640, preview_height=480, capture_width=1920, capture_height=1080)

        self.is_processing = False
        self.auto_read_enabled = tk.BooleanVar(value=True)
        self.crop_to_guide = tk.BooleanVar(value=True)
        self.speed_var = tk.StringVar(value="Calm (120 WPM)")
        self.last_audio_file = None
        self.steady_start_time = None
        self.auto_read_cooldown = 0.0
        self.prev_gray_roi = None
        self._last_af_trigger = 0.0

        self._build_ui()
        self._start_video_loop()

    def _on_speed_changed(self, event=None):
        val = self.speed_var.get()
        rate_map = {
            "Calm (120 WPM)": (120, 80),
            "Normal (140 WPM)": (140, 90),
            "Fast (165 WPM)": (165, 105),
        }
        rate, pico_lvl = rate_map.get(val, (125, 82))
        from config import TTSConfig
        TTSConfig.RATE = rate
        TTSConfig.PICO_SPEED_LEVEL = pico_lvl
        if self.pipeline.tts_engine:
            self.pipeline.tts_engine.rate = rate

    def _build_ui(self):
        self.root.title("Smart Reading Glasses — College IDP Review 2")
        self.root.geometry("1120x680")
        self.root.minsize(920, 580)
        self.root.configure(bg="#18181f")

        # Top Header Bar
        header = tk.Frame(self.root, bg="#22222b", height=48)
        header.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header,
            text="👓 SMART READING GLASSES",
            font=("Helvetica", 15, "bold"),
            fg="#00d4ff",
            bg="#22222b"
        )
        title_lbl.pack(side=tk.LEFT, padx=14, pady=8)

        self.backend_lbl = tk.Label(
            header,
            text=f"Camera: {self.stream.backend}",
            font=("Helvetica", 9),
            fg="#9090aa",
            bg="#22222b"
        )
        self.backend_lbl.pack(side=tk.RIGHT, padx=14, pady=8)

        # Main Layout: 2 Columns
        container = tk.Frame(self.root, bg="#18181f")
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Left Column: Camera Viewfinder & Focus Indicator
        left_col = tk.Frame(container, bg="#101014", relief=tk.RIDGE, bd=2)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        cam_title_bar = tk.Frame(left_col, bg="#101014")
        cam_title_bar.pack(fill=tk.X, padx=8, pady=(6, 2))

        tk.Label(
            cam_title_bar,
            text="LIVE CAMERA VIEWFINDER",
            font=("Helvetica", 10, "bold"),
            fg="#ffffff",
            bg="#101014"
        ).pack(side=tk.LEFT)

        self.fps_lbl = tk.Label(
            cam_title_bar,
            text="30 FPS",
            font=("Helvetica", 9),
            fg="#60ff90",
            bg="#101014"
        )
        self.fps_lbl.pack(side=tk.RIGHT)

        # Viewfinder Canvas (640x480)
        self.viewfinder = tk.Label(left_col, bg="#000000", width=640, height=480)
        self.viewfinder.pack(padx=6, pady=4)

        # Focus & Guide Bar
        focus_bar = tk.Frame(left_col, bg="#101014")
        focus_bar.pack(fill=tk.X, padx=8, pady=(4, 8))

        self.focus_lbl = tk.Label(
            focus_bar,
            text="Focus: Checking...",
            font=("Helvetica", 9, "bold"),
            fg="#ffaa00",
            bg="#101014"
        )
        self.focus_lbl.pack(side=tk.LEFT)

        tk.Label(
            focus_bar,
            text="Shortcut: Press [SPACE] to Read",
            font=("Helvetica", 9),
            fg="#808090",
            bg="#101014"
        ).pack(side=tk.RIGHT)

        # Right Column: Controls & Text Panel
        right_col = tk.Frame(container, bg="#202028", relief=tk.RIDGE, bd=2)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Action Buttons Area
        btn_area = tk.Frame(right_col, bg="#202028")
        btn_area.pack(fill=tk.X, padx=12, pady=10)

        main_btn_row = tk.Frame(btn_area, bg="#202028")
        main_btn_row.pack(fill=tk.X, pady=(0, 8))

        self.read_btn = tk.Button(
            main_btn_row,
            text="📖  READ BOOK PAGE  [SPACE]",
            font=("Helvetica", 12, "bold"),
            bg="#00a859",
            fg="#ffffff",
            activebackground="#008044",
            activeforeground="#ffffff",
            relief=tk.RAISED,
            bd=3,
            cursor="hand2",
            command=self.trigger_read_page
        )
        self.read_btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        self.refocus_btn = tk.Button(
            main_btn_row,
            text="🎯  REFOCUS  [F]",
            font=("Helvetica", 11, "bold"),
            bg="#0284c7",
            fg="#ffffff",
            activebackground="#0369a1",
            activeforeground="#ffffff",
            relief=tk.RAISED,
            bd=3,
            cursor="hand2",
            command=self.trigger_refocus
        )
        self.refocus_btn.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(0, 0))

        # Secondary Control Row 1: Auto-Read & Crop Options
        ctrl_row = tk.Frame(btn_area, bg="#202028")
        ctrl_row.pack(fill=tk.X, pady=(0, 4))

        auto_chk = tk.Checkbutton(
            ctrl_row,
            text="⚡ Auto-Read (Hands-Free)",
            variable=self.auto_read_enabled,
            font=("Helvetica", 9),
            fg="#e0e0e0",
            bg="#202028",
            selectcolor="#141419",
            activebackground="#202028",
            activeforeground="#00d4ff"
        )
        auto_chk.pack(side=tk.LEFT)

        self.crop_guide_chk = tk.Checkbutton(
            ctrl_row,
            text="📐 Crop to Guide (Eliminates Desk)",
            variable=self.crop_to_guide,
            font=("Helvetica", 9),
            fg="#e0e0e0",
            bg="#202028",
            selectcolor="#141419",
            activebackground="#202028",
            activeforeground="#00d4ff"
        )
        self.crop_guide_chk.pack(side=tk.LEFT, padx=10)

        # Secondary Control Row 2: Speech Speed & Audio Actions
        ctrl_row2 = tk.Frame(btn_area, bg="#202028")
        ctrl_row2.pack(fill=tk.X, pady=(4, 0))

        tk.Label(
            ctrl_row2,
            text="Voice Speed:",
            font=("Helvetica", 9),
            fg="#a0a0b0",
            bg="#202028"
        ).pack(side=tk.LEFT)

        self.speed_combo = ttk.Combobox(
            ctrl_row2,
            textvariable=self.speed_var,
            values=["Calm (120 WPM)", "Normal (140 WPM)", "Fast (165 WPM)"],
            state="readonly",
            width=15
        )
        self.speed_combo.pack(side=tk.LEFT, padx=6)
        self.speed_combo.bind("<<ComboboxSelected>>", self._on_speed_changed)

        self.replay_btn = tk.Button(
            ctrl_row2,
            text="🔊 Replay Speech",
            font=("Helvetica", 9, "bold"),
            bg="#3b3b4f",
            fg="#ffffff",
            activebackground="#505070",
            relief=tk.GROOVE,
            command=self.replay_audio
        )
        self.replay_btn.pack(side=tk.RIGHT, padx=4)

        clear_btn = tk.Button(
            ctrl_row2,
            text="🧹 Clear",
            font=("Helvetica", 9),
            bg="#2e2e38",
            fg="#c0c0c0",
            activebackground="#404050",
            relief=tk.GROOVE,
            command=self.clear_text
        )
        clear_btn.pack(side=tk.RIGHT, padx=4)

        # Recognized Text Title & Meta
        text_hdr = tk.Frame(right_col, bg="#202028")
        text_hdr.pack(fill=tk.X, padx=12, pady=(10, 2))

        tk.Label(
            text_hdr,
            text="RECOGNIZED BOOK TEXT",
            font=("Helvetica", 10, "bold"),
            fg="#00d4ff",
            bg="#202028"
        ).pack(side=tk.LEFT)

        self.meta_lbl = tk.Label(
            text_hdr,
            text="0 words | 0% conf",
            font=("Helvetica", 9),
            fg="#9090aa",
            bg="#202028"
        )
        self.meta_lbl.pack(side=tk.RIGHT)

        # Large-font Readable Text Box
        text_frame = tk.Frame(right_col, bg="#202028")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        scroll = tk.Scrollbar(text_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_box = tk.Text(
            text_frame,
            wrap=tk.WORD,
            yscrollcommand=scroll.set,
            font=("Georgia", 13),
            bg="#14141a",
            fg="#f0f0f5",
            insertbackground="#ffffff",
            relief=tk.SUNKEN,
            bd=2,
            padx=10,
            pady=10
        )
        self.text_box.pack(fill=tk.BOTH, expand=True)
        scroll.config(command=self.text_box.yview)

        # Bottom Status Banner
        self.status_bar = tk.Label(
            self.root,
            text="● READY — Aim Camera Module 3 at your book page and press [SPACE] to read.",
            font=("Helvetica", 9),
            fg="#00ff88",
            bg="#121216",
            anchor=tk.W,
            padx=14,
            pady=5
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # Hotkeys
        self.root.bind("<space>", lambda event: self.trigger_read_page())
        self.root.bind("<Return>", lambda event: self.trigger_read_page())
        self.root.bind("<f>", lambda event: self.trigger_refocus())
        self.root.bind("<F>", lambda event: self.trigger_refocus())
        self.root.bind("<q>", lambda event: self.root.destroy())
        self.root.bind("<Escape>", lambda event: self.root.destroy())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def trigger_refocus(self):
        """Forces an immediate fast autofocus sweep on Camera Module 3."""
        self.status_bar.config(text="🎯 Autofocusing Camera Module 3 on book page...", fg="#00d4ff")
        self.focus_lbl.config(text="Focus: [AUTOFOCUSING...]", fg="#00d4ff")
        self.stream.trigger_autofocus()

    def _start_video_loop(self):
        """Pulls camera frames and updates viewfinder display in Tkinter mainloop."""
        from preprocessing.image_quality import detect_text_presence

        def update_frame():
            if not self.root.winfo_exists():
                return

            frame = self.stream.read_preview_frame()
            if frame is not None:
                h, w = frame.shape[:2]

                # Center reading guide ROI (70% center)
                bx1, by1 = int(w * 0.15), int(h * 0.15)
                bx2, by2 = int(w * 0.85), int(h * 0.85)
                roi = frame[by1:by2, bx1:bx2]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray_roi, cv2.CV_64F).var()

                # Detect if printed text is present in the viewing area
                has_text, edge_density, text_blocks = detect_text_presence(gray_roi)

                preview = frame.copy()
                now = time.time()

                # Dynamic focus & text detection feedback
                if has_text and sharpness >= QualityConfig.BLUR_THRESHOLD:
                    self.focus_lbl.config(
                        text=f"Focus: {sharpness:.0f} [TEXT IN VIEW | SHARP & READY]",
                        fg="#00ff88"
                    )
                    box_color = (0, 255, 0)  # Green
                elif has_text:
                    self.focus_lbl.config(
                        text=f"Focus: {sharpness:.0f} [TEXT IN VIEW | FOCUSING...]",
                        fg="#ffaa00"
                    )
                    box_color = (0, 165, 255)  # Orange
                else:
                    self.focus_lbl.config(
                        text=f"Focus: {sharpness:.0f} [AIM AT PRINTED BOOK PAGE]",
                        fg="#8888aa"
                    )
                    box_color = (120, 120, 120)  # Gray

                cv2.rectangle(preview, (bx1, by1), (bx2, by2), box_color, 2)

                # Auto-refocus trigger and Hands-Free Auto-Read
                if self.prev_gray_roi is not None:
                    # Apply Gaussian blur before differencing to eliminate CMOS sensor noise
                    curr_blur = cv2.GaussianBlur(gray_roi, (5, 5), 0)
                    prev_blur = cv2.GaussianBlur(self.prev_gray_roi, (5, 5), 0)
                    diff = cv2.absdiff(curr_blur, prev_blur)
                    motion = float(np.mean(diff))

                    # 1. Quick Auto-Refocus on Camera Module 3 when page is steady but focus is soft
                    if motion < 5.0 and sharpness < 45.0 and (now - self._last_af_trigger) > 2.0:
                        self._last_af_trigger = now
                        self.stream.trigger_autofocus()

                    # 2. Hands-Free Auto-Read: triggers automatically when text is steady & sharp!
                    if self.auto_read_enabled.get() and not self.is_processing and now > self.auto_read_cooldown:
                        is_steady = motion < 6.5
                        is_sharp = sharpness >= 45.0

                        if has_text and is_sharp and is_steady:
                            if self.steady_start_time is None:
                                self.steady_start_time = now
                            elapsed = now - self.steady_start_time

                            # Visual countdown progress bar on screen (0.55s)
                            countdown_pct = min(1.0, elapsed / 0.55)
                            bar_w = int((bx2 - bx1) * countdown_pct)
                            cv2.rectangle(preview, (bx1, by2 - 16), (bx1 + bar_w, by2), (0, 255, 100), -1)
                            cv2.putText(
                                preview,
                                f"HOLD STEADY - READING IN {max(0.0, 0.55 - elapsed):.1f}s",
                                (bx1 + 10, by2 - 22),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (0, 255, 100),
                                2
                            )

                            if elapsed >= 0.55:
                                self.steady_start_time = None
                                self.auto_read_cooldown = now + 4.0  # 4.0s cooldown before next page
                                self.trigger_read_page()
                        else:
                            self.steady_start_time = None
                self.prev_gray_roi = gray_roi.copy()

                # Render frame onto Tkinter label
                rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                img_tk = ImageTk.PhotoImage(image=img)
                self.viewfinder.img_tk = img_tk
                self.viewfinder.config(image=img_tk)

            # Schedule next frame (~30 FPS)
            self.root.after(33, update_frame)

        self.root.after(50, update_frame)

    def trigger_read_page(self, captured_frame: Optional[np.ndarray] = None):
        """Triggers the full pipeline using high-resolution sensor capture."""
        if self.is_processing:
            return

        # Capture high-resolution frame (1080p) for accurate OCR
        frame = captured_frame if captured_frame is not None else self.stream.capture_highres_frame()
        if frame is None:
            frame = self.stream.read_preview_frame()

        if frame is None:
            messagebox.showwarning("Camera Error", "No video frame could be captured from the camera.")
            return

        self.is_processing = True
        self.read_btn.config(text="⏳ READING PAGE...", bg="#d97706", state=tk.DISABLED)
        self.status_bar.config(
            text="● PROCESSING: Enhancing image, recognizing text with Tesseract, generating speech...",
            fg="#ffaa00"
        )

        threading.Thread(target=self._run_pipeline_worker, args=(frame.copy(),), daemon=True).start()

    def _run_pipeline_worker(self, frame: np.ndarray):
        """Background worker thread to run OpenCV + Tesseract + TTS without blocking GUI."""
        import traceback
        try:
            print("[INFO] Initiating page capture and recognition...")
            INPUT_DIR.mkdir(parents=True, exist_ok=True)
            snapshot_path = INPUT_DIR / "camera_live_capture.jpg"

            # Crop to the focused reading guide box if enabled (removes desk/fingers)
            if self.crop_to_guide.get():
                h, w = frame.shape[:2]
                bx1, by1 = int(w * 0.10), int(h * 0.10)
                bx2, by2 = int(w * 0.90), int(h * 0.90)
                ocr_frame = frame[by1:by2, bx1:bx2]
            else:
                ocr_frame = frame

            cv2.imwrite(str(snapshot_path), ocr_frame)

            # 1. Run pipeline with play_audio=False so recognized text displays immediately!
            result = self.pipeline.run_on_image(
                snapshot_path,
                play_audio=False,
                save_audio=True,
                binarization_method=self.method
            )

            text = result.cleaned_text.cleaned_text
            word_count = result.ocr.word_count
            conf = result.ocr.mean_confidence
            self.last_audio_file = result.audio_path

            # If guide crop yielded 0 words, auto-fallback to full uncropped frame
            if (word_count == 0 or not text.strip()) and self.crop_to_guide.get():
                print("[INFO] Guide crop yielded 0 words. Retrying with full uncropped frame...")
                cv2.imwrite(str(snapshot_path), frame)
                result = self.pipeline.run_on_image(
                    snapshot_path,
                    play_audio=False,
                    save_audio=True,
                    binarization_method=self.method
                )
                text = result.cleaned_text.cleaned_text
                word_count = result.ocr.word_count
                conf = result.ocr.mean_confidence
                self.last_audio_file = result.audio_path

            print(f"[INFO] OCR Completed: {word_count} words recognized ({conf:.1f}% confidence).")

            # 2. IMMEDIATELY update UI on main thread with recognized text!
            self.root.after(0, self._on_pipeline_success, text, word_count, conf)

            # 3. Play speech audio asynchronously in background if TTS is enabled and words were found
            if self.enable_tts and self.last_audio_file and word_count > 0:
                print(f"[INFO] Speaking recognized text: {self.last_audio_file}")
                play_audio_file(self.last_audio_file)

        except Exception as e:
            traceback.print_exc()
            self.root.after(0, self._on_pipeline_error, str(e))

    def _on_pipeline_success(self, text: str, word_count: int, confidence: float):
        """Called on main thread when OCR succeeds."""
        self.read_btn.config(text="📖  READ BOOK PAGE  [SPACE]", bg="#00a859", state=tk.NORMAL)
        self.is_processing = False

        if word_count == 0 or not text or not text.strip():
            help_msg = (
                "[No clear text recognized on this page]\n\n"
                "Helpful tips:\n"
                "• Position the book page inside the green guide box (approx. 20-30 cm from camera).\n"
                "• Press [F] or click '🎯 REFOCUS' to lock sharp macro focus on the printed words.\n"
                "• Ensure standard room lighting without harsh glare or deep shadows.\n"
                "• Press [SPACE] to capture and read again."
            )
            self.text_box.delete("1.0", tk.END)
            self.text_box.insert(tk.END, help_msg)
            self.meta_lbl.config(text="0 words | 0% conf", fg="#ffaa00")
            self.status_bar.config(
                text="⚠ No text detected. Check lighting, press [F] to Refocus, and press [SPACE] again.",
                fg="#ffaa00"
            )
        else:
            self.text_box.delete("1.0", tk.END)
            self.text_box.insert(tk.END, text.strip())
            self.meta_lbl.config(text=f"{word_count} words | {confidence:.0f}% confidence", fg="#00e5ff")
            self.status_bar.config(
                text=f"✔ COMPLETED — Reading {word_count} words aloud with {confidence:.0f}% confidence.",
                fg="#00ff88"
            )

    def _on_pipeline_error(self, err_msg: str):
        """Called on main thread when an error occurs."""
        self.status_bar.config(text=f"✖ ERROR: {err_msg[:80]}", fg="#ff4444")
        self.read_btn.config(text="📖  READ BOOK PAGE  [SPACE]", bg="#00a859", state=tk.NORMAL)
        self.is_processing = False

        err_display = (
            f"[Error During Page Processing]\n\n"
            f"Details: {err_msg}\n\n"
            "Troubleshooting Steps:\n"
            "1. Tesseract OCR missing: Open terminal on Raspberry Pi and run:\n"
            "   sudo apt install -y tesseract-ocr tesseract-ocr-eng\n"
            "2. Camera Module 3: Ensure ribbon cable is seated firmly in CAM/DISP 0.\n"
            "3. Try pressing [SPACE] again or click '📖 READ BOOK PAGE'."
        )
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert(tk.END, err_display)

    def replay_audio(self):
        """Replays the last spoken audio file through the speakers."""
        if self.last_audio_file and os.path.exists(self.last_audio_file):
            threading.Thread(target=play_audio_file, args=(self.last_audio_file,), daemon=True).start()
            self.status_bar.config(text="🔊 Replaying audio through speakers...", fg="#00d4ff")
        else:
            self.status_bar.config(text="No audio available to replay yet.", fg="#ffaa00")

    def clear_text(self):
        """Clears the recognized text display."""
        self.text_box.delete("1.0", tk.END)
        self.meta_lbl.config(text="0 words | 0% conf")

    def on_close(self):
        """Safely shuts down camera and closes application."""
        self.stream.close()
        self.root.destroy()


def start_live_reader_gui(method: str = "auto", enable_tts: bool = True):
    """Entry point to launch the native desktop GUI."""
    root = tk.Tk()
    app = SmartReadingGlassesApp(root, method=method, enable_tts=enable_tts)
    root.mainloop()


if __name__ == "__main__":
    start_live_reader_gui()
