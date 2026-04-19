"""
Beard Twisting Habit Detector
Runs in the background and alerts you when you touch/twist your beard.
Uses MediaPipe for hand + face landmark detection.
"""

import cv2
import mediapipe as mp
import time
import json
import os
import sys
import shlex
import signal
import argparse
import subprocess
import threading
from datetime import datetime
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────

CONFIG = {
    "trigger_delay_seconds": 2.0,    # how long hand must stay near face before alert
    "cooldown_seconds": 10.0,        # minimum time between alerts
    "sensitivity": 0.18,             # how close hand must be to face (ratio of frame width)
    "log_file": "detections.json",
    "camera_index": 0,
}

# ─── Logging ──────────────────────────────────────────────────────────────────

LOG_PATH = Path(__file__).parent / CONFIG["log_file"]

def log_detection():
    entry = {"timestamp": datetime.now().isoformat(), "type": "beard_touch"}
    logs = []
    if LOG_PATH.exists():
        try:
            with open(LOG_PATH) as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(entry)
    with open(LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2)
    print(f"[{entry['timestamp']}] Detection logged.")

# ─── Alert ────────────────────────────────────────────────────────────────────

NOTIFICATION_TITLE = "Beard Alert"
NOTIFICATION_MESSAGE = "You're touching your beard again. Put your hand down!"
NOTIFICATION_SOUND = "Submarine"  # any file in /System/Library/Sounds


def _notify_macos(title: str, message: str, sound: str | None) -> bool:
    """Native macOS notification via osascript. Returns True on success."""
    try:
        script = f'display notification {json.dumps(message)} with title {json.dumps(title)}'
        if sound:
            script += f' sound name {json.dumps(sound)}'
        subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
            timeout=3,
        )
        return True
    except Exception:
        return False


def _notify_terminal_notifier(title: str, message: str, sound: str | None) -> bool:
    """Fallback path using `terminal-notifier` if installed (nicer on macOS)."""
    try:
        cmd = ["terminal-notifier", "-title", title, "-message", message]
        if sound:
            cmd += ["-sound", sound]
        subprocess.run(cmd, check=True, capture_output=True, timeout=3)
        return True
    except Exception:
        return False


def _notify_linux(title: str, message: str) -> bool:
    try:
        subprocess.run(
            ["notify-send", title, message],
            check=True,
            capture_output=True,
            timeout=3,
        )
        return True
    except Exception:
        return False


def _notify_plyer(title: str, message: str) -> bool:
    try:
        from plyer import notification
        notification.notify(title=title, message=message, timeout=4)
        return True
    except Exception:
        return False


def send_alert():
    """Send a real desktop notification. Platform-aware with fallbacks."""
    title, message, sound = NOTIFICATION_TITLE, NOTIFICATION_MESSAGE, NOTIFICATION_SOUND

    if sys.platform == "darwin":
        if _notify_terminal_notifier(title, message, sound):
            return
        if _notify_macos(title, message, sound):
            return
    elif sys.platform.startswith("linux"):
        if _notify_linux(title, message):
            return
    elif sys.platform == "win32":
        if _notify_plyer(title, message):
            return

    if _notify_plyer(title, message):
        return
    print(f"\n⚠️  {title}: {message}\n")

# ─── System Tray ──────────────────────────────────────────────────────────────

tray_icon = None
detector_running = True

def create_tray_icon(stop_callback):
    try:
        import pystray
        from PIL import Image, ImageDraw

        # Create a simple icon (red circle)
        img = Image.new("RGB", (64, 64), color=(30, 30, 30))
        draw = ImageDraw.Draw(img)
        draw.ellipse([12, 12, 52, 52], fill=(220, 80, 80))

        menu = pystray.Menu(
            pystray.MenuItem("Beard Detector — Running", lambda: None, enabled=False),
            pystray.MenuItem("Stop Detector", stop_callback),
        )
        icon = pystray.Icon("beard_detector", img, "Beard Detector", menu)
        icon.run()
    except ImportError:
        print("pystray not installed — running without tray icon. Ctrl+C to stop.")

# ─── Detection Logic ──────────────────────────────────────────────────────────

def get_face_chin_y(face_landmarks, frame_h):
    """Return the Y pixel coordinate of the chin (landmark 152)."""
    if face_landmarks:
        chin = face_landmarks.landmark[152]
        return int(chin.y * frame_h)
    return None

def get_face_center_x(face_landmarks, frame_w):
    """Return the X pixel coordinate of the nose tip (landmark 1)."""
    if face_landmarks:
        nose = face_landmarks.landmark[1]
        return int(nose.x * frame_w)
    return None

def hand_near_beard(hand_landmarks, face_landmarks, frame_w, frame_h):
    """
    Returns True if any hand landmark is in the beard zone:
    - Below the nose
    - Above the collarbone (estimated as chin + 30% of frame height)
    - Within sensitivity distance of face center X
    """
    if not hand_landmarks or not face_landmarks:
        return False

    chin_y = get_face_chin_y(face_landmarks, frame_h)
    face_cx = get_face_center_x(face_landmarks, frame_w)

    if chin_y is None or face_cx is None:
        return False

    # Beard zone: from chin upward ~10px, downward ~chin + 25% frame
    beard_top_y = chin_y - 20
    beard_bottom_y = chin_y + int(frame_h * 0.15)
    x_tolerance = int(frame_w * CONFIG["sensitivity"])

    for lm in hand_landmarks.landmark:
        hx = int(lm.x * frame_w)
        hy = int(lm.y * frame_h)
        if (beard_top_y <= hy <= beard_bottom_y and
                abs(hx - face_cx) < x_tolerance):
            return True
    return False


def run_detector(headless: bool = False):
    global detector_running

    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(CONFIG["camera_index"])
    if not cap.isOpened():
        print("❌ Could not open webcam. Check camera index in CONFIG.")
        return

    hand_near_start = None
    last_alert_time = 0

    print("✅ Beard Detector running. Watching for beard touches...")
    if headless:
        print("   Headless mode — no preview window. Send SIGINT/SIGTERM to stop.\n")
    else:
        print("   Press Q in the preview window to quit, or use tray icon.\n")

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands, mp_face.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
    ) as face_mesh:

        while detector_running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_results = hands.process(rgb)
            face_results = face_mesh.process(rgb)

            face_lms = (
                face_results.multi_face_landmarks[0]
                if face_results.multi_face_landmarks
                else None
            )

            touching = False

            if hand_results.multi_hand_landmarks:
                for hand_lms in hand_results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(100, 220, 100), thickness=1, circle_radius=3),
                        mp_draw.DrawingSpec(color=(50, 150, 50), thickness=1),
                    )
                    if hand_near_beard(hand_lms, face_lms, frame_w, frame_h):
                        touching = True

            now = time.time()

            if touching:
                if hand_near_start is None:
                    hand_near_start = now

                elapsed = now - hand_near_start
                remaining = CONFIG["trigger_delay_seconds"] - elapsed

                # Progress bar
                progress = min(elapsed / CONFIG["trigger_delay_seconds"], 1.0)
                bar_w = int(frame_w * 0.6)
                bar_x = int(frame_w * 0.2)
                bar_y = frame_h - 40
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 16), (50, 50, 50), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 16), (80, 200, 120), -1)
                cv2.putText(frame, f"Hand near beard... {remaining:.1f}s", (bar_x, bar_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                if elapsed >= CONFIG["trigger_delay_seconds"]:
                    if now - last_alert_time >= CONFIG["cooldown_seconds"]:
                        last_alert_time = now
                        log_detection()
                        threading.Thread(target=send_alert, daemon=True).start()
                        # Flash red
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0), (frame_w, frame_h), (0, 0, 200), -1)
                        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                        cv2.putText(frame, "⚠ BEARD TOUCH!", (frame_w//2 - 140, frame_h//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            else:
                hand_near_start = None

            if not headless:
                # Status overlay
                status = "WATCHING" if not touching else "HAND NEAR BEARD"
                color = (80, 200, 120) if not touching else (80, 180, 255)
                cv2.putText(frame, f"Beard Detector  |  {status}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                cv2.imshow("Beard Detector (Q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    detector_running = False
    cap.release()
    cv2.destroyAllWindows()
    if tray_icon:
        tray_icon.stop()
    print("Detector stopped.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Beard Twisting Habit Detector")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without preview window (for background use). Stop with SIGINT/SIGTERM.",
    )
    args = parser.parse_args()

    def stop_from_tray(icon, item):
        global detector_running
        detector_running = False
        icon.stop()

    def handle_signal(signum, frame):
        global detector_running
        print(f"\nReceived signal {signum}, shutting down...")
        detector_running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Skip tray on macOS: pystray (NSStatusItem) and cv2.imshow both require
    # the main thread, so they can't coexist. Use Q in the preview to quit.
    if not args.headless and sys.platform != "darwin":
        tray_thread = threading.Thread(
            target=create_tray_icon,
            args=(stop_from_tray,),
            daemon=True,
        )
        tray_thread.start()

    run_detector(headless=args.headless)
