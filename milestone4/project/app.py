# app.py
import time
import threading
from collections import deque
from math import sqrt
from typing import Optional, Tuple

import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify

# MediaPipe Hands
import mediapipe as mp

# PyCAW for Windows system volume
from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

app = Flask(__name__)

# ----------------------------
# Video & MediaPipe setup
# ----------------------------
FRAME_W, FRAME_H = 256, 192
TARGET_FPS_MIN, TARGET_FPS_MAX = 5, 10
TARGET_FRAME_INTERVAL = 1.0 / 8.0  # Aim ~8 FPS

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       model_complexity=0,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)

# ----------------------------
# System volume (PyCAW)
# ----------------------------
def init_volume_interface():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))

endpoint_volume = None
try:
    endpoint_volume = init_volume_interface()
except Exception as e:
    print("Failed to initialize system volume control. Ensure PyCAW is installed and you're on Windows.")
    endpoint_volume = None

def set_system_volume_percent(percent: float):
    """Set Windows master volume (0-100%)."""
    if endpoint_volume is None:
        return
    p = max(0.0, min(100.0, percent))
    scalar = p / 100.0
    try:
        endpoint_volume.SetMasterVolumeLevelScalar(scalar, None)
    except Exception:
        pass

def get_system_volume_percent() -> float:
    if endpoint_volume is None:
        return 0.0
    try:
        return float(endpoint_volume.GetMasterVolumeLevelScalar() * 100.0)
    except Exception:
        return 0.0

# ----------------------------
# Smoothing & calibration
# ----------------------------
# Deques to store recent values (performance-friendly)
DIST_WINDOW = 15
VOL_WINDOW = 10
FPS_WINDOW = 30

distance_history = deque(maxlen=300)  # for graph
dist_smooth_window = deque(maxlen=DIST_WINDOW)
vol_smooth_window = deque(maxlen=VOL_WINDOW)
fps_window = deque(maxlen=FPS_WINDOW)

# Calibration defaults
calib_min_px = 20.0
calib_max_px = 200.0
observed_min_px = float('inf')
observed_max_px = 0.0

# Rate limiting for volume updates
last_volume_set_time = 0.0
MIN_SET_INTERVAL = 0.08  # seconds
MIN_DELTA_PERCENT = 1.5  # change threshold to avoid micro-flutter

# Shared state for metrics
lock = threading.Lock()
latest_metrics = {
    "fps": 0.0,
    "response_ms": 0.0,
    "distance_px": 0.0,
    "smoothed_volume_percent": 0.0,
    "gesture": "None",
    "semantic_level": "LOW",
    "distance_graph": []
}

# ----------------------------
# Gesture recognition helpers
# ----------------------------
FINGER_TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
FINGER_PIPS = [3, 6, 10, 14, 18]  # Thumb IP, Index PIP, Middle PIP, Ring PIP, Pinky PIP

def is_finger_extended(landmarks_px, tip_idx, pip_idx) -> bool:
    tip = landmarks_px[tip_idx]
    pip = landmarks_px[pip_idx]
    # In image coordinates, y grows downward.
    return tip[1] < pip[1]  # tip higher (smaller y) than pip => extended

def count_extended_fingers(landmarks_px) -> int:
    # Thumb: special case due to lateral extension; heuristic using x-position
    # Compare thumb tip to thumb IP relative to wrist to infer extension direction.
    wrist = landmarks_px[0]
    thumb_tip = landmarks_px[4]
    thumb_ip = landmarks_px[3]
    thumb_extended = abs(thumb_tip[0] - thumb_ip[0]) > 12  # lateral spread

    # Other fingers: tip above PIP
    idx_ext = is_finger_extended(landmarks_px, 8, 6)
    mid_ext = is_finger_extended(landmarks_px, 12, 10)
    ring_ext = is_finger_extended(landmarks_px, 16, 14)
    pinky_ext = is_finger_extended(landmarks_px, 20, 18)
    return int(thumb_extended) + int(idx_ext) + int(mid_ext) + int(ring_ext) + int(pinky_ext)

def classify_gesture(distance_px: float, landmarks_px) -> str:
    # Pinch: very small distance
    if distance_px <= max(10.0, 0.08 * (calib_max_px - calib_min_px) + calib_min_px):
        return "Pinch"
    # If landmarks present, estimate finger spread
    if landmarks_px is not None:
        ext_count = count_extended_fingers(landmarks_px)
        if ext_count >= 4:
            return "Open Hand"
        if ext_count <= 1 and distance_px < (calib_min_px + 0.2 * (calib_max_px - calib_min_px)):
            return "Closed Fist"
    return "Neutral"

def semantic_level(volume_percent: float) -> str:
    if volume_percent <= 25.0:
        return "LOW"
    elif volume_percent <= 65.0:
        return "MEDIUM"
    else:
        return "HIGH"

# ----------------------------
# Distance-volume mapping
# ----------------------------
def map_distance_to_volume(distance_px: float, dmin: float, dmax: float) -> float:
    # Linear map: dmin -> 0%, dmax -> 100%
    if dmax <= dmin:
        return 0.0
    norm = (distance_px - dmin) / (dmax - dmin)
    return float(max(0.0, min(1.0, norm)) * 100.0)

# ----------------------------
# Video streaming generator
# ----------------------------
def video_stream():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Ensure exact frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)  # driver hint; we throttle manually

    prev_time = time.time()

    global observed_min_px, observed_max_px
    global last_volume_set_time

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            # If capture fails, yield a blank frame to keep stream alive
            frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

        # Resize and flip for a more natural webcam feel
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        landmarks_px = None
        distance_px: Optional[float] = None

        if results.multi_hand_landmarks:
            # Detect exactly one hand (configured with max_num_hands=1)
            hand_landmarks = results.multi_hand_landmarks[0]
            # Convert normalized landmarks to pixel coordinates
            landmarks_px = []
            for lm in hand_landmarks.landmark:
                x = int(lm.x * FRAME_W)
                y = int(lm.y * FRAME_H)
                landmarks_px.append((x, y))

            # Draw all 21 landmarks and connections (thin single lines)
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(180, 180, 180), thickness=1, circle_radius=1)
            )

            # Highlighted thin line between thumb tip (4) and index tip (8)
            p4 = landmarks_px[4]
            p8 = landmarks_px[8]
            cv2.line(frame, p4, p8, (0, 255, 255), 2)

            # Euclidean distance
            distance_px = sqrt((p4[0] - p8[0]) ** 2 + (p4[1] - p8[1]) ** 2)

            # Update observed ranges for auto-calibrate
            observed_min_px = min(observed_min_px, distance_px)
            observed_max_px = max(observed_max_px, distance_px)

        # Smoothing and volume logic
        smoothed_volume = get_system_volume_percent()  # start from actual system state
        gesture = "None"

        if distance_px is not None:
            dist_smooth_window.append(distance_px)
            # Moving average for distance
            avg_distance = float(np.mean(dist_smooth_window)) if dist_smooth_window else distance_px
            distance_history.append(avg_distance)

            # Map to volume using current calibration
            target_volume = map_distance_to_volume(avg_distance, calib_min_px, calib_max_px)

            vol_smooth_window.append(target_volume)
            smoothed_volume = float(np.mean(vol_smooth_window)) if vol_smooth_window else target_volume
            smoothed_volume = max(0.0, min(100.0, smoothed_volume))

            gesture = classify_gesture(avg_distance, landmarks_px)

            # Controlled updates to prevent sudden jumps
            now = time.time()
            current_system_vol = get_system_volume_percent()
            if (abs(smoothed_volume - current_system_vol) >= MIN_DELTA_PERCENT) and (now - last_volume_set_time >= MIN_SET_INTERVAL):
                set_system_volume_percent(smoothed_volume)
                last_volume_set_time = now

            # Annotate distance and volume
            cv2.putText(frame, f"Distance: {avg_distance:.1f}px", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Volume: {smoothed_volume:.1f}%", (6, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Gesture: {gesture}", (6, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Level: {semantic_level(smoothed_volume)}", (6, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1, cv2.LINE_AA)
        else:
            # Safe handling when no hand is detected
            cv2.putText(frame, "No hand detected", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Metrics: FPS and response time
        end_time = time.time()
        response_ms = (end_time - start_time) * 1000.0

        # FPS (moving average)
        dt = end_time - prev_time
        prev_time = end_time
        inst_fps = 1.0 / dt if dt > 0 else TARGET_FPS_MAX
        fps_window.append(inst_fps)
        avg_fps = float(np.mean(fps_window)) if fps_window else inst_fps

        # Store metrics safely
        with lock:
            latest_metrics["fps"] = round(avg_fps, 2)
            latest_metrics["response_ms"] = round(response_ms, 1)
            latest_metrics["distance_px"] = round(float(dist_smooth_window[-1]), 1) if dist_smooth_window else 0.0
            latest_metrics["smoothed_volume_percent"] = round(smoothed_volume, 1)
            latest_metrics["gesture"] = gesture
            latest_metrics["semantic_level"] = semantic_level(smoothed_volume)
            # Provide last 120 points for the graph
            latest_metrics["distance_graph"] = list(distance_history)[-120:]

        # Encode as MJPEG
        ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Throttle to target FPS range for stability
        elapsed = time.time() - start_time
        if elapsed < TARGET_FRAME_INTERVAL:
            time.sleep(TARGET_FRAME_INTERVAL - elapsed)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/metrics')
def metrics():
    with lock:
        return jsonify(latest_metrics)


@app.route('/calibrate', methods=['POST'])
def calibrate():
    global calib_min_px, calib_max_px
    data = request.json or {}
    try:
        min_px = float(data.get('min_px', calib_min_px))
        max_px = float(data.get('max_px', calib_max_px))
        # Clamp and ensure valid range
        min_px = max(0.0, min(min_px, 500.0))
        max_px = max(0.0, min(max_px, 500.0))
        if max_px <= min_px:
            return jsonify({"ok": False, "message": "Max must be greater than Min."}), 400
        calib_min_px = min_px
        calib_max_px = max_px
        return jsonify({"ok": True, "min_px": calib_min_px, "max_px": calib_max_px})
    except Exception:
        return jsonify({"ok": False, "message": "Invalid calibration values."}), 400


@app.route('/auto_calibrate', methods=['POST'])
def auto_calibrate():
    global calib_min_px, calib_max_px, observed_min_px, observed_max_px
    # Use observed range, but apply sensible bounds
    if observed_max_px == 0.0 or observed_min_px == float('inf'):
        return jsonify({"ok": False, "message": "Insufficient data for auto-calibration. Show your hand and pinch/open."}), 400
    # Pad ranges slightly to avoid extremes
    dmin = max(10.0, observed_min_px * 0.95)
    dmax = min(450.0, observed_max_px * 1.05)
    if dmax <= dmin:
        return jsonify({"ok": False, "message": "Auto-calibration failed. Try again."}), 400
    calib_min_px, calib_max_px = dmin, dmax
    # Reset observed for next rounds
    observed_min_px = float('inf')
    observed_max_px = 0.0
    return jsonify({"ok": True, "min_px": calib_min_px, "max_px": calib_max_px})


if __name__ == '__main__':
    # Ensure initial volume window has the current system volume to avoid initial spike
    if endpoint_volume is not None:
        vol_smooth_window.append(get_system_volume_percent())
    app.run(host='127.0.0.1', port=5000, threaded=True)