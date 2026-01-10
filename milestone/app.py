from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

app = Flask(__name__)
app.secret_key = 'touchless_gesture_control_secret_key_2024'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_interface = cast(interface, POINTER(IAudioEndpointVolume))

volume_range = volume_interface.GetVolumeRange()
min_volume = volume_range[0]
max_volume = volume_range[1]

class GestureVolumeController:
    def __init__(self):
        self.cap = None
        self.volume_history = deque(maxlen=10)
        self.distance_history = deque(maxlen=50)
        self.last_volume = 50
        self.is_locked = False
        self.gesture_state = "None"
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0
        self.auto_min_distance = 20
        self.auto_max_distance = 200
        self.manual_min_distance = 20
        self.manual_max_distance = 200
        self.calibration_mode = "auto"
        self.detected_distances = deque(maxlen=100)
        self.gesture_timer = 0
        self.last_gesture = "None"
        self.response_times = deque(maxlen=20)

    def initialize_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)
            self.cap.set(cv2.CAP_PROP_FPS, 10)

    def calculate_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def detect_gesture(self, landmarks, hand_landmarks):
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        thumb_ip = landmarks[3]
        index_pip = landmarks[6]
        middle_pip = landmarks[10]
        ring_pip = landmarks[14]
        pinky_pip = landmarks[18]

        fingers_extended = 0
        if thumb_tip[0] < thumb_ip[0]:
            fingers_extended += 1
        if index_tip[1] < index_pip[1]:
            fingers_extended += 1
        if middle_tip[1] < middle_pip[1]:
            fingers_extended += 1
        if ring_tip[1] < ring_pip[1]:
            fingers_extended += 1
        if pinky_tip[1] < pinky_pip[1]:
            fingers_extended += 1

        distance = self.calculate_distance(thumb_tip, index_tip)

        if distance < 30:
            gesture = "Pinch"
        elif fingers_extended >= 4:
            gesture = "Open Hand"
        elif fingers_extended <= 1:
            gesture = "Closed Hand"
        else:
            gesture = "Partial"

        current_time = time.time()
        if gesture == self.last_gesture:
            self.gesture_timer = current_time
        else:
            if current_time - self.gesture_timer > 0.5:
                self.last_gesture = gesture
                self.gesture_timer = current_time

        return self.last_gesture

    def update_calibration(self, distance):
        self.detected_distances.append(distance)
        if len(self.detected_distances) > 20:
            sorted_distances = sorted(self.detected_distances)
            self.auto_min_distance = max(20, sorted_distances[5])
            self.auto_max_distance = min(200, sorted_distances[-5])

    def map_distance_to_volume(self, distance):
        if self.calibration_mode == "auto":
            min_dist = self.auto_min_distance
            max_dist = self.auto_max_distance
        else:
            min_dist = self.manual_min_distance
            max_dist = self.manual_max_distance

        if distance < min_dist:
            distance = min_dist
        if distance > max_dist:
            distance = max_dist

        volume_percent = ((distance - min_dist) / (max_dist - min_dist)) * 100
        return max(0, min(100, volume_percent))

    def get_volume_level(self, volume):
        if volume <= 25:
            return "LOW"
        elif volume <= 65:
            return "MEDIUM"
        else:
            return "HIGH"

    def set_system_volume(self, volume_percent):
        volume_value = min_volume + (volume_percent / 100) * (max_volume - min_volume)
        volume_interface.SetMasterVolumeLevel(volume_value, None)

    def generate_frames(self):
        self.initialize_camera()

        while True:
            start_time = time.time()
            success, frame = self.cap.read()
            if not success:
                break

            frame = cv2.resize(frame, (256, 192))
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            finger_distance = 0
            hand_detected = False

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                h, w, c = frame.shape

                landmarks = []
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append((cx, cy))

                connections = mp_hands.HAND_CONNECTIONS
                for connection in connections:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    start_point = landmarks[start_idx]
                    end_point = landmarks[end_idx]
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 1)

                thumb_tip = landmarks[4]
                index_tip = landmarks[8]

                cv2.circle(frame, thumb_tip, 4, (255, 0, 0), -1)
                cv2.circle(frame, index_tip, 4, (0, 0, 255), -1)
                cv2.line(frame, thumb_tip, index_tip, (255, 255, 0), 1)

                finger_distance = self.calculate_distance(thumb_tip, index_tip)
                self.distance_history.append(finger_distance)

                self.gesture_state = self.detect_gesture(landmarks, hand_landmarks)

                if self.gesture_state == "Closed Hand":
                    self.is_locked = True
                elif self.gesture_state == "Open Hand":
                    self.is_locked = False

                if not self.is_locked:
                    if self.calibration_mode == "auto":
                        self.update_calibration(finger_distance)

                    volume_percent = self.map_distance_to_volume(finger_distance)
                    self.volume_history.append(volume_percent)

                    if len(self.volume_history) >= 3:
                        smoothed_volume = sum(self.volume_history) / len(self.volume_history)
                        self.last_volume = smoothed_volume
                        self.set_system_volume(smoothed_volume)

                hand_detected = True

            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_time)
                self.frame_count = 0
                self.last_time = current_time

            response_time = (time.time() - start_time) * 1000
            self.response_times.append(response_time)

            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def get_metrics(self):
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0

        return {
            'volume': round(self.last_volume, 1),
            'distance': round(self.distance_history[-1], 1) if self.distance_history else 0,
            'distance_history': list(self.distance_history),
            'fps': round(self.fps, 1),
            'gesture': self.gesture_state,
            'locked': self.is_locked,
            'volume_level': self.get_volume_level(self.last_volume),
            'response_time': round(avg_response_time, 1),
            'auto_min': round(self.auto_min_distance, 1),
            'auto_max': round(self.auto_max_distance, 1),
            'manual_min': self.manual_min_distance,
            'manual_max': self.manual_max_distance,
            'calibration_mode': self.calibration_mode
        }

controller = GestureVolumeController()

@app.route('/')
def index():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == 'touchless' and password == 'notouch@123':
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/video_feed')
def video_feed():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return Response(controller.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/metrics')
def metrics():
    if 'logged_in' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify(controller.get_metrics())

@app.route('/set_calibration', methods=['POST'])
def set_calibration():
    if 'logged_in' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.json
    controller.calibration_mode = data.get('mode', 'auto')

    if controller.calibration_mode == 'manual':
        controller.manual_min_distance = int(data.get('min', 20))
        controller.manual_max_distance = int(data.get('max', 200))

    return jsonify({'success': True})

@app.route('/toggle_lock', methods=['POST'])
def toggle_lock():
    if 'logged_in' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    controller.is_locked = not controller.is_locked
    return jsonify({'locked': controller.is_locked})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
