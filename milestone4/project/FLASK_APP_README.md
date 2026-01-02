# Gesture-Based Volume Control System

A fully-featured Flask web application that provides real-time hand gesture recognition and volume control using MediaPipe and OpenCV.

## Features

### Core Functionality
- **Real-Time Webcam Feed**: 256×192 resolution at 5-10 FPS for smooth continuous tracking
- **MediaPipe Hand Tracking**: Detects all 21 hand landmarks with real-time visualization
- **Distance Calculation**: Measures Euclidean distance between thumb and index finger
- **Dynamic Volume Mapping**: Converts finger distance to volume percentage (0-100%)

### Advanced Features
- **Gesture Recognition**: Detects Pinch, Open Hand, Closed Fist, and Neutral gestures
- **Live Distance Graph**: Real-time plotting of distance values over time
- **Calibration Module**:
  - Manual min/max distance input
  - Auto-calibration based on current hand position
- **Performance Metrics Dashboard**:
  - Accuracy percentage
  - Response time (ms)
  - Finger distance (px)
  - Current volume (%)

### UI/UX
- Modern gradient background with professional styling
- Responsive layout that works on multiple screen sizes
- Smooth animations and hover effects
- Intuitive circular gauge displays for metrics
- Real-time volume bar visualization
- No-hand detection indicator

## Project Structure

```
.
├── app.py                          # Flask backend application
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                 # Frontend HTML/CSS/JS
└── FLASK_APP_README.md            # This file
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Webcam connected to your computer
- 500MB+ free disk space (for dependencies)

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Flask: Web framework
- OpenCV: Video capture and image processing
- MediaPipe: Hand landmark detection
- NumPy: Numerical computations

### Step 2: Run the Application

```bash
python app.py
```

You should see output like:
```
 * Running on http://127.0.0.1:5000
 * Press CTRL+C to quit
```

### Step 3: Access the Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

## Usage Guide

### Basic Operation
1. **Position Your Hand**: Show your hand to the webcam
2. **Track Recognition**: The system automatically detects your hand and draws 21 landmarks
3. **Control Volume**:
   - Bring thumb and index finger close together (pinch) to lower volume
   - Spread them apart to increase volume
4. **View Metrics**: Monitor performance in real-time on the dashboard

### Calibration

The system comes with default calibration (20-200px), but you can adjust for your specific setup:

**Auto-Calibrate**:
- Click "Auto Calibrate" button
- The system reads your current hand distance as a baseline
- Automatically sets min and max ranges

**Manual Calibrate**:
- Enter desired minimum distance (pixels)
- Enter desired maximum distance (pixels)
- Click "Apply" to save

### Gesture Recognition
- **Pinch**: Thumb and index finger very close (< 30px)
- **Open Hand**: Fingers spread wide apart
- **Closed Fist**: All fingers close together
- **Neutral**: Relaxed hand position

### Performance Metrics

The dashboard displays four key metrics:

1. **Accuracy** (0-100%): Detection confidence and tracking quality
2. **Response Time**: Processing time in milliseconds (lower is better)
3. **Finger Distance**: Current distance between thumb and index in pixels
4. **Current Volume**: Calculated volume percentage based on finger distance

## System Requirements

### Minimum Requirements
- Processor: Intel i5 / AMD Ryzen 5 or equivalent
- RAM: 4GB
- Webcam: 720p or higher

### Recommended
- Processor: Intel i7 / AMD Ryzen 7 or better
- RAM: 8GB+
- Webcam: 1080p
- Low-latency internet (if streaming)

## Troubleshooting

### "No module named 'mediapipe'"
```bash
pip install --upgrade mediapipe
```

### Webcam Not Detected
- Check if another application is using your webcam
- Verify camera permissions in system settings
- Try running: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

### High Response Time
- Reduce other background processes
- Lower your webcam resolution (if possible)
- Check CPU usage in task manager

### Hand Not Detected
- Ensure adequate lighting (not backlighting)
- Move hand into frame center
- Keep hand in clear view (not partially out of frame)
- Adjust confidence threshold in app.py if needed

### Graph Not Updating
- Ensure JavaScript is enabled in your browser
- Check browser console for errors (F12 → Console tab)
- Try refreshing the page

## Performance Optimization

### For Better Accuracy
1. Improve lighting conditions
2. Use a higher-quality webcam
3. Position yourself 30-60cm from camera
4. Perform auto-calibration for your setup

### For Lower Latency
1. Close unnecessary applications
2. Disable other webcam access
3. Reduce browser extensions
4. Use wired internet if available

## API Endpoints

### GET `/`
Returns the main HTML interface

### GET `/video_feed`
Streaming endpoint for webcam video feed
- MIME type: `multipart/x-mixed-replace; boundary=frame`
- Frame rate: 5-10 FPS
- Resolution: 256×192px

### GET `/api/metrics`
Returns current metrics data in JSON format
```json
{
  "distance": 45.32,
  "volume": 65.5,
  "gesture": "Neutral",
  "accuracy": 92.3,
  "response_time": 8.5,
  "min_distance": 20,
  "max_distance": 200,
  "distance_history": [40, 42, 45, ...],
  "volume_percentage": 65.5
}
```

### POST `/api/calibrate`
Performs auto-calibration
- Returns: `{"status": "calibrated", "metrics": {...}}`

### POST `/api/set_calibration`
Sets manual calibration values
- Body: `{"min_distance": 20, "max_distance": 200}`
- Returns: `{"status": "set", "metrics": {...}}`

## Architecture

### Backend (Flask)
- Real-time frame capture from webcam
- MediaPipe hand detection and landmark extraction
- Gesture recognition logic
- Distance calculation and volume mapping
- Multi-threaded frame processing
- REST API for frontend communication

### Frontend (HTML/CSS/JavaScript)
- Responsive UI with gradient design
- Real-time metric updates via AJAX
- Canvas-based distance graph
- Circular gauge displays
- Interactive calibration controls
- Live video stream display

### Data Flow
```
Webcam Input
    ↓
OpenCV Capture (256×192)
    ↓
MediaPipe Hand Detection
    ↓
Landmark Extraction + Visualization
    ↓
Distance Calculation
    ↓
Volume Mapping
    ↓
Gesture Recognition
    ↓
JSON API Response
    ↓
Frontend Updates
```

## Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

JavaScript features used:
- Fetch API
- Canvas API
- requestAnimationFrame
- CSS Grid & Flexbox

## Limitations

1. Single-hand detection only (for volume control accuracy)
2. FPS capped at 8-10 for processing stability
3. Requires local webcam (no remote webcam support)
4. Browser-based visualization only (doesn't control system volume)

## Future Enhancements

Possible improvements:
- Multi-hand gesture recognition
- Custom gesture training
- System volume integration (OS-specific APIs)
- Hand pose estimation for advanced gestures
- Cloud storage of calibration profiles
- Mobile app with remote hand tracking
- ML-based gesture learning

## License

This project is provided as-is for educational and development purposes.

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Verify all dependencies are installed: `pip list`
3. Ensure webcam is properly connected
4. Check browser console for JavaScript errors
5. Review Flask server logs for backend errors

## Technical Details

### Hand Landmarks (MediaPipe)
The system tracks 21 hand landmarks:
- Palm: landmarks 0-8
- Thumb: landmarks 0-4
- Index: landmarks 5-8
- Middle: landmarks 9-12
- Ring: landmarks 13-16
- Pinky: landmarks 17-20

Key landmarks for volume control:
- Thumb tip: landmark 4
- Index tip: landmark 8

### Distance Mapping Algorithm
```
Distance (px) → Volume (%)
20px → 0%
200px → 100%

Linear mapping: volume = (distance - 20) / 180 * 100
```

### Performance Metrics Calculation
- **Accuracy**: Based on detection confidence (0-100%)
- **Response Time**: Frame processing time in milliseconds
- **Distance**: Euclidean distance in pixels
- **Volume**: Mapped percentage based on calibration

## Version History

- v1.0: Initial release with core features
  - Real-time hand tracking
  - Gesture recognition
  - Calibration module
  - Performance dashboard
