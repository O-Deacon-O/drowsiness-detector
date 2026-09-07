# Drowsiness Detector

A real-time driver drowsiness detector using OpenCV and Google's MediaPipe.

## Features

- Real-time face and eye landmark detection from a webcam
- Eye aspect ratio monitoring for closed-eye detection
- On-screen eye-ratio, closed-eye duration, and FPS displays
- Wake-up warning after the configured closed-eye duration
- Configurable camera resolution and detection thresholds

## Requirements

- Python 3.9-3.12
- Webcam/video input device
- Windows/Mac/Linux

## Installation

1. Clone or download this repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On Mac/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```bash
python src/main.py
```

**Controls:**
- `q` - Quit the application

The application opens the default webcam and displays the detected face mesh. 
Keep your face visible to the camera for eye tracking.

## Configuration

Edit `src/config.py` to customize:
- **Video resolution**: `TARGET_WIDTH`, `TARGET_HEIGHT`
- **Eye landmark indices**: `LEFT_EYE`, `RIGHT_EYE`
- **Closed-eye threshold**: `CLOSED_THRESHOLD`
- **Required closed-eye duration**: `CLOSED_TIME_REQUIRED`
- **Text size, colors, and thickness**: `TEXT_SCALES`, `TEXT_COLORS`, `TEXT_THICKNESSES`

## How It Works

1. Captures video from your webcam
2. Uses MediaPipe Face Mesh to locate facial landmarks
3. Measures the vertical-to-horizontal eye landmark distance ratio for both eyes
4. Tracks how long the average eye ratio remains below `CLOSED_THRESHOLD`
5. Displays `WAKE UP!!!` when the eyes remain closed for `CLOSED_TIME_REQUIRED` seconds

## Troubleshooting

**Face or eyes are not detected?**
- Improve the lighting in front of your face
- Keep your face within the camera frame
- Make sure no other application is using the webcam

**Webcam not opening?**
- Check if another app is using the camera
- Check that your system has a working camera

**Poor performance?**
- Lower `TARGET_WIDTH` and `TARGET_HEIGHT`
- Close other applications using the camera or CPU

## License

MIT License - Feel free to use this project