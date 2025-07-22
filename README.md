# Grounded SAM2 ROS2

This repository provides the hand-eye correction function, currently has the eye in hand method.

---

## Installation

### 1. Install this repository 
```bash
git clone https://github.com/TKUwengkunduo/Hand-Eye-Calibration.git
```

### 2. Docker
```bash
cd docker/ubuntu
bash build.sh
bash run.sh
```


---

## Operation process
### 1. Collecting information
You will get a `hand_eye_data/data.json` file and some photos.
```bash
colcon build
source install/setup.bash
ros2 run hand_eye_data_collector collector
```


### 2. Make a correction
```bash
python3 hand_eye_calibration.py
```