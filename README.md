# Beard twisting detector
A python script that uses OpenCV and MediaPipe to detected whether the user is twisting or playing with his beard and sending system notifications on detection.

Background: This script was written for me to help me tackle a very bad and annoying habit of constantly plucking my beard while working, watching TV etc. Maybe it can help some of you as well :D

## Usage
```
python3 -m venv venv
source venv/bin/activate
python3 detector.py
```

## Stats / Analytics
The detector script logs each detection into detections.json. Run the following script to print some stats:
```
python3 analytics.py
```
