"""
DUNOT - 1/2

A donut detector and image processor

    Made by Sasha Bédard, Laura-Ann Gendron-Blais, Noémi Larouche, Rachel Pelletier and Ugo Jutras
    For Frédéric Maheux, Rhétorique des Médias, UQÀM, Médias Interactifs, 2025.

        This script captures video from a webcam, detects donuts in the frames using a YOLOv11 model,
        crops the detected donuts, and saves them to a specified directory. It also displays the detected donuts in real-time with a confidence slider.
        The script uses OpenCV for video capture and image processing, and the YOLOv11 model for object detection.
        the script also includes a feature to display the last 3 cropped donuts in a sidebar.
        The donut detection is done using a pre-trained YOLOv11 model, and the detected donuts are saved with a timestamp in the filename.
        The script is designed to run indefinitely until the user presses 'q' to quit.
"""

from ultralytics import YOLO
import cv2
import time
import os
import numpy as np
from collections import deque

# Load YOLOv11 model
model = YOLO("models/yolo11n.pt")

output_dir = "cropped_donuts"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
donut_id = 0

# Confidence slider
cv2.namedWindow("Dunot Detector")
cv2.createTrackbar("Confidence %", "Dunot Detector", 10, 100, lambda x: None)

# Queue to store last 3 cropped donuts
recent_crops = deque(maxlen=3)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get confidence threshold from slider
    slider_val = cv2.getTrackbarPos("Confidence %", "Dunot Detector")
    conf = max(slider_val / 100.0, 0.01)

    results = model.predict(source=frame, conf=conf, verbose=False)
    names = model.names

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            score = float(box.conf[0])

            if cls_name.lower() != "donut":
                continue

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            label = f"DuNOT {score * 100:.1f}%"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 100), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)

            # Crop donut
            cropped = frame[y1:y2, x1:x2].copy()
            cropped = cv2.resize(cropped, (160, 160))  # Standardize display size
            recent_crops.appendleft(cropped)  # Add to recent queue

            # Save cropped image
            timestamp = int(time.time() * 1000)
            filename = os.path.join(output_dir, f"donut{timestamp}_{donut_id}.jpg")
            cv2.imwrite(filename, cropped)
            print(f"[+] Saved cropped donut: {filename}")
            donut_id += 1

    # Compose sidebar from recent crops
       # Compose sidebar from recent crops
    if recent_crops:
        padding = np.zeros((10, 160, 3), dtype=np.uint8)
        sidebar = padding.copy()
        for img in recent_crops:
            sidebar = np.vstack((sidebar, img, padding))

        # Match sidebar height to webcam frame
        h_frame = frame.shape[0]
        sidebar_resized = cv2.resize(sidebar, (160 + 10, h_frame))  # 160 width + border

        combined = np.hstack((frame, sidebar_resized))
    else:
        combined = frame


    # Show the combined display
    cv2.imshow("Dunot Detector", combined)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
