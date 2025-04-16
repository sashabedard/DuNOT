# DUNOT

## Moshing script for image processing and video generation.

    Made by Sasha Bédard, Laura-Ann Gendron-Blais, Noémi Larouche, Rachel Pelletier and Ugo Jutras
    For Frédéric Maheux, Rhétorique des Médias, UQÀM, Médias Interactifs, 2025.

### Moshing.py
        This script applies various glitch effects to images of donuts, saves the glitched images, and compiles them into a video.
        It uses OpenCV for image processing and video writing.
        The script includes functions for RGB offset, pixel sorting, datamoshing, and more.
        The script also handles image resizing and background color changes.
        It is designed to run indefinitely, processing new images as they are added to the input folder.


### Detector.py
        This script captures video from a webcam, detects donuts in the frames using a YOLOv11 model,
        crops the detected donuts, and saves them to a specified directory. It also displays the detected donuts in real-time with a confidence slider.
        The script uses OpenCV for video capture and image processing, and the YOLOv11 model for object detection.
        the script also includes a feature to display the last 3 cropped donuts in a sidebar.
        The donut detection is done using a pre-trained YOLOv11 model, and the detected donuts are saved with a timestamp in the filename.
        The script is designed to run indefinitely until the user presses 'q' to quit.
