"""
DUNOT - 2/2

Moshing script for image processing and video generation.

    Made by Sasha Bédard, Laura-Ann Gendron-Blais, Noémi Larouche, Rachel Pelletier and Ugo Jutras
    For Frédéric Maheux, Rhétorique des Médias, UQÀM, Médias Interactifs, 2025.

        This script applies various glitch effects to images of donuts, saves the glitched images, and compiles them into a video.
        It uses OpenCV for image processing and video writing.
        The script includes functions for RGB offset, pixel sorting, datamoshing, and more.
        The script also handles image resizing and background color changes.
        It is designed to run indefinitely, processing new images as they are added to the input folder.
"""

import os
import time
import cv2
import numpy as np

# Dossier d'entrée et de sortie
input_folder = "cropped_donuts"
output_folder = "dunot_glitched"
video_output_folder = "video_output"  # Nouveau dossier pour la vidéo

os.makedirs(output_folder, exist_ok=True)
os.makedirs(video_output_folder, exist_ok=True)  # Création automatique du dossier vidéo

seen_files = set()
glitch_cache = {}
rewinding = False
rewind_start_time = None

# Initialiser VideoWriter pour enregistrer la vidéo au format MP4 dans le dossier video_output
video_filename = os.path.join(video_output_folder, 'output_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 12
frame_width = 1024
frame_height = 1024
video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

def rgb_offset(img, shift_range=5):
    b, g, r = cv2.split(img)
    b_shift = np.roll(b, np.random.randint(-shift_range, shift_range), axis=1)
    g_shift = np.roll(g, np.random.randint(-shift_range, shift_range), axis=0)
    r_shift = np.roll(r, np.random.randint(-shift_range, shift_range), axis=1)
    return cv2.merge((b_shift, g_shift, r_shift))

def glitch_blend(current, previous, alpha=0.3):
    return cv2.addWeighted(current, 1 - alpha, previous, alpha, 0)

def inversion_pulse(img, chance=0.6):
    if np.random.rand() < chance:
        inverted = 255 - img
        contrast_boost = cv2.convertScaleAbs(inverted, alpha=1.5, beta=10)
        return contrast_boost
    return img

def pixel_sort(img, sort_chance=0.4):
    output = img.copy()
    for row in range(output.shape[0]):
        if np.random.rand() > sort_chance:
            continue
        brightness = np.sum(output[row], axis=1)
        sorted_indices = np.argsort(brightness)
        output[row] = output[row][sorted_indices]
    return output

def databend_simulation(img, strength=5):
    output = img.copy()
    rows, cols = img.shape[:2]
    for _ in range(strength):
        y = np.random.randint(0, rows)
        h = np.random.randint(1, 4)
        shift = np.random.randint(-cols//8, cols//8)
        output[y:y+h] = np.roll(output[y:y+h], shift, axis=1)
    return output

def preserve_original(original, glitched, alpha=0.7):
    return cv2.addWeighted(glitched, alpha, original, 1 - alpha, 0)

def datamosh(img, fname):
    if img is None:
        return None
    glitched = img.copy()
    for i in range(0, img.shape[0], 10):
        offset = np.random.randint(-10, 10)
        glitched[i:i+10] = np.roll(glitched[i:i+10], offset, axis=1)
    if np.random.rand() > 0.5:
        glitched = glitched[..., np.random.permutation(3)]
    glitched = rgb_offset(glitched)
    glitched = pixel_sort(glitched)
    glitched = databend_simulation(glitched)
    glitched = inversion_pulse(glitched, chance=0.6)
    if fname in glitch_cache:
        glitched = glitch_blend(glitched, glitch_cache[fname])
    glitched = preserve_original(img, glitched, alpha=0.7)
    glitch_cache[fname] = glitched.copy()
    return glitched

def upscale_image(img, size=(1024, 1024)):
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

def change_background_color_and_deform(img, t):
    h, w = img.shape[:2]
    shift_x = np.sin(t / 20.0) * 20
    shift_y = np.cos(t / 30.0) * 20
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    deform = cv2.warpAffine(img, M, (w, h))
    hsv = cv2.cvtColor(deform, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = (hsv[..., 0] + t / 2) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

rewind_start_time = time.time()
rewinding = False

try:
    while True:
        current_files = set(f for f in os.listdir(input_folder) if f.endswith(".jpg"))
        new_files = current_files - seen_files

        for fname in sorted(new_files):
            path = os.path.join(input_folder, fname)
            img = cv2.imread(path)
            if img is None:
                print(f"[!] Couldn't read image: {fname}")
                continue

            current_time = time.time()

            # Adjusted rewind time to 5 seconds
            if current_time - rewind_start_time < 5:
                rewinding = True
                img = change_background_color_and_deform(img, current_time)

            glitched = datamosh(img, fname)
            if glitched is not None:
                glitched_resized = upscale_image(glitched)
                out_path = os.path.join(output_folder, f"glitched_{fname}")
                cv2.imwrite(out_path, glitched_resized)
                print(f"[+] Glitched image saved: {out_path}")
                cv2.imshow("DUNOT", glitched_resized)
                cv2.waitKey(50)

                video_writer.write(glitched_resized)
                print(f"Frame written to video")

            seen_files.add(fname)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[X] Quit by user")
            break

        time.sleep(0.5)

finally:
    print("Releasing video writer and closing windows...")
    video_writer.release()
    cv2.destroyAllWindows()
