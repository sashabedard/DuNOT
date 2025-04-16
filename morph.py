import os
import cv2
import numpy as np

input_folder = "cropped_donuts"
video_filename = 'trippy_donut_morphs_clean.mp4'
output_size = (512, 512)
fps = 12
steps_per_morph = 10

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_filename, fourcc, fps, output_size)

def print_progress_bar(iteration, total, length=40):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = "█" * filled_length + "-" * (length - filled_length)
    print(f"\rProgress |{bar}| {percent}% Complete", end="")

def warp_blend(img1, img2, alpha):
    h, w = img1.shape[:2]
    flow = np.random.randn(h, w, 2).astype(np.float32) * 5 * (1 - abs(0.5 - alpha))  # Reduced flow intensity

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)

    warped1 = cv2.remap(img1, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    warped2 = cv2.remap(img2, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Directly blend the images without adjusting brightness
    return cv2.addWeighted(warped1, 1 - alpha, warped2, alpha, 0)  # No change in brightness

def add_frame_echo(current_frame, previous_frame, decay=0.85):
    # Ensure decay doesn't affect the brightness
    return cv2.addWeighted(current_frame, 1.0, previous_frame, 0, 0)  # No decay, no brightness change

def add_noise(img, strength=3):  # Reduced noise strength
    noise = np.random.randint(-strength, strength, img.shape, dtype=np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy

def morph_trippy(img1, img2, steps=10):
    frames = []
    prev = img1.copy()
    for i, alpha in enumerate(np.linspace(0, 1, steps)):
        warped = warp_blend(img1, img2, alpha)
        noisy = add_noise(warped, strength=3)  # Reduced noise strength
        trippy = add_frame_echo(noisy, prev)
        frames.append(trippy)
        prev = trippy.copy()
    return frames

# Load and resize all donut images
image_files = sorted(f for f in os.listdir(input_folder) if f.endswith(".jpg"))
images = []
for fname in image_files:
    path = os.path.join(input_folder, fname)
    img = cv2.imread(path)
    if img is not None:
        img = cv2.resize(img, output_size)
        images.append(img)

if len(images) < 2:
    print("[⚠️] Not enough images in folder.")
    exit()

# Morph through all images
total_pairs = len(images) - 1
frame_count = 0
for i in range(total_pairs):
    print_progress_bar(i, total_pairs)
    img1, img2 = images[i], images[i + 1]
    morph_frames = morph_trippy(img1, img2, steps_per_morph)
    for frame in morph_frames:
        video_writer.write(frame)
        cv2.imshow("TRIPPY DONUT DREAM", frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print_progress_bar(total_pairs, total_pairs)
print(f"\n[✅] Done! {frame_count} frames written to {video_filename}")

video_writer.release()
cv2.destroyAllWindows()
