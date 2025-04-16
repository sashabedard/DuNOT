from ultralytics import YOLO
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
results = model.predict(source="0", show=True, save=True)  # predict on an image, video, or webcam