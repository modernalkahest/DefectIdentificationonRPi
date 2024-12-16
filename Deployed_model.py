import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration({"size": (640, 640)})
picam2.configure(config)

# Enable autofocus (continuous mode)
picam2.set_controls({"AfMode": 2})

# Load YOLOv8 model
model = YOLO("/home/mohithemaprasad/Desktop/best_yolov11_small.pt")

# Define a confidence threshold
CONFIDENCE_THRESHOLD = 0.5

def capture_images(num_frames=5):
    images = []
    picam2.start()
    for _ in range(num_frames):
        frame = picam2.capture_array()
        frame = frame[:, :, :3] if frame.shape[2] == 4 else frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(frame)
    picam2.stop()
    return images

def run_inference(model, images):
    results = model.predict(images, classes=[0])  # Directly pass image array
    return results
i=0
while True:
    images = capture_images(num_frames=6)
    results = run_inference(model, images)

    for result in results:
        annotated_frame = result.plot()
        cv2.imwrite(f'/home/mohithemaprasad/Desktop/Model Results/Results{i}.jpg',annotated_frame)
        cv2.imshow("Camera", annotated_frame)
        i+=1

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
