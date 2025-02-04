import cv2
import math
import time
from ultralytics import YOLO
from google.colab.patches import cv2_imshow  # Fix for Colab image display

# Load the trained YOLO model
model_path = "/kaggle/input/pitcher-detection/pytorch/default/1/yolov8_trained_pitcher.pt"
model = YOLO(model_path)

# Load the video
video_path = "/kaggle/input/videos/vid1.mp4"  # Adjust path if using Google Drive
cap = cv2.VideoCapture(video_path)

# Check if video is opened correctly
if not cap.isOpened():
    print("❌ Error: Could not open video file.")
    exit()

# Get video FPS (for timing)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1 / fps if fps > 0 else 0.03  # Default to 30 FPS if unknown

# Initialize variables
prev_position = None
speed_list = []
pixel_to_meter_ratio = 0.01  # Adjust based on real-world scale
prev_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()  # Ensure a new frame is actually read
    if not ret:
        print("✅ Video processing complete!")
        break  # Exit loop if video ends

    frame_count += 1
    print(f"Processing Frame: {frame_count}")  # Debugging: Confirm frame is updating

    # Run YOLOv8 detection only if frame is available
    results = model(frame)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())  # Get class ID
            if class_id == 0:  # Assuming class 0 = baseball (Check with print(model.names))
                x1, y1, x2, y2 = box.xyxy[0]
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2

                # Calculate time difference
                current_time = time.time()
                time_diff = current_time - prev_time
                prev_time = current_time  # Update for next frame

                if prev_position is not None and time_diff > 0:
                    prev_x, prev_y = prev_position
                    distance_pixels = math.sqrt((x_center - prev_x) ** 2 + (y_center - prev_y) ** 2)

                    # Convert to real-world speed
                    distance_meters = distance_pixels * pixel_to_meter_ratio
                    speed_mps = distance_meters / time_diff  # Speed in meters per second
                    speed_mph = speed_mps * 2.23694  # Convert to mph

                    speed_list.append(speed_mph)
                    print(f"Frame {frame_count} | Speed: {speed_mph:.2f} mph")

                prev_position = (x_center, y_center)

    # Display frame in Colab
    cv2_imshow(frame)  # Fix for Colab

    # Add slight delay for stability
    time.sleep(0.01)

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Print average speed
if speed_list:
    avg_speed = sum(speed_list) / len(speed_list)
    print(f"🏎️ Average Speed: {avg_speed:.2f} mph")
