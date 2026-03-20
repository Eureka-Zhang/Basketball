import cv2
import supervision as sv
from roboflow import Roboflow
import os
import numpy as np

# Initialize Roboflow API
from dotenv import load_dotenv
load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)

# Load the player detection model
PLAYER_DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo"
project = rf.workspace().project(PLAYER_DETECTION_MODEL_ID)
PLAYER_DETECTION_MODEL = project.version(4).model

# Define color palette and annotators
COLOR = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
    "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])
box_annotator = sv.BoxAnnotator(color=COLOR, thickness=2)
label_annotator = sv.LabelAnnotator(color=COLOR, text_color=sv.Color.BLACK)

# Video path
VIDEO_PATH = "../test_video/clip3_shoot.mp4"

# Function to check if ball is in basket
def is_ball_in_basket(detections):
    for detection in detections:
        if detection['class'] == "ball-in-basket":
            return True
    return False

# Initialize debounce counter
debounce_counter = 0

# Define class name to integer mapping
class_name_to_id = {
    "player": 0,
    "ball-in-basket": 1
}

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Failed to open video.")
        return

    frame_count = 0
    makes = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video.")
            break

        # Predict using the model
        results = PLAYER_DETECTION_MODEL.predict(frame, confidence=0.35, overlap=0.5).json()
        detections = results['predictions']

        # Check if ball is in basket
        if is_ball_in_basket(detections):
            debounce_counter += 1
            if debounce_counter >= 1: # Require 3 consecutive frames to confirm a make
                makes += 1
                print(f"[INFO] Frame {frame_count}: Ball in basket! Total makes: {makes}")
                debounce_counter = 0  # Reset debounce counter after counting a make
        else:
            debounce_counter = 0  # Reset debounce counter if ball is not detected

        # Convert detections using supervision's Detections.from_inference
        sv_detections = sv.Detections.from_inference(results)

        # Annotate frame
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=sv_detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=sv_detections
        )

        # Display frame
        cv2.imshow("Frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Total makes: {makes}")

if __name__ == "__main__":
    main()