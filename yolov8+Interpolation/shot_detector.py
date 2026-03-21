# Avi Shah - Basketball Shot Detector/Tracker - July 2023

from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import sys
import types
import os
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device
from roboflow import Roboflow
from dotenv import load_dotenv
import supervision as sv

# Initialize Roboflow API
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

class ShotDetector:
    def __init__(self):
        # Load the YOLO model created from main.py - change text to your relative path
        self.overlay_text = "Waiting..."

        # Compatibility shim:
        # Some older YOLOv8 checkpoints reference `ultralytics.nn.modules.conv`,
        # but newer ultralytics exposes those layers directly from `ultralytics.nn.modules`
        # (a single modules.py file, not a package).
        # By creating a lightweight `ultralytics.nn.modules.conv` module on-the-fly, we
        # allow torch to unpickle the checkpoint successfully.
        try:
            import ultralytics.nn.modules as modules_pkg

            # Make parent look like a package so `ultralytics.nn.modules.conv` can be imported.
            if not hasattr(modules_pkg, "__path__"):
                modules_pkg.__path__ = []  # type: ignore[attr-defined]

            # Create a few likely legacy submodules that checkpoints may reference.
            # We map everything exported from `modules.py` to each submodule.
            for sub in ["conv", "block", "head"]:
                legacy_mod_name = f"ultralytics.nn.modules.{sub}"
                legacy_mod = types.ModuleType(legacy_mod_name)
                for attr in dir(modules_pkg):
                    if not attr.startswith("_"):
                        try:
                            setattr(legacy_mod, attr, getattr(modules_pkg, attr))
                        except Exception:
                            pass
                sys.modules[legacy_mod_name] = legacy_mod
        except Exception:
            # If this shim fails, ultralytics will raise the original checkpoint loading error.
            pass

        self.model = PLAYER_DETECTION_MODEL
        self.class_names = [
            'ball', 'ball-in-basket', 'number', 'player', 'player-in-possession',
            'player-jump-shot', 'player-layup-dunk', 'player-shot-block', 'referee', 'rim'
        ]
        self.device = get_device()
        # Uncomment line below to use webcam (I streamed to my iPhone using Iriun Webcam)
        # self.cap = cv2.VideoCapture(0)

        # Use video - replace text with your video path
        video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_video", "clip4_shoot.mp4")
        print(f"[INFO] Video path: {video_path}")
        self.cap = cv2.VideoCapture(video_path)
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        if not self.cap.isOpened():
            print("[ERROR] VideoCapture failed. Check video path or camera index.")

        self.ball_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.hoop_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)

        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        # Used to detect shots (upper and lower region)
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        # Used for green and red colors after make/miss
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        self.run()

    def run(self):
        while True:
            ret, self.frame = self.cap.read()

            if not ret:
                # End of the video or an error occurred
                print("[INFO] Failed to read frame. Video may have ended or source is not available.")
                break

            # Ensure the first frame triggers the window event loop quickly.
            if self.frame_count == 0:
                cv2.imshow("Frame", self.frame)
                cv2.waitKey(1)

            results = self.model.predict(self.frame, confidence=0.35, overlap=0.5).json()

            for prediction in results['predictions']:
                # Bounding box
                x1, y1, x2, y2 = (
                    prediction['x'] - prediction['width'] / 2,
                    prediction['y'] - prediction['height'] / 2,
                    prediction['x'] + prediction['width'] / 2,
                    prediction['y'] + prediction['height'] / 2,
                )
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = prediction['confidence']

                # Class Name
                current_class = prediction['class']

                center = (int(x1 + w / 2), int(y1 + h / 2))

                # Only create ball points if high confidence or near hoop
                if (conf > .3 or (in_hoop_region(center, self.hoop_pos) and conf > 0.15)) and current_class == "ball":
                    self.ball_pos.append((center, self.frame_count, w, h, conf))
                    cvzone.cornerRect(self.frame, (x1, y1, w, h))

                # Create hoop points if high confidence
                if conf > .5 and current_class == "rim":
                    self.hoop_pos.append((center, self.frame_count, w, h, conf))
                    cvzone.cornerRect(self.frame, (x1, y1, w, h))
            
            # Add frame number to the frame
            frame_number_text = f"Frame: {self.frame_count + 1}"
            cv2.putText(
                self.frame, 
                frame_number_text, 
                (10, 30),  # Position (x, y)
                cv2.FONT_HERSHEY_SIMPLEX,  # Font
                1,  # Font scale
                (0, 255, 0),  # Color (Green)
                2  # Thickness
            )


            self.clean_motion()
            self.shot_detection()
            self.display_score()
            self.frame_count += 1

            cv2.imshow('Frame', self.frame)

            # Close if 'q' is clicked
            if cv2.waitKey(1) & 0xFF == ord('q'):  # higher waitKey slows video down, use 1 for webcam
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self._print_final_summary()

    def _print_final_summary(self):
        """程序结束时在终端输出最终统计。"""
        print()
        print("=" * 50)
        print(f"最终进球数 (Makes):     {self.makes}")
        print(f"投篮尝试次数 (Attempts): {self.attempts}")
        if self.attempts > 0:
            pct = 100.0 * self.makes / self.attempts
            print(f"命中率:                 {pct:.1f}%")
        else:
            print("命中率:                 —（无投篮尝试）")
        print("=" * 50)

    def clean_motion(self):
        # Clean and display ball motion
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(0, len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)

        # Clean hoop motion and display current hoop center
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            # Detecting when ball is in 'up' and 'down' area - ball can only be in 'down' area after it is in 'up'
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]

            # If ball goes from 'up' area to 'down' area in that order, increase attempt and reset
            if self.frame_count % 10 == 0:
                if self.up and self.down and self.up_frame < self.down_frame:
                    self.attempts += 1
                    self.up = False
                    self.down = False

                    # If it is a make, put a green overlay and display "完美"
                    if score(self.ball_pos, self.hoop_pos):
                        self.makes += 1
                        self.overlay_color = (0, 255, 0)  # Green for make
                        self.overlay_text = "Make"
                        self.fade_counter = self.fade_frames

                    else:
                        self.overlay_color = (255, 0, 0)  # Red for miss
                        self.overlay_text = "Miss"
                        self.fade_counter = self.fade_frames

    def display_score(self):
        # Add text
        text = str(self.makes) + " / " + str(self.attempts)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        # Add overlay text for shot result if it exists
        if hasattr(self, 'overlay_text'):
            # Calculate text size to position it at the right top corner
            (text_width, text_height), _ = cv2.getTextSize(self.overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)
            text_x = self.frame.shape[1] - text_width - 40  # Right alignment with some margin
            text_y = 100  # Top margin

            # Display overlay text with color (overlay_color)
            cv2.putText(self.frame, self.overlay_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        self.overlay_color, 6)
            # cv2.putText(self.frame, self.overlay_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        # Gradually fade out color after shot
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, np.full_like(self.frame, self.overlay_color), alpha, 0)
            self.fade_counter -= 1


if __name__ == "__main__":
    ShotDetector()

