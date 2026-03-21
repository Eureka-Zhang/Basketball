import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY)

# Helper functions
def detect(frame):
    """Run detection on the frame and return the basket bounding box."""
    # Save the frame as a temporary image
    temp_image_path = "temp_frame.jpg"
    cv2.imwrite(temp_image_path, frame)

    # Perform inference using the HTTP client
    results = CLIENT.infer(temp_image_path, model_id="robocon-bb/2")

    basket_bbox = None
    for prediction in results['predictions']:
        if prediction['class'] == 'basket':
            basket_bbox = {
                'x': prediction['x'],
                'y': prediction['y'],
                'width': prediction['width'],
                'height': prediction['height'],
                'confidence': prediction['confidence']
            }
            break  # Only one basket is expected, so we can exit early

    return basket_bbox

def calc_flow(prev_frame, frame):
    """Calculate optical flow between two frames."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def compute_magnitude(flow):
    """Compute the magnitude of optical flow."""
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag

def smooth(series, window_size=5):
    """Smooth a time series using a moving average."""
    return np.convolve(series, np.ones(window_size)/window_size, mode='valid')

def ball_pass_hoop(ball_positions, hoop_y):
    """Detect when the ball passes the hoop."""
    for t, pos in enumerate(ball_positions):
        if pos > hoop_y:
            return t
    return None

def detect_score(peaks, t_cross):
    """Determine if a score occurred based on peaks and ball crossing time."""
    if t_cross is None:
        return False
    for peak in peaks:
        if abs(peak - t_cross) < 5:  # Allow small time difference
            return True
    return False

def main():
    video_path = "../test_video/clip4_shoot.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] Failed to open video.")
        return

    motion_series = []
    ball_positions = []
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video.")
            break

        if prev_frame is None:
            prev_frame = frame
            continue

        # 1. YOLO detection
        basket_bbox = detect(frame)

        if basket_bbox:
            ball_positions.append(basket_bbox['y'])

        # 2. Optical flow calculation
        flow = calc_flow(prev_frame, frame)

        if basket_bbox:
            x1 = int(basket_bbox['x'] - basket_bbox['width'] / 2)
            y1 = int(basket_bbox['y'] - basket_bbox['height'] / 2)
            x2 = int(basket_bbox['x'] + basket_bbox['width'] / 2)
            y2 = int(basket_bbox['y'] + basket_bbox['height'] / 2)
            roi = flow[y1:y2, x1:x2]
            mag = compute_magnitude(roi)
            motion_series.append(np.mean(mag))

        prev_frame = frame

    # ---- Time series processing ----
    motion_smooth = smooth(motion_series)
    peaks, _ = find_peaks(motion_smooth, height=2.0)

    if basket_bbox:
        hoop_y = basket_bbox['y']
        t_cross = ball_pass_hoop(ball_positions, hoop_y)

        if detect_score(peaks, t_cross):
            print("进球！")
        else:
            print("未进球。")

    cap.release()

if __name__ == "__main__":
    main()