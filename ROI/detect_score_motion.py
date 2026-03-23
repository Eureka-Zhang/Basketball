import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
from scipy.signal import find_peaks
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
    results = CLIENT.infer(temp_image_path, model_id="basketball-players-fy4c2/25")

    basket_bbox = None
    for prediction in results['predictions']:
        if prediction['class'] == 'Hoop':
            basket_bbox = {
                'x': prediction['x'],
                'y': prediction['y'],
                'width': prediction['width'],
                'height': prediction['height'],
                'confidence': prediction['confidence']
            }
            break  # Only one basket is expected, so we can exit early

    # 将检测结果标在 temp_frame.jpg 上（便于直接打开该文件查看）
    out = frame.copy()
    if basket_bbox is not None:
        h, w = frame.shape[:2]
        x1 = max(0, int(basket_bbox['x'] - basket_bbox['width'] / 2))
        y1 = max(0, int(basket_bbox['y'] - basket_bbox['height'] / 2))
        x2 = min(w, int(basket_bbox['x'] + basket_bbox['width'] / 2))
        y2 = min(h, int(basket_bbox['y'] + basket_bbox['height'] / 2))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"basket {basket_bbox['confidence']:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    cv2.imwrite(temp_image_path, out)

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
    basket_bbox = None
    frame_idx = 0
    gui_available = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video.")
            break

        if prev_frame is None:
            prev_frame = frame
            continue

        frame_idx += 1
        display_frame = frame.copy()

        # 1. YOLO detection
        basket_bbox = detect(frame)

        if basket_bbox:
            ball_positions.append(basket_bbox['y'])

        # 2. Optical flow calculation
        flow = calc_flow(prev_frame, frame)

        if basket_bbox:
            x1 = max(0, int(basket_bbox['x'] - basket_bbox['width'] / 2))
            y1 = max(0, int(basket_bbox['y'] - basket_bbox['height'] / 2))
            x2 = min(frame.shape[1], int(basket_bbox['x'] + basket_bbox['width'] / 2))
            y2 = min(frame.shape[0], int(basket_bbox['y'] + basket_bbox['height'] / 2))
            roi = flow[y1:y2, x1:x2]
            if roi.size > 0:
                mag = compute_magnitude(roi)
                flow_value = float(np.mean(mag))
                motion_series.append(flow_value)
                print(f"[FLOW] Frame {frame_idx}: {flow_value:.4f}")

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display_frame,
                f"basket {basket_bbox['confidence']:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        else:
            print(f"[FLOW] Frame {frame_idx}: basket not detected")

        if gui_available:
            try:
                cv2.imshow("Basket Detection + Optical Flow", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Interrupted by user.")
                    break
            except cv2.error:
                gui_available = False
                print("[WARN] OpenCV GUI is unavailable; disabling on-screen display.")

        prev_frame = frame

    # ---- Time series processing ----
    if not motion_series:
        print("[WARN] No valid optical-flow ROI collected.")
        cap.release()
        cv2.destroyAllWindows()
        return

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
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()