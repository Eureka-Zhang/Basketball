"""
基于本地 YOLOv8 + ByteTrack 轨迹追踪的篮球进球检测
用法: python score_tracker.py --video ../test_video/clip4_shoot.mp4 --model path/to/best.pt
"""

import argparse
import sys
import os
import math
import numpy as np
import cv2
import supervision as sv
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

# 复用父目录的 utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'yolov8+Interpolation'))
from utils import score, detect_up, detect_down, in_hoop_region, clean_ball_pos, clean_hoop_pos

# ── 类别名（与训练标签对齐，按需修改） ──────────────────────────────────────
BALL_CLASSES = {'ball', 'ball-in-basket'}
RIM_CLASSES  = {'rim'}


def run(video_path: str, model_path: str, conf: float = 0.35):
    model = YOLO(model_path)
    predict_fn = lambda frame: model(frame, conf=conf, verbose=False)[0]
    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    gui_available = True
    try:
        cv2.namedWindow("Tracker", cv2.WINDOW_NORMAL)
    except cv2.error:
        gui_available = False
        print("[WARN] OpenCV GUI is unavailable; disabling on-screen display.")

    ball_pos, hoop_pos = [], []
    frame_count = 0
    makes, attempts = 0, 0
    up, down = False, False
    up_frame, down_frame = 0, 0
    fade_counter, fade_frames = 0, 20
    overlay_color = (0, 0, 0)
    overlay_text  = "Waiting..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── 1. 模型推理（本地 YOLOv8） ───────────────────────────────────
        results = predict_fn(frame)
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        names = model.names  # {id: 'class_name'}

        for i, (xyxy, _, det_conf, cls_id, track_id, _) in enumerate(detections):
            x1, y1, x2, y2 = map(int, xyxy)
            w, h = x2 - x1, y2 - y1
            center = (x1 + w // 2, y1 + h // 2)
            cls_name = names[int(cls_id)]

            # ── 2. 收集球的位置 ───────────────────────────────────────────
            if cls_name in BALL_CLASSES:
                if det_conf > 0.3 or (in_hoop_region(center, hoop_pos) and det_conf > 0.15):
                    ball_pos.append((center, frame_count, w, h, det_conf))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    if track_id is not None:
                        cv2.putText(frame, f"ball#{track_id}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            # ── 3. 收集篮筐位置 ───────────────────────────────────────────
            elif cls_name in RIM_CLASSES and det_conf > 0.5:
                hoop_pos.append((center, frame_count, w, h, det_conf))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 0), 2)

        # ── 4. 清洗轨迹数据 ───────────────────────────────────────────────
        ball_pos = clean_ball_pos(ball_pos, frame_count)
        if len(hoop_pos) > 1:
            hoop_pos = clean_hoop_pos(hoop_pos)

        # 绘制球的历史轨迹
        for bp in ball_pos:
            cv2.circle(frame, bp[0], 3, (0, 0, 255), -1)
        if hoop_pos:
            cv2.circle(frame, hoop_pos[-1][0], 4, (128, 128, 0), -1)

        # ── 5. 进球判断（up→down 状态机） ────────────────────────────────
        if hoop_pos and ball_pos:
            if not up:
                up = detect_up(ball_pos, hoop_pos)
                if up:
                    up_frame = ball_pos[-1][1]

            if up and not down:
                down = detect_down(ball_pos, hoop_pos)
                if down:
                    down_frame = ball_pos[-1][1]

            if frame_count % 10 == 0 and up and down and up_frame < down_frame:
                attempts += 1
                if score(ball_pos, hoop_pos):
                    makes += 1
                    overlay_color = (0, 255, 0)
                    overlay_text  = "Make!"
                else:
                    overlay_color = (0, 0, 255)
                    overlay_text  = "Miss"
                fade_counter = fade_frames
                up = down = False

        # ── 6. 显示比分与结果 ─────────────────────────────────────────────
        score_text = f"{makes} / {attempts}"
        cv2.putText(frame, score_text, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 6)
        cv2.putText(frame, score_text, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 3)

        (tw, _), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
        cv2.putText(frame, overlay_text, (frame.shape[1] - tw - 30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, overlay_color, 4)

        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if fade_counter > 0:
            alpha = 0.2 * (fade_counter / fade_frames)
            overlay = np.full_like(frame, overlay_color)
            frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
            fade_counter -= 1

        if gui_available:
            try:
                cv2.imshow("Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                gui_available = False
                print("[WARN] OpenCV GUI is unavailable; disabling on-screen display.")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 40)
    print(f"进球数:   {makes}")
    print(f"投篮次数: {attempts}")
    if attempts:
        print(f"命中率:   {100*makes/attempts:.1f}%")
    print("=" * 40)


def run_roboflow(
    video_path: str,
    project_id: str,
    version: int,
    conf: float = 0.35
):
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("未找到 ROBOFLOW_API_KEY，请在 .env 中配置。")

    rf = Roboflow(api_key=api_key)
    rf_model = rf.workspace().project(project_id).version(version).model

    tracker = sv.ByteTrack()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    gui_available = True
    try:
        cv2.namedWindow("Tracker", cv2.WINDOW_NORMAL)
    except cv2.error:
        gui_available = False
        print("[WARN] OpenCV GUI is unavailable; disabling on-screen display.")

    ball_pos, hoop_pos = [], []
    frame_count = 0
    makes, attempts = 0, 0
    up, down = False, False
    up_frame, down_frame = 0, 0
    fade_counter, fade_frames = 0, 20
    overlay_color = (0, 0, 0)
    overlay_text = "Waiting..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── 1. 模型推理（Roboflow Hosted） ───────────────────────────────
        inference = rf_model.predict(frame, confidence=conf, overlap=0.5).json()
        detections = sv.Detections.from_inference(inference)
        detections = tracker.update_with_detections(detections)
        names = {i: name for i, name in enumerate(inference.get("class_names", []))}

        for xyxy, _, det_conf, cls_id, track_id, _ in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            w, h = x2 - x1, y2 - y1
            center = (x1 + w // 2, y1 + h // 2)
            cls_name = names.get(int(cls_id), "unknown")

            if cls_name in BALL_CLASSES:
                if det_conf > 0.3 or (in_hoop_region(center, hoop_pos) and det_conf > 0.15):
                    ball_pos.append((center, frame_count, w, h, det_conf))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    if track_id is not None:
                        cv2.putText(
                            frame,
                            f"ball#{track_id}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 165, 255),
                            1,
                        )

            elif cls_name in RIM_CLASSES and det_conf > 0.5:
                hoop_pos.append((center, frame_count, w, h, det_conf))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 0), 2)

        ball_pos = clean_ball_pos(ball_pos, frame_count)
        if len(hoop_pos) > 1:
            hoop_pos = clean_hoop_pos(hoop_pos)

        for bp in ball_pos:
            cv2.circle(frame, bp[0], 3, (0, 0, 255), -1)
        if hoop_pos:
            cv2.circle(frame, hoop_pos[-1][0], 4, (128, 128, 0), -1)

        if hoop_pos and ball_pos:
            if not up:
                up = detect_up(ball_pos, hoop_pos)
                if up:
                    up_frame = ball_pos[-1][1]

            if up and not down:
                down = detect_down(ball_pos, hoop_pos)
                if down:
                    down_frame = ball_pos[-1][1]

            if frame_count % 10 == 0 and up and down and up_frame < down_frame:
                attempts += 1
                if score(ball_pos, hoop_pos):
                    makes += 1
                    overlay_color = (0, 255, 0)
                    overlay_text = "Make!"
                else:
                    overlay_color = (0, 0, 255)
                    overlay_text = "Miss"
                fade_counter = fade_frames
                up = down = False

        score_text = f"{makes} / {attempts}"
        cv2.putText(frame, score_text, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(frame, score_text, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        (tw, _), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
        cv2.putText(frame, overlay_text, (frame.shape[1] - tw - 30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, overlay_color, 4)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if fade_counter > 0:
            alpha = 0.2 * (fade_counter / fade_frames)
            overlay = np.full_like(frame, overlay_color)
            frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
            fade_counter -= 1

        if gui_available:
            try:
                cv2.imshow("Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                gui_available = False
                print("[WARN] OpenCV GUI is unavailable; disabling on-screen display.")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 40)
    print(f"进球数:   {makes}")
    print(f"投篮次数: {attempts}")
    if attempts:
        print(f"命中率:   {100 * makes / attempts:.1f}%")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="../test_video/clip4_shoot.mp4", help="视频路径")
    parser.add_argument("--model", help="YOLOv8 权重路径 (.pt)")
    parser.add_argument("--conf",  type=float, default=0.35, help="置信度阈值")
    parser.add_argument("--backend", choices=["yolo", "roboflow"], default="yolo", help="推理后端")
    parser.add_argument("--rf-project", help="Roboflow 项目ID，例如 basketball-player-detection-3-ycjdo")
    parser.add_argument("--rf-version", type=int, help="Roboflow 版本号，例如 4")
    args = parser.parse_args()
    # 将相对路径统一解析到当前脚本目录，避免从不同工作目录运行导致找不到文件
    video_path = args.video
    if not os.path.isabs(video_path):
        video_path = os.path.normpath(os.path.join(os.path.dirname(__file__), video_path))
    if args.backend == "roboflow":
        if not args.rf_project or not args.rf_version:
            raise ValueError("使用 --backend roboflow 时必须提供 --rf-project 和 --rf-version")
        run_roboflow(video_path, args.rf_project, args.rf_version, args.conf)
    else:
        if not args.model:
            raise ValueError("使用 --backend yolo 时必须提供 --model")
        run(video_path, args.model, args.conf)
