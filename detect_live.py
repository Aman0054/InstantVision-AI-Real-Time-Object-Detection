import argparse
from typing import Optional, List, Tuple
from collections import deque
import numpy as np

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live object detection from webcam using YOLOv8"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="Path or name of YOLOv8 model (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt). Larger models are more accurate but slower.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index (0 is default)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for predictions (0.0 - 1.0). Higher values reduce false positives.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IOU threshold for Non-Maximum Suppression (0.0 - 1.0). Lower values reduce overlapping boxes.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for inference. Larger values (e.g., 1280) improve accuracy but are slower.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Number of frames to average for temporal smoothing (0 to disable). Reduces flickering.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use half precision (FP16) for faster inference on GPU (may improve accuracy on some GPUs).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on, e.g., 'cpu', 'cuda', '0', '0,1'. Defaults to auto",
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Overlay FPS on the stream",
    )
    return parser.parse_args()


class TemporalSmoother:
    """Smooths detections across frames to reduce flickering."""
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.detection_history: deque = deque(maxlen=window_size)
    
    def smooth(self, boxes: List, confidences: List, class_ids: List) -> Tuple[List, List, List]:
        """Average detections over the last N frames."""
        if self.window_size == 0 or len(boxes) == 0:
            return boxes, confidences, class_ids
        
        # Store current detections
        current_detections = {
            'boxes': boxes,
            'confidences': confidences,
            'class_ids': class_ids
        }
        self.detection_history.append(current_detections)
        
        if len(self.detection_history) < 2:
            return boxes, confidences, class_ids
        
        # Average boxes and confidences for same class IDs
        # Simple approach: use the most recent detection but with averaged confidence
        if len(self.detection_history) >= self.window_size:
            # Group by class and average
            class_detections = {}
            for det in self.detection_history:
                for box, conf, cls_id in zip(det['boxes'], det['confidences'], det['class_ids']):
                    cls_id_int = int(cls_id)
                    if cls_id_int not in class_detections:
                        class_detections[cls_id_int] = {'boxes': [], 'confs': []}
                    class_detections[cls_id_int]['boxes'].append(box)
                    class_detections[cls_id_int]['confs'].append(conf)
            
            # Average the detections
            smoothed_boxes = []
            smoothed_confs = []
            smoothed_ids = []
            
            for cls_id, data in class_detections.items():
                if len(data['boxes']) >= 2:  # Only smooth if we have multiple detections
                    # Average box coordinates
                    avg_box = np.mean(data['boxes'], axis=0).tolist()
                    # Average confidence
                    avg_conf = np.mean(data['confs'])
                    smoothed_boxes.append(avg_box)
                    smoothed_confs.append(avg_conf)
                    smoothed_ids.append(cls_id)
                elif len(data['boxes']) == 1:
                    # Use single detection as-is
                    smoothed_boxes.append(data['boxes'][0])
                    smoothed_confs.append(data['confs'][0])
                    smoothed_ids.append(cls_id)
            
            return smoothed_boxes, smoothed_confs, smoothed_ids
        
        return boxes, confidences, class_ids


def draw_detections(frame, boxes, confidences, class_ids, class_names, fps_text: Optional[str] = None):
    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = class_names[int(cls_id)] if (class_names and int(cls_id) < len(class_names)) else str(cls_id)
        caption = f"{label} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 170, 255), 2)
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), (0, 170, 255), -1)
        cv2.putText(frame, caption, (x1 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    if fps_text:
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2)

    return frame


def main():
    args = parse_args()

    # Load model (auto-downloads weights if needed)
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    # Warmup read to get frame size
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Failed to read from camera. Check the index or permissions.")

    # Inference loop
    prev_tick = cv2.getTickCount()
    tick_freq = cv2.getTickFrequency()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(
                source=frame,
                conf=args.conf,
                verbose=False,
                device=args.device if args.device is not None else None,
            )

            boxes = []
            confidences = []
            class_ids = []
            class_names = results[0].names if results and len(results) > 0 else None

            if results and len(results) > 0:
                r = results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    for b in r.boxes:
                        xyxy = b.xyxy[0].tolist()
                        conf = float(b.conf[0].item()) if b.conf is not None else 0.0
                        cls_id = int(b.cls[0].item()) if b.cls is not None else -1
                        boxes.append(xyxy)
                        confidences.append(conf)
                        class_ids.append(cls_id)

            fps_text = None
            if args.show_fps:
                curr_tick = cv2.getTickCount()
                dt = (curr_tick - prev_tick) / tick_freq
                prev_tick = curr_tick
                if dt > 0:
                    fps_text = f"FPS: {1.0 / dt:.1f}"

            annotated = draw_detections(frame, boxes, confidences, class_ids, class_names, fps_text)
            cv2.imshow("YOLOv8 Live - Press ESC to quit", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


