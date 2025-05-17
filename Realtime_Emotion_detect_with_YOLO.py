from ultralytics import YOLO
import cv2
import numpy as np

# Load trained YOLOv8 model
model = YOLO(r"./YOLO files/best.pt")
class_names = model.names  # e.g. {0: 'angry', 1: 'disgust', ...}

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=640)[0]
    boxes = results.boxes

    emotion_scores = {name: 0.0 for name in class_names.values()}
    emotion_counts = {name: 0 for name in class_names.values()}

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = class_names[cls_id]

        emotion_scores[label] += conf
        emotion_counts[label] += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label_text = f"{label}: {conf*100:.1f}%"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # Draw background rectangle
        cv2.rectangle(frame, (x1, y1 - text_height - 8), (x1 + text_width, y1), (0, 0, 0), -1)
        # Draw label text
        cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display all emotion stats on left
    start_y = 30
    for idx, (label, total_conf) in enumerate(emotion_scores.items()):
        count = emotion_counts[label]
        avg_conf = (total_conf / count * 100) if count > 0 else 0.0
        text = f"{label}: {avg_conf:.1f}%"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_pos = start_y + idx * 25

        # Background rectangle for stats
        cv2.rectangle(frame, (5, y_pos - text_height - 4), (5 + text_width + 5, y_pos + 5), (0, 0, 0), -1)
        # Draw emotion stat text
        cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("YOLOv8 Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()