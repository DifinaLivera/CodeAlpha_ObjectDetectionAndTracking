from detection.detector import PersonDetector
import cv2

video_path = r"videos\test_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

detector = PersonDetector()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    output = detector.draw_detections(frame, detections)

    frame_count += 1
    print(f"Frame {frame_count}: {len(detections)} detections")

    cv2.imshow("Detection Test", output)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Saved video test complete.")