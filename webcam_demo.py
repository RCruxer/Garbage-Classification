# webcam_demo.py
import cv2
from ultralytics import YOLO
MODEL_PATH = "runs/detect/train/weights/best.pt"  # or last.pt
DEVICE = "cpu"   # change to "0" if you have GPU and torch.cuda is configured
IMGSZ = 640
CONF = 0.25
IOU = 0.45

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, device=DEVICE, imgsz=IMGSZ, conf=CONF, iou=IOU)
        annotated = results[0].plot()  # BGR numpy image
        cv2.imshow("YOLO Webcam Demo", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
