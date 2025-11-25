import cv2
import easyocr
import utils
import os
import time
import warnings
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# Mute FutureWarnings and reduce logging noise
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["ULTRALYTICS_LOGGING"] = "False"  # Optional: Suppress YOLO logs

# ─────────────────────────────────────────────────────────────────────────────
# Load YOLOv8 trained model
model = YOLO("C:/Users/satis/OneDrive/Documents/ANPR_YOLO_EasyOCR/runs/detect/train3/weights/best.pt")

# Initialize EasyOCR
reader = easyocr.Reader(["en"], gpu=False)

# Load prebooked plates
with open("prebooked_plates.txt", "r") as f:
    prebooked = set(line.strip() for line in f if line.strip())

print("🚀 ANPR system started. Press 'q' to quit.\n")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open camera.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame. Retrying…")
            continue

        # Run YOLOv8 detection
        results = model(frame)

        # Process detections and run OCR
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = frame[y1:y2, x1:x2]

                # Run OCR on cropped plate
                ocr_result = reader.readtext(plate_crop, detail=0, paragraph=False)
                for text in ocr_result:
                    plate = utils.clean_text(text)
                    if not plate:
                        continue

                    # Draw bounding box and plate text on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, plate, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 0), 2)

                    # Print status to console
                    print(f"\nDetected Plate: {plate}")
                    if plate in prebooked:
                        print("✅ Prebooked.\n✅ OPEN GATE.")
                    else:
                        print("❌ Not Prebooked.\n🚫 GO & PREBOOK.\nDON’T OPEN GATE.")

        # Show video feed window
        cv2.imshow("ANPR Camera Feed", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 ANPR stopped by user.")
            break

        time.sleep(0.2)  # CPU load control

finally:
    cap.release()
    cv2.destroyAllWindows()

 