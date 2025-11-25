import cv2
import easyocr
import numpy as np
import time
import os
import utils
import datetime
import torch

# Configuration
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SAVE_PATH = "detected_plates"
MODEL_PATH = "runs/detect/train3/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.5

# Initialize EasyOCR
print("Initializing EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)
print("EasyOCR initialized")

# Load YOLOv5 model
print(f"Loading YOLOv8 model from {MODEL_PATH}...")
try:
    # Direct loading of the PyTorch model
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    if isinstance(model, dict) and 'model' in model:
        model = model['model']
    model.float().eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    # Try alternative loading method
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully using YOLO")
    except Exception as e2:
        print(f"Error with alternative loading method: {e2}")
        exit(1)

# Load prebooked plates
with open("prebooked_plates.txt", "r") as f:
    prebooked = set(line.strip() for line in f if line.strip())
print(f"Loaded {len(prebooked)} prebooked plates")

# Create directory for saving plates
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    print(f"Created directory: {SAVE_PATH}")

def detect_and_recognize_plates():
    """Detect and recognize license plates using YOLOv5 model"""
    # Initialize webcam
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera opened successfully. Press 'q' to quit")
    
    frame_count = 0
    last_detection_time = time.time()
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Process every 5 frames to reduce CPU usage
            if frame_count % 5 == 0:
                try:
                    # Check if we're using the YOLO class or direct PyTorch model
                    if 'YOLO' in str(type(model)):
                        # Using YOLO class
                        results = model(frame)
                        boxes = results[0].boxes
                        detections = []
                        
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = float(box.conf[0])
                            if confidence >= CONFIDENCE_THRESHOLD:
                                detections.append({
                                    'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                                    'confidence': confidence
                                })
                    else:
                        # Using direct PyTorch model
                        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
                        img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
                        
                        with torch.no_grad():
                            preds = model(img)
                        
                        detections = []
                        if hasattr(preds, 'xyxy'):
                            # YOLOv5 format
                            for *xyxy, conf, cls in preds.xyxy[0]:
                                if conf >= CONFIDENCE_THRESHOLD:
                                    x1, y1, x2, y2 = map(int, xyxy)
                                    detections.append({
                                        'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                                        'confidence': float(conf)
                                    })
                        else:
                            # Fallback to traditional contour-based detection
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            filtered = cv2.bilateralFilter(gray, 11, 17, 17)
                            edged = cv2.Canny(filtered, 30, 200)
                            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
                            
                            for contour in contours:
                                peri = cv2.arcLength(contour, True)
                                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                                
                                if len(approx) == 4 and cv2.contourArea(contour) > 1000:
                                    x, y, w, h = cv2.boundingRect(contour)
                                    aspect_ratio = w / float(h)
                                    if 2.0 < aspect_ratio < 6.0:
                                        detections.append({
                                            'xmin': x, 'ymin': y, 'xmax': x+w, 'ymax': y+h,
                                            'confidence': 0.5  # Default confidence
                                        })
                    
                    # Process detections
                    for detection in detections:
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                        confidence = detection['confidence']
                        
                        # Extract the license plate region
                        plate_img = frame[y1:y2, x1:x2]
                        
                        if plate_img.size > 0:
                            # Draw rectangle around the license plate
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Check if enough time has passed since last detection
                            if current_time - last_detection_time > 2:
                                # Run OCR on the detected plate
                                ocr_results = reader.readtext(plate_img, detail=0, paragraph=False)
                                
                                for text in ocr_results:
                                    # Clean and normalize text
                                    plate_text = utils.clean_text(text)
                                    
                                    if plate_text:
                                        # Save the plate image
                                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                        plate_file = f"{SAVE_PATH}/plate_{timestamp}_{plate_text}.jpg"
                                        cv2.imwrite(plate_file, plate_img)
                                        
                                        # Display information
                                        print(f"\nDetected plate: {plate_text} (Confidence: {confidence:.2f})")
                                        
                                        # Check if plate is prebooked
                                        if plate_text in prebooked:
                                            status = "✅ PREBOOKED"
                                            color = (0, 255, 0)  # Green
                                            print("✅ Prebooked. OPEN GATE.")
                                        else:
                                            status = "❌ NOT PREBOOKED"
                                            color = (0, 0, 255)  # Red
                                            print("❌ Not prebooked. DON'T OPEN GATE.")
                                        
                                        # Draw text on the frame
                                        cv2.putText(frame, plate_text, (x1, y1 - 10), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                                        cv2.putText(frame, status, (x1, y2 + 20),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                        
                                        last_detection_time = current_time
                                        break  # Stop after finding one plate
                except Exception as e:
                    print(f"Error in detection: {e}")
                
                # Add FPS information
                fps = 1.0 / (time.time() - current_time + 0.001)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow("ANPR System", frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")
        
if __name__ == "__main__":
    print("\n=== License Plate Recognition System ===\n")
    detect_and_recognize_plates()
        


