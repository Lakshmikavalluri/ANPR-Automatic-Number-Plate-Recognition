from google.cloud import vision
import os
import cv2
import numpy as np
import sys
import time
import re
from threading import Thread
from queue import Queue

def check_python_version():
    """Check if we're running on Python 3.13"""
    if sys.version_info.major != 3 or sys.version_info.minor != 13:
        print(f"Warning: This code is running on Python {sys.version_info.major}.{sys.version_info.minor}")
        print("Some features might not work as expected.")
    else:
        print("Running on Python 3.13 - All features supported!")

# Set Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'parknsecure-459805-660b517a2be3.json'

class PlateDetector:
    def __init__(self):
        self.client = vision.ImageAnnotatorClient()
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.debug_queue = Queue(maxsize=2)
        self.running = True
        self.detection_thread = Thread(target=self._detection_worker, daemon=True)
        self.detection_thread.start()
        
        # Indian license plate patterns
        self.patterns = [
            r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$',  # KA19EQ0001
            r'^[A-Z]{2}\d{2}[A-Z]{2}\d{3}$',  # KA01AB123
            r'^[A-Z]{2}\d{2}[A-Z]\d{4}$',     # KA01A1234
            r'^[A-Z]{2}\d{2}[A-Z]{2}\d{2}$',  # KA01AB12
            r'^[A-Z]{2}\d{1}[A-Z]{2}\d{4}$'   # DL7CQ1939
        ]
        print("Plate detector initialized successfully")

    def _is_valid_plate(self, text):
        """Check if the text matches any of the Indian license plate patterns"""
        # Remove spaces and convert to uppercase
        text = text.replace(" ", "").upper()
        
        # Try to match any of the patterns
        for pattern in self.patterns:
            if re.match(pattern, text):
                return True
        return False

    def _detection_worker(self):
        """Worker thread for plate detection"""
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                if frame is None:
                    continue
                
                try:
                    # Create debug image
                    debug_img = frame.copy()
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Apply Gaussian blur
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    
                    # Apply Canny edge detection
                    edges = cv2.Canny(blurred, 50, 150)
                    
                    # Find contours
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Sort contours by area
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    
                    # Process top 5 largest contours
                    for contour in contours[:5]:
                        # Get rectangle bounding contour
                        [x, y, w, h] = cv2.boundingRect(contour)
                        
                        # Draw rectangle for all contours
                        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Calculate aspect ratio
                        aspect_ratio = w / float(h)
                        
                        # Check if contour could be a license plate
                        if 2.0 < aspect_ratio < 5.0 and w > 100 and h > 20:
                            # Extract region
                            roi = frame[y:y+h, x:x+w]
                            
                            # Try to detect text
                            plate_number = self._process_frame(roi)
                            if plate_number and self._is_valid_plate(plate_number):
                                print(f"Detected valid Indian plate: {plate_number}")
                                self.result_queue.put(plate_number)
                                # Draw red rectangle for detected plate
                                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                                cv2.putText(debug_img, plate_number, (x, y-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                break
                            elif plate_number:
                                print(f"Detected text but not a valid plate: {plate_number}")
                    
                    # Add debug information
                    cv2.putText(debug_img, f"Contours found: {len(contours)}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Add debug image to queue
                    if not self.debug_queue.full():
                        self.debug_queue.put(debug_img)
                
                except Exception as e:
                    print(f"Error in detection worker: {str(e)}")

    def _process_frame(self, frame):
        """Process a single frame for plate detection"""
        try:
            # Resize image for better detection
            frame = cv2.resize(frame, (0, 0), fx=2, fy=2)
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Convert the image to bytes
            _, img_encoded = cv2.imencode('.jpg', thresh, [cv2.IMWRITE_JPEG_QUALITY, 95])
            content = img_encoded.tobytes()

            # Detect text
            image = vision.Image(content=content)
            response = self.client.text_detection(image=image)
            texts = response.text_annotations

            if texts and len(texts) > 0:
                detected_text = texts[0].description
                cleaned_text = ' '.join(detected_text.split())
                
                # Basic validation
                if len(cleaned_text) >= 5:
                    return cleaned_text
            
            return None

        except Exception as e:
            print(f"Error in text detection: {str(e)}")
            return None

    def add_frame(self, frame):
        """Add a frame to the processing queue"""
        if not self.frame_queue.full():
            self.frame_queue.put(frame)

    def get_result(self):
        """Get the latest detection result"""
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None

    def get_debug_frame(self):
        """Get the latest debug frame"""
        if not self.debug_queue.empty():
            return self.debug_queue.get()
        return None

    def stop(self):
        """Stop the detector"""
        self.running = False
        self.detection_thread.join()

def main():
    # Check Python version
    check_python_version()
    
    print("\nANPR (Automatic Number Plate Recognition) System")
    print("===============================================")
    print("Press 'q' to quit the program")
    print("\nSupported Indian License Plate Formats:")
    print("1. KA19EQ0001 (State + 2 digits + 2 letters + 4 digits)")
    print("2. KA01AB123  (State + 2 digits + 2 letters + 3 digits)")
    print("3. KA01A1234  (State + 2 digits + 1 letter + 4 digits)")
    print("4. KA01AB12   (State + 2 digits + 2 letters + 2 digits)")
    print("5. DL7CQ1939  (State + 1 digit + 2 letters + 4 digits)")
    
    # Try different webcam indices
    cap = None
    for i in range(2):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Successfully opened webcam {i}")
            break
    
    if not cap or not cap.isOpened():
        print("Error: Could not open any webcam")
        return
    
    try:
        # Set webcam resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize plate detector
        detector = PlateDetector()
        
        # FPS calculation variables
        frame_count = 0
        start_time = time.time()
        fps = 0
        last_plate = None
        
        print("Starting main loop...")
        
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                cap.release()
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    print("Failed to reconnect to webcam")
                    break
                continue
            
            # Calculate FPS
            frame_count += 1
            if frame_count >= 30:
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()
            
            # Add frame to detector queue
            detector.add_frame(frame)
            
            # Get detection result
            plate_number = detector.get_result()
            if plate_number:
                last_plate = plate_number
                print(f"New plate detected: {plate_number}")
            
            # Get debug frame
            debug_frame = detector.get_debug_frame()
            if debug_frame is not None:
                frame = debug_frame
            
            # Display FPS and plate number
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if last_plate:
                cv2.putText(frame, f"Plate: {last_plate}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('ANPR System', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        # Release resources
        if detector:
            detector.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("\nProgram terminated successfully")

if __name__ == '__main__':
    main()