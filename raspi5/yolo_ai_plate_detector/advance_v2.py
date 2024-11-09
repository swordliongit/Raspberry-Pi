
import cv2
from picamera2 import Picamera2
from libcamera import controls
from ultralytics import YOLO
import time
import numpy as np
import pytesseract
from datetime import datetime
import os

class LicensePlateReader:
    def __init__(self):
        # Initialize camera
        self.picam2 = Picamera2()
        preview_config = self.picam2.create_preview_configuration(
            main={"format": 'RGB888', "size": (640, 640)}
        )
        self.picam2.configure(preview_config)
        
        # Load YOLO model
        self.model = YOLO("license_plate_detector_ncnn_model")
        
        # OCR configuration
        self.last_plate = None
        self.last_read_time = 0
        self.read_cooldown = 1  # Seconds between OCR attempts

        # Focus control variables
        self.last_focus_time = 0
        self.last_detection_time = time.time()
        self.focus_cooldown = 5        # Minimum seconds between focus attempts
        self.detection_timeout = 3     # Seconds without detection before focusing

        # Output file setup
        self.output_file = "detections.txt"
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def initial_focus(self):
        """Perform focus cycle"""
        print("Focusing camera...")
        self.picam2.set_controls({"AfMode": 1, "AfTrigger": 0})
        time.sleep(1.5)
        self.picam2.set_controls({"AfMode": 0})
    
    def preprocess_plate(self, plate_img):
        """Preprocess license plate image for better OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Reduce noise
        thresh = cv2.medianBlur(thresh, 3)
        
        # Dilation to make characters thicker
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        
        return thresh

    def read_plate(self, plate_img):
        """Perform OCR on preprocessed plate image"""
        # Preprocess the plate image
        processed = self.preprocess_plate(plate_img)
        
        # Configure pytesseract for license plates
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        # Perform OCR
        try:
            text = pytesseract.image_to_string(processed, config=custom_config)
            # Clean the text
            text = ''.join(c for c in text if c.isalnum())
            return text if text else None
        except:
            return None

    def should_read_plate(self):
        """Check if enough time has passed for new OCR attempt"""
        current_time = time.time()
        if current_time - self.last_read_time >= self.read_cooldown:
            self.last_read_time = current_time
            return True
        return False

    def save_detection(self, plate_text, confidence):
        """Save detection to file with timestamp and confidence"""
        with open(self.output_file, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - Plate: {plate_text}, Confidence: {confidence:.2f}\n")

    def run(self):
        self.picam2.start()
        print("Starting detection...")
        print("Controls:")
        print("f - Force refocus")
        print("q - Quit")
        
        while True:
            frame = self.picam2.capture_array()
            current_time = time.time()
            detection_found = False
            
            results = self.model(frame)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Skip low confidence detections
                    if confidence < 0.5:
                        continue
                        
                    detection_found = True
                    plate_img = frame[y1:y2, x1:x2]
                    
                    if self.should_read_plate():
                        plate_text = self.read_plate(plate_img)
                        if plate_text and len(plate_text) > 3:  # Minimum length check
                            if plate_text != self.last_plate:
                                print(f"License Plate Detected: {plate_text} (Confidence: {confidence:.2f})")
                                self.save_detection(plate_text, confidence)
                                self.last_plate = plate_text
                            
                            # Draw rectangle around plate
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Draw text background
                            text = f"{plate_text} ({confidence:.2f})"
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                            cv2.rectangle(frame, 
                                        (x1, y1 - 25), 
                                        (x1 + text_size[0], y1), 
                                        (0, 255, 0), 
                                        -1)
                            
                            # Draw plate text
                            cv2.putText(frame, text,
                                      (x1, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.8, (0, 0, 0), 2)
                    else:
                        # Always draw the bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # If we have a last plate, show it
                        if self.last_plate:
                            text = f"{self.last_plate} ({confidence:.2f})"
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                            cv2.rectangle(frame, 
                                        (x1, y1 - 25), 
                                        (x1 + text_size[0], y1), 
                                        (0, 255, 0), 
                                        -1)
                            cv2.putText(frame, text,
                                      (x1, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.8, (0, 0, 0), 2)
            
            # Check if we should refocus
            if (not detection_found and 
                current_time - self.last_detection_time > self.detection_timeout and 
                current_time - self.last_focus_time > self.focus_cooldown):
                self.initial_focus()
                self.last_focus_time = current_time
                
            # Display frame
            cv2.imshow("License Plate Detection", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                self.initial_focus()
                self.last_focus_time = current_time
         
        cv2.destroyAllWindows()
        self.picam2.stop()

if __name__ == "__main__":
    reader = LicensePlateReader()
    reader.run()
