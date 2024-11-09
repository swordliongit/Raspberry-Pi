import cv2
from picamera2 import Picamera2
from libcamera import controls
from ultralytics import YOLO
import time
import numpy as np
import pytesseract
from datetime import datetime
import os
import re

class LicensePlateReader:
    def __init__(self):
        # Initialize camera
        self.picam2 = Picamera2()
        preview_config = self.picam2.create_preview_configuration(
            main={"format": 'RGB888', "size": (1024, 1024)}
        )
        self.picam2.configure(preview_config)
        
        # Set continuous autofocus mode
        self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        
        # Load YOLO model
        self.model = YOLO("license_plate_detector_ncnn_model")
        
        # OCR configuration
        self.last_plate = None
        self.last_read_time = 0
        self.read_cooldown = 0.2  # Fast updates
        
        # Generate all valid Turkish city codes (01-81)
        self.valid_city_codes = set(f"{i:02d}" for i in range(1, 82))
        
        # Generate prefix corrections for all city codes
        self.prefix_corrections = {}
        for city_code in self.valid_city_codes:
            # Common patterns to check for each city code
            patterns = [
                f"H{city_code}",  # H34 -> 34
                f"W{city_code}",  # W34 -> 34
                f"Y{city_code}",  # Y34 -> 34
                f"B{city_code}",  # B34 -> 34
                f"E{city_code}",  # E34 -> 34
                f"A{city_code}",  # A34 -> 34
                f"S{city_code}",  # S34 -> 34
                f"1{city_code}",  # 134 -> 34
                f"2{city_code}",  # 234 -> 34
                f"3{city_code}",  # 334 -> 34
                f"4{city_code}",  # 434 -> 34
                f"5{city_code}",  # 534 -> 34
                # Also handle O vs 0 confusion
                city_code.replace('0', 'O'),  # Convert "06" -> "O6"
                f"H{city_code.replace('0', 'O')}",  # HO6 -> 06
                f"W{city_code.replace('0', 'O')}",  # WO6 -> 06
            ]
            
            # Add each pattern to corrections dict
            for pattern in patterns:
                self.prefix_corrections[pattern] = city_code
        
        # Common OCR mistakes in characters
        self.char_corrections = {
            'O': '0',
            'I': '1',
            'S': '5',
            'B': '8',
            'Z': '2',
            'G': '6',
            'T': '7',
            'L': '1',
            'D': '0',
            'Q': '0',
            'U': '0',
            'V': '7',
            'H': '4'
        }
        # Output file setup
        self.output_file = "detections.txt"
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def clean_plate_text(self, text):
        """Clean and format plate text with strict Turkish plate rules"""
        if not text:
            return None
            
        # Convert to uppercase and remove spaces
        text = text.upper().replace(' ', '')
        
        # Clean the text (remove non-alphanumeric)
        text = ''.join(c for c in text if c.isalnum())
        
        # Try prefix corrections first
        for wrong, right in self.prefix_corrections.items():
            if text.startswith(wrong):
                text = right + text[len(wrong):]
                break
        
        # Extract parts
        try:
            # Find the first two digits
            first_two = ''
            remaining = text
            
            # Handle common prefixes that might appear before the city code
            if len(text) > 2:
                if text[0].isdigit():
                    if text[1].isdigit():
                        first_two = text[:2]
                        remaining = text[2:]
                    elif text[2].isdigit():
                        first_two = text[1:3]
                        remaining = text[3:]
                elif len(text) > 3 and text[1].isdigit() and text[2].isdigit():
                    first_two = text[1:3]
                    remaining = text[3:]
            
            # Apply corrections to potential city code
            if first_two:
                # Clean up the city code
                city_code = ''
                for c in first_two:
                    if c in self.char_corrections:
                        city_code += self.char_corrections[c]
                    else:
                        city_code += c
                
                # Validate city code
                if city_code in self.valid_city_codes:
                    # Find the letters part (2-3 letters)
                    letters = ''
                    numbers = ''
                    letter_count = 0
                    
                    for c in remaining:
                        if letter_count < 3 and c.isalpha():
                            letters += c
                            letter_count += 1
                        elif c.isdigit() or c in self.char_corrections:
                            if c in self.char_corrections:
                                numbers += self.char_corrections[c]
                            else:
                                numbers += c
                    
                    # Validate parts
                    if 2 <= len(letters) <= 3 and 2 <= len(numbers) <= 4:
                        return f"{city_code} {letters} {numbers}"
        
        except Exception:
            pass
        
        return None
    
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
        """Perform OCR with Turkish plate validation"""
        processed = self.preprocess_plate(plate_img)
        
        # Configure pytesseract for license plates
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        try:
            text = pytesseract.image_to_string(processed, config=custom_config)
            cleaned_text = self.clean_plate_text(text)
            return cleaned_text
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
        print("Press 'q' to quit")
        
        while True:
            frame = self.picam2.capture_array()
            results = self.model(frame)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    if confidence < 0.55:  # Confidence threshold
                        continue
                    
                    plate_img = frame[y1:y2, x1:x2]
                    
                    if self.should_read_plate():
                        plate_text = self.read_plate(plate_img)
                        if plate_text:
                            print(f"License Plate Detected: {plate_text} (Confidence: {confidence:.2f})")
                            self.save_detection(plate_text, confidence)
                            self.last_plate = plate_text
                    
                    # Always draw the detection
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    if self.last_plate:
                        # Draw text background
                        text = f"{self.last_plate} ({confidence:.2f})"
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
            
            # Display frame
            cv2.imshow("License Plate Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        self.picam2.stop()

if __name__ == "__main__":
    reader = LicensePlateReader()
    reader.run()