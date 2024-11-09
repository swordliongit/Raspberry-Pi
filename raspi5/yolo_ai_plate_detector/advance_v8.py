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
        
        # Detection tracking
        self.last_plate = None
        self.last_box = None  # Store last bounding box
        self.last_confidence = None
        self.last_read_time = 0
        self.read_cooldown = 0.01  # Faster updates
        self.reset_threshold = 0.4  # Position change threshold
        
        # Plate locking mechanism
        self.locked_plate = None
        self.lock_time = None
        self.lock_duration = 0.1  # Short lock duration for faster transitions
        self.stable_readings = []
        self.max_stable = 1
        self.stability_threshold = 1    
        
        # Valid Turkish city codes (01-81)
        self.valid_city_codes = set(f"{i:02d}" for i in range(1, 82))
        
        # Valid Turkish letters for the middle part
        self.valid_letters = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        # Common OCR mistakes in Turkish plates
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
        
        # Prefix corrections for common misreadings
        self.prefix_corrections = {}
        for city_code in self.valid_city_codes:
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
                city_code.replace('0', 'O'),  # Convert "06" -> "O6"
                f"H{city_code.replace('0', 'O')}",  # HO6 -> 06
                f"W{city_code.replace('0', 'O')}",  # WO6 -> 06
            ]
            
            for pattern in patterns:
                self.prefix_corrections[pattern] = city_code

        # Output file setup
        self.output_file = "detections.txt"
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def should_reset_detection(self, current_box, current_conf):
        """Check if we should reset detection based on box position or confidence"""
        if self.last_box is None or self.last_confidence is None:
            return True
            
        # Calculate box center
        current_center = ((current_box[0] + current_box[2])/2, (current_box[1] + current_box[3])/2)
        last_center = ((self.last_box[0] + self.last_box[2])/2, (self.last_box[1] + self.last_box[3])/2)
        
        # Check for significant position change
        position_change = abs(current_center[0] - last_center[0]) + abs(current_center[1] - last_center[1])
        if position_change > self.reset_threshold * (current_box[2] - current_box[0]):
            return True
        
        # Check for significant confidence change
        if abs(current_conf - self.last_confidence) > 0.1:
            return True
        
        return False

    def clean_plate_text(self, text):
        """Clean and format plate text with strict Turkish plate rules:
        Format: XX ABC YYYY or XX ABC YY
        where: XX = city code (01-81)
               ABC = 1-3 letters
               YYYY or YY = 2 or 4 numbers"""
        if not text:
            return None
            
        # Convert to uppercase and remove spaces
        text = text.upper().replace(' ', '')
        
        # Clean the text (remove non-alphanumeric)
        text = ''.join(c for c in text if c.isalnum())
        
        try:
            # Step 1: Extract and validate city code (first two digits)
            if len(text) < 5:  # Minimum length: 2 (city) + 1 (letter) + 2 (numbers)
                return None
                
            city_code = ''
            remaining = text
            
            # Handle first two characters for city code
            for i in range(min(3, len(text))):
                if text[i:i+2].isdigit():
                    potential_code = text[i:i+2]
                    # Apply corrections to potential city code
                    cleaned_code = ''
                    for c in potential_code:
                        cleaned_code += self.char_corrections.get(c, c)
                    
                    if cleaned_code in self.valid_city_codes:
                        city_code = cleaned_code
                        remaining = text[i+2:]
                        break
            
            if not city_code:
                return None
                
            # Step 2: Extract letters (1-3 letters after city code)
            letters = ''
            numbers = ''
            letter_count = 0
            
            # Process remaining characters
            number_start = False
            for c in remaining:
                # Once we start seeing numbers, don't accept any more letters
                if number_start and c.isalpha():
                    return None
                    
                if not number_start and c.isalpha() and letter_count < 3:
                    if c in self.valid_letters:
                        letters += c
                        letter_count += 1
                elif c.isdigit() or c in self.char_corrections:
                    number_start = True
                    if c in self.char_corrections:
                        numbers += self.char_corrections[c]
                    else:
                        numbers += c
                else:
                    return None
            
            # If exactly 4 digits starting with '00', remove one zero
            if len(numbers) == 4 and numbers.startswith('00'):
                numbers = '0' + numbers[2:]
            
            # Validate final format
            if (1 <= len(letters) <= 3) and (len(numbers) == 2 or len(numbers) == 4):
                if letters.isalpha() and numbers.isdigit():
                    return f"{city_code} {letters} {numbers}"
            
        except Exception as e:
            print(f"Error in plate cleaning: {e}")
            return None
        
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
        """Perform OCR with plate locking"""
        current_time = time.time()
        
        # Check if we have a locked plate and if the lock is still valid
        if self.locked_plate and self.lock_time:
            if current_time - self.lock_time < self.lock_duration:
                return self.locked_plate
            else:
                # Lock expired, clear it
                self.locked_plate = None
                self.lock_time = None
                self.stable_readings = []
        
        # Normal plate reading process
        processed = self.preprocess_plate(plate_img)
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        try:
            text = pytesseract.image_to_string(processed, config=custom_config)
            cleaned_text = self.clean_plate_text(text)
            
            if cleaned_text:
                # Add to stable readings
                self.stable_readings.append(cleaned_text)
                if len(self.stable_readings) > self.max_stable:
                    self.stable_readings.pop(0)
                
                # Check for stability
                if len(self.stable_readings) >= self.stability_threshold:
                    most_common = max(set(self.stable_readings), key=self.stable_readings.count)
                    if self.stable_readings.count(most_common) >= self.stability_threshold:
                        # We have a stable reading, lock it
                        self.locked_plate = most_common
                        self.lock_time = current_time
                        return most_common
                
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
                    
                    if confidence < 0.55:
                        continue
                    
                    # Check if detection parameters changed significantly
                    if self.should_reset_detection([x1, y1, x2, y2], confidence):
                        self.last_plate = None
                        print("Reset detection due to significant change")
                    
                    plate_img = frame[y1:y2, x1:x2]
                    
                    if self.should_read_plate():
                        plate_text = self.read_plate(plate_img)
                        if plate_text:
                            print(f"License Plate Detected: {plate_text} (Confidence: {confidence:.2f})")
                            self.save_detection(plate_text, confidence)
                            self.last_plate = plate_text
                            
                    # Update tracking variables
                    self.last_box = [x1, y1, x2, y2]
                    self.last_confidence = confidence
                    
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