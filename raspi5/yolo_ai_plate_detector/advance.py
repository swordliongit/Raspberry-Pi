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
            main={"format": 'RGB888', "size": (640, 640)}
        )
        self.picam2.configure(preview_config)
        
        # Set up continuous autofocus
        self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        
        # Load YOLO model
        self.model = YOLO("license_plate_detector_ncnn_model")
        
        # OCR configuration
        self.last_plate = None
        self.last_read_time = 0
        self.read_cooldown = 0.5  # Reduced cooldown for faster detection
        
        # Turkish plate pattern (2 numbers + 1-3 letters + 2-4 numbers)
        self.plate_pattern = re.compile(r'^(\d{2})([A-Z]{1,3})(\d{2,4})$')
        
        # Common OCR mistakes in Turkish plates
        self.char_corrections = {
            '0': 'O',
            'O': '0',
            'I': '1',
            '1': 'I',
            'S': '5',
            '5': 'S',
            'B': '8',
            '8': 'B'
        }

        # Output file setup
        self.output_file = "detections.txt"
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def correct_common_mistakes(self, text, position):
        """Correct common OCR mistakes based on position in plate"""
        if not text:
            return text
            
        # First two characters should be numbers
        if position == 'prefix' and len(text) == 2:
            return ''.join('0' if c == 'O' else '1' if c == 'I' else c for c in text)
            
        # Middle section should be letters
        elif position == 'letters':
            return ''.join('O' if c == '0' else 'I' if c == '1' else c for c in text)
            
        # Last section should be numbers
        elif position == 'suffix':
            return ''.join('0' if c == 'O' else '1' if c == 'I' else c for c in text)
            
        return text

    def validate_plate(self, text):
        """Validate and format plate number according to Turkish format"""
        if not text:
            return None
            
        # Remove any spaces and convert to uppercase
        text = text.upper().replace(' ', '')
        
        # Try different combinations of common OCR mistakes
        variations = [text]
        for i, char in enumerate(text):
            if char in self.char_corrections:
                new_text = text[:i] + self.char_corrections[char] + text[i+1:]
                variations.append(new_text)
        
        # Check each variation against the pattern
        for variant in variations:
            match = self.plate_pattern.match(variant)
            if match:
                numbers_prefix = self.correct_common_mistakes(match.group(1), 'prefix')
                letters = self.correct_common_mistakes(match.group(2), 'letters')
                numbers_suffix = self.correct_common_mistakes(match.group(3), 'suffix')
                return f"{numbers_prefix} {letters} {numbers_suffix}"
        
        return None

    def preprocess_plate(self, plate_img):
        """Enhanced preprocessing for Turkish plates"""
        # Resize to a standard size
        plate_img = cv2.resize(plate_img, (240, 80))
        
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        gray = cv2.equalizeHist(gray)
        
        # Apply bilateral filter to reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
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
        """Improved OCR with multiple attempts"""
        processed = self.preprocess_plate(plate_img)
        
        # Try multiple PSM modes and configurations
        psm_modes = [7, 8, 6]  # Different page segmentation modes
        for psm in psm_modes:
            config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            try:
                text = pytesseract.image_to_string(processed, config=config)
                text = ''.join(c for c in text if c.isalnum())
                
                # Validate and format the plate number
                valid_plate = self.validate_plate(text)
                if valid_plate:
                    return valid_plate
                    
            except Exception as e:
                continue
                
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
                    
                    if confidence < 0.5:  # Skip low confidence detections
                        continue
                        
                    plate_img = frame[y1:y2, x1:x2]
                    
                    if self.should_read_plate():
                        plate_text = self.read_plate(plate_img)
                        if plate_text and plate_text != self.last_plate:
                            print(f"License Plate Detected: {plate_text} (Confidence: {confidence:.2f})")
                            self.save_detection(plate_text, confidence)
                            self.last_plate = plate_text
                        
                        if plate_text:
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Draw background for text
                            text = f"{plate_text} ({confidence:.2f})"
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(frame, 
                                        (x1, y1 - 25), 
                                        (x1 + text_size[0], y1), 
                                        (0, 255, 0), 
                                        -1)
                            
                            # Draw text
                            cv2.putText(frame, text,
                                      (x1, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.6, (0, 0, 0), 2)

            cv2.imshow("License Plate Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        self.picam2.stop()

if __name__ == "__main__":
    reader = LicensePlateReader()
    reader.run()