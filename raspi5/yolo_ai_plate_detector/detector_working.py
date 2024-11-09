import cv2
from picamera2 import Picamera2
from libcamera import controls
from ultralytics import YOLO
import time
import numpy as np
import pytesseract

class LicensePlateReader:
    def __init__(self):
        # Initialize camera
        self.picam2 = Picamera2()
        preview_config = self.picam2.create_preview_configuration(
            main={"format": 'RGB888', "size": (640, 640)}
        )
        self.picam2.configure(preview_config)
        
        # Load YOLO model
        self.model = YOLO("license_plate_detector_ncnn_model")  # Replace with your model path
        
        # Focus control variables
        self.consecutive_blurry = 0
        self.blur_threshold = 100
        self.last_focus_time = 0
        self.focus_cooldown = 2
        
        # OCR configuration
        self.last_plate = None
        self.last_read_time = 0
        self.read_cooldown = 1  # Seconds between OCR attempts
        
    def initial_focus(self):
        """Perform initial focus and wait for it to settle"""
        print("Setting initial focus...")
        self.picam2.set_controls({"AfMode": 1, "AfTrigger": 0})
        time.sleep(2)
        self.picam2.set_controls({"AfMode": 0})
        print("Focus locked.")
        
    def is_blurry(self, image, threshold=100):
        """Check if image is blurry using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold
    
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

    def run(self):
        # Start camera
        self.picam2.start()
        
        # Initial warmup and focus
        print("Warming up camera...")
        time.sleep(2)
        self.initial_focus()
        
        print("Starting detection...")
        print("Controls:")
        print("f - Force refocus")
        print("q - Quit")
        
        while True:
            # Capture frame
            frame = self.picam2.capture_array()
            
            # Check focus and refocus if needed
            if self.is_blurry(frame):
                self.consecutive_blurry += 1
                if self.consecutive_blurry >= 3:
                    print("Refocusing due to blur...")
                    self.initial_focus()
                    self.consecutive_blurry = 0
                    continue
            else:
                self.consecutive_blurry = 0
            
            # Run YOLO detection
            results = self.model(frame)
            
            # Process each detected plate
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Extract plate region
                    plate_img = frame[y1:y2, x1:x2]
                    
                    # Perform OCR if cooldown has passed
                    if self.should_read_plate():
                        plate_text = self.read_plate(plate_img)
                        if plate_text and len(plate_text) > 3:  # Minimum length check
                            if plate_text != self.last_plate:  # Only print if different from last read
                                print(f"License Plate Detected: {plate_text} (Confidence: {confidence:.2f})")
                                self.last_plate = plate_text
            
            # Draw results
            annotated_frame = results[0].plot()
            
            # Display frame
            cv2.imshow("License Plate Detection", annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                print("Manual refocus triggered...")
                self.initial_focus()
        
        # Cleanup
        cv2.destroyAllWindows()
        self.picam2.stop()

if __name__ == "__main__":
    reader = LicensePlateReader()
    reader.run()