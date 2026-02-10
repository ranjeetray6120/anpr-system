import cv2
import re
import os
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR
from difflib import SequenceMatcher
from .base import BaseTrafficService

class ANPRService(BaseTrafficService):
    def __init__(self, base_model="yolo11n.pt", anpr_model="anpr_plat.pt"):
        super().__init__(model_path=base_model)
        self.anpr_model = YOLO(anpr_model).to(self.device)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.seen_plates = set()
        self.blacklist = self._load_blacklist()

    def _load_blacklist(self):
        # Look in multiple locations relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(base_dir) # Go up one level to ANPR-System
        
        paths = [
            os.path.join(root_dir, "blacklist.txt"),
            os.path.join(base_dir, "blacklist.txt"),
            r"d:\Python\Smart-Traffic-Analytics\ANPR-System\blacklist.txt"
        ]
        loaded = set()
        for path in paths:
            if os.path.exists(path):
                try:
                    # Use utf-8-sig to handle BOM if present
                    with open(path, "r", encoding='utf-8-sig') as f:
                        for line in f:
                            # Clean each line
                            clean = re.sub(r'[^A-Z0-9]', '', line.upper())
                            if clean:
                                loaded.add(clean)
                    self.log_debug(f"Loaded blacklist from {path}: {len(loaded)} entries")
                except Exception as e:
                    self.log_debug(f"Error loading {path}: {e}")
        
        final_list = list(loaded)
        self.log_debug(f"Final Blacklist ({len(final_list)}): {sorted(final_list)}")
        return final_list

    def clean_plate_text(self, text):
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def validate_indian_plate(self, text):
        std_pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{4}$'
        bh_pattern = r'^[0-9]{2}BH[0-9]{1,4}[A-Z]{1,2}[0-9]{1,4}$'
        if len(text) < 8 or len(text) > 12: return False
        return bool(re.match(std_pattern, text) or re.match(bh_pattern, text))

    def is_similar(self, text, threshold=0.8):
        for seen_text in self.seen_plates:
            if SequenceMatcher(None, text, seen_text).ratio() > threshold:
                return True, seen_text
        return False, None

    def process_plate(self, plate_img):
        if plate_img.size == 0: return ""
        
        # Advanced Preprocessing (Cleaned up based on anpr_final.py)
        plate_processed = cv2.resize(plate_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(plate_processed, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        plate_input = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        h, w = plate_img.shape[:2]
        aspect_ratio = w / h
        detected_text = ""

        # Logic from anpr_final.py
        if aspect_ratio < 2.5: # Potential 2-row plate
            half_h = plate_input.shape[0] // 2
            top_half = plate_input[0:half_h, :]
            bottom_half = plate_input[half_h:, :]

            res_top = self.ocr.ocr(top_half, cls=True)
            res_bottom = self.ocr.ocr(bottom_half, cls=True)
            
            text_top = " ".join([l[1][0] for l in res_top[0]]) if res_top and res_top[0] else ""
            text_bottom = " ".join([l[1][0] for l in res_bottom[0]]) if res_bottom and res_bottom[0] else ""
            
            raw_combined = text_top + text_bottom
            clean_combined = self.clean_plate_text(raw_combined)
            
            if self.validate_indian_plate(clean_combined):
                detected_text = raw_combined
            else:
                 # Fallback to full image if split failed
                res_full = self.ocr.ocr(plate_input, cls=True)
                if res_full and res_full[0]:
                    detected_text = " ".join([l[1][0] for l in res_full[0]])
        else:
            res_full = self.ocr.ocr(plate_input, cls=True)
            if res_full and res_full[0]:
                detected_text = " ".join([l[1][0] for l in res_full[0]])
        
        return detected_text

    def run_detection(self, frame, frame_count):
        results = self.anpr_model(frame, verbose=False)[0]
        detected_plates = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            plate_img = frame[y1:y2, x1:x2]
            
    def extract_indian_plate(self, text):
        """
        Attempts to extract a valid Indian License Plate number from a noisy string.
        Handles cases like 'KA05MU7549IND', 'KA05NG6006NO', etc.
        """
        # Standard Pattern: 2 chars (State), 1-2 digits (District), 0-3 chars (Series), 4 digits (Number)
        # Relaxed slightly for extraction
        std_pattern = r'[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}'
        # BH Series
        bh_pattern = r'[0-9]{2}BH[0-9]{1,4}[A-Z]{1,2}[0-9]{1,4}'
        
        # Search for patterns
        std_match = re.search(std_pattern, text)
        if std_match:
            return std_match.group(0)
            
        bh_match = re.search(bh_pattern, text)
        if bh_match:
            return bh_match.group(0)
            
        return None

    def log_debug(self, message):
        with open("anpr_debug.log", "a") as f:
            f.write(f"{datetime.now()} - {message}\n")

    def run_detection(self, frame, frame_count):
        results = self.anpr_model(frame, verbose=False)[0]
        detected_plates = []
        
        self.log_debug(f"Frame {frame_count}: Detected {len(results.boxes)} potential plates.")

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            plate_img = frame[y1:y2, x1:x2]
            
            if plate_img.size > 0:
                raw_text = self.process_plate(plate_img)
                clean_text = self.clean_plate_text(raw_text)
                
                self.log_debug(f"Frame {frame_count}: Box [{x1},{y1},{x2},{y2}] Conf {conf:.2f} Raw '{raw_text}' Clean '{clean_text}'")

                # Try to extract valid plate from clean_text if it's too long or noisy
                extracted = self.extract_indian_plate(clean_text)
                if extracted:
                    self.log_debug(f"  -> Extracted '{extracted}' from '{clean_text}'")
                    clean_text = extracted

                is_new = False
                
                if clean_text:
                    # Check for "Seen" status
                    if clean_text in self.seen_plates:
                        is_new = False
                        self.log_debug(f"  -> Exact duplicate: {clean_text}")
                    else:
                        # Check similarity
                        similar, original = self.is_similar(clean_text)
                        if similar:
                            is_new = False
                            clean_text = original # Normalize to the original seen text
                            self.log_debug(f"  -> Fuzzy duplicate of {original}")
                        else:
                            is_new = True

                    is_valid = self.validate_indian_plate(clean_text)
                    self.log_debug(f"  -> Valid: {is_valid}")
                    
                    if is_valid:
                        if is_new:
                            self.seen_plates.add(clean_text)
                            self.log_debug(f"Frame {frame_count}: NEW PLATE ACCEPTED: {clean_text} (Conf: {conf:.2f})")
                        
                        # Check Blacklist
                        alert = clean_text in self.blacklist
                        if alert:
                            self.log_debug(f"Frame {frame_count}: BLACKLIST ALERT: {clean_text}")
                        else:
                            self.log_debug(f"  -> Blacklist Check: {clean_text} NOT found in {len(self.blacklist)} entries")

                        detected_plates.append({
                            "frame": int(frame_count),
                            "text": str(clean_text),
                            "conf": float(conf),
                            "valid": "Yes",
                            "box": [int(x1), int(y1), int(x2), int(y2)],
                            "is_new": is_new,
                            "alert": alert,
                            "alert_type": "SECURITY ALERT" if alert else None
                        })
                    elif len(clean_text) > 4:
                        self.log_debug(f"Frame {frame_count}: REJECTED: {clean_text} (Valid: False)")
                        pass
                else:
                    self.log_debug(f"Frame {frame_count}: No text extracted.")
                    pass
        return detected_plates
