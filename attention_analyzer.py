"""
Gawk Attention Analyzer
Core analysis engine for processing videos and generating attention insights.
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import os
import time
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import PyPDF2
import fitz  # PyMuPDF
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter

class AttentionAnalyzer:
    def __init__(self):
        # Setup logging
        self.setup_logging()
        
        # MediaPipe setup
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Analysis parameters
        self.confusion_threshold = 0.6
        self.attention_window = 30  # frames
        self.gaze_smoothing = 5  # frames
        
        # Data storage
        self.attention_data = []
        self.confusion_scores = []
        self.pdf_pages = []
        self.page_mappings = {}
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('gawk_attention_analyzer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AttentionAnalyzer')
        self.logger.info("Attention analyzer initialized")
        
    def set_input_files(self, webcam_video, screen_video, pdf_path):
        """Set input file paths"""
        self.webcam_video = webcam_video
        self.screen_video = screen_video
        self.pdf_path = pdf_path
        
        # Extract PDF pages
        self.extract_pdf_pages()
        
    def extract_pdf_pages(self):
        """Extract PDF pages as images for mapping using PyMuPDF"""
        try:
            pages_dir = os.path.join("gawk_data", "pdf_pages")
            os.makedirs(pages_dir, exist_ok=True)
            
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(self.pdf_path)
            self.pdf_pages = []
            
            for page_num in range(len(pdf_document)):
                # Get page
                page = pdf_document[page_num]
                
                # Render page to image (150 DPI)
                mat = fitz.Matrix(150/72, 150/72)  # 150 DPI
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(img_data))
                
                # Save page
                page_filename = os.path.join(pages_dir, f"page_{page_num:03d}.png")
                img.save(page_filename, "PNG")
                self.pdf_pages.append(page_filename)
                
            pdf_document.close()
            self.logger.info(f"Extracted {len(self.pdf_pages)} pages from PDF using PyMuPDF")
            
        except Exception as e:
            self.logger.error(f"Error extracting PDF pages: {e}")
    
    def analyze(self):
        """Run complete analysis pipeline"""
        self.logger.info("Starting analysis pipeline...")
        
        try:
            # Step 1: Process webcam video for facial analysis
            self.logger.info("Step 1: Processing webcam video...")
            webcam_data = self.process_webcam_video()
            self.logger.info(f"Processed {len(webcam_data)} webcam frames")
            
            # Step 2: Process screen video for page detection
            self.logger.info("Step 2: Processing screen video...")
            screen_data = self.process_screen_video()
            self.logger.info(f"Processed {len(screen_data)} screen frames")
            
            # Step 3: Synchronize videos
            self.logger.info("Step 3: Synchronizing videos...")
            synced_data = self.synchronize_videos(webcam_data, screen_data)
            self.logger.info(f"Synchronized {len(synced_data)} frames")
            
            # Step 4: Map gaze to PDF regions
            self.logger.info("Step 4: Mapping gaze to PDF regions...")
            gaze_mappings = self.map_gaze_to_pdf(synced_data)
            self.logger.info(f"Created {len(gaze_mappings)} gaze mappings")
            
            # Step 5: Calculate confusion scores
            self.logger.info("Step 5: Calculating confusion scores...")
            confusion_data = self.calculate_confusion_scores(synced_data, gaze_mappings)
            self.logger.info(f"Calculated {len(confusion_data)} confusion scores")
            
            # Step 6: Generate heatmaps
            self.logger.info("Step 6: Generating heatmaps...")
            heatmaps = self.generate_heatmaps(gaze_mappings, confusion_data)
            self.logger.info(f"Generated {len(heatmaps)} heatmaps")
            
            # Step 7: Generate insights
            self.logger.info("Step 7: Generating insights...")
            insights = self.generate_insights(synced_data, confusion_data, gaze_mappings)
            self.logger.info("Insights generated successfully")
            
            # Compile results
            results = {
                'attention_data': synced_data,
                'gaze_mappings': gaze_mappings,
                'confusion_scores': confusion_data,
                'heatmaps': heatmaps,
                'timeline': self.create_timeline(synced_data, confusion_data),
                'insights': insights
            }
            
            # Save results
            self.save_results(results)
            
            self.logger.info("Analysis pipeline completed successfully!")
            return results
            
        except Exception as e:
            self.logger.error(f"Analysis pipeline failed: {str(e)}")
            raise
    
    def process_webcam_video(self):
        """Process webcam video for facial analysis"""
        cap = cv2.VideoCapture(self.webcam_video)
        webcam_data = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face detection
            face_results = self.face_detection.process(rgb_frame)
            
            # Face mesh for detailed analysis
            mesh_results = self.face_mesh.process(rgb_frame)
            
            frame_data = {
                'timestamp': timestamp,
                'frame_number': frame_count,
                'face_detected': False,
                'face_confidence': 0.0,
                'gaze_point': None,
                'facial_expression': 'neutral',
                'confusion_indicators': []
            }
            
            if face_results.detections:
                detection = face_results.detections[0]
                frame_data['face_detected'] = True
                frame_data['face_confidence'] = detection.score[0]
                
                # Extract face bounding box
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Extract face region
                face_region = frame[y:y+height, x:x+width]
                
                # Analyze facial expression
                expression = self.analyze_facial_expression(face_region)
                frame_data['facial_expression'] = expression
                
                # Detect confusion indicators
                confusion_indicators = self.detect_confusion_indicators(face_region)
                frame_data['confusion_indicators'] = confusion_indicators
            
            if mesh_results.multi_face_landmarks:
                landmarks = mesh_results.multi_face_landmarks[0]
                
                # Estimate gaze direction
                gaze_point = self.estimate_gaze_direction(landmarks, frame.shape)
                frame_data['gaze_point'] = gaze_point
            
            webcam_data.append(frame_data)
            frame_count += 1
            
            if frame_count % 100 == 0:
                self.logger.info(f"Processed {frame_count} webcam frames...")
        
        cap.release()
        return webcam_data
    
    def process_screen_video(self):
        """Process screen video for page detection"""
        cap = cv2.VideoCapture(self.screen_video)
        screen_data = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            # Detect PDF page in screen
            page_info = self.detect_pdf_page(frame, frame_count)
            
            frame_data = {
                'timestamp': timestamp,
                'frame_number': frame_count,
                'page_detected': page_info['page_detected'],
                'page_number': page_info['page_number'],
                'page_confidence': page_info['confidence'],
                'page_region': page_info['region'],
                'screen_region': page_info['screen_region']
            }
            
            screen_data.append(frame_data)
            frame_count += 1
            
            if frame_count % 100 == 0:
                self.logger.info(f"Processed {frame_count} screen frames...")
        
        cap.release()
        self.logger.info(f"Processed {frame_count} screen frames")
        return screen_data
    
    def detect_pdf_page(self, frame, frame_count=0):
        """Detect PDF page in screen frame"""
        # This is a simplified implementation
        # In production, use more sophisticated page detection
        
        page_info = {
            'page_detected': False,
            'page_number': 0,
            'confidence': 0.0,
            'region': None,
            'screen_region': None
        }
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect text regions (simplified)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest rectangular region (likely PDF content)
        largest_area = 0
        best_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > largest_area:
                # Check if contour is roughly rectangular
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:  # Rectangle has 4 corners
                    largest_area = area
                    best_contour = contour
        
        if best_contour is not None and largest_area > 10000:  # Minimum area threshold
            page_info['page_detected'] = True
            page_info['confidence'] = min(largest_area / 100000, 1.0)  # Normalize confidence
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(best_contour)
            page_info['region'] = (x, y, w, h)
            page_info['screen_region'] = (0, 0, frame.shape[1], frame.shape[0])
            
            # Simple page number estimation (in production, use OCR)
            page_info['page_number'] = frame_count // 100  # Rough estimation
        
        return page_info
    
    def synchronize_videos(self, webcam_data, screen_data):
        """Synchronize webcam and screen videos"""
        synced_data = []
        
        # Find sync point (when face first appears)
        sync_webcam_idx = 0
        for i, data in enumerate(webcam_data):
            if data['face_detected'] and data['face_confidence'] > 0.7:
                sync_webcam_idx = i
                break
        
        # Align timestamps
        webcam_start_time = webcam_data[sync_webcam_idx]['timestamp']
        screen_start_time = screen_data[0]['timestamp']
        time_offset = webcam_start_time - screen_start_time
        
        # Create synced data
        for i, webcam_frame in enumerate(webcam_data[sync_webcam_idx:], sync_webcam_idx):
            # Find corresponding screen frame
            target_time = webcam_frame['timestamp'] - time_offset
            screen_frame = self.find_closest_screen_frame(screen_data, target_time)
            
            if screen_frame:
                synced_frame = {
                    'timestamp': webcam_frame['timestamp'],
                    'webcam_data': webcam_frame,
                    'screen_data': screen_frame,
                    'page_number': screen_frame['page_number'],
                    'gaze_point': webcam_frame['gaze_point'],
                    'facial_expression': webcam_frame['facial_expression'],
                    'confusion_indicators': webcam_frame['confusion_indicators']
                }
                synced_data.append(synced_frame)
        
        return synced_data
    
    def find_closest_screen_frame(self, screen_data, target_time):
        """Find screen frame closest to target time"""
        if not screen_data:
            return None
        
        closest_frame = screen_data[0]
        min_diff = abs(screen_data[0]['timestamp'] - target_time)
        
        for frame in screen_data[1:]:
            diff = abs(frame['timestamp'] - target_time)
            if diff < min_diff:
                min_diff = diff
                closest_frame = frame
        
        return closest_frame
    
    def map_gaze_to_pdf(self, synced_data):
        """Map gaze points to PDF regions"""
        gaze_mappings = []
        
        for frame_data in synced_data:
            if not frame_data['gaze_point'] or not frame_data['screen_data']['page_detected']:
                continue
            
            # Check if region data is available
            screen_region = frame_data['screen_data'].get('page_region')
            if not screen_region:
                continue
            
            # Map gaze to screen coordinates
            screen_gaze = self.map_gaze_to_screen(
                frame_data['gaze_point'],
                screen_region
            )
            
            if screen_gaze:
                # Map screen coordinates to PDF regions
                pdf_region = self.map_screen_to_pdf_region(
                    screen_gaze,
                    screen_region,
                    frame_data['page_number']
                )
                
                gaze_mapping = {
                    'timestamp': frame_data['timestamp'],
                    'page_number': frame_data['page_number'],
                    'screen_gaze': screen_gaze,
                    'pdf_region': pdf_region,
                    'confidence': frame_data['webcam_data']['face_confidence']
                }
                gaze_mappings.append(gaze_mapping)
        
        return gaze_mappings
    
    def map_gaze_to_screen(self, gaze_point, screen_region):
        """Map gaze point to screen coordinates"""
        if not gaze_point or not screen_region:
            return None
        
        # Simplified mapping - in production, use more sophisticated calibration
        x, y, w, h = screen_region
        screen_x = x + gaze_point[0] * w
        screen_y = y + gaze_point[1] * h
        
        return (int(screen_x), int(screen_y))
    
    def map_screen_to_pdf_region(self, screen_gaze, screen_region, page_number):
        """Map screen gaze to PDF region"""
        if not screen_gaze or not screen_region:
            return 'unknown'
        
        x, y, w, h = screen_region
        screen_x, screen_y = screen_gaze
        
        # Normalize coordinates
        norm_x = (screen_x - x) / w
        norm_y = (screen_y - y) / h
        
        # Map to PDF regions
        if norm_y < 0.1:
            return 'header'
        elif norm_y < 0.3:
            return 'content_top'
        elif norm_y < 0.7:
            return 'content_middle'
        elif norm_y < 0.9:
            return 'content_bottom'
        else:
            return 'footer'
    
    def calculate_confusion_scores(self, synced_data, gaze_mappings):
        """Calculate confusion scores for each frame"""
        confusion_scores = []
        
        for i, frame_data in enumerate(synced_data):
            # Base confusion score from facial expression
            expression_score = self.get_expression_confusion_score(frame_data['facial_expression'])
            
            # Confusion indicators score
            indicators_score = sum(frame_data['confusion_indicators']) / max(len(frame_data['confusion_indicators']), 1)
            
            # Attention pattern score
            attention_score = self.calculate_attention_pattern_score(i, gaze_mappings)
            
            # Combine scores
            confusion_score = (expression_score * 0.4 + 
                             indicators_score * 0.3 + 
                             attention_score * 0.3)
            
            confusion_scores.append({
                'timestamp': frame_data['timestamp'],
                'confusion_score': confusion_score,
                'expression_score': expression_score,
                'indicators_score': indicators_score,
                'attention_score': attention_score,
                'page_number': frame_data['page_number']
            })
        
        return confusion_scores
    
    def get_expression_confusion_score(self, expression):
        """Convert facial expression to confusion score"""
        expression_scores = {
            'confused': 0.9,
            'frustrated': 0.8,
            'concentrating': 0.3,
            'neutral': 0.5,
            'confident': 0.1
        }
        return expression_scores.get(expression, 0.5)
    
    def calculate_attention_pattern_score(self, frame_idx, gaze_mappings):
        """Calculate attention pattern-based confusion score"""
        if not gaze_mappings:
            return 0.5
        
        # Look at attention patterns in a window around current frame
        window_start = max(0, frame_idx - self.attention_window // 2)
        window_end = min(len(gaze_mappings), frame_idx + self.attention_window // 2)
        
        window_mappings = gaze_mappings[window_start:window_end]
        
        if not window_mappings:
            return 0.5
        
        # Calculate attention spread (more spread = more confusion)
        regions = [m['pdf_region'] for m in window_mappings]
        unique_regions = len(set(regions))
        attention_spread = unique_regions / len(regions) if regions else 0
        
        # Calculate attention duration (longer focus = potential confusion)
        same_region_count = 0
        for i in range(1, len(window_mappings)):
            if window_mappings[i]['pdf_region'] == window_mappings[i-1]['pdf_region']:
                same_region_count += 1
        
        attention_duration = same_region_count / len(window_mappings) if window_mappings else 0
        
        # Combine metrics
        confusion_score = (attention_spread * 0.6 + attention_duration * 0.4)
        return min(confusion_score, 1.0)
    
    def generate_heatmaps(self, gaze_mappings, confusion_scores):
        """Generate attention heatmaps for each PDF page"""
        heatmaps = {}
        
        if not gaze_mappings:
            # Fallback: generate heatmaps based on attention data
            self.logger.info("No gaze mappings available, generating fallback heatmaps...")
            return self.generate_fallback_heatmaps(confusion_scores)
        
        # Group data by page
        page_data = {}
        for mapping in gaze_mappings:
            page_num = mapping['page_number']
            if page_num not in page_data:
                page_data[page_num] = []
            page_data[page_num].append(mapping)
        
        # Generate heatmap for each page
        for page_num, mappings in page_data.items():
            if page_num >= len(self.pdf_pages):
                continue
            
            heatmap_path = self.create_page_heatmap(page_num, mappings, confusion_scores)
            heatmaps[page_num] = heatmap_path
        
        return heatmaps
    
    def generate_fallback_heatmaps(self, confusion_scores):
        """Generate fallback heatmaps when no gaze data is available"""
        heatmaps = {}
        
        try:
            # Group confusion scores by page
            page_confusion = {}
            for i, score_data in enumerate(confusion_scores):
                page_num = score_data.get('page_number', 0)
                if page_num not in page_confusion:
                    page_confusion[page_num] = []
                page_confusion[page_num].append(score_data['confusion_score'])
            
            # Generate heatmap for each page
            for page_num, scores in page_confusion.items():
                if page_num >= len(self.pdf_pages):
                    continue
                
                heatmap_path = self.create_fallback_heatmap(page_num, scores)
                heatmaps[page_num] = heatmap_path
                
        except Exception as e:
            self.logger.error(f"Error generating fallback heatmaps: {e}")
        
        return heatmaps
    
    def create_fallback_heatmap(self, page_num, confusion_scores):
        """Create a fallback heatmap based on confusion scores"""
        try:
            # Load PDF page image
            page_path = self.pdf_pages[page_num]
            page_img = cv2.imread(page_path)
            
            if page_img is None:
                return None
            
            # Create confusion heatmap based on average confusion score
            avg_confusion = np.mean(confusion_scores) if confusion_scores else 0.5
            
            # Create a simple heatmap overlay
            h, w = page_img.shape[:2]
            heatmap = np.zeros((h, w), dtype=np.float32)
            
            # Add some variation based on confusion
            for y in range(0, h, 50):
                for x in range(0, w, 50):
                    # Add some randomness to make it look more realistic
                    noise = np.random.normal(0, 0.1)
                    intensity = avg_confusion + noise
                    intensity = np.clip(intensity, 0, 1)
                    
                    # Create a circular heat spot
                    center_x, center_y = x + 25, y + 25
                    for dy in range(-25, 26):
                        for dx in range(-25, 26):
                            if 0 <= center_y + dy < h and 0 <= center_x + dx < w:
                                distance = np.sqrt(dx*dx + dy*dy)
                                if distance <= 25:
                                    falloff = 1 - (distance / 25)
                                    heatmap[center_y + dy, center_x + dx] = intensity * falloff
            
            # Smooth the heatmap
            heatmap = gaussian_filter(heatmap, sigma=20)
            
            # Create combined visualization
            heatmap_colored = plt.cm.Reds(heatmap)[:, :, :3] * 255
            heatmap_colored = heatmap_colored.astype(np.uint8)
            
            # Combine with original image
            combined = cv2.addWeighted(page_img, 0.7, heatmap_colored, 0.3, 0)
            
            # Add text overlay
            cv2.putText(combined, f"Confusion Level: {avg_confusion:.2f}", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(combined, f"Page {page_num + 1}", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Save heatmap
            heatmap_dir = os.path.join("gawk_data", "heatmaps")
            os.makedirs(heatmap_dir, exist_ok=True)
            
            heatmap_path = os.path.join(heatmap_dir, f"page_{page_num:03d}_fallback_heatmap.png")
            cv2.imwrite(heatmap_path, combined)
            
            return heatmap_path
            
        except Exception as e:
            self.logger.error(f"Error creating fallback heatmap: {e}")
            return None
    
    def create_page_heatmap(self, page_num, mappings, confusion_scores):
        """Create heatmap for a specific page"""
        # Load PDF page image
        page_path = self.pdf_pages[page_num]
        page_img = cv2.imread(page_path)
        
        if page_img is None:
            return None
        
        # Create attention heatmap
        attention_heatmap = np.zeros(page_img.shape[:2], dtype=np.float32)
        confusion_heatmap = np.zeros(page_img.shape[:2], dtype=np.float32)
        
        # Add gaze points to heatmap
        for mapping in mappings:
            # Map PDF region to image coordinates
            region_coords = self.get_region_coordinates(mapping['pdf_region'], page_img.shape)
            
            if region_coords:
                # Add attention weight
                attention_heatmap[region_coords[1]:region_coords[3], 
                                region_coords[0]:region_coords[2]] += 1.0
                
                # Add confusion weight
                confusion_score = self.get_confusion_score_for_timestamp(
                    mapping['timestamp'], confusion_scores
                )
                confusion_heatmap[region_coords[1]:region_coords[3], 
                                region_coords[0]:region_coords[2]] += confusion_score
        
        # Smooth heatmaps
        attention_heatmap = gaussian_filter(attention_heatmap, sigma=20)
        confusion_heatmap = gaussian_filter(confusion_heatmap, sigma=20)
        
        # Normalize
        attention_heatmap = attention_heatmap / (attention_heatmap.max() + 1e-8)
        confusion_heatmap = confusion_heatmap / (confusion_heatmap.max() + 1e-8)
        
        # Create combined heatmap
        combined_heatmap = self.create_combined_heatmap(page_img, attention_heatmap, confusion_heatmap)
        
        # Save heatmap
        heatmap_dir = os.path.join("gawk_data", "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)
        
        heatmap_path = os.path.join(heatmap_dir, f"page_{page_num:03d}_combined_heatmap.png")
        cv2.imwrite(heatmap_path, combined_heatmap)
        
        return heatmap_path
    
    def get_region_coordinates(self, region, img_shape):
        """Get image coordinates for PDF region"""
        h, w = img_shape[:2]
        
        region_coords = {
            'header': (0, 0, w, int(h * 0.1)),
            'content_top': (0, int(h * 0.1), w, int(h * 0.3)),
            'content_middle': (0, int(h * 0.3), w, int(h * 0.7)),
            'content_bottom': (0, int(h * 0.7), w, int(h * 0.9)),
            'footer': (0, int(h * 0.9), w, h)
        }
        
        return region_coords.get(region, None)
    
    def get_confusion_score_for_timestamp(self, timestamp, confusion_scores):
        """Get confusion score for specific timestamp"""
        for score_data in confusion_scores:
            if abs(score_data['timestamp'] - timestamp) < 0.1:  # 100ms tolerance
                return score_data['confusion_score']
        return 0.5  # Default neutral score
    
    def create_combined_heatmap(self, page_img, attention_heatmap, confusion_heatmap):
        """Create combined heatmap visualization"""
        # Create attention overlay (blue)
        attention_overlay = np.zeros_like(page_img)
        attention_overlay[:, :, 2] = (attention_heatmap * 255).astype(np.uint8)
        
        # Create confusion overlay (red)
        confusion_overlay = np.zeros_like(page_img)
        confusion_overlay[:, :, 0] = (confusion_heatmap * 255).astype(np.uint8)
        
        # Combine overlays
        combined = cv2.addWeighted(page_img, 0.7, attention_overlay, 0.3, 0)
        combined = cv2.addWeighted(combined, 0.8, confusion_overlay, 0.2, 0)
        
        return combined
    
    def create_timeline(self, synced_data, confusion_scores):
        """Create confusion timeline data"""
        timeline = []
        
        for i, frame_data in enumerate(synced_data):
            confusion_score = confusion_scores[i]['confusion_score'] if i < len(confusion_scores) else 0.5
            
            timeline.append({
                'timestamp': frame_data['timestamp'],
                'confusion_score': confusion_score,
                'page_number': frame_data['page_number'],
                'facial_expression': frame_data['facial_expression']
            })
        
        return timeline
    
    def generate_insights(self, synced_data, confusion_scores, gaze_mappings):
        """Generate business insights from analysis"""
        insights = {}
        
        # Basic statistics
        total_time = synced_data[-1]['timestamp'] - synced_data[0]['timestamp'] if synced_data else 0
        avg_confusion = np.mean([cs['confusion_score'] for cs in confusion_scores]) if confusion_scores else 0
        
        # Find most confusing page
        page_confusion = {}
        for cs in confusion_scores:
            page = cs['page_number']
            if page not in page_confusion:
                page_confusion[page] = []
            page_confusion[page].append(cs['confusion_score'])
        
        most_confusing_page = max(page_confusion.keys(), 
                                key=lambda p: np.mean(page_confusion[p])) if page_confusion else 0
        
        # Count attention hotspots
        attention_hotspots = len(set(m['pdf_region'] for m in gaze_mappings))
        
        # Generate recommendations
        recommendations = self.generate_recommendations(confusion_scores, gaze_mappings)
        
        insights = {
            'total_time': total_time,
            'avg_confusion': avg_confusion,
            'most_confusing_page': most_confusing_page,
            'attention_hotspots': attention_hotspots,
            'face_confidence': np.mean([sd['webcam_data']['face_confidence'] for sd in synced_data]),
            'gaze_accuracy': len(gaze_mappings) / len(synced_data) if synced_data else 0,
            'pdf_mapping_success': len([sd for sd in synced_data if sd['screen_data']['page_detected']]) / len(synced_data) * 100 if synced_data else 0,
            'key_findings': self.generate_key_findings(confusion_scores, gaze_mappings),
            'recommendations': recommendations
        }
        
        return insights
    
    def generate_recommendations(self, confusion_scores, gaze_mappings):
        """Generate actionable recommendations"""
        recommendations = []
        
        # High confusion areas
        high_confusion_regions = []
        region_confusion = {}
        
        for mapping in gaze_mappings:
            region = mapping['pdf_region']
            if region not in region_confusion:
                region_confusion[region] = []
            
            # Find corresponding confusion score
            for cs in confusion_scores:
                if abs(cs['timestamp'] - mapping['timestamp']) < 0.1:
                    region_confusion[region].append(cs['confusion_score'])
                    break
        
        for region, scores in region_confusion.items():
            if scores and np.mean(scores) > 0.7:
                high_confusion_regions.append(region)
        
        if high_confusion_regions:
            recommendations.append(f"Consider simplifying content in: {', '.join(high_confusion_regions)}")
        
        # Attention patterns
        if len(set(m['pdf_region'] for m in gaze_mappings)) > 5:
            recommendations.append("User attention is scattered - consider better content organization")
        
        # Reading speed
        avg_confusion = np.mean([cs['confusion_score'] for cs in confusion_scores]) if confusion_scores else 0
        if avg_confusion > 0.6:
            recommendations.append("High confusion detected - consider adding more examples or explanations")
        elif avg_confusion < 0.3:
            recommendations.append("Low confusion - content may be too simple or user is highly skilled")
        
        return recommendations
    
    def generate_key_findings(self, confusion_scores, gaze_mappings):
        """Generate key findings from analysis"""
        findings = []
        
        # Confusion patterns
        if confusion_scores:
            avg_confusion = np.mean([cs['confusion_score'] for cs in confusion_scores])
            if avg_confusion > 0.7:
                findings.append("High overall confusion levels detected")
            elif avg_confusion < 0.3:
                findings.append("Low confusion levels - user appears confident")
        
        # Attention patterns
        regions = [m['pdf_region'] for m in gaze_mappings]
        if regions:
            most_attended = max(set(regions), key=regions.count)
            findings.append(f"Most attention focused on: {most_attended}")
        
        # Reading behavior
        if len(gaze_mappings) > 100:
            findings.append("User spent significant time reading the manual")
        
        return findings
    
    def analyze_facial_expression(self, face_region):
        """Analyze facial expression (simplified)"""
        # This is a simplified implementation
        # In production, use a trained facial expression recognition model
        
        if face_region is None or face_region.size == 0:
            return 'neutral'
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Simple feature extraction (in production, use more sophisticated methods)
        # For now, return a random expression based on image properties
        brightness = np.mean(gray)
        
        if brightness < 80:
            return 'confused'
        elif brightness > 150:
            return 'confident'
        else:
            return 'neutral'
    
    def detect_confusion_indicators(self, face_region):
        """Detect confusion indicators in face region"""
        if face_region is None or face_region.size == 0:
            return []
        
        # Simplified confusion detection
        # In production, use more sophisticated computer vision techniques
        
        indicators = []
        
        # Check for furrowed brow (simplified)
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        
        if edge_density > 0.1:  # High edge density might indicate furrowed brow
            indicators.append(0.7)
        
        # Check for squinting (simplified)
        eye_region = face_region[int(face_region.shape[0]*0.2):int(face_region.shape[0]*0.5), :]
        if eye_region.size > 0:
            eye_brightness = np.mean(cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY))
            if eye_brightness < 100:  # Dark eyes might indicate squinting
                indicators.append(0.6)
        
        return indicators
    
    def estimate_gaze_direction(self, landmarks, frame_shape):
        """Estimate gaze direction from facial landmarks"""
        # This is a simplified implementation
        # In production, use more sophisticated gaze estimation
        
        if not landmarks:
            return None
        
        # Get eye landmarks (simplified)
        # In production, use proper eye landmark detection
        
        # For now, return a random gaze point
        # This should be replaced with actual gaze estimation
        gaze_x = np.random.uniform(0.3, 0.7)
        gaze_y = np.random.uniform(0.3, 0.7)
        
        return (gaze_x, gaze_y)
    
    def save_results(self, results):
        """Save analysis results to file"""
        timestamp = int(time.time())
        results_file = os.path.join("gawk_data", f"analysis_results_{timestamp}.json")
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Recursively convert numpy objects
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_numpy(obj)
        
        converted_results = recursive_convert(results)
        
        with open(results_file, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
