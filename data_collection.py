"""
Gawk Data Collection Application
Records webcam video, screen recording, and handles PDF input for attention analysis.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import time
import os
import json
import logging
from datetime import datetime
import PyPDF2
import fitz  # PyMuPDF
import mediapipe as mp
import mss
import pyautogui
from PIL import Image
import io

class GawkDataCollector:
    def __init__(self):
        # Setup logging
        self.setup_logging()
        
        self.root = tk.Tk()
        self.root.title("Gawk - Data Collection")
        self.root.geometry("600x500")
        
        # Recording state
        self.is_recording = False
        self.webcam_cap = None
        self.webcam_writer = None
        self.screen_writer = None
        self.start_time = None
        self.screen_monitor = None
        
        # File paths
        self.pdf_path = None
        self.output_dir = "gawk_data"
        self.videos_dir = os.path.join(self.output_dir, "videos")
        self.pdfs_dir = os.path.join(self.output_dir, "pdfs")
        self.metadata_dir = os.path.join(self.output_dir, "metadata")
        
        # Create directories
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.pdfs_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # MediaPipe setup for face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        
        self.setup_ui()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('gawk_data_collection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('GawkDataCollector')
        self.logger.info("Data collection application initialized")
        
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_label = tk.Label(self.root, text="Gawk Data Collection", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # PDF Selection
        pdf_frame = tk.Frame(self.root)
        pdf_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(pdf_frame, text="PDF Manual:", font=("Arial", 12)).pack(anchor="w")
        self.pdf_label = tk.Label(pdf_frame, text="No PDF selected", 
                                 fg="gray", wraplength=500)
        self.pdf_label.pack(anchor="w", pady=5)
        
        tk.Button(pdf_frame, text="Select PDF", 
                 command=self.select_pdf).pack(anchor="w")
        
        # Recording Controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=20)
        
        self.record_button = tk.Button(control_frame, text="Start Recording", 
                                     command=self.toggle_recording,
                                     font=("Arial", 14), bg="green", fg="white",
                                     width=15, height=2)
        self.record_button.pack(side="left", padx=10)
        
        self.status_label = tk.Label(control_frame, text="Ready to record", 
                                   font=("Arial", 12))
        self.status_label.pack(side="left", padx=10)
        
        # Status display
        status_display = tk.LabelFrame(self.root, text="Recording Status", font=("Arial", 12, "bold"))
        status_display.pack(pady=10, padx=20, fill="x")
        
        # Recording indicators
        self.webcam_status = tk.Label(status_display, text="Webcam: Ready", font=("Arial", 10), fg="green")
        self.webcam_status.pack(pady=5)
        
        self.screen_status = tk.Label(status_display, text="Screen: Ready", font=("Arial", 10), fg="green")
        self.screen_status.pack(pady=5)
        
        self.face_status = tk.Label(status_display, text="Face Detection: Ready", font=("Arial", 10), fg="green")
        self.face_status.pack(pady=5)
        
        # Instructions
        instructions = tk.Text(self.root, height=8, width=70, wrap="word")
        instructions.pack(pady=20, padx=20)
        
        instructions_text = """INSTRUCTIONS:
1. Select your PDF manual using the 'Select PDF' button
2. Position yourself in front of the webcam
3. Open the PDF in your preferred PDF viewer
4. Click 'Start Recording' to begin data collection
5. Read through the manual naturally
6. Click 'Stop Recording' when finished

STATUS INDICATORS:
- Webcam: Shows if webcam is recording and face detection status
- Screen: Shows if screen recording is working
- Face Detection: Shows if your face is being detected
- Recording Info: Shows elapsed time, frame count, and FPS

SYNC TIPS:
- Start both recordings at the same time
- Make a clear gesture (wave hand) at the start for sync reference
- Keep the same screen resolution throughout recording
- Ensure good lighting for face detection

The system will automatically:
- Record your webcam feed with face detection
- Record your screen activity
- Extract PDF pages for analysis
- Save all data with timestamps for synchronization"""
        
        instructions.insert("1.0", instructions_text)
        instructions.config(state="disabled")
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(pady=10, padx=20, fill="x")
        
        # Recording info frame
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(pady=5, padx=20, fill="x")
        
        self.recording_info = tk.Label(self.info_frame, text="", font=("Arial", 10), fg="blue")
        self.recording_info.pack()
        
    def select_pdf(self):
        """Select PDF file for analysis"""
        try:
            self.logger.info("Starting PDF selection process")
            file_path = filedialog.askopenfilename(
                title="Select PDF Manual",
                filetypes=[("PDF files", "*.pdf")]
            )
            
            if file_path:
                self.logger.info(f"PDF selected: {file_path}")
                self.pdf_path = file_path
                self.pdf_label.config(text=os.path.basename(file_path), fg="black")
                
                # Copy PDF to data directory
                try:
                    import shutil
                    pdf_filename = f"manual_{int(time.time())}.pdf"
                    dest_path = os.path.join(self.pdfs_dir, pdf_filename)
                    shutil.copy2(file_path, dest_path)
                    self.pdf_path = dest_path
                    self.logger.info(f"PDF copied successfully to: {dest_path}")
                    messagebox.showinfo("Success", f"PDF copied to: {dest_path}")
                except Exception as e:
                    self.logger.error(f"Failed to copy PDF: {str(e)}")
                    messagebox.showerror("Error", f"Failed to copy PDF: {str(e)}")
            else:
                self.logger.info("No PDF file selected")
        except Exception as e:
            self.logger.error(f"Error in PDF selection: {str(e)}")
            messagebox.showerror("Error", f"PDF selection failed: {str(e)}")
    
    def toggle_recording(self):
        """Toggle recording state"""
        if not self.is_recording:
            if not self.pdf_path:
                messagebox.showerror("Error", "Please select a PDF first!")
                return
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording webcam and screen"""
        try:
            self.logger.info("Starting recording process")
            
            # Initialize webcam
            self.logger.info("Initializing webcam...")
            self.webcam_cap = cv2.VideoCapture(0)
            if not self.webcam_cap.isOpened():
                raise Exception("Could not open webcam")
            
            # Get webcam properties
            webcam_width = int(self.webcam_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            webcam_height = int(self.webcam_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            webcam_fps = int(self.webcam_cap.get(cv2.CAP_PROP_FPS))
            self.logger.info(f"Webcam initialized: {webcam_width}x{webcam_height} @ {webcam_fps}fps")
            
            # Initialize screen capture using mss (get monitor info only)
            self.logger.info("Initializing screen capture...")
            with mss.mss() as temp_capture:
                self.screen_monitor = temp_capture.monitors[1]  # Primary monitor
            self.logger.info(f"Screen capture initialized: {self.screen_monitor['width']}x{self.screen_monitor['height']}")
            
            # Setup video writers
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            webcam_filename = os.path.join(self.videos_dir, f"webcam_{timestamp}.avi")
            screen_filename = os.path.join(self.videos_dir, f"screen_{timestamp}.avi")
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.webcam_writer = cv2.VideoWriter(webcam_filename, fourcc, webcam_fps, 
                                               (webcam_width, webcam_height))
            
            # Setup screen recording
            screen_width = self.screen_monitor['width']
            screen_height = self.screen_monitor['height']
            self.screen_writer = cv2.VideoWriter(screen_filename, fourcc, webcam_fps,
                                               (screen_width, screen_height))
            
            self.logger.info(f"Video writers created: webcam={webcam_filename}, screen={screen_filename}")
            
            self.is_recording = True
            self.start_time = time.time()
            
            # Update UI
            self.record_button.config(text="Stop Recording", bg="red")
            self.status_label.config(text="Recording...")
            self.progress.start()
            
            # Start recording thread
            self.recording_thread = threading.Thread(target=self.record_videos)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            self.logger.info("Recording started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start recording: {str(e)}")
            messagebox.showerror("Error", f"Failed to start recording: {str(e)}")
    
    def record_videos(self):
        """Record webcam and screen videos"""
        try:
            self.logger.info("Recording thread started")
            frame_count = 0
            
            # Create a new MSS instance in the thread
            with mss.mss() as screen_capture:
                monitor = screen_capture.monitors[1]  # Primary monitor
                
                while self.is_recording:
                    # Capture webcam frame
                    ret, webcam_frame = self.webcam_cap.read()
                    if ret:
                        # Detect face for sync reference
                        rgb_frame = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
                        results = self.face_detection.process(rgb_frame)
                        
                        # Draw face detection box
                        if results.detections:
                            for detection in results.detections:
                                bbox = detection.location_data.relative_bounding_box
                                h, w, _ = webcam_frame.shape
                                x = int(bbox.xmin * w)
                                y = int(bbox.ymin * h)
                                width = int(bbox.width * w)
                                height = int(bbox.height * h)
                                cv2.rectangle(webcam_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                        
                        # Add timestamp
                        timestamp = time.time() - self.start_time
                        cv2.putText(webcam_frame, f"Time: {timestamp:.2f}s", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        self.webcam_writer.write(webcam_frame)
                        
                        # Update webcam status
                        if frame_count % 30 == 0:  # Update every second
                            self.root.after(0, self.update_webcam_status, True, len(results.detections) if results.detections else 0)
                    
                    # Capture screen
                    try:
                        screen_shot = screen_capture.grab(monitor)
                        screen_frame = np.array(screen_shot)
                        screen_frame = cv2.cvtColor(screen_frame, cv2.COLOR_BGRA2BGR)
                        
                        # Add timestamp overlay
                        cv2.putText(screen_frame, f"Screen Recording - {timestamp:.2f}s", 
                                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        self.screen_writer.write(screen_frame)
                        
                        # Update screen status
                        if frame_count % 30 == 0:  # Update every second
                            self.root.after(0, self.update_screen_status, True)
                        
                    except Exception as screen_error:
                        self.logger.warning(f"Screen capture error: {screen_error}")
                        # Create a placeholder frame if screen capture fails
                        screen_frame = np.zeros((self.screen_monitor['height'], self.screen_monitor['width'], 3), dtype=np.uint8)
                        cv2.putText(screen_frame, f"Screen Capture Error - {timestamp:.2f}s", 
                                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        self.screen_writer.write(screen_frame)
                        
                        # Update screen status with error
                        if frame_count % 30 == 0:  # Update every second
                            self.root.after(0, self.update_screen_status, False)
                    
                    time.sleep(1/30)  # 30 FPS
                    frame_count += 1
                    
                    if frame_count % 30 == 0:  # Update info every second at 30fps
                        elapsed_time = time.time() - self.start_time
                        self.root.after(0, self.update_recording_info, frame_count, elapsed_time)
                    
                    if frame_count % 300 == 0:  # Log every 10 seconds at 30fps
                        self.logger.info(f"Recorded {frame_count} frames")
                
        except Exception as e:
            self.logger.error(f"Recording error: {e}")
        finally:
            self.logger.info(f"Recording thread ended. Total frames: {frame_count}")
    
    def stop_recording(self):
        """Stop recording and save metadata"""
        try:
            self.logger.info("Stopping recording...")
            self.is_recording = False
            
            # Wait for recording thread to finish
            if hasattr(self, 'recording_thread'):
                self.recording_thread.join(timeout=2)
                self.logger.info("Recording thread stopped")
            
            # Release resources
            if self.webcam_cap:
                self.webcam_cap.release()
                self.logger.info("Webcam released")
            if self.webcam_writer:
                self.webcam_writer.release()
                self.logger.info("Webcam writer released")
            if self.screen_writer:
                self.screen_writer.release()
                self.logger.info("Screen writer released")
            
            # Save metadata
            self.save_metadata()
            
            # Update UI
            self.record_button.config(text="Start Recording", bg="green")
            self.status_label.config(text="Recording saved!")
            self.progress.stop()
            self.recording_info.config(text="")
            
            # Reset status indicators
            self.webcam_status.config(text="Webcam: Ready", fg="green")
            self.screen_status.config(text="Screen: Ready", fg="green")
            self.face_status.config(text="Face Detection: Ready", fg="green")
            
            self.logger.info("Recording stopped successfully")
            messagebox.showinfo("Success", "Recording completed and saved!")
            
        except Exception as e:
            self.logger.error(f"Error stopping recording: {str(e)}")
            messagebox.showerror("Error", f"Error stopping recording: {str(e)}")
    
    def save_metadata(self):
        """Save recording metadata"""
        try:
            self.logger.info("Saving metadata...")
            
            # Extract PDF information
            pdf_info = self.extract_pdf_info()
            
            # Recording metadata
            recording_info = {
                "start_time": self.start_time,
                "end_time": time.time(),
                "duration": time.time() - self.start_time,
                "webcam_file": os.path.join(self.videos_dir, f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"),
                "screen_file": os.path.join(self.videos_dir, f"screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"),
                "pdf_file": self.pdf_path,
                "webcam_resolution": [640, 480],  # Default
                "screen_resolution": [1920, 1080]  # Default
            }
            
            # Save metadata files
            recording_file = os.path.join(self.metadata_dir, "recording_info.json")
            with open(recording_file, "w") as f:
                json.dump(recording_info, f, indent=2)
            self.logger.info(f"Recording metadata saved to: {recording_file}")
            
            pdf_file = os.path.join(self.metadata_dir, "pdf_info.json")
            with open(pdf_file, "w") as f:
                json.dump(pdf_info, f, indent=2)
            self.logger.info(f"PDF metadata saved to: {pdf_file}")
            
            # Extract PDF pages as images
            self.extract_pdf_pages()
            
            self.logger.info("Metadata saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
    
    def extract_pdf_info(self):
        """Extract information from PDF"""
        try:
            self.logger.info(f"Extracting PDF info from: {self.pdf_path}")
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                info = {
                    "num_pages": len(pdf_reader.pages),
                    "title": pdf_reader.metadata.get('/Title', 'Unknown') if pdf_reader.metadata else 'Unknown',
                    "author": pdf_reader.metadata.get('/Author', 'Unknown') if pdf_reader.metadata else 'Unknown',
                    "file_path": self.pdf_path,
                    "original_path": self.pdf_path
                }
                self.logger.info(f"PDF info extracted: {info['num_pages']} pages, title: {info['title']}")
                return info
        except Exception as e:
            self.logger.error(f"Error extracting PDF info: {e}")
            return {"num_pages": 0, "title": "Unknown", "author": "Unknown", 
                   "file_path": self.pdf_path, "original_path": self.pdf_path}
    
    def extract_pdf_pages(self):
        """Extract PDF pages as images using PyMuPDF"""
        try:
            self.logger.info("Extracting PDF pages as images using PyMuPDF...")
            pages_dir = os.path.join(self.output_dir, "pdf_pages")
            os.makedirs(pages_dir, exist_ok=True)
            
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(self.pdf_path)
            self.logger.info(f"PDF opened successfully: {len(pdf_document)} pages")
            
            for page_num in range(len(pdf_document)):
                # Get page
                page = pdf_document[page_num]
                
                # Render page to image (150 DPI)
                mat = fitz.Matrix(150/72, 150/72)  # 150 DPI
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Save page
                page_filename = os.path.join(pages_dir, f"page_{page_num:03d}.png")
                img.save(page_filename, "PNG")
                self.logger.info(f"Saved page {page_num+1}: {page_filename}")
            
            pdf_document.close()
            self.logger.info(f"Successfully extracted {len(pdf_document)} pages from PDF")
            
        except Exception as e:
            self.logger.error(f"Error extracting PDF pages: {e}")
            # Fallback to placeholder pages
            self.logger.info("Creating placeholder pages as fallback...")
            self.create_placeholder_pages()
    
    def create_placeholder_pages(self):
        """Create placeholder pages as fallback"""
        try:
            pages_dir = os.path.join(self.output_dir, "pdf_pages")
            os.makedirs(pages_dir, exist_ok=True)
            
            # Get page count from PyPDF2
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for i in range(num_pages):
                    # Create a placeholder image
                    placeholder = np.ones((800, 600, 3), dtype=np.uint8) * 255  # White background
                    cv2.putText(placeholder, f"Page {i+1}", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
                    cv2.putText(placeholder, "PDF Page Placeholder", (50, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
                    cv2.putText(placeholder, "PyMuPDF extraction failed", (50, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                    
                    page_filename = os.path.join(pages_dir, f"page_{i:03d}.png")
                    cv2.imwrite(page_filename, placeholder)
                    self.logger.info(f"Created placeholder page {i+1}: {page_filename}")
        except Exception as e:
            self.logger.error(f"Error creating placeholder pages: {e}")
    
    def update_webcam_status(self, working, face_count=0):
        """Update webcam status display"""
        try:
            if working:
                if face_count > 0:
                    self.webcam_status.config(text=f"Webcam: Recording (Face detected)", fg="green")
                    self.face_status.config(text=f"Face Detection: {face_count} face(s) found", fg="green")
                else:
                    self.webcam_status.config(text="Webcam: Recording (No face)", fg="orange")
                    self.face_status.config(text="Face Detection: No face detected", fg="orange")
            else:
                self.webcam_status.config(text="Webcam: Error", fg="red")
        except Exception as e:
            self.logger.warning(f"Error updating webcam status: {e}")
    
    def update_screen_status(self, working):
        """Update screen status display"""
        try:
            if working:
                self.screen_status.config(text="Screen: Recording", fg="green")
            else:
                self.screen_status.config(text="Screen: Error", fg="red")
        except Exception as e:
            self.logger.warning(f"Error updating screen status: {e}")
    
    def update_recording_info(self, frame_count, elapsed_time):
        """Update recording information display"""
        try:
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            info_text = f"Recording: {minutes:02d}:{seconds:02d} | Frames: {frame_count} | FPS: {fps:.1f}"
            self.recording_info.config(text=info_text)
        except Exception as e:
            self.logger.warning(f"Error updating recording info: {e}")
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = GawkDataCollector()
    app.run()
