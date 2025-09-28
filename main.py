"""
Gawk Main Analysis Application
Analyzes recorded videos and PDF to generate attention heatmaps and confusion scores.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import os
import time
import logging
import threading
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
import mediapipe as mp
from PIL import Image, ImageTk
from attention_analyzer import AttentionAnalyzer

class GawkAnalyzer:
    def __init__(self):
        # Setup logging
        self.setup_logging()
        
        self.root = tk.Tk()
        self.root.title("Gawk - Analysis Dashboard")
        self.root.geometry("1200x800")
        
        # Analysis components
        self.analyzer = AttentionAnalyzer()
        self.analysis_results = None
        
        # File paths
        self.webcam_video = None
        self.screen_video = None
        self.pdf_path = None
        self.output_dir = "gawk_data"
        
        self.setup_ui()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('gawk_analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('GawkAnalyzer')
        self.logger.info("Analysis application initialized")
        
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_label = tk.Label(self.root, text="Gawk Analysis Dashboard", 
                              font=("Arial", 18, "bold"))
        title_label.pack(pady=10)
        
        # File selection frame
        file_frame = tk.LabelFrame(self.root, text="Input Files", font=("Arial", 12, "bold"))
        file_frame.pack(pady=10, padx=20, fill="x")
        
        # Webcam video
        webcam_frame = tk.Frame(file_frame)
        webcam_frame.pack(pady=5, fill="x")
        tk.Label(webcam_frame, text="Webcam Video:").pack(side="left")
        self.webcam_label = tk.Label(webcam_frame, text="No file selected", fg="gray")
        self.webcam_label.pack(side="left", padx=10)
        tk.Button(webcam_frame, text="Browse", 
                 command=lambda: self.select_file("webcam")).pack(side="right")
        
        # Screen video
        screen_frame = tk.Frame(file_frame)
        screen_frame.pack(pady=5, fill="x")
        tk.Label(screen_frame, text="Screen Video:").pack(side="left")
        self.screen_label = tk.Label(screen_frame, text="No file selected", fg="gray")
        self.screen_label.pack(side="left", padx=10)
        tk.Button(screen_frame, text="Browse", 
                 command=lambda: self.select_file("screen")).pack(side="right")
        
        # PDF file
        pdf_frame = tk.Frame(file_frame)
        pdf_frame.pack(pady=5, fill="x")
        tk.Label(pdf_frame, text="PDF Manual:").pack(side="left")
        self.pdf_label = tk.Label(pdf_frame, text="No file selected", fg="gray")
        self.pdf_label.pack(side="left", padx=10)
        tk.Button(pdf_frame, text="Browse", 
                 command=lambda: self.select_file("pdf")).pack(side="right")
        
        # Analysis controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=20)
        
        self.analyze_button = tk.Button(control_frame, text="Start Analysis", 
                                       command=self.start_analysis,
                                       font=("Arial", 14), bg="blue", fg="white",
                                       width=15, height=2)
        self.analyze_button.pack(side="left", padx=10)
        
        self.status_label = tk.Label(control_frame, text="Ready for analysis", 
                                   font=("Arial", 12))
        self.status_label.pack(side="left", padx=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(pady=10, padx=20, fill="x")
        
        # Results frame
        results_frame = tk.LabelFrame(self.root, text="Analysis Results", 
                                     font=("Arial", 12, "bold"))
        results_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Heatmap tab
        self.heatmap_frame = tk.Frame(self.notebook)
        self.notebook.add(self.heatmap_frame, text="Attention Heatmaps")
        
        # Timeline tab
        self.timeline_frame = tk.Frame(self.notebook)
        self.notebook.add(self.timeline_frame, text="Confusion Timeline")
        
        # Insights tab
        self.insights_frame = tk.Frame(self.notebook)
        self.notebook.add(self.insights_frame, text="Business Insights")
        
        # Initialize empty results
        self.setup_results_ui()
        
    def select_file(self, file_type):
        """Select input files"""
        if file_type == "webcam":
            file_path = filedialog.askopenfilename(
                title="Select Webcam Video",
                filetypes=[("Video files", "*.avi *.mp4 *.mov")]
            )
            if file_path:
                self.webcam_video = file_path
                self.webcam_label.config(text=os.path.basename(file_path), fg="black")
        
        elif file_type == "screen":
            file_path = filedialog.askopenfilename(
                title="Select Screen Recording",
                filetypes=[("Video files", "*.avi *.mp4 *.mov")]
            )
            if file_path:
                self.screen_video = file_path
                self.screen_label.config(text=os.path.basename(file_path), fg="black")
        
        elif file_type == "pdf":
            file_path = filedialog.askopenfilename(
                title="Select PDF Manual",
                filetypes=[("PDF files", "*.pdf")]
            )
            if file_path:
                self.pdf_path = file_path
                self.pdf_label.config(text=os.path.basename(file_path), fg="black")
    
    def start_analysis(self):
        """Start the analysis process"""
        if not all([self.webcam_video, self.screen_video, self.pdf_path]):
            messagebox.showerror("Error", "Please select all required files!")
            return
        
        # Update UI
        self.analyze_button.config(state="disabled")
        self.status_label.config(text="Analyzing...")
        self.progress.start()
        
        # Start analysis in separate thread
        self.analysis_thread = threading.Thread(target=self.run_analysis)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def run_analysis(self):
        """Run the complete analysis"""
        try:
            # Initialize analyzer with input files
            self.analyzer.set_input_files(
                webcam_video=self.webcam_video,
                screen_video=self.screen_video,
                pdf_path=self.pdf_path
            )
            
            # Run analysis
            self.analysis_results = self.analyzer.analyze()
            
            # Update UI with results
            self.root.after(0, self.update_results_ui)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.root.after(0, lambda: self.show_error(error_msg))
    
    def show_error(self, message):
        """Show error message"""
        self.analyze_button.config(state="normal")
        self.status_label.config(text="Analysis failed")
        self.progress.stop()
        messagebox.showerror("Error", message)
    
    def update_results_ui(self):
        """Update UI with analysis results"""
        self.analyze_button.config(state="normal")
        self.status_label.config(text="Analysis complete!")
        self.progress.stop()
        
        # Update results display
        self.display_heatmaps()
        self.display_timeline()
        self.display_insights()
        
        messagebox.showinfo("Success", "Analysis completed successfully!")
    
    def setup_results_ui(self):
        """Setup empty results UI"""
        # Heatmap frame
        self.heatmap_canvas = tk.Canvas(self.heatmap_frame, bg="white")
        self.heatmap_canvas.pack(fill="both", expand=True)
        
        # Timeline frame
        self.timeline_canvas = tk.Canvas(self.timeline_frame, bg="white")
        self.timeline_canvas.pack(fill="both", expand=True)
        
        # Insights frame
        self.insights_text = tk.Text(self.insights_frame, wrap="word", font=("Arial", 10))
        self.insights_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add scrollbar for insights
        scrollbar = tk.Scrollbar(self.insights_text)
        scrollbar.pack(side="right", fill="y")
        self.insights_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.insights_text.yview)
    
    def display_heatmaps(self):
        """Display attention heatmaps"""
        if not self.analysis_results:
            return
        
        # Clear canvas
        self.heatmap_canvas.delete("all")
        
        # Get heatmap data
        heatmaps = self.analysis_results.get('heatmaps', {})
        
        if not heatmaps:
            self.heatmap_canvas.create_text(300, 200, text="No heatmap data available", 
                                          font=("Arial", 14))
            return
        
        # Display first page heatmap as example
        first_page = list(heatmaps.keys())[0]
        heatmap_path = heatmaps[first_page]
        
        if os.path.exists(heatmap_path):
            try:
                # Load image using PIL and convert to PhotoImage
                pil_img = Image.open(heatmap_path)
                
                # Resize if too large
                max_width, max_height = 500, 400
                if pil_img.width > max_width or pil_img.height > max_height:
                    pil_img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                img = ImageTk.PhotoImage(pil_img)
                
                # Display image
                canvas_width = self.heatmap_canvas.winfo_reqwidth() or 600
                canvas_height = self.heatmap_canvas.winfo_reqheight() or 400
                self.heatmap_canvas.create_image(canvas_width//2, canvas_height//2, image=img)
                self.heatmap_canvas.image = img  # Keep reference to prevent garbage collection
                
                # Add page info
                self.heatmap_canvas.create_text(canvas_width//2, 30, 
                                              text=f"Heatmap - Page {first_page + 1}", 
                                              font=("Arial", 14, "bold"))
                
            except Exception as e:
                self.heatmap_canvas.create_text(300, 200, 
                                              text=f"Error loading heatmap: {str(e)}", 
                                              font=("Arial", 12), fill="red")
    
    def display_timeline(self):
        """Display confusion timeline"""
        if not self.analysis_results:
            return
        
        # Clear canvas
        self.timeline_canvas.delete("all")
        
        # Get timeline data
        timeline_data = self.analysis_results.get('timeline', [])
        
        if not timeline_data:
            self.timeline_canvas.create_text(300, 200, text="No timeline data available", 
                                           font=("Arial", 14))
            return
        
        # Create improved timeline visualization
        canvas_width = self.timeline_canvas.winfo_reqwidth() or 600
        canvas_height = self.timeline_canvas.winfo_reqheight() or 300
        
        # Title
        self.timeline_canvas.create_text(canvas_width//2, 30, text="Confusion Timeline", 
                                       font=("Arial", 16, "bold"))
        
        # Y-axis label
        self.timeline_canvas.create_text(20, canvas_height//2, text="Confusion\nLevel", 
                                       font=("Arial", 10), angle=90)
        
        # X-axis label
        self.timeline_canvas.create_text(canvas_width//2, canvas_height - 20, text="Time (seconds)", 
                                       font=("Arial", 10))
        
        # Draw timeline bars with better spacing
        y_start = 60
        y_end = canvas_height - 40
        bar_area_height = y_end - y_start
        
        # Show up to 30 data points for better visibility
        max_points = min(30, len(timeline_data))
        bar_width = max(15, (canvas_width - 100) // max_points - 2)
        
        for i, data_point in enumerate(timeline_data[:max_points]):
            confusion = data_point.get('confusion_score', 0)
            bar_height = confusion * bar_area_height  # Scale to available height
            x = 60 + i * (bar_width + 2)
            y = y_end - bar_height
            
            # Color coding with more levels
            if confusion > 0.8:
                color = "#FF0000"  # Red - High confusion
            elif confusion > 0.6:
                color = "#FF6600"  # Orange-Red
            elif confusion > 0.4:
                color = "#FFAA00"  # Orange
            elif confusion > 0.2:
                color = "#FFFF00"  # Yellow
            else:
                color = "#00FF00"  # Green - Low confusion
            
            # Draw bar
            self.timeline_canvas.create_rectangle(x, y, x + bar_width, y_end, 
                                                fill=color, outline="black", width=1)
            
            # Add value label on top of bar (every 5th bar to avoid clutter)
            if i % 5 == 0:
                self.timeline_canvas.create_text(x + bar_width//2, y - 15, 
                                               text=f"{confusion:.2f}", 
                                               font=("Arial", 8))
        
        # Add legend
        legend_y = 20
        legend_items = [
            ("High Confusion (>0.8)", "#FF0000"),
            ("Medium-High (0.6-0.8)", "#FF6600"),
            ("Medium (0.4-0.6)", "#FFAA00"),
            ("Low-Medium (0.2-0.4)", "#FFFF00"),
            ("Low Confusion (<0.2)", "#00FF00")
        ]
        
        for i, (label, color) in enumerate(legend_items):
            x_pos = canvas_width - 200 + (i % 2) * 100
            y_pos = legend_y + (i // 2) * 20
            self.timeline_canvas.create_rectangle(x_pos, y_pos, x_pos + 15, y_pos + 15, 
                                                fill=color, outline="black")
            self.timeline_canvas.create_text(x_pos + 20, y_pos + 7, text=label, 
                                           font=("Arial", 8), anchor="w")
    
    def display_insights(self):
        """Display business insights"""
        if not self.analysis_results:
            return
        
        # Clear text
        self.insights_text.delete("1.0", tk.END)
        
        # Get insights data
        insights = self.analysis_results.get('insights', {})
        
        insights_text = f"""
BUSINESS INSIGHTS REPORT
========================

Overall Analysis:
- Total reading time: {insights.get('total_time', 0):.1f} seconds
- Average confusion score: {insights.get('avg_confusion', 0):.2f}
- Most confusing page: {insights.get('most_confusing_page', 'N/A')}
- Attention hotspots: {insights.get('attention_hotspots', 0)} regions identified

Key Findings:
{insights.get('key_findings', 'No specific findings available')}

Recommendations:
{insights.get('recommendations', 'No recommendations available')}

Technical Details:
- Face detection confidence: {insights.get('face_confidence', 0):.2f}
- Gaze tracking accuracy: {insights.get('gaze_accuracy', 0):.2f}
- PDF mapping success: {insights.get('pdf_mapping_success', 0):.1f}%
        """
        
        self.insights_text.insert("1.0", insights_text)
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    import threading
    app = GawkAnalyzer()
    app.run()
