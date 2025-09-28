"""
Gawk Launcher Application
Main entry point for the Gawk attention analysis system.
"""

import tkinter as tk
from tkinter import messagebox
import os
import sys

class GawkLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gawk - Attention Analysis System")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # Center the window
        self.center_window()
        
        self.setup_ui()
        
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_label = tk.Label(self.root, text="Gawk", 
                              font=("Arial", 24, "bold"), fg="#2E86AB")
        title_label.pack(pady=20)
        
        subtitle_label = tk.Label(self.root, text="Attention Analysis System", 
                                 font=("Arial", 14), fg="#666666")
        subtitle_label.pack(pady=(0, 30))
        
        # Description
        desc_text = """Real-time computer vision system for analyzing user attention 
while reading PDF manuals. Generate attention heatmaps, confusion scores, 
and business insights to optimize manual content."""
        
        desc_label = tk.Label(self.root, text=desc_text, 
                             font=("Arial", 10), wraplength=400, justify="center")
        desc_label.pack(pady=(0, 30))
        
        # Buttons frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20, fill="x")
        
        # Data Collection Button
        collect_btn = tk.Button(button_frame, text="Data Collection", 
                               command=self.launch_data_collection,
                               font=("Arial", 12, "bold"),
                               bg="#4CAF50", fg="white",
                               width=20, height=3,
                               relief="raised", bd=3)
        collect_btn.pack(pady=10, fill="x", padx=50)
        
        collect_desc = tk.Label(button_frame, text="Record webcam, screen, and PDF data", 
                               font=("Arial", 9), fg="#666666")
        collect_desc.pack()
        
        # Analysis Button
        analyze_btn = tk.Button(button_frame, text="Analysis Dashboard", 
                               command=self.launch_analysis,
                               font=("Arial", 12, "bold"),
                               bg="#2196F3", fg="white",
                               width=20, height=3,
                               relief="raised", bd=3)
        analyze_btn.pack(pady=10, fill="x", padx=50)
        
        analyze_desc = tk.Label(button_frame, text="Analyze recorded data and generate insights", 
                               font=("Arial", 9), fg="#666666")
        analyze_desc.pack()
        
        # Status frame
        status_frame = tk.Frame(self.root)
        status_frame.pack(pady=20, fill="x")
        
        # Check for existing data
        self.check_existing_data()
        
        # Instructions
        instructions = tk.Text(self.root, height=6, width=60, wrap="word", 
                              font=("Arial", 9), bg="#F5F5F5")
        instructions.pack(pady=10, padx=20, fill="x")
        
        instructions_text = """WORKFLOW:
1. First, use 'Data Collection' to record your reading session
2. Then, use 'Analysis Dashboard' to process the recorded data
3. View attention heatmaps, confusion scores, and business insights

SYNC TIPS:
- Start both recordings simultaneously
- Make a clear gesture at the beginning for sync reference
- Keep consistent screen resolution throughout recording
- Ensure good lighting for face detection"""
        
        instructions.insert("1.0", instructions_text)
        instructions.config(state="disabled")
        
        # Exit button
        exit_btn = tk.Button(self.root, text="Exit", 
                            command=self.root.quit,
                            font=("Arial", 10),
                            bg="#F44336", fg="white",
                            width=10, height=1)
        exit_btn.pack(pady=10)
    
    def check_existing_data(self):
        """Check for existing data and show status"""
        data_dir = "gawk_data"
        if os.path.exists(data_dir):
            # Check for videos
            videos_dir = os.path.join(data_dir, "videos")
            if os.path.exists(videos_dir):
                video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.avi', '.mp4', '.mov'))]
                if video_files:
                    status_text = f"Found {len(video_files)} video file(s) ready for analysis"
                    status_color = "#4CAF50"
                else:
                    status_text = "No video files found - start with data collection"
                    status_color = "#FF9800"
            else:
                status_text = "No data directory found - start with data collection"
                status_color = "#FF9800"
        else:
            status_text = "No data found - start with data collection"
            status_color = "#FF9800"
        
        status_label = tk.Label(self.root, text=status_text, 
                               font=("Arial", 10), fg=status_color)
        status_label.pack(pady=5)
    
    def launch_data_collection(self):
        """Launch data collection application"""
        try:
            # Import and run data collection
            from data_collection import GawkDataCollector
            self.root.withdraw()  # Hide launcher window
            
            collector = GawkDataCollector()
            collector.run()
            
            # Show launcher again when data collection is done
            self.root.deiconify()
            self.check_existing_data()  # Update status
            
        except ImportError as e:
            messagebox.showerror("Error", f"Failed to import data collection module: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch data collection: {e}")
            self.root.deiconify()
    
    def launch_analysis(self):
        """Launch analysis dashboard"""
        try:
            # Check if data exists
            data_dir = "gawk_data"
            if not os.path.exists(data_dir):
                messagebox.showerror("Error", "No data found. Please run data collection first.")
                return
            
            videos_dir = os.path.join(data_dir, "videos")
            if not os.path.exists(videos_dir) or not os.listdir(videos_dir):
                messagebox.showerror("Error", "No video files found. Please run data collection first.")
                return
            
            # Import and run analysis
            from main import GawkAnalyzer
            self.root.withdraw()  # Hide launcher window
            
            analyzer = GawkAnalyzer()
            analyzer.run()
            
            # Show launcher again when analysis is done
            self.root.deiconify()
            
        except ImportError as e:
            messagebox.showerror("Error", f"Failed to import analysis module: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch analysis: {e}")
            self.root.deiconify()
    
    def run(self):
        """Run the launcher application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = GawkLauncher()
    app.run()
