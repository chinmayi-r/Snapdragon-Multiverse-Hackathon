"""
Simple test launcher to verify buttons are visible
"""

import tkinter as tk
from tkinter import messagebox

class TestLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Test Launcher")
        self.root.geometry("400x300")
        
        # Title
        title_label = tk.Label(self.root, text="Gawk Test", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        # Data Collection Button
        collect_btn = tk.Button(button_frame, text="Data Collection", 
                               command=self.test_collect,
                               font=("Arial", 12, "bold"),
                               bg="green", fg="white",
                               width=20, height=2)
        collect_btn.pack(pady=10)
        
        # Analysis Button
        analyze_btn = tk.Button(button_frame, text="Analysis Dashboard", 
                               command=self.test_analyze,
                               font=("Arial", 12, "bold"),
                               bg="blue", fg="white",
                               width=20, height=2)
        analyze_btn.pack(pady=10)
        
        # Status
        self.status = tk.Label(self.root, text="Click a button to test", 
                              font=("Arial", 10))
        self.status.pack(pady=10)
    
    def test_collect(self):
        self.status.config(text="Data Collection button clicked!")
        messagebox.showinfo("Test", "Data Collection button works!")
    
    def test_analyze(self):
        self.status.config(text="Analysis Dashboard button clicked!")
        messagebox.showinfo("Test", "Analysis Dashboard button works!")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = TestLauncher()
    app.run()
