# Gawk Setup Instructions

## Prerequisites

### Python Dependencies
Install all required Python packages:
```bash
pip install -r requirements.txt
```

### System Dependencies

#### Windows
For PDF processing, you need to install Poppler for Windows:

1. Download Poppler for Windows from: https://github.com/oschwartz10612/poppler-windows/releases/
2. Extract the zip file to a folder (e.g., `C:\poppler`)
3. Add the `bin` folder to your system PATH:
   - Open System Properties → Advanced → Environment Variables
   - Add `C:\poppler\bin` to the PATH variable
4. Restart your command prompt/PowerShell

#### Alternative Windows Installation (using conda):
```bash
conda install -c conda-forge poppler
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt-get install poppler-utils
```

#### macOS:
```bash
brew install poppler
```

## Testing Installation

Run the test script to verify everything is working:
```bash
python test_gawk.py
```

This will test:
- All Python imports
- Webcam access
- Screen capture functionality
- Face detection
- File system permissions

## Running the Application

1. **Data Collection**:
   ```bash
   python gawk_launcher.py
   ```
   Click "Data Collection" to record videos and PDF

2. **Analysis**:
   ```bash
   python gawk_launcher.py
   ```
   Click "Analysis Dashboard" to analyze recorded data

## Troubleshooting

### Common Issues

1. **"No module named pdf2image"**:
   - Install pdf2image: `pip install pdf2image`
   - Install poppler (see system dependencies above)

2. **Webcam not detected**:
   - Ensure webcam is connected and not used by other applications
   - Check camera permissions in Windows settings

3. **Screen recording fails**:
   - Run as administrator for screen capture permissions
   - Ensure no other screen recording software is running

4. **Face detection not working**:
   - Ensure good lighting
   - Check that MediaPipe is properly installed
   - Verify webcam is working

### Log Files

The application creates detailed log files:
- `gawk_data_collection.log` - Data collection logs
- `gawk_analysis.log` - Analysis logs  
- `gawk_attention_analyzer.log` - Core analysis engine logs

Check these files if you encounter issues.

## Video Synchronization Tips

1. **Start both recordings simultaneously** - Click "Start Recording" when ready
2. **Make a sync gesture** - Wave your hand clearly at the start
3. **Keep consistent screen resolution** - Don't change display settings during recording
4. **Ensure good lighting** - Face should be clearly visible for detection

## Performance Optimization

- Close unnecessary applications during recording
- Use a well-lit environment
- Ensure stable internet connection (for MediaPipe models)
- Use SSD storage for better video writing performance
