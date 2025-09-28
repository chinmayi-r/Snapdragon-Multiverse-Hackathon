# Gawk 

A real-time computer vision system for analyzing user attention while reading PDF manuals. Built for the Snapdragon Multiverse Hackathon focusing on real-time computer vision applications.

## Features

- **Real-time Data Collection**: Records webcam video, screen recording, and PDF input simultaneously
- **Facial Expression Recognition**: Analyzes facial expressions to detect confusion and engagement
- **Gaze Tracking**: Maps user gaze to specific PDF regions (header, content, footer)
- **Attention Heatmaps**: Generates visual heatmaps showing attention patterns on PDF pages
- **Confusion Scoring**: Calculates confusion scores based on multiple indicators
- **Business Insights**: Provides actionable recommendations for manual optimization
- **Interactive Dashboard**: User-friendly interface for data collection and analysis

## System Architecture

### Data Collection Phase
1. **Webcam Recording**: Captures user's face for facial analysis
2. **Screen Recording**: Records screen activity while reading PDF
3. **PDF Processing**: Extracts pages and metadata from the manual
4. **Synchronization**: Ensures videos are properly aligned in time

### Analysis Phase
1. **Facial Analysis**: Detects expressions and confusion indicators
2. **Gaze Mapping**: Maps gaze points to PDF regions
3. **Page Detection**: Identifies which PDF page is currently being viewed
4. **Heatmap Generation**: Creates attention visualizations
5. **Insight Generation**: Produces business recommendations

## Installation

### Prerequisites
- Python 3.8 or higher
- Windows 10/11 (for executable version)
- Webcam
- PDF viewer application

### Python Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd Snapdragon-Multiverse-Hackathon
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python gawk_launcher.py
```

### Windows Executable
1. Build the executable:
```bash
python build_exe.py
```

2. Run the installer:
```bash
install.bat
```

## Usage

### Step 1: Data Collection
1. Launch the application using `gawk_launcher.py`
2. Click "Data Collection"
3. Select your PDF manual
4. Position yourself in front of the webcam
5. Open the PDF in your preferred viewer
6. Click "Start Recording"
7. Read through the manual naturally
8. Click "Stop Recording" when finished

### Step 2: Analysis
1. Click "Analysis Dashboard"
2. Select the recorded webcam video
3. Select the recorded screen video
4. Select the PDF manual
5. Click "Start Analysis"
6. View results in the dashboard tabs:
   - **Attention Heatmaps**: Visual representation of attention patterns
   - **Confusion Timeline**: Timeline showing confusion levels over time
   - **Business Insights**: Actionable recommendations and findings

## Video Synchronization

The system includes several methods to ensure proper synchronization:

1. **Automatic Sync**: The system automatically detects when your face first appears and aligns the videos
2. **Timestamp Matching**: Both videos are recorded with precise timestamps
3. **Manual Sync Reference**: Make a clear gesture (wave hand) at the start for additional sync reference
4. **Consistent Resolution**: Keep the same screen resolution throughout recording

## Confusion Scoring Algorithm

The confusion score is calculated using multiple components:

1. **Facial Expression Analysis** (40% weight):
   - Detects confused, frustrated, or concentrated expressions
   - Uses MediaPipe for facial landmark detection

2. **Confusion Indicators** (30% weight):
   - Furrowed brow detection
   - Squinting detection
   - Eye movement patterns

3. **Attention Patterns** (30% weight):
   - Attention spread across regions
   - Duration of focus on same location
   - Gaze movement patterns

## PDF Region Mapping

The system maps user attention to five PDF regions:
- **Header**: Top 10% of page
- **Content Top**: 10-30% of page
- **Content Middle**: 30-70% of page
- **Content Bottom**: 70-90% of page
- **Footer**: Bottom 10% of page

## Business Insights

The system generates actionable insights including:
- Overall reading time and confusion levels
- Most confusing pages and regions
- Attention hotspot identification
- Content organization recommendations
- Reading behavior analysis

## Technical Details

### Dependencies
- OpenCV: Computer vision processing
- MediaPipe: Facial analysis and landmark detection
- NumPy: Numerical computations
- Matplotlib/Seaborn: Visualization
- Scikit-learn: Machine learning algorithms
- PyPDF2: PDF processing
- MSS: Screen capture
- Tkinter: GUI framework

### Performance Optimization
- Real-time processing with 30 FPS target
- Efficient face detection using MediaPipe
- Optimized video encoding (XVID codec)
- Memory-efficient data structures
- Parallel processing where possible

## File Structure

```
Snapdragon-Multiverse-Hackathon/
├── gawk_launcher.py          # Main launcher application
├── data_collection.py        # Data collection module
├── main.py                   # Analysis dashboard
├── attention_analyzer.py     # Core analysis engine
├── build_exe.py             # Windows executable builder
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── gawk_data/              # Data directory
    ├── videos/             # Recorded videos
    ├── pdfs/               # PDF files
    ├── pdf_pages/          # Extracted PDF pages
    ├── heatmaps/           # Generated heatmaps
    └── metadata/           # Analysis metadata
```

## Troubleshooting

### Common Issues

1. **Webcam not detected**:
   - Ensure webcam is connected and not used by other applications
   - Check camera permissions in Windows settings

2. **Screen recording issues**:
   - Run as administrator for screen capture permissions
   - Ensure no other screen recording software is running

3. **PDF processing errors**:
   - Install poppler-utils for PDF to image conversion
   - Ensure PDF is not password protected

4. **Performance issues**:
   - Close unnecessary applications
   - Reduce video resolution if needed
   - Ensure adequate lighting for face detection

### Sync Issues

If videos appear out of sync:
1. Check that both recordings started simultaneously
2. Look for the sync reference gesture in both videos
3. Verify timestamps in the metadata files
4. Use the manual sync adjustment in the analysis dashboard

## Future Enhancements

- Advanced gaze tracking with eye-tracking hardware
- Machine learning models for better expression recognition
- Real-time analysis during recording
- Cloud-based processing for large datasets
- Integration with popular PDF viewers
- Mobile app for data collection

## Contributing

This project was developed for the Snapdragon Multiverse Hackathon. For contributions or questions, please contact the development team.

## License

This project is developed for educational and research purposes as part of the Snapdragon Multiverse Hackathon.

## Acknowledgments

- Snapdragon Multiverse Hackathon organizers
- MediaPipe team for facial analysis tools
- OpenCV community for computer vision libraries
- All open-source contributors whose work made this project possible

