# 🧬 SpermApp - Sperm Detection and Motility Classification

A Streamlit-based web application for automated sperm detection, tracking, and motility classification using computer vision and machine learning.

## 🎯 Features

- **Object Detection & Tracking**: YOLO-based sperm detection with persistent tracking across video frames
- **Motility Classification**: Machine learning-powered classification into three motility categories:
  - Progressive Motility
  - Intermediate Motility  
  - Hyperactivated Motility
- **Kinematic Metrics**: Calculation of key sperm motility parameters:
  - VSL (Straight Line Velocity)
  - VCL (Curvilinear Velocity)
  - VAP (Average Path Velocity)
  - LIN (Linearity)
  - WOB (Wobble)
  - STR (Straightness)
  - ALH (Amplitude of Lateral Head Displacement)
- **Video Processing**: Support for multiple video formats (MP4, AVI, MOV, MKV)
- **Results Export**: Download annotated videos and CSV data for further analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- FFmpeg (for video processing)


### Usage

Application can be accessed here: https://juliannalamm-spermapp-streamlit-app-0avwqu.streamlit.app/


## 📊 Understanding the Results

### Detection & Tracking Tab
- **Annotated Video**: Original video with bounding boxes and tracking IDs
- **Tracking Data CSV**: Frame-by-frame coordinates and confidence scores

### Motility Metrics Tab  
- **Classified Video**: Color-coded tracks showing motility classification
- **Kinematic Metrics CSV**: Calculated motility parameters for each sperm

### Key Metrics Explained

- **VSL**: Straight-line distance from start to end point divided by time
- **VCL**: Total path length divided by time (actual distance traveled)
- **VAP**: Smoothed path length divided by time
- **LIN**: VSL/VCL ratio (measure of path straightness)
- **WOB**: VAP/VCL ratio (measure of path smoothness)
- **STR**: VSL/VAP ratio (indication of the relationship between the net space gain and the general trajectory of the spermatozoon) 
- **ALH**: Amplitude of lateral head movement

## ⚙️ Configuration

### For Custom Videos

When uploading your own videos, you'll need to provide:

1. **Pixels per micron**: Calibration factor for accurate measurements
   - **Recommended method**: Use ImageJ
     1. Open video in ImageJ
     2. Use "Set Scale" tool (Analyze > Set Scale)
     3. Draw line over known distance
     4. Enter known distance and units
     5. Use resulting scale (pixels/unit)

2. **Video duration**: Total length in seconds
3. **Image dimensions**: Width and height in pixels

### Default Settings

Demo videos use these default parameters:
- Pixels per micron: 0.48
- Duration: 30 seconds
- Resolution: 640x480 pixels

## 🏗️ Project Structure

```
SpermApp/
├── streamlit_app.py          # Main Streamlit application
├── process_csv_metrics.py    # Motility analysis and classification
├── models/                   # ML models and scalers
│   ├── best.pt              # YOLO detection model
│   ├── minmax_scaler.pkl    # Feature scaler
│   ├── subcluster_model.pkl # Motility classifier
│   └── cluster_label_map.pkl # Cluster label mappings
├── demo_videos/             # Sample videos for testing
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 Technical Details

### Dependencies

- **Streamlit**: Web application framework
- **OpenCV**: Video processing and computer vision
- **Ultralytics**: YOLO object detection
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning (joblib for model loading)
- **Matplotlib**: Visualization and color mapping

### Model Architecture

1. **Detection**: YOLO model trained on sperm datasets
2. **Tracking**: Custom BoTSORT implementation for persistent tracking
3. **Classification**: Supervised learning model using kinematic features
4. **Feature Engineering**: Rolling averages, displacement calculations, and statistical measures


### Performance Tips

- Use shorter video segments for faster processing
- Ensure adequate RAM (4GB+ recommended)
- Close other applications during processing
- Use SSD storage for better I/O performance

---

**Note**: This application is designed for research purposes. Always validate results with domain experts and follow appropriate laboratory protocols. 