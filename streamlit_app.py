import streamlit as st
import tempfile
import cv2
import csv
from ultralytics import YOLO
import time
import os
import uuid
import subprocess
from process_csv_metrics import process_csv_metrics

# --- UI Custom Logo and Title ---
st.title("🧬 Sperm Detection and Motility Classification")
st.markdown(
"Analyze sperm movement directly from video—this app detects, tracks, and classifies motility behaviors to support more granular fertility insig"hts."
)
with st.expander("ℹ️ About this app"):
    st.markdown("""
    This app performs automated sperm detection, tracking, and motility classification from uploaded microscope videos using deep learning and computer vision techniques.

    **Key features:**
    - **YOLO-based object detection** and **BoT-SORT tracking**
    - **Kinematic metric calculations**: VCL, VAP, VSL, ALH, LIN, WOB, STR
    - **Cluster-based motility classification** (e.g., Progressive, Hyperactivated)
    - **Downloadable Results**
        - **Annotated Video**: Original video overlaid with bounding boxes and tracking IDs
        - **Tracking Data CSV**: Frame-by-frame coordinates and confidence scores for each detection
        - **Classified Video**: Tracks color-coded by motility subtype
        - **Kinematic Metrics CSV**: Detailed metrics per sperm, with class labels

    Built as part of a research-driven effort to enhance sperm motility assessment through machine learning and quantitative analysis.
    """)

# --- Session State ---
if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.output_video_path = None
    st.session_state.csv_output_path = None
    st.session_state.frames_written = 0
    st.session_state.video_bytes = None
    st.session_state.csv_bytes = None

# --- Video Processing Function ---
def process_video(video_path, model, tracker, progress_bar, frame_info):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    output_video_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_video_path = output_video_temp.name

    csv_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.csv').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    csv_file = open(csv_output_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'id', 'x1', 'y1', 'x2', 'y2', 'confidence', 'class'])

    frames_written = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, tracker=tracker)
        boxes = results[0].boxes
        if boxes.id is not None and len(boxes.id) > 0:
            for i in range(len(boxes.id)):
                track_id = int(boxes.id[i])
                x1, y1, x2, y2 = [float(x) for x in boxes.xyxy[i]]
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])
                csv_writer.writerow([frame_count, track_id, x1, y1, x2, y2, conf, cls])

        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        frames_written += 1
        frame_count += 1
        progress_bar.progress(min(frame_count / total_frames, 1.0))
        frame_info.text(f"Processing frame {frame_count}/{total_frames}")

    cap.release()
    out.release()
    csv_file.close()

    fixed_output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fixed_output_path = fixed_output_temp.name
    subprocess.run([
        "ffmpeg", "-y", "-i", output_video_path,
        "-vcodec", "libx264", "-acodec", "aac", fixed_output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with open(fixed_output_path, 'rb') as f:
        video_bytes = f.read()
    with open(csv_output_path, 'rb') as f:
        csv_bytes = f.read()

    os.remove(output_video_path)
    os.remove(fixed_output_path)

    return None, None, frames_written, video_bytes, csv_bytes

# --- Video Selection ---
demo_dir = "demo_videos"
demo_videos = [f for f in os.listdir(demo_dir) if f.endswith((".mp4", ".avi", ".mov", ".mkv"))]
demo_videos.insert(0, "None (upload your own)")

st.subheader("Select a demo video or upload your own:")
demo_choice = st.selectbox("Demo video:", demo_videos)
uploaded_file = st.file_uploader("Or upload a video", type=["mp4", "avi", "mov", "mkv"])

video_path = None
if demo_choice != "None (upload your own)":
    video_path = os.path.join(demo_dir, demo_choice)
elif uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

# --- Advanced Metadata ---
use_defaults = (demo_choice != "None (upload your own)")
if not use_defaults:
    with st.expander("Advanced Settings (custom upload only)"):
        st.subheader("Enter Metadata for Uploaded Video")
        image_j_scale = st.number_input(
            "Pixels per micron", 
            value=0.48, 
            min_value=0.01,
            help="Using ImageJ: 1) Open your video in ImageJ 2) Use the 'Set Scale' tool (Analyze > Set Scale) 3) Draw a line over a known distance (e.g., 100 microns) 4) Enter the known distance and units 5) Use the resulting scale (pixels/unit). Alternative: Measure pixels manually and divide by known distance."
        )
        total_duration_seconds = st.number_input("Total video duration (sec)", value=30.0, min_value=0.1)
        img_w = st.number_input("Image width (px)", value=640)
        img_h = st.number_input("Image height (px)", value=480)
else:
    image_j_scale = 0.48
    total_duration_seconds = 30
    img_w = 640
    img_h = 480

# --- Processing Trigger ---
start = st.button("Start Processing")

# --- Video Preview ---
if video_path and not start:
    st.subheader("Video Preview")
    st.video(video_path)

if start and video_path:
    model = YOLO("models/best.pt")
    tracker = "models/custom_botsort.yaml"
    progress_bar = st.progress(0)
    frame_info = st.empty()

    with st.spinner('Processing: Object Detection and Tracking...'):
        (_, _, frames_written, video_bytes, csv_bytes) = process_video(
            video_path, model, tracker, progress_bar, frame_info
        )
        st.session_state.processed = True
        st.session_state.video_bytes = video_bytes
        st.session_state.csv_bytes = csv_bytes
        st.session_state.frames_written = frames_written

    with st.spinner("Classifying motility, just a few more moments..."):
        annotated_video, metrics_csv = process_csv_metrics(
            tracked_csv_bytes=st.session_state.csv_bytes,
            original_video_path=video_path,
            image_j_scale=image_j_scale,
            total_duration_seconds=total_duration_seconds,
            img_w=img_w,
            img_h=img_h
        )
        st.session_state.annotated_video = annotated_video
        st.session_state.metrics_csv = metrics_csv

    st.success(f"Processing complete! {st.session_state.frames_written} frames processed.")

# --- Results Display ---
if st.session_state.get('processed', False):
    st.markdown("---")
    st.header("Results")
    tab1, tab2 = st.tabs(["📊 Motility Metrics","🔍 Detection & Tracking"])
    
    with tab1:
        st.subheader("Motility Classification")
        if st.session_state.get('annotated_video'):
            st.video(st.session_state.annotated_video)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download Motility Classification Video", st.session_state.annotated_video, "motility_classified_video.mp4", mime="video/mp4")
        with col2:
            st.download_button("Download Kinematic Metrics CSV", st.session_state.metrics_csv, "kinematic_metrics.csv", mime="text/csv")

    with tab2:
        st.subheader("Object Detection & Tracking")
        if st.session_state.video_bytes:
            st.video(st.session_state.video_bytes)
        col1, col2 = st.columns(2)
        with col1: 
            st.download_button("Download Annotated Video", st.session_state.video_bytes, "annotated_video.mp4", mime="video/mp4")
        with col2:
            st.download_button("Download Tracking Data (CSV)", st.session_state.csv_bytes, "tracked_coordinates.csv", mime="text/csv")

    if st.button("Process New Video"):
        for key in [
            'processed', 'output_video_path', 'csv_output_path', 'frames_written',
            'video_bytes', 'csv_bytes', 'annotated_video', 'metrics_csv']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
