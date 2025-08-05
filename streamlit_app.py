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
"Analyze sperm movement directly from video - this app detects, tracks, and classifies motility behaviors to support more granular fertility insights."
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
def process_video(video_path, model, tracker, progress_bar, frame_info, max_seconds=2):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frames to process (2 seconds worth)
    frames_to_process = min(fps * max_seconds, total_frames)
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
    while cap.isOpened() and frame_count < frames_to_process:
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
        progress_bar.progress(min(frame_count / frames_to_process, 1.0))
        frame_info.text(f"Processing frame {frame_count}/{frames_to_process}")

    # Close video writer after processing 2 seconds
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

st.subheader("Select a demo video:")
demo_choice = st.selectbox("Demo video:", demo_videos)

video_path = None

video_path = os.path.join(demo_dir, demo_choice)


# --- Advanced Metadata ---



image_j_scale = 0.48
total_duration_seconds = 2  # Analyze only 2 seconds
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
        try:
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
        except Exception as e:
            st.error(f"Error during motility classification: {str(e)}")
            st.warning("No sperm were detected or processed. Please try with a different video.")
            # Set empty results to prevent further errors
            st.session_state.annotated_video = b""
            st.session_state.metrics_csv = b""

    st.success(f"Processing complete! {st.session_state.frames_written} frames processed.")

# --- Results Display ---
if st.session_state.get('processed', False):
    st.markdown("---")
    st.header("Results")
    tab1, tab2 = st.tabs(["📊 Motility Metrics","🔍 Detection & Tracking"])
    
    with tab1:
        st.subheader("Motility Classification")
        if st.session_state.get('annotated_video') and len(st.session_state.annotated_video) > 0:
            st.video(st.session_state.annotated_video)
        else:
            st.warning("No motility classification results available. This may indicate no sperm were detected or processed.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.get('annotated_video') and len(st.session_state.annotated_video) > 0:
                st.download_button("Download Motility Classification Video", st.session_state.annotated_video, "motility_classified_video.mp4", mime="video/mp4")
            else:
                st.download_button("Download Motility Classification Video", b"", "motility_classified_video.mp4", mime="video/mp4", disabled=True)
        with col2:
            if st.session_state.get('metrics_csv') and len(st.session_state.metrics_csv) > 0:
                st.download_button("Download Kinematic Metrics CSV", st.session_state.metrics_csv, "kinematic_metrics.csv", mime="text/csv")
            else:
                st.download_button("Download Kinematic Metrics CSV", b"", "kinematic_metrics.csv", mime="text/csv", disabled=True)

    with tab2:
        st.subheader("Object Detection & Tracking")
        if st.session_state.video_bytes and len(st.session_state.video_bytes) > 0:
            st.video(st.session_state.video_bytes)
        else:
            st.warning("No detection and tracking results available.")
        col1, col2 = st.columns(2)
        with col1: 
            if st.session_state.video_bytes and len(st.session_state.video_bytes) > 0:
                st.download_button("Download Annotated Video", st.session_state.video_bytes, "annotated_video.mp4", mime="video/mp4")
            else:
                st.download_button("Download Annotated Video", b"", "annotated_video.mp4", mime="video/mp4", disabled=True)
        with col2:
            if st.session_state.csv_bytes and len(st.session_state.csv_bytes) > 0:
                st.download_button("Download Tracking Data (CSV)", st.session_state.csv_bytes, "tracked_coordinates.csv", mime="text/csv")
            else:
                st.download_button("Download Tracking Data (CSV)", b"", "tracked_coordinates.csv", mime="text/csv", disabled=True)

    if st.button("Process New Video"):
        for key in [
            'processed', 'output_video_path', 'csv_output_path', 'frames_written',
            'video_bytes', 'csv_bytes', 'annotated_video', 'metrics_csv']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
