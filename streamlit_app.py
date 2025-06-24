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

# Initialize session state for storing results
if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.output_video_path = None
    st.session_state.csv_output_path = None
    st.session_state.frames_written = 0
    st.session_state.video_bytes = None
    st.session_state.csv_bytes = None

def process_video(video_path, model, tracker, progress_bar, max_seconds=10):
    import subprocess

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

    cap.release()
    out.release()
    csv_file.close()

    # Re-encode to browser-safe format
    fixed_output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fixed_output_path = fixed_output_temp.name
    subprocess.run([
        "ffmpeg", "-y", "-i", output_video_path,
        "-vcodec", "libx264", "-acodec", "aac", fixed_output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Read to memory
    with open(fixed_output_path, 'rb') as f:
        video_bytes = f.read()
    with open(csv_output_path, 'rb') as f:
        csv_bytes = f.read()

    # Clean up all temp files
    os.remove(output_video_path)
    os.remove(fixed_output_path)
    # os.remove(csv_output_path)

    return None, None, frames_written, video_bytes, csv_bytes


# Streamlit UI

demo_dir = "demo_videos"
demo_videos = [f for f in os.listdir(demo_dir) if f.endswith((".mp4", ".avi", ".mov", ".mkv"))]
demo_videos.insert(0, "None (upload your own)")

st.title("YOLO Sperm Tracking App")

# 1. Demo selector/upload at the top
st.subheader("Select a demo video or upload your own:")
demo_choice = st.selectbox("Demo video:", demo_videos)
uploaded_file = st.file_uploader("Or upload a video", type=["mp4", "avi", "mov", "mkv"])

# 2. Determine which video to use
video_path = None
if demo_choice != "None (upload your own)":
    video_path = os.path.join(demo_dir, demo_choice)
elif uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

# 3. Metadata input (if needed)
use_defaults = (demo_choice != "None (upload your own)")
if not use_defaults:
    st.subheader("Enter Metadata for Uploaded Video")
    image_j_scale = st.number_input("Pixels per micron", value=0.48, min_value=0.01)
    total_duration_seconds = st.number_input("Total video duration (sec)", value=30.0, min_value=0.1)
    img_w = st.number_input("Image width (px)", value=640)
    img_h = st.number_input("Image height (px)", value=480)
else:
    image_j_scale = 0.48
    total_duration_seconds = 30
    img_w = 640
    img_h = 480

# 4. Start Processing button
start = st.button("Start Processing")

# 5. Processing logic
if start and video_path:
    # Load model
    model = YOLO("models/best.pt")
    tracker = "models/custom_botsort.yaml"

    # Create progress bar
    progress_bar = st.progress(0)

    # Process video (object detection/tracking)
    with st.spinner('Processing: Object Detection and Tracking...'):
        (output_video_path, 
         csv_output_path, 
         frames_written,
         video_bytes,
         csv_bytes) = process_video(
            video_path, model, tracker, progress_bar
        )
        st.session_state.processed = True
        st.session_state.video_bytes = video_bytes
        st.session_state.csv_bytes = csv_bytes
        st.session_state.frames_written = frames_written

    # Process motility classification and kinematic metrics
    with st.spinner("Processing: Motility Classification and Kinematic Metrics..."):
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

# 6. Results layout: two columns, each with video and download buttons
if st.session_state.get('processed', False):
    st.markdown("---")
    st.header("Results")
    col1, col2 = st.columns(2)

    # Left: Detection/Tracking
    with col1:
        st.subheader("Object Detection & Tracking")
        if st.session_state.video_bytes:
            st.video(st.session_state.video_bytes)
        st.download_button(
            label="Download Annotated Video",
            data=st.session_state.video_bytes,
            file_name="annotated_video.mp4",
            mime="video/mp4"
        )
        st.download_button(
            label="Download Tracking Data (CSV)",
            data=st.session_state.csv_bytes,
            file_name="tracked_coordinates.csv",
            mime="text/csv"
        )

    # Right: Motility Classification
    with col2:
        st.subheader("Motility Classification & Kinematic Metrics")
        if st.session_state.get('annotated_video'):
            st.video(st.session_state.annotated_video)
        st.download_button(
            label="Download Motility Classification Video",
            data=st.session_state.annotated_video,
            file_name="motility_classified_video.mp4",
            mime="video/mp4"
        )
        st.download_button(
            label="Download Kinematic Metrics CSV",
            data=st.session_state.metrics_csv,
            file_name="kinematic_metrics.csv",
            mime="text/csv"
        )

    # Option to process a new video
    if st.button("Process New Video"):
        for key in [
            'processed', 'output_video_path', 'csv_output_path', 'frames_written',
            'video_bytes', 'csv_bytes', 'annotated_video', 'metrics_csv']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
