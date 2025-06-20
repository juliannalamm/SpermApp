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
demo_videos = [f for f in os.listdir(demo_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
demo_videos.insert(0, "None (upload your own)")

st.title("YOLO Sperm Tracking App")
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

# Demo video selector
demo_choice = st.selectbox("Or select a demo video:", demo_videos)

# File uploader

# Determine which video to use
video_path = None


if demo_choice != "None (upload your own)":
    video_path = os.path.join(demo_dir, demo_choice)
elif uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

# Show preview if a demo is selected
start = st.button("Start Processing")  # Add a Start button

# Show preview if a demo is selected and processing hasn't started
if demo_choice != "None (upload your own)" and not st.session_state.processed:
    st.video(video_path)

# After processing, always show the last processed video (if available)
if st.session_state.processed and st.session_state.video_bytes:
    st.video(st.session_state.video_bytes)

# Only process if Start is pressed and a video is selected
if start and video_path:
    # Load model
    model = YOLO("train_experiments/031025/run12/weights/best.pt")
    tracker = "custom_trackers/custom_botsort.yaml"

    # Create progress bar
    progress_bar = st.progress(0)

    # Process video
    with st.spinner('Processing video...'):
        (st.session_state.output_video_path, 
         st.session_state.csv_output_path, 
         st.session_state.frames_written,
         st.session_state.video_bytes,
         st.session_state.csv_bytes) = process_video(
            video_path, model, tracker, progress_bar
        )
        st.session_state.processed = True

    st.success(f"Processing complete! {st.session_state.frames_written} frames processed.")

    if st.session_state.video_bytes:
        st.video(st.session_state.video_bytes)
    else:
        st.warning("Video could not be loaded.")

    # Add a button to process a new video
    if st.session_state.processed:
        col1, col2 = st.columns(2)
    
    col1.download_button(
        label="Download Annotated Video",
        data=st.session_state.video_bytes,  
        file_name="annotated_video.mp4",
        mime="video/mp4"
    )
    col2.download_button(
        label="Download Tracking Data (CSV)",
        data=st.session_state.csv_bytes,
        file_name="tracked_coordinates.csv",
        mime="text/csv"
    )
        
    if st.button("Process New Video"):
        st.session_state.processed = False
        st.rerun()
        

# -------------------------------
# Step 2: Optional Classification
# -------------------------------
if st.session_state.processed:
    st.markdown("---")
    st.header("Optional: Classify and Calculate Kinematic Metrics")

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

    if st.button("Classify and Calculate Kinematic Metrics"):
        with st.spinner("Classifying motility and generating annotated video..."):
            annotated_video, metrics_csv = process_csv_metrics(
                tracked_csv_bytes=st.session_state.csv_bytes,
                original_video_path=video_path,
                image_j_scale=image_j_scale,
                total_duration_seconds=total_duration_seconds,
                img_w=img_w,img_h=img_h
                )

        st.video(annotated_video)

        col1, col2 = st.columns(2)
        col1.download_button(
            label="Download Kinematic Metrics CSV",
            data=metrics_csv,
            file_name="kinematic_metrics.csv",
            mime="text/csv"
        )
        col2.download_button(
            label="Download Motility Classification Video",
            data=annotated_video,
            file_name="motility_classified_video.mp4",
            mime="video/mp4"
        )
