import pandas as pd
import numpy as np 
import cv2
import tempfile
import os 
import joblib
import matplotlib.pyplot as plt
from matplotlib import cm
import io
import subprocess

def compute_centers(df,pixel_size):
    df['center_x_pixels']= (df['x1']+ df['x2'])/2 
    df['center_y_pixels']= (df['y1']+ df['y2'])/2 
    df['center_x_um']= df['center_x_pixels']*pixel_size
    df['center_y_um']= df['center_y_pixels']*pixel_size
    return df

def compute_frame_and_deltas(df):
  df['frame_number'] = df['frame'].astype(int)
  df = df.sort_values(by=['id','frame_number'])
  df['delta_x'] = df.groupby('id')['center_x_um'].diff().fillna(0)
  df['delta_y'] = df.groupby('id')['center_y_um'].diff().fillna(0)
  df.drop(columns=['frame'], inplace=True)
  return df

def calc_VSL(df,total_duration_seconds):
  total_frames = df['frame_number'].nunique()
  frame_rate = total_frames/total_duration_seconds
  last_frames = df.groupby('id').last()
  first_frames = df.groupby('id').first()
  displacement = abs(last_frames[['center_x_um','center_y_um']] - first_frames[['center_x_um','center_y_um']])
  displacement.columns = ['x_displacement','y_displacement']
  total_time_in_frame = (last_frames['frame_number'] - first_frames['frame_number'])/frame_rate
  total_displacement = np.sqrt(displacement['x_displacement']**2 + displacement['y_displacement']**2)
  VSL_df = pd.DataFrame({
      'total_displacement': total_displacement,
      'VSL' : total_displacement/total_time_in_frame,
  })
  return VSL_df

def calc_VCL(df, total_duration_seconds):
    df['delta_x'] = df.groupby('id')['center_x_um'].diff().fillna(0)
    df['delta_y'] = df.groupby('id')['center_y_um'].diff().fillna(0)

    total_frames = df['frame_number'].nunique()
    frame_rate = total_frames / total_duration_seconds

    first_frames = df.groupby('id').first()
    last_frames = df.groupby('id').last()
  # does a sperm still exist if you cant see it? yes. 
    total_time_in_frame = (last_frames['frame_number'] - first_frames['frame_number']) / frame_rate 

    df['distance'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2)
    total_distance_by_id = df.groupby('id')['distance'].sum()

    VCL_df = pd.DataFrame({
        'total_distance': total_distance_by_id,
        'VCL': total_distance_by_id / total_time_in_frame
    })
    return VCL_df

def calc_VAP(df, total_duration_seconds, window_size):
    df['center_x_um_smooth'] = df.groupby('id')['center_x_um'].transform(lambda x: x.rolling(window=window_size, center=True).mean())
    df['center_y_um_smooth'] = df.groupby('id')['center_y_um'].transform(lambda x: x.rolling(window=window_size, center=True).mean())
    df['delta_x_smooth'] = df.groupby('id')['center_x_um_smooth'].diff().fillna(0)
    df['delta_y_smooth'] = df.groupby('id')['center_y_um_smooth'].diff().fillna(0)
    df['distance_smooth'] = np.sqrt(df['delta_x_smooth']**2 + df['delta_y_smooth']**2)

    total_frames = df['frame_number'].nunique()
    frame_rate = total_frames / total_duration_seconds

    first_frames = df.groupby('id').first()
    last_frames = df.groupby('id').last()
    total_time_by_id = (last_frames['frame_number'] - first_frames['frame_number']) / frame_rate

    distance_sum_smooth = df.groupby('id')['distance_smooth'].sum()

    VAP_df = pd.DataFrame({
        'VAP': distance_sum_smooth / total_time_by_id
    })
    return VAP_df

def moving_average(series,window_size = 3):
    return series.rolling(window=window_size, min_periods=1, center=True).mean()

def apply_moving_average(df, window_size = 3):
    df['avg_center_x_um'] = df.groupby('id')['center_x_um'].transform(lambda x: moving_average(x, window_size))
    df['avg_center_y_um'] = df.groupby('id')['center_y_um'].transform(lambda x: moving_average(x, window_size))
    return df

def distance_between_segments(seg1_start, seg1_end, seg2_start, seg2_end):
    midpoint1 = ((seg1_start[0] + seg1_end[0]) / 2, (seg1_start[1] + seg1_end[1]) / 2)
    midpoint2 = ((seg2_start[0] + seg2_end[0]) / 2, (seg2_start[1] + seg2_end[1]) / 2)
    return np.sqrt((midpoint1[0] - midpoint2[0])**2 + (midpoint1[1] - midpoint2[1])**2)

def calculate_risers(df, wSize):
    df['riser'] = np.nan
    for id, group in df.groupby('id'):
        for j in range(len(group) - wSize - 1):
            min_dist = float('inf')
            for k in range(j, min(j + wSize - 1, len(group) - 1)):
                seg_actual_start = (group.iloc[k]['center_x_um'], group.iloc[k]['center_y_um'])
                seg_actual_end = (group.iloc[k + 1]['center_x_um'], group.iloc[k + 1]['center_y_um'])
                seg_avg_start = (group.iloc[j]['avg_center_x_um'], group.iloc[j]['avg_center_y_um'])
                seg_avg_end = (group.iloc[j + 1]['avg_center_x_um'], group.iloc[j + 1]['avg_center_y_um'])
                dist = distance_between_segments(seg_actual_start, seg_actual_end, seg_avg_start, seg_avg_end)
                min_dist = min(min_dist, dist)
            df.loc[group.index[j], 'riser'] = min_dist
    return df

def calculate_alh(df, pixel_size):
    alh_results = []
    
    for id, group in df.groupby('id'):
        alh_max = group['riser'].max()
        alh_mean = group['riser'].mean()
        alh_results.append({
            'id': id,
            'ALH Mean': 2 * alh_mean * pixel_size, 
            'ALH Max': 2 * alh_max * pixel_size
        })
    return pd.DataFrame(alh_results)


def process_csv_metrics(tracked_csv_path=None, tracked_csv_bytes=None, original_video_path=None, image_j_scale=None, total_duration_seconds=None, img_w=None, img_h=None, window_size=10):
    pixel_size = 1/image_j_scale
    if tracked_csv_bytes is not None:
        df = pd.read_csv(io.BytesIO(tracked_csv_bytes))
    elif tracked_csv_path is not None:
        df = pd.read_csv(tracked_csv_path)
    else:
        raise ValueError("Must provide either tracked_csv_path or tracked_csv_bytes")

    df = compute_centers(df, pixel_size)
    df = compute_frame_and_deltas(df)
    vsl_df = calc_VSL(df, total_duration_seconds)
    vcl_df = calc_VCL(df, total_duration_seconds)
    vap_df = calc_VAP(df, total_duration_seconds, window_size)
    df = apply_moving_average(df, window_size)
    df = calculate_risers(df, window_size)
    alh_df = calculate_alh(df, pixel_size)

    metrics_df = vsl_df.join([vcl_df, vap_df, alh_df.set_index('id')], how='inner')
    metrics_df = metrics_df.reset_index()
    metrics_df['LIN'] = (metrics_df['VSL'] / metrics_df['VCL']) * 100
    metrics_df['WOB'] = (metrics_df['VAP'] / metrics_df['VCL']) * 100
    metrics_df['STR'] = (metrics_df['VSL'] / metrics_df['VAP']) * 100
    metrics_df.replace([np.inf, -np.inf], 0, inplace=True)
    metrics_df.fillna(0, inplace=True)
    metrics_df = metrics_df[metrics_df['total_displacement'] >= 20]

    scaler = joblib.load("minmax_scaler.pkl")
    model = joblib.load("subcluster_model.pkl")
    features = metrics_df[['VCL', 'VAP', 'VSL', 'LIN', 'WOB', 'STR', 'ALH Mean', 'ALH Max']]
    X_scaled = scaler.transform(features)
    metrics_df['Cluster'] = model.predict(X_scaled)
    cluster_names = {0: "Intermediate Motility", 1: "Hyperactivated Motility", 2: "Progressive Motility"}
    metrics_df['Motility Class'] = pd.Series(metrics_df['Cluster']).map(cluster_names)

    cap = cv2.VideoCapture(original_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (img_w, img_h))
    color_map = cm.get_cmap('Set1', len(np.unique(metrics_df['Cluster'])))
    id_to_color = {
        row['id']: tuple((np.array(color_map(row['Cluster']))[:3] * 255).astype(np.uint8))
        for _, row in metrics_df.iterrows()
    }

    # --- Begin new video annotation logic ---
    # Prepare color map and cluster label map
    cluster_ids = pd.Series(metrics_df['Cluster']).unique()
    n_clusters = len(cluster_ids)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i / n_clusters) for i in range(n_clusters)]
    cluster_colors = {cluster_id: tuple(int(255 * x) for x in colors[i][:3][::-1]) for i, cluster_id in enumerate(cluster_ids)}
    cluster_label_map = {
        0: "Intermediate Motility",
        1: "Hyperactivated Motility",
        2: "Progressive Motility"
    }
    # Map id to cluster
    id_to_cluster = metrics_df.set_index('id')['Cluster'].to_dict()
    # Track history for polylines
    track_history = {}
    # Prepare frame_dict for fast lookup
    frame_dict = {f: [] for f in df['frame_number'].unique()}
    for _, row in df.iterrows():
        frame_dict[row['frame_number']].append(row)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_tracks = frame_dict.get(frame_idx, [])
        for row in current_tracks:
            track_id = row['id']
            if track_id in id_to_cluster:
                x = int(row['center_x_pixels'])
                y = int(row['center_y_pixels'])
                # Initialize track history if needed
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append((x, y))
                # Get cluster color
                cluster = id_to_cluster[track_id]
                color = cluster_colors[cluster]
                # Draw track history as polyline
                if len(track_history[track_id]) > 1:
                    points = np.array(track_history[track_id], np.int32)
                    points = points.reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], False, color, 2)
                # Draw current position as circle
                cv2.circle(frame, (x, y), 3, color, -1)
        # Draw legend (top-left corner)
        legend_x, legend_y = 5, 10
        for cluster_id in sorted(cluster_ids):
            color = cluster_colors[cluster_id]
            label = cluster_label_map.get(cluster_id, f"Cluster {cluster_id}")
            cv2.rectangle(frame, (legend_x, legend_y), (legend_x + 20, legend_y + 20), color, -1)
            cv2.putText(frame, label, (legend_x + 30, legend_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            legend_y += 30
        out.write(frame)
        frame_idx += 1
    # --- End new video annotation logic ---

    cap.release()
    out.release()

    # Re-encode to browser-safe format (H.264 in MP4 container)
    fixed_output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fixed_output_path = fixed_output_temp.name
    subprocess.run([
        "ffmpeg", "-y", "-i", video_output_path,
        "-vcodec", "libx264", "-acodec", "aac", fixed_output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    metrics_output_path = tempfile.NamedTemporaryFile(delete=False, suffix="_metrics.csv").name
    metrics_df.to_csv(metrics_output_path, index=False)

    with open(fixed_output_path, "rb") as f:
        annotated_video_bytes = f.read()
    with open(metrics_output_path, "rb") as f:
        metrics_csv_bytes = f.read()

    os.remove(video_output_path)
    os.remove(fixed_output_path)
    os.remove(metrics_output_path)

    return annotated_video_bytes, metrics_csv_bytes
