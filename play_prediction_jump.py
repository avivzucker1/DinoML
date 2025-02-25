import cv2
import keyboard
import sys
import win32gui
import win32ui
import win32con
from ctypes import windll
import time
import os
import torch
from screen_capture_service import capture_active_window
from obstacle_compute_metrics_service import SpeedAnalyzer
import joblib
import pandas as pd
import win32api
from datetime import datetime
from config import Config

# Configuration
CONFIG = {
    'model_type': 'gb',  # Model type for jump prediction: 'rf' (Random Forest), 'gb' (Gradient Boosting), 'dt' (Decision Tree)
    'confidence_threshold': 0.92
}

# Initialize analyzer
analyzer = SpeedAnalyzer()
prev_frame_objects = None
recent_speeds = {}

# Load trained model based on configuration
model_path = f"models/{CONFIG['model_type']}_classifier.joblib"
print(f"Loading {CONFIG['model_type']} model from {model_path}")
model = joblib.load(model_path)
metadata = joblib.load('models/model_metadata.joblib')
print(f"Model {CONFIG['model_type']} loaded successfully")

def create_video_writer(frame):
    if not Config.ENABLE_VIDEO_CAPTURE:
        return None
        
    height, width = frame.shape[:2]
    # Create fourcc code from codec string
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        Config.VIDEO_CONFIG['filename'],
        fourcc,  # Use fourcc instead of codec string
        Config.VIDEO_CONFIG['fps'],
        (width, height)
    )
    print(f"Recording video to: {Config.VIDEO_CONFIG['filename']}")
    return writer

def predict_jump(metrics):
    if not metrics:
        return False
    
    # Find the closest obstacle's metrics
    closest_obstacle = None
    min_distance = float('inf')
    
    for cls, metric in metrics.items():
        distance = metric.distance_to_dino
        if distance < min_distance:
            min_distance = distance
            closest_obstacle = metric
    
    if not closest_obstacle:
        return False
        
    effective_speed = max(closest_obstacle.speed, closest_obstacle.avg_speed)
    time_to_impact = closest_obstacle.distance_to_dino / effective_speed
    clearance_ratio = closest_obstacle.distance_to_dino / closest_obstacle.group_width
    safety_margin = (closest_obstacle.distance_to_dino - effective_speed * 0.5) / closest_obstacle.group_width
    weighted_time = time_to_impact * (100 / effective_speed)
    jump_urgency = (effective_speed ** 2) / (closest_obstacle.distance_to_dino * 100)
    
    features = pd.DataFrame([{
        'obstacle_speed': closest_obstacle.speed,
        'obstacle_avg_speed': closest_obstacle.avg_speed,
        'distance_to_obstacle': closest_obstacle.distance_to_dino,
        'obstacle_width': closest_obstacle.group_width,
        'obstacle_elevation': closest_obstacle.y2,
        'time_to_impact': time_to_impact,
        'clearance_ratio': clearance_ratio,
        'safety_margin': safety_margin,
        'weighted_time': weighted_time,
        'jump_urgency': jump_urgency
    }])
    
    prediction = model.predict_proba(features)[0]
    confidence = prediction[1]
    should_jump = confidence > CONFIG['confidence_threshold']
    
    return should_jump

def create_monitor_window():
    # Create window without decorations
    cv2.namedWindow("Game Monitor", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    
    # Get game window position and size
    game_hwnd = win32gui.GetForegroundWindow()
    game_rect = win32gui.GetWindowRect(game_hwnd)
    client_rect = win32gui.GetClientRect(game_hwnd)
    game_x, game_y, game_right, game_bottom = game_rect
    
    # Get actual game client area size
    client_width = client_rect[2] - client_rect[0]
    client_height = client_rect[3] - client_rect[1]
    
    # Get screen size
    screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    
    # Get first frame to determine size
    first_frame = capture_active_window()
    if first_frame is not None:
        # Position monitor window to the right of game window if there's space,
        # otherwise position it to the left
        monitor_x = game_right + 10  # Try right first
        if monitor_x + client_width > screen_width:
            monitor_x = game_x - client_width - 10  # Try left instead
        
        # Remove window decorations first
        hwnd = win32gui.FindWindow(None, "Game Monitor")
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
        style &= ~(win32con.WS_CAPTION | win32con.WS_THICKFRAME)
        win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)
        
        # Set exact window size and position
        win32gui.SetWindowPos(hwnd, None, 
                            monitor_x, game_y,
                            client_width, client_height,
                            win32con.SWP_NOZORDER)
        
    return first_frame

# Initialize monitor window and video writer only if video capture is enabled
if Config.ENABLE_VIDEO_CAPTURE:
    print("Video capture enabled, creating monitor window...")
    initial_frame = create_monitor_window()
    if initial_frame is None:
        print("Failed to create monitor window")
        sys.exit(1)
    video_writer = create_video_writer(initial_frame)
else:
    print("Video capture disabled, skipping monitor window creation")
    initial_frame = None
    video_writer = None
print("Start playing...")

def show_detections(frame, objects, metrics, jump_prediction):
    # Skip visualization if video capture is disabled
    if not Config.ENABLE_VIDEO_CAPTURE:
        return
        
    if frame is None:
        return
        
    # Make a copy for drawing
    vis_frame = frame.copy()
    
    # Draw boxes for each detected object
    for cls in objects:
        color = Config.COLORS[cls]
        for obj in objects[cls]:
            x1 = int(float(obj[Config.DETECTION_METRIX_INDEX['X1']]))
            x2 = int(float(obj[Config.DETECTION_METRIX_INDEX['X2']]))
            y1 = int(float(obj[Config.DETECTION_METRIX_INDEX['Y1']])) + Config.GAME_AREA_TOP
            y2 = int(float(obj[Config.DETECTION_METRIX_INDEX['Y2']])) + Config.GAME_AREA_TOP
            
            # Draw rectangle and label with thinner lines
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), Config.COLORS[cls], 1)
            cv2.putText(vis_frame, Config.CLASS_NAMES[cls], 
                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, color, 1)
            
            # Add time_to_impact and distance labels for obstacles
            if cls != 0 and metrics and cls in metrics:  # If not dino and metrics exist
                metric = metrics[cls]
                effective_speed = max(metric.speed, metric.avg_speed)
                time_to_impact = metric.distance_to_dino / effective_speed
                
                # Position labels under the dino
                dino_x = 50  # Approximate dino x position
                label_y = y2 + 70  # Start 70 pixels below the box (was 20)
                
                # Draw time to impact
                time_text = f"Time to impact: {time_to_impact:.2f}s"
                cv2.putText(vis_frame, time_text,
                           (dino_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, (0, 0, 0), 1)  # Black color
                
                # Draw distance
                dist_text = f"Distance to obstacle: {metric.distance_to_dino:.1f}px"
                cv2.putText(vis_frame, dist_text,
                           (dino_x, label_y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, (0, 0, 0), 1)  # Black color

                # Draw obstacle speed
                speed_text = f"Obstacle speed: {metric.speed:.1f}px/s"
                cv2.putText(vis_frame, speed_text,
                           (dino_x, label_y + 30), cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, (0, 0, 0), 1)  # Black color
                
                # Draw obstacle width
                width_text = f"Obstacle width: {metric.group_width:.1f}px"
                cv2.putText(vis_frame, width_text,
                           (dino_x, label_y + 45), cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, (0, 0, 0), 1)  # Black color
                
                # Draw jump prediction
                jump_text = "Prediction: JUMP!" if jump_prediction else "Prediction: No jump"
                jump_color = (0, 0, 255) if jump_prediction else (0, 0, 0)  # Red if jump, black if no jump
                cv2.putText(vis_frame, jump_text,
                           (dino_x, label_y + 60), cv2.FONT_HERSHEY_SIMPLEX,  # Was +30
                           0.4, jump_color, 1)
    
    # Show frame in monitor window
    cv2.imshow("Game Monitor", vis_frame)
    
    # Only write video if enabled and writer exists
    if video_writer is not None:
        video_writer.write(vis_frame)
    
    cv2.waitKey(1)

while True:
    if keyboard.is_pressed('q'):
        print("Quitting...")
        break

    try:
        # 1. Capture frame
        frame = capture_active_window()
        
        # Check if frame is valid
        if frame is None or frame.size == 0 or 0 in frame.shape:
            print("Invalid frame captured, skipping...")
            continue
            
        # 2. Get metrics from analyzer
        current_frame_objects, metrics = analyzer.compute_metrics(frame, prev_frame_objects, recent_speeds)
        prev_frame_objects = current_frame_objects

        # 3. Predict jump
        jump_prediction = predict_jump(metrics)
        
        if jump_prediction:
            keyboard.press('space')
        
        # Show detections in monitor window (only if video capture is enabled)
        if current_frame_objects and Config.ENABLE_VIDEO_CAPTURE:
            show_detections(frame, current_frame_objects, metrics, jump_prediction)

    except KeyboardInterrupt:
        print("Interrupted by user")
        break

print("Exiting...")

# At the end
if video_writer is not None:
    video_writer.release()
if Config.ENABLE_VIDEO_CAPTURE:
    cv2.destroyAllWindows()
