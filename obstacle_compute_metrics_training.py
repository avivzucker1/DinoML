# This script processes recorded gameplay screenshots to generate training data for the jump prediction model.
# It analyzes sequences of screenshots to:
# 1. Detect obstacles and the dino
# 2. Track obstacles between frames to compute their speed
# 3. Calculate metrics like distance to dino, obstacle width, and speed
# 4. Record whether each jump was successful
# The output is a CSV file containing these metrics along with the jump success/failure label.

import os
from ultralytics import YOLO
import cv2
from datetime import datetime
import re
import csv
import torch
from config import Config

class SpeedAnalyzer:
    # Analyzes gameplay screenshots to generate training data.
    # Processes sequences of screenshots to compute obstacle metrics and record jump outcomes.
    
    def __init__(self):
        # Initialize YOLO model and create output CSV file
        self.model = YOLO(Config.MODEL_PATH)
        self.class_names = Config.CLASS_NAMES
        
        # Create/overwrite CSV file for storing training data
        # Each row will contain metrics for one obstacle when a jump occurred
        csv_path = os.path.join(Config.SCREENSHOT_FOLDER, Config.TRAINING_RESULTS_FILE)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(Config.TRAINING_HEADERS)
    
    def parse_timestamp(self, filename):
        # Extract timestamp from screenshot filename
        # Used to order screenshots chronologically and compute time differences
        match = re.match(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{3})', filename)
        if match:
            return datetime.strptime(match.group(1), '%Y_%m_%d_%H_%M_%S_%f')
        return None
    
    def track_objects(self, prev_objects, curr_objects, max_distance=50):
        # Match objects between consecutive frames to track their movement
        # This allows us to compute obstacle speeds and ensure we're tracking the same obstacle
        tracked = []
        
        # Sort by x-position to process left-to-right
        sorted_curr_objects = sorted(curr_objects, key=lambda obj: obj[Config.DETECTION_METRIX_INDEX['X1']])
        
        for curr_obj in sorted_curr_objects:
            best_match = None
            min_dist = max_distance
            
            # Find matching object from previous frame based on position
            for prev_obj in prev_objects:
                dist = abs(curr_obj[Config.DETECTION_METRIX_INDEX['X1']] - prev_obj[Config.DETECTION_METRIX_INDEX['X1']])
                if dist < min_dist:
                    min_dist = dist
                    best_match = prev_obj
            
            if best_match:
                tracked.append((curr_obj, best_match))
                prev_objects.remove(best_match)
        
        return tracked
        
    def compute_metrics(self):
        # Main function that processes all screenshots to generate training data
        # Looks for screenshots with jump markers to record training examples
        
        # Start processing from specified file (useful for multiple training sessions)
        start_timestamp = self.parse_timestamp(Config.TRAINING_START_FILE)
        
        # Get all screenshots and sort by timestamp
        files = []
        for filename in os.listdir(Config.SCREENSHOT_FOLDER):
            if filename.endswith('.png'):
                timestamp = self.parse_timestamp(filename)
                if timestamp and timestamp >= start_timestamp:
                    files.append((timestamp, filename))
        files.sort()
        
        # Initialize tracking dictionaries
        prev_frame_objects = {}  # Previous frame's objects for tracking
        obstacle_metrics = {}    # Computed metrics for each obstacle
        recent_speeds = {}      # Speed history for running averages
        
        # Process each screenshot in chronological order
        for timestamp, filename in files:
            # Check if this frame contains a jump and its outcome
            is_jump_success = Config.JUMP_SUCCESS_MARKER in filename
            is_jump_failure = Config.JUMP_FAILURE_MARKER in filename
            is_jump = is_jump_success or is_jump_failure
            
            # Load and process screenshot
            img_path = os.path.join(Config.SCREENSHOT_FOLDER, filename)
            img = cv2.imread(img_path)
            game_area = img[Config.GAME_AREA_TOP:Config.GAME_AREA_BOTTOM, :]
            
            # Run object detection
            with torch.no_grad():
                results = self.model(game_area)[0]
            
            # Process detected objects
            current_frame_objects = {}
            for detection in results.boxes.data:
                x1, y1, x2, y2, conf, cls = detection
                cls = int(cls)
                
                if conf < Config.OBSTACLE_CONFIDENCE_THRESHOLD:
                    continue
                
                if cls == 0:
                    dino_x2 = x2
                    continue
                
                x1, x2 = float(x1), float(x2)
                y1, y2 = float(y1), float(y2)
                
                # Group objects by class
                if cls not in current_frame_objects:
                    current_frame_objects[cls] = []
                current_frame_objects[cls].append((timestamp, filename, x1, x2, y1, y2))
            
            # Compare with previous frame to compute metrics
            for cls in current_frame_objects:
                # Initialize tracking for new classes
                if cls not in prev_frame_objects:
                    prev_frame_objects[cls] = current_frame_objects[cls]
                    continue
                
                # Match objects between frames based on position
                tracked_pairs = self.track_objects(prev_frame_objects[cls], 
                                                current_frame_objects[cls])
                
                # Filter pairs where current object is ahead of dino
                valid_pairs = [pair for pair in tracked_pairs if pair[0][Config.DETECTION_METRIX_INDEX['X1']] > dino_x2]

                # If no valid pairs, reset recent speeds
                if not valid_pairs:
                    recent_speeds[cls] = []

                if valid_pairs:
                    sorted_pairs = sorted(valid_pairs, key=lambda pair: pair[0][Config.DETECTION_METRIX_INDEX['X1']])
                    curr_obj, prev_obj = valid_pairs[0]
                    
                    time_diff = (curr_obj[Config.DETECTION_METRIX_INDEX['TIMESTAMP']] - 
                               prev_obj[Config.DETECTION_METRIX_INDEX['TIMESTAMP']]).total_seconds()
                    if time_diff > 0:
                        # Objects move right to left, so previous - current
                        distance = (prev_obj[Config.DETECTION_METRIX_INDEX['X1']] - 
                                  curr_obj[Config.DETECTION_METRIX_INDEX['X1']])
                        speed = distance / time_diff
                        
                        # Store valid speeds (moving left)
                        if speed > 0:
                            if cls not in obstacle_metrics:
                                obstacle_metrics[cls] = []
                            
                            # For jump frames, include distance to dino and y2
                            if is_jump:
                                distance_to_dino = curr_obj[Config.DETECTION_METRIX_INDEX['X1']] - dino_x2
                                
                                # Calculate group width
                                group_width = (curr_obj[Config.DETECTION_METRIX_INDEX['X2']] - 
                                             curr_obj[Config.DETECTION_METRIX_INDEX['X1']])
                                
                                if len(valid_pairs) > 1:
                                    group_x1 = curr_obj[Config.DETECTION_METRIX_INDEX['X1']]
                                    group_x2 = curr_obj[Config.DETECTION_METRIX_INDEX['X2']]
                                    
                                    for next_pair in sorted_pairs[1:]:
                                        next_x1 = next_pair[0][Config.DETECTION_METRIX_INDEX['X1']]
                                        if next_x1 - group_x2 < Config.GROUP_DISTANCE_THRESHOLD:
                                            group_x2 = next_pair[0][Config.DETECTION_METRIX_INDEX['X2']]
                                        else:
                                            break
                                    
                                    if group_x2 > group_x1:
                                        group_width = group_x2 - group_x1
                                
                                # Update running average of speeds
                                if cls not in recent_speeds:
                                    recent_speeds[cls] = []
                                recent_speeds[cls].append(speed)
                                recent_speeds[cls] = recent_speeds[cls][-Config.SPEED_HISTORY_SIZE:]
                                avg_speed = sum(recent_speeds[cls]) / len(recent_speeds[cls])
                                
                                # {class_id: [(timestamp, filename, speed, avg_speed, distance_to_dino, y2, group_width, is_jump), ...]}
                                obstacle_metrics[cls].append((curr_obj[Config.DETECTION_METRIX_INDEX['TIMESTAMP']], curr_obj[Config.DETECTION_METRIX_INDEX['FILENAME']], 
                                                            speed, avg_speed, distance_to_dino, 
                                                            curr_obj[Config.DETECTION_METRIX_INDEX['Y2']], group_width, is_jump_success))
                                
                                print(f"Obstacle {cls} in {curr_obj[Config.DETECTION_METRIX_INDEX['FILENAME']]} with speed {speed} pixels/second, "
                                      f"running avg speed {avg_speed:.2f}, distance {distance_to_dino} pixels, "
                                      f"y2 {curr_obj[Config.DETECTION_METRIX_INDEX['Y2']]}, obstacle width: {group_width}")
                                
                                # Write training example to CSV
                                csv_path = os.path.join(Config.SCREENSHOT_FOLDER, Config.TRAINING_RESULTS_FILE)
                                with open(csv_path, 'a', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([
                                        int(speed),                  # Current speed of obstacle
                                        int(avg_speed),             # Running average of speed
                                        int(distance_to_dino),      # Distance from dino to obstacle
                                        int(group_width),           # Total width of obstacle group
                                        int(curr_obj[Config.DETECTION_METRIX_INDEX['Y2']]),  # Vertical position
                                        int(is_jump_success)        # Whether the jump was successful
                                    ])
                            else:
                                # For non-jump frames, store None for avg_speed and group_width
                                obstacle_metrics[cls].append((curr_obj[Config.DETECTION_METRIX_INDEX['TIMESTAMP']], curr_obj[Config.DETECTION_METRIX_INDEX['FILENAME']], 
                                                            speed, None, None, curr_obj[Config.DETECTION_METRIX_INDEX['Y2']], None, None))
                
                # Update previous frame objects for next iteration
                prev_frame_objects[cls] = current_frame_objects[cls]
        
if __name__ == "__main__":
    analyzer = SpeedAnalyzer()
    analyzer.compute_metrics()
