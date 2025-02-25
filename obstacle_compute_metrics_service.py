# This service analyzes game frames to detect and track obstacles, computing metrics needed for jump decisions.
# It uses YOLO for object detection and tracks objects between frames to calculate their speed and other properties.

from ultralytics import YOLO
from datetime import datetime
import torch
from config import Config
from config import ObstacleMetrics
import numpy as np
import cv2

class SpeedAnalyzer:
    # Main class for analyzing obstacles in the Chrome Dino game.
    # Detects objects (dino, cactus, bird), tracks them between frames,
    # and computes metrics like speed, distance, and group width (if obstacles are adjacent).
    
    def __init__(self):
        # Initialize YOLO model for object detection
        self.model = YOLO(Config.MODEL_PATH, verbose=False)
        self.class_names = Config.CLASS_NAMES
        
    def track_objects(self, prev_objects, curr_objects):
        # Matches objects between consecutive frames to track their movement.
        # Uses x-coordinate distance to determine which objects are the same between frames.
        tracked = []
        
        # Sort current objects by x1 position (leftmost first)
        sorted_curr_objects = sorted(curr_objects, key=lambda obj: obj[Config.DETECTION_METRIX_INDEX['X1']])
        
        for curr_obj in sorted_curr_objects:
            best_match = None
            min_dist = Config.MAX_TRACKING_DISTANCE
            
            # Find the closest object from previous frame based on x-position
            for prev_obj in prev_objects:
                # Calculate horizontal distance between objects
                dist = abs(curr_obj[Config.DETECTION_METRIX_INDEX['X1']] - prev_obj[Config.DETECTION_METRIX_INDEX['X1']])
                
                # Update best match if this distance is smaller
                if dist < min_dist:
                    min_dist = dist
                    best_match = prev_obj
            
            if best_match:
                tracked.append((curr_obj, best_match))
                prev_objects.remove(best_match)
        
        return tracked
        
    def get_frame_objects(self, frame):
        # Process a single frame to detect objects (dino, cactus, bird)
        # Returns a dictionary of detected objects by class and dino's right edge position
        current_frame_objects = {}  # Store current frame's objects by class
        dino_x2 = None
        timestamp = datetime.now()
        
        # Run detection on game area only to improve performance
        game_area = frame[Config.GAME_AREA_TOP:Config.GAME_AREA_BOTTOM, :]
        
        with torch.no_grad():
            results = self.model(game_area, verbose=False)[0]  # Set verbose=False to reduce output
        
        # Process each detected object
        for detection in results.boxes.data:
            x1, y1, x2, y2, conf, cls = detection
            cls = int(cls)

            # Filter low confidence detections
            if conf < Config.OBSTACLE_CONFIDENCE_THRESHOLD:
                continue
            
            # Store dino's right edge position separately
            if cls == 0:  # Dino
                dino_x2 = float(x2)
                continue
            
            # Convert coordinates to float for precise calculations
            x1, x2 = float(x1), float(x2)
            y1, y2 = float(y1), float(y2)
            
            # Group objects by class
            if cls not in current_frame_objects:
                current_frame_objects[cls] = []
            current_frame_objects[cls].append((timestamp, None, x1, x2, y1, y2))
        
        return current_frame_objects, dino_x2
    
    def compute_metrics(self, frame, prev_frame_objects=None, recent_speeds=None):
        # Main function to compute metrics for jump decisions
        # Analyzes current frame and compares with previous frame to calculate:
        # - Object speeds
        # - Distances to dino
        # - Group widths (for multiple adjacent obstacles)
        # - Running average speeds
        
        if recent_speeds is None:
            recent_speeds = {}
        
        # Get current frame objects and dino position
        current_frame_objects, dino_x2 = self.get_frame_objects(frame)
        
        if dino_x2 is None or not prev_frame_objects:
            return current_frame_objects, {}
        
        metrics = {}
        
        # Compare with previous frame for each class to compute speeds and metrics
        for cls in current_frame_objects:
            if cls not in prev_frame_objects:
                continue
            
            # Match objects between frames to track movement
            tracked_pairs = self.track_objects(prev_frame_objects[cls], current_frame_objects[cls])
            
            # Only consider objects ahead of the dino
            valid_pairs = [pair for pair in tracked_pairs if pair[0][Config.DETECTION_METRIX_INDEX['X1']] > dino_x2]
            if valid_pairs:
                # Process the closest obstacle to the dino
                sorted_pairs = sorted(valid_pairs, key=lambda pair: pair[0][Config.DETECTION_METRIX_INDEX['X1']])
                curr_obj, prev_obj = valid_pairs[0]
                
                # Calculate speed based on position change over time
                time_diff = (curr_obj[Config.DETECTION_METRIX_INDEX['TIMESTAMP']] - prev_obj[Config.DETECTION_METRIX_INDEX['TIMESTAMP']]).total_seconds()
                if time_diff > 0:
                    # Objects move right to left, so previous - current
                    distance = prev_obj[Config.DETECTION_METRIX_INDEX['X1']] - curr_obj[Config.DETECTION_METRIX_INDEX['X1']]
                    speed = distance / time_diff  # pixels per second
                    
                    if speed > 0:
                        # Calculate distance to dino and other metrics
                        distance_to_dino = curr_obj[Config.DETECTION_METRIX_INDEX['X1']] - dino_x2
                        
                        # Calculate total width of obstacle group
                        group_width = curr_obj[Config.DETECTION_METRIX_INDEX['X2']] - curr_obj[Config.DETECTION_METRIX_INDEX['X1']]
                        if len(valid_pairs) > 1:
                            # Check for adjacent obstacles that form a group
                            sorted_pairs = sorted(valid_pairs, key=lambda pair: pair[0][Config.DETECTION_METRIX_INDEX['X1']])
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
                        
                        # Update running average of speeds for smoother predictions
                        if cls not in recent_speeds:
                            recent_speeds[cls] = []
                        recent_speeds[cls].append(speed)
                        recent_speeds[cls] = recent_speeds[cls][-Config.SPEED_HISTORY_SIZE:]
                        avg_speed = sum(recent_speeds[cls]) / len(recent_speeds[cls])
                        
                        # Store all computed metrics for this obstacle
                        metrics[cls] = ObstacleMetrics(
                            speed=speed,
                            avg_speed=avg_speed,
                            distance_to_dino=distance_to_dino,
                            y1=curr_obj[Config.DETECTION_METRIX_INDEX['Y1']],
                            y2=curr_obj[Config.DETECTION_METRIX_INDEX['Y2']],
                            group_width=group_width
                        )
        
        return current_frame_objects, metrics

