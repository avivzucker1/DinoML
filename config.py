from datetime import datetime
from dataclasses import dataclass
from enum import Enum, auto

class Config:
    # Model detection settings
    MODEL_PATH = 'runs/detect/train2/weights/best.pt'  # Path to trained YOLO model
    
    # Object visualization colors (in BGR format)
    COLORS = {
        0: (0, 0, 255),    # Red for dino
        1: (0, 255, 0),    # Green for cactus
        2: (255, 0, 0)     # Blue for bird
    }
    
    # Class names for object detection
    CLASS_NAMES = ["dino", "cactus", "bird"]
    
    # Game window area settings (for screenshot capture). Used to crop the game area in order to make the predictions faster.
    # Did it because I run it on a laptop without a GPU. You may consider removing it if you are running it on a machine with a GPU.
    GAME_AREA_TOP = 140     # Y-coordinate where game area starts
    GAME_AREA_BOTTOM = 260  # Y-coordinate where game area ends
    
    # Video recording configuration
    ENABLE_VIDEO_CAPTURE = False  # Default to off
    VIDEO_CONFIG = {
        'fps': 30,         # Frames per second for recorded video
        'filename': f'game_recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'  # Dynamic filename with timestamp
    }
    
    # Screenshot capture settings
    SCREENSHOT_FOLDER = 'screenshots'  # Directory to store captured screenshots
    FAST_CAPTURE_COUNT = 5            # Number of screenshots to take in burst mode
    CAPTURE_INTERVAL = 0.05           # Time between screenshots (in seconds)
    
    # Screenshot filename markers
    JUMP_SUCCESS_MARKER = '_jump_success'  # Added to filename when jump was successful
    JUMP_FAILURE_MARKER = '_jump_failure'  # Added to filename when jump failed
    
    # Obstacle detection settings
    OBSTACLE_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence score for obstacle detection
    MAX_TRACKING_DISTANCE = 50  # Maximum pixel distance to consider same object between frames
    GROUP_DISTANCE_THRESHOLD = 15  # Maximum distance between obstacles to consider them as a group
    SPEED_HISTORY_SIZE = 10  # Number of recent speed measurements to keep for averaging 
    
    # Object detection metrics
    DETECTION_METRIX_INDEX = {
        'TIMESTAMP': 0,  # Index for timestamp in object tuples
        'FILENAME': 1,   # Index for filename in object tuples
        'X1': 2,        # Index for left x coordinate
        'X2': 3,        # Index for right x coordinate
        'Y1': 4,        # Index for top y coordinate
        'Y2': 5         # Index for bottom y coordinate
    }
    
    # Jump prediction training settings
    TRAINING_START_FILE = "2025_02_17_18_18_40_740.png"  # First file to process in training (in case of multiple iterations)
    TRAINING_RESULTS_FILE = 'jump_results.csv'  # CSV file to store training results
    TRAINING_HEADERS = [
        'obstacle_speed', 
        'obstacle_avg_speed', 
        'distance_to_obstacle', 
        'obstacle_width', 
        'obstacle_elevation', 
        'success_or_fail'
    ]

class MetricKeys(Enum):
    SPEED = 'speed'
    AVG_SPEED = 'avg_speed'
    DISTANCE_TO_DINO = 'distance_to_dino'
    Y1 = 'y1'
    Y2 = 'y2'
    GROUP_WIDTH = 'group_width'

@dataclass
class ObstacleMetrics:
    speed: float
    avg_speed: float
    distance_to_dino: float
    y1: float
    y2: float
    group_width: float 