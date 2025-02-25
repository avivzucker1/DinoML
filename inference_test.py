from ultralytics import YOLO
import cv2
import numpy as np
import os
from config import Config
# Load the trained model
model = YOLO(Config.MODEL_PATH)  # path to your trained weights

# Class names
class_names = Config.CLASS_NAMES

# Load and run inference on an image
def process_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return

    # Run inference
    results = model(img)[0]  # returns a list of Results objects

    # Plot results on image
    for result in results.boxes.data:
        x1, y1, x2, y2, conf, cls = result
        
        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Get color for this class
        class_idx = int(cls)
        color = Config.COLORS[class_idx]
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Add label with confidence
        label = f"{class_names[class_idx]} {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show image
    cv2.imshow("Detection Result", img)
    # Wait for 500ms, return True if 'q' is pressed
    key = cv2.waitKey(100)
    return key == ord('q')

def process_directory(directory_path):
    # Get all image files in directory
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Processing {len(image_files)} images in {directory_path}")
    
    # Process each image
    for image_file in sorted(image_files):
        image_path = os.path.join(directory_path, image_file)
        print(f"Processing: {image_file}")
        
        # If process_image returns True (q pressed), stop processing
        if process_image(image_path):
            return True
    
    return False

# Process training and validation directories
directories = [
    "./datasets/train/images",
    "./datasets/val/images"
]

# Process each directory
for directory in directories:
    print(f"\nProcessing directory: {directory}")
    # If user pressed 'q', stop processing
    if process_directory(directory):
        break

cv2.destroyAllWindows()