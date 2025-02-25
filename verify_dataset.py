import cv2
import os
from config import Config

# Path to dataset
image_folder = "./datasets/val/images/"
label_folder = "./datasets/val/labels/"

# Get all image filenames
image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    label_path = os.path.join(label_folder, image_file.replace(".png", ".txt"))
    
    # Skip if label file doesn't exist
    if not os.path.exists(label_path):
        continue

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        continue

    h, w, _ = img.shape

    # Read label file
    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        data = line.strip().split()
        class_id = int(data[0])
        x_center, y_center, width, height = map(float, data[1:])

        # Convert from YOLO format to pixel values
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        # Draw bounding box and label with class-specific colors
        color = Config.COLORS[class_id]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, Config.CLASS_NAMES[class_id], (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Labeled Image", img)
    cv2.waitKey(100)  # Show for 0.1 sec

cv2.destroyAllWindows()
