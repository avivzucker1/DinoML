# Chrome Dino Game AI Player

This project uses computer vision and machine learning to play the Chrome Dino game by predicting when to jump over obstacles.

## Setup Instructions

1. **Record Training Data for Obstacle Detection**
   - Use the `capture_screenshots.py` script to record training data. It has multiple options to record the data:
     - Continuous capture with pause/resume functionality ('c', 'p').
     - Single screenshot capture ('s').
     - Burst capture (multiple screenshots in a row) ('f').
     - Screenshots are saved to screenshots folder (configurable), with timestamps, postfix _jump if space was pressed.
   - Run the game and take snapshots with different obstacles. I took ~110 snapshots for the training. You can see a recorded dataset under datasets/[train/val]/images folder for reference.
   
2. **Label Training Data**
   - Install labelImg: `pip install labelImg`.
   - Open labelImg and load the screenshots folder.
   - Label three classes:
     - "dino": The player character.
     - "cactus": Ground obstacles.
     - "bird": Flying obstacles.
   - Save annotations in YOLO format.
   - Once you labeled all the images, split them into training and validation sets. You can see the labeled dataset under datasets/[train/val]/labels folder for reference.

3. **Verify Labeled Data**
   - Run `verify_dataset.py` to verify the labeled data. It will show you the images with the labels.

4. **Train Obstacle Detection**
   - Train YOLO model on the labeled data (`obstacle_detection_training.py`).
   - The model learns to detect the dino and obstacles.
   - Trained weights are saved to `runs/detect/train[#]/weights/best.pt`.
   - You can see the output of the trained model under runs/detect/. In my case I saved the last run (train2). \runs\detect\train2\weights\best.pt is the trained model and will be used for the jump prediction (configured in `config.py`).
   - To visualize the results run `inference_test.py`.

5. **Record Training Data for Jump Prediction**
   - Run `capture_screenshots.py` to record the training data for the jump prediction (see options above).
   - Play while you record, to capture jumps and failures, for multiple speeds.
   - Each screenshot has a timestamp, and is saved with a postfix _jump if space was pressed. You need to go over all jumps and classify and success or failure.
   - You can see sample recorded data under datasets/screenshots folder for reference. I used about 10K screenshots for the training... this was the tedious part.

6. **Train Jump Predictor**
   - Run `obstacle_compute_metrics_training.py` to process labeled screenshots.
   - This generates `jump_results.csv` with metrics for each jump. This is the training data for the jump predictor.
   - Run `jump_prediction_training.py` to train the jump prediction model.
   - Models are saved to the `models/` directory.

7. **Play the Game**
   - Run `play_prediction_jump.py` to start the AI player.
   - The program will:
     - Capture the game window.
     - Detect obstacles using YOLO.
     - Compute metrics (speed, distance, etc.).
     - Predict when to jump.
     - Display detection visualization.
