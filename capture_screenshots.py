import time
import keyboard
import os
from screen_capture_service import capture_active_window, save_screenshot
from config import Config

class ScreenshotCapture:
    # Handles screenshot capture functionality for the Chrome Dino game.
    # Supports single, burst, and continuous capture modes with pause/resume capability.
    
    def __init__(self):
        # Track space key state for jump detection
        self.space_pressed = False
        # Ensure screenshot directory exists
        os.makedirs(Config.SCREENSHOT_FOLDER, exist_ok=True)
    
    def on_key_press(self, event):
        # Global keyboard event handler for screenshot controls.
        # Args:
        #     event: Keyboard event containing the key name that was pressed
        
        if event.name == 'q':
            print("\nCapture cancelled by user")
            keyboard.unhook_all()
            os._exit(0)  # Force exit for clean shutdown
        elif event.name == 's':
            self._capture_single()  # Single screenshot mode
        elif event.name == 'f':
            self._capture_fast()    # Burst capture mode
        elif event.name == 'c':
            self.continuous_capture()  # Continuous capture mode
    
    def _capture_single(self):
        # Takes and saves a single screenshot of the game window
        frame = capture_active_window()
        filepath = save_screenshot(frame, Config.SCREENSHOT_FOLDER)
        print(f"Captured {filepath}")
    
    def _capture_fast(self):
        # Burst mode: Captures multiple screenshots in rapid succession.
        # Uses FAST_CAPTURE_COUNT and CAPTURE_INTERVAL from Config.
        for _ in range(Config.FAST_CAPTURE_COUNT):
            frame = capture_active_window()
            filepath = save_screenshot(frame, Config.SCREENSHOT_FOLDER)
            print(f"Captured {filepath}")
            time.sleep(Config.CAPTURE_INTERVAL)
    
    def continuous_capture(self):
        # Continuous capture mode with pause/resume functionality.
        # Monitors space key for jump detection and labels frames accordingly.
        
        paused = False  # Track pause state
        last_space_time = 0  # Debounce timer for space key
        
        # Print control instructions
        print("Starting continuous capture")
        print("Controls:")
        print("  'q' to stop")
        print("  'p' to pause/unpause")
        
        try:
            while True:
                current_time = time.time()
                
                # Check for quit command
                if keyboard.is_pressed('q'):
                    print("\nContinuous capture stopped by user")
                    break
                
                # Handle pause/resume
                if self._handle_pause_resume(paused):
                    paused = not paused
                    continue
                
                # Skip processing while paused
                if paused:
                    time.sleep(0.1)  # Reduce CPU usage while paused
                    continue
                
                # Detect jumps with debouncing
                if keyboard.is_pressed('space') or self.space_pressed:
                    if current_time - last_space_time > 0.1:  # 100ms debounce
                        self.space_pressed = True
                        last_space_time = current_time
                
                # Capture and save the current frame
                self._capture_continuous_frame()
                time.sleep(Config.CAPTURE_INTERVAL)
                
        except KeyboardInterrupt:
            print("\nContinuous capture stopped")
        finally:
            keyboard.unhook_all()  # Cleanup keyboard hooks
    
    def _handle_pause_resume(self, current_pause_state):
        # Handles pause/resume logic for continuous capture.
        # Args:
        #     current_pause_state: Boolean indicating if capture is currently paused
        # Returns:
        #     Boolean indicating if pause state should be toggled
        
        if keyboard.is_pressed('p') or (current_pause_state and keyboard.is_pressed('c')):
            new_state = not current_pause_state
            print("Capture paused" if new_state else "Capture resumed")
            time.sleep(0.3)  # Debounce pause/resume key
            return True
        return False
    
    def _capture_continuous_frame(self):
        # Captures a single frame during continuous capture mode.
        # Labels the frame if a jump was detected.
        
        frame = capture_active_window()
        postfix = "_jump" if self.space_pressed else ""
        filepath = save_screenshot(frame, Config.SCREENSHOT_FOLDER, postfix)
        self.space_pressed = False  # Reset jump detection
        print(f"Captured {filepath}")

def main():
    # Main entry point for the screenshot capture tool.
    # Sets up keyboard handlers and displays control instructions.
    
    capture = ScreenshotCapture()
    
    # Print available commands
    print("\nScreenshot Capture Controls:")
    print("  's' to capture single screenshot")
    print("  'f' for fast capture (5 shots)")
    print("  'c' for continuous capture")
    print("  'p' to pause/unpause continuous capture")
    print("  'q' to quit")
    
    # Set up keyboard handlers
    keyboard.on_press(capture.on_key_press)
    keyboard.wait('q')  # Wait for quit command

if __name__ == "__main__":
    main()

