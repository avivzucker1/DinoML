# This service handles screen capture functionality for the Chrome Dino game.
# It uses Windows API (win32gui, win32ui) to efficiently capture the game window,
# process the image, and save screenshots when needed.

import win32gui
import win32ui
import win32con
from ctypes import windll
import numpy as np
from PIL import Image
import os
from datetime import datetime
import time
from config import Config

# Initialize Windows GDI+ library
gdi32 = windll.gdi32
user32 = windll.user32

def capture_active_window():
    # Capture the currently active window (Chrome with Dino game)
    # Returns a numpy array containing the window's image data
    
    # Get handle of the active window
    hwnd = win32gui.GetForegroundWindow()
    
    # Get window dimensions
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top
    
    # Create device context
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    
    # Create bitmap
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)
    
    # Copy screen to bitmap
    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)
    
    # Convert to numpy array
    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)
    
    # Clean up
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)
    
    # Convert to numpy array
    img_array = np.frombuffer(bmpstr, dtype='uint8').reshape(height, width, 4)
    return img_array[:,:,:3]  # Return BGR format for OpenCV

def save_screenshot(frame, marker=''):
    # Save a screenshot with optional marker in filename
    # Used to mark frames where jumps occurred and their outcomes
    
    # Generate filename with timestamp and optional marker
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
    filename = f"{timestamp}{marker}.png"
    filepath = os.path.join(Config.SCREENSHOT_FOLDER, filename)
    
    # Ensure screenshots directory exists
    os.makedirs(Config.SCREENSHOT_FOLDER, exist_ok=True)
    
    # Save the image
    Image.fromarray(frame).save(filepath)
    return filename

def capture_burst(count=Config.FAST_CAPTURE_COUNT, interval=Config.CAPTURE_INTERVAL):
    # Capture multiple screenshots in rapid succession
    # Used to get detailed motion data for obstacle tracking
    
    frames = []
    for _ in range(count):
        frame = capture_active_window()
        frames.append(frame)
        time.sleep(interval)
    return frames