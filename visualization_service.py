import cv2
import win32gui
import win32con
import win32api
from config import Config

class VisualizationService:
    def __init__(self):
        self.window_name = "Game Monitor"
        self._setup_window()
        
    def _setup_window(self):
        """Create and position the monitoring window"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        
        # Get game window info
        game_hwnd = win32gui.GetForegroundWindow()
        game_rect = win32gui.GetWindowRect(game_hwnd)
        client_rect = win32gui.GetClientRect(game_hwnd)
        game_x, _, game_right, _ = game_rect
        
        # Calculate dimensions
        client_width = client_rect[2] - client_rect[0]
        client_height = client_rect[3] - client_rect[1]
        
        # Position window
        monitor_x = self._calculate_monitor_position(game_x, game_right, client_width)
        
        # Remove decorations and set position
        self._set_window_properties(monitor_x, game_rect[1], client_width, client_height)
    
    def _calculate_monitor_position(self, game_x, game_right, width):
        """Calculate the best position for the monitor window"""
        screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        monitor_x = game_right + 10
        if monitor_x + width > screen_width:
            monitor_x = game_x - width - 10
        return monitor_x
    
    def _set_window_properties(self, x, y, width, height):
        """Set window properties including position and style"""
        hwnd = win32gui.FindWindow(None, self.window_name)
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
        style &= ~(win32con.WS_CAPTION | win32con.WS_THICKFRAME)
        win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)
        win32gui.SetWindowPos(hwnd, None, x, y, width, height, win32con.SWP_NOZORDER)
    
    def show_frame(self, frame, objects, metrics, jump_prediction):
        """Display frame with detections and metrics"""
        if frame is None:
            return
            
        vis_frame = frame.copy()
        self._draw_detections(vis_frame, objects)
        self._draw_metrics(vis_frame, metrics, jump_prediction)
        
        cv2.imshow(self.window_name, vis_frame)
        cv2.waitKey(1)
        
        return vis_frame
    
    def _draw_detections(self, frame, objects):
        """Draw detection boxes and labels"""
        for cls in objects:
            color = Config.COLORS[cls]
            for obj in objects[cls]:
                x1, x2 = int(float(obj[2])), int(float(obj[3]))
                y1, y2 = int(float(obj[4])) + Config.GAME_AREA_TOP, int(float(obj[5])) + Config.GAME_AREA_TOP
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(frame, Config.CLASS_NAMES[cls], 
                           (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, color, 1)
    
    def _draw_metrics(self, frame, metrics, jump_prediction):
        """Draw metrics and predictions"""
        for cls in metrics:
            metric = metrics[cls]
            if cls == 0:  # Skip dino metrics
                continue
                
            effective_speed = max(metric['speed'], metric['avg_speed'])
            time_to_impact = metric['distance_to_dino'] / effective_speed
            
            # Position labels
            dino_x = 50
            label_y = 200  # Fixed position for metrics
            
            self._draw_metric_labels(frame, dino_x, label_y, metric, time_to_impact, jump_prediction)
    
    def _draw_metric_labels(self, frame, x, y, metric, time_to_impact, jump_prediction):
        """Draw individual metric labels"""
        labels = [
            f"Time to impact: {time_to_impact:.2f}s",
            f"Distance to obstacle: {metric['distance_to_dino']:.1f}px",
            f"Obstacle speed: {metric['speed']:.1f}px/s",
            f"Obstacle width: {metric['group_width']:.1f}px"
        ]
        
        for i, label in enumerate(labels):
            cv2.putText(frame, label, (x, y + i*15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Draw jump prediction
        jump_text = "Prediction: JUMP!" if jump_prediction else "Prediction: No jump"
        jump_color = (0, 0, 255) if jump_prediction else (0, 0, 0)
        cv2.putText(frame, jump_text, (x, y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, jump_color, 1) 