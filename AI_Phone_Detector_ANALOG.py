"""
AI PHONE DETECTOR PRO v3.0 - ANALOG EDITION
============================================
Features:
- Enhanced GPU detection for RTX 3060/3070/3080/3090/4000
- YouTube/Twitch stream analysis support
- Advanced phone detection with YOLOv8m model
- Retro analog CRT monitor aesthetic
- Multi-stream support

Author: AI Camera Solutions
Version: 3.0.0 ANALOG
"""

import sys
import cv2
import numpy as np
import torch
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QTextEdit, QGroupBox, QGridLayout, QCheckBox,
                             QMessageBox, QComboBox, QSlider, QSpinBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen
import time
from datetime import datetime
import re

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import yt_dlp
    YOUTUBE_SUPPORT = True
except ImportError:
    YOUTUBE_SUPPORT = False


# CRT effect removed - clean video display


def get_youtube_stream_url(url):
    """Extract direct stream URL from YouTube/Twitch"""
    if not YOUTUBE_SUPPORT:
        return None
    
    try:
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # Get best video URL
            if 'url' in info:
                return info['url']
            elif 'formats' in info:
                for fmt in info['formats']:
                    if fmt.get('vcodec') != 'none':
                        return fmt.get('url')
        
        return None
    except Exception as e:
        print(f"YouTube extraction error: {e}")
        return None


class DetectionThread(QThread):
    """Enhanced AI Detection Thread"""
    frame_ready = pyqtSignal(np.ndarray)
    stats_update = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.stream_url = ""
        self.running = False
        self.model = None
        self.device = 'cpu'
        self.conf_threshold = 0.20  # Lower for better phone detection
        self.detect_persons = True
        self.detect_phones = True
        self.use_fp16 = False
        self.model_size = 'n'  # n, s, m, l, x
        
    def set_stream_url(self, url):
        self.stream_url = url
        
    def force_gpu_detection(self):
        """Enhanced GPU detection specifically for RTX cards"""
        try:
            # Check CUDA availability
            if not torch.cuda.is_available():
                self.status_signal.emit("⚠️ CUDA not available - using CPU")
                self.device = 'cpu'
                return False
            
            # Get GPU details
            gpu_count = torch.cuda.device_count()
            self.status_signal.emit(f"🔍 Found {gpu_count} GPU(s)")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                
                self.status_signal.emit(f"GPU {i}: {gpu_name} ({vram:.1f} GB VRAM)")
                
                # Force use first GPU
                if i == 0:
                    self.device = f'cuda:{i}'
                    torch.cuda.set_device(i)
                    
                    # Test GPU
                    test = torch.randn(100, 100).to(self.device)
                    result = test @ test
                    del test, result
                    
                    torch.cuda.empty_cache()
                    
                    # Enable FP16 for RTX cards
                    if any(x in gpu_name.upper() for x in ['RTX', 'TITAN', 'TESLA', 'A100', 'A6000']):
                        self.use_fp16 = True
                        self.status_signal.emit("⚡ FP16 acceleration ENABLED")
                    
                    cuda_version = torch.version.cuda
                    pytorch_version = torch.__version__
                    
                    self.status_signal.emit(f"✅ GPU ACTIVE: {gpu_name}")
                    self.status_signal.emit(f"🔧 CUDA {cuda_version} | PyTorch {pytorch_version}")
                    
                    return True
            
            return False
            
        except Exception as e:
            self.status_signal.emit(f"❌ GPU setup failed: {e}")
            self.device = 'cpu'
            return False
    
    def load_model(self):
        """Load YOLOv8 model with better phone detection"""
        try:
            if not YOLO_AVAILABLE:
                self.error_signal.emit("Ultralytics not installed!\n\nRun: pip install ultralytics")
                return False
            
            self.status_signal.emit("📦 Loading AI model...")
            
            # Force GPU detection
            gpu_success = self.force_gpu_detection()
            
            # Load larger model for better phone detection
            model_name = f'yolov8{self.model_size}.pt'
            self.status_signal.emit(f"Loading {model_name}...")
            
            self.model = YOLO(model_name)
            
            # Move to device
            if self.device != 'cpu':
                self.model.to(self.device)
                
                # Apply FP16
                if self.use_fp16:
                    try:
                        self.model.model.half()
                        self.status_signal.emit("✅ FP16 mode activated")
                    except Exception as e:
                        self.status_signal.emit(f"⚠️ FP16 failed: {e}")
                        self.use_fp16 = False
                
                # Extended GPU warmup
                self.status_signal.emit("🔥 GPU warmup (this may take 30 seconds)...")
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                
                for i in range(5):
                    self.status_signal.emit(f"Warmup {i+1}/5...")
                    _ = self.model.predict(
                        dummy, 
                        device=self.device,
                        half=self.use_fp16,
                        verbose=False,
                        conf=self.conf_threshold
                    )
                    time.sleep(0.5)
                
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                
                self.status_signal.emit("✅ GPU warmup complete!")
            else:
                self.status_signal.emit("ℹ️ Running on CPU (slower)")
            
            self.status_signal.emit("✅ Model ready for detection!")
            return True
            
        except Exception as e:
            self.error_signal.emit(f"Model load failed:\n{e}")
            return False
    
    def connect_to_stream(self):
        """Connect to RTSP or YouTube stream"""
        url = self.stream_url.strip()
        
        # Check if YouTube/Twitch URL
        if any(domain in url.lower() for domain in ['youtube.com', 'youtu.be', 'twitch.tv']):
            if not YOUTUBE_SUPPORT:
                self.error_signal.emit(
                    "YouTube support not installed!\n\n"
                    "Install with: pip install yt-dlp"
                )
                return None
            
            self.status_signal.emit("🎥 Detecting YouTube/Twitch stream...")
            stream_url = get_youtube_stream_url(url)
            
            if not stream_url:
                self.error_signal.emit("Failed to extract stream URL from YouTube/Twitch")
                return None
            
            self.status_signal.emit("✅ Stream URL extracted!")
            url = stream_url
        
        # Connect to stream
        self.status_signal.emit(f"🔗 Connecting to stream...")
        cap = cv2.VideoCapture(url)
        
        # Configure capture
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            return None
        
        # Get stream info
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        self.status_signal.emit(f"✅ Stream: {width}x{height} @ {fps} FPS")
        
        return cap
    
    def run(self):
        """Main detection loop"""
        self.running = True
        
        # Load model
        if not self.load_model():
            return
        
        # Connect to stream
        cap = self.connect_to_stream()
        if cap is None:
            self.error_signal.emit(
                "Stream connection failed!\n\n"
                "Check:\n"
                "• RTSP URL format correct?\n"
                "• Camera/stream online?\n"
                "• Network connection?\n"
                "• YouTube URL valid?"
            )
            return
        
        self.status_signal.emit("🚀 Detection started!")
        
        fps_list = []
        last_time = time.time()
        frame_count = 0
        reconnect_attempts = 0
        
        while self.running:
            try:
                ret, frame = cap.read()
                
                if not ret:
                    reconnect_attempts += 1
                    if reconnect_attempts > 5:
                        self.error_signal.emit("Stream lost after 5 reconnect attempts")
                        break
                    
                    self.status_signal.emit("⚠️ Stream interrupted - reconnecting...")
                    time.sleep(2)
                    cap.release()
                    cap = self.connect_to_stream()
                    if cap is None:
                        break
                    continue
                
                reconnect_attempts = 0
                frame_count += 1
                
                # FPS calculation
                current_time = time.time()
                if last_time > 0:
                    fps_list.append(1.0 / (current_time - last_time))
                    if len(fps_list) > 30:
                        fps_list.pop(0)
                last_time = current_time
                fps = sum(fps_list) / len(fps_list) if fps_list else 0
                
                # AI Detection with enhanced settings
                results = self.model.predict(
                    frame,
                    device=self.device,
                    half=self.use_fp16,
                    conf=self.conf_threshold,
                    iou=0.45,  # Better overlap handling
                    classes=[0, 67],  # person, cell phone
                    verbose=False,
                    imgsz=640
                )
                
                result = results[0]
                
                persons = []
                phones = []
                
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        
                        if cls == 0 and self.detect_persons:
                            persons.append({
                                'box': [x1, y1, x2, y2],
                                'conf': conf
                            })
                        elif cls == 67 and self.detect_phones:
                            phones.append({
                                'box': [x1, y1, x2, y2],
                                'conf': conf
                            })
                
                # Enhanced phone-person matching
                critical_count = 0
                high_count = 0
                medium_count = 0
                
                for phone in phones:
                    px1, py1, px2, py2 = phone['box']
                    phone_cx = (px1 + px2) // 2
                    phone_cy = (py1 + py2) // 2
                    phone['matched'] = False
                    
                    # Find closest person
                    min_distance = float('inf')
                    closest_person = None
                    
                    for person in persons:
                        p_x1, p_y1, p_x2, p_y2 = person['box']
                        person_cx = (p_x1 + p_x2) // 2
                        person_cy = (p_y1 + p_y2) // 2
                        
                        # Check if phone is within expanded person area
                        if (p_x1 - 100 <= phone_cx <= p_x2 + 100 and
                            p_y1 - 100 <= phone_cy <= p_y2 + 100):
                            
                            distance = np.sqrt(
                                (phone_cx - person_cx)**2 + 
                                (phone_cy - person_cy)**2
                            )
                            
                            if distance < min_distance:
                                min_distance = distance
                                closest_person = person
                    
                    if closest_person:
                        p_x1, p_y1, p_x2, p_y2 = closest_person['box']
                        person_height = p_y2 - p_y1
                        
                        # Enhanced position analysis
                        head_zone = p_y1 + person_height * 0.20
                        chest_zone = p_y1 + person_height * 0.55
                        
                        phone['matched'] = True
                        
                        if phone_cy <= head_zone:
                            phone['severity'] = 'CRITICAL'
                            phone['color'] = (0, 0, 255)
                            phone['label'] = 'HEAD LEVEL - ACTIVE USE'
                            critical_count += 1
                        elif phone_cy <= chest_zone:
                            phone['severity'] = 'HIGH'
                            phone['color'] = (0, 140, 255)
                            phone['label'] = 'CHEST LEVEL - SUSPICIOUS'
                            high_count += 1
                        else:
                            phone['severity'] = 'MEDIUM'
                            phone['color'] = (0, 255, 255)
                            phone['label'] = 'LAP AREA - POSSIBLE USE'
                            medium_count += 1
                    else:
                        phone['severity'] = 'UNMATCHED'
                        phone['color'] = (255, 255, 0)
                        phone['label'] = 'PHONE DETECTED'
                
                # Draw detections
                vis = frame.copy()
                
                # Draw persons
                for person in persons:
                    x1, y1, x2, y2 = person['box']
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label = f"PERSON {person['conf']:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(vis, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(vis, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Draw phones with enhanced visualization
                for phone in phones:
                    x1, y1, x2, y2 = phone['box']
                    color = phone.get('color', (255, 255, 0))
                    label = phone.get('label', 'PHONE')
                    
                    # Thicker box for critical alerts
                    thickness = 4 if phone.get('severity') == 'CRITICAL' else 3
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                    
                    # Label background
                    label_text = f"{label} {phone['conf']:.2f}"
                    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(vis, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0] + 10, y1), color, -1)
                    cv2.putText(vis, label_text, (x1 + 5, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    # Blinking effect for critical
                    if phone.get('severity') == 'CRITICAL' and frame_count % 20 < 10:
                        cv2.circle(vis, ((x1+x2)//2, (y1+y2)//2), 20, (0, 0, 255), -1)
                        cv2.putText(vis, "!", ((x1+x2)//2 - 8, (y1+y2)//2 + 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                # Enhanced info overlay - ANALOG STYLE
                h, w = vis.shape[:2]
                
                # Create darker overlay for analog feel
                overlay = np.zeros((100, w, 3), dtype=np.uint8)
                overlay[:] = (20, 20, 20)
                
                # Add green scanline effect
                for i in range(0, 100, 2):
                    cv2.line(overlay, (0, i), (w, i), (0, 50, 0), 1)
                
                vis = np.vstack([overlay, vis])
                
                # ANALOG STYLE TEXT - Green terminal look
                cv2.putText(vis, f"[FPS: {fps:.1f}]", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv2.putText(vis, f"PERSONS: {len(persons)}", (10, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
                
                cv2.putText(vis, f"PHONES: {len(phones)}", (280, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
                
                # Alert indicators
                if critical_count > 0:
                    cv2.putText(vis, f"!!! CRITICAL: {critical_count} !!!", (550, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                if high_count > 0:
                    cv2.putText(vis, f"HIGH: {high_count}", (550, 75),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
                
                # Device info
                device_text = f"[{self.device.upper()}]"
                if self.use_fp16:
                    device_text += " FP16"
                cv2.putText(vis, device_text, (w - 200, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Model info
                cv2.putText(vis, f"YOLOv8{self.model_size}", (w - 200, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Send frame and stats (NO EFFECTS - CLEAN VIDEO)
                self.frame_ready.emit(vis)
                
                self.stats_update.emit({
                    'fps': fps,
                    'persons': len(persons),
                    'phones': len(phones),
                    'critical': critical_count,
                    'high': high_count,
                    'medium': medium_count
                })
                
                # GPU memory management
                if self.device != 'cpu' and frame_count % 50 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                self.status_signal.emit(f"⚠️ Frame error: {e}")
                continue
        
        # Cleanup
        cap.release()
        if self.device != 'cpu':
            torch.cuda.empty_cache()
        self.status_signal.emit("🛑 Detection stopped")
    
    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    """ANALOG RETRO UI"""
    
    def __init__(self):
        super().__init__()
        self.detection_thread = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("◈ AI PHONE DETECTOR PRO ◈ ANALOG EDITION v3.0")
        self.setGeometry(50, 50, 1600, 950)
        
        # ANALOG/RETRO THEME - Green terminal + CRT monitor aesthetic
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a0a0a;
                background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAABZJREFUeNpi/P//PwMDAwMjSAAIMAADPgMDAwMjSJCBgYHBwQMCQiAAIAAsAAQARKAGaFlkAAAAASUVORK5CYII=);
            }
            QGroupBox {
                border: 3px solid #00ff00;
                border-radius: 0px;
                margin-top: 15px;
                padding: 15px;
                font-weight: bold;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                background-color: rgba(0, 40, 0, 0.8);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px;
                color: #00ff00;
                background-color: #0a0a0a;
            }
            QLabel {
                color: #00ff00;
                font-family: 'Courier New', monospace;
                background-color: transparent;
            }
            QLineEdit {
                background-color: #001100;
                border: 2px solid #00aa00;
                border-radius: 0px;
                padding: 10px;
                color: #00ff00;
                font-size: 11pt;
                font-family: 'Courier New', monospace;
                selection-background-color: #00aa00;
            }
            QLineEdit:focus {
                border: 2px solid #00ff00;
            }
            QPushButton {
                background-color: #003300;
                color: #00ff00;
                border: 3px solid #00aa00;
                padding: 15px;
                border-radius: 0px;
                font-weight: bold;
                font-size: 12pt;
                font-family: 'Courier New', monospace;
            }
            QPushButton:hover {
                background-color: #005500;
                border: 3px solid #00ff00;
            }
            QPushButton:pressed {
                background-color: #00aa00;
                color: #000000;
            }
            QPushButton:disabled {
                background-color: #111111;
                color: #444444;
                border: 3px solid #333333;
            }
            QTextEdit {
                background-color: #000000;
                border: 2px solid #00aa00;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
                selection-background-color: #00aa00;
            }
            QCheckBox {
                color: #00ff00;
                font-family: 'Courier New', monospace;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #00aa00;
                background-color: #001100;
            }
            QCheckBox::indicator:checked {
                background-color: #00ff00;
            }
            QComboBox {
                background-color: #001100;
                border: 2px solid #00aa00;
                color: #00ff00;
                padding: 8px;
                font-family: 'Courier New', monospace;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #001100;
                border: 2px solid #00aa00;
                color: #00ff00;
                selection-background-color: #00aa00;
            }
        """)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(20)
        
        # Left panel
        left = QVBoxLayout()
        
        # ASCII Art Title
        title = QLabel("╔═══════════════════════════════════════╗\n"
                      "║  AI PHONE DETECTOR PRO - ANALOG ED  ║\n"
                      "╚═══════════════════════════════════════╝")
        title.setStyleSheet("""
            font-size: 11pt;
            font-weight: bold;
            color: #00ff00;
            font-family: 'Courier New', monospace;
            padding: 15px;
            background-color: rgba(0, 40, 0, 0.6);
            border: 2px solid #00aa00;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(title)
        
        # Video display
        video_group = QGroupBox("▼ LIVE SURVEILLANCE FEED")
        video_layout = QVBoxLayout()
        
        self.video_label = QLabel(">>> SYSTEM READY - AWAITING INPUT <<<")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(1000, 600)
        self.video_label.setStyleSheet("""
            background-color: #000000;
            border: 4px solid #00ff00;
            color: #00aa00;
            font-size: 16pt;
            font-family: 'Courier New', monospace;
            padding: 20px;
        """)
        video_layout.addWidget(self.video_label)
        
        video_group.setLayout(video_layout)
        left.addWidget(video_group)
        
        # Controls
        controls_group = QGroupBox("▼ STREAM CONFIGURATION")
        controls_layout = QVBoxLayout()
        
        # Stream type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("TYPE:"))
        self.stream_type = QComboBox()
        self.stream_type.addItems(["RTSP Camera", "YouTube Live", "Twitch Stream"])
        type_layout.addWidget(self.stream_type)
        controls_layout.addLayout(type_layout)
        
        # URL input
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("URL:"))
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("rtsp://admin:pass@192.168.1.100:554/stream")
        url_layout.addWidget(self.url_input)
        controls_layout.addLayout(url_layout)
        
        # Model selector
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("MODEL:"))
        self.model_selector = QComboBox()
        self.model_selector.addItems(["Nano (Fast)", "Small (Balanced)", "Medium (Accurate)", "Large (Best)"])
        self.model_selector.setCurrentIndex(2)  # Default to Medium
        model_layout.addWidget(self.model_selector)
        controls_layout.addLayout(model_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ INITIATE DETECTION")
        self.start_btn.clicked.connect(self.start_detection)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("■ TERMINATE")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        controls_layout.addLayout(button_layout)
        controls_group.setLayout(controls_layout)
        left.addWidget(controls_group)
        
        main_layout.addLayout(left, 3)
        
        # Right panel
        right = QVBoxLayout()
        
        # System status
        status_group = QGroupBox("▼ SYSTEM STATUS")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("INITIALIZING...")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 10pt; padding: 10px;")
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)
        right.addWidget(status_group)
        
        # Stats
        stats_group = QGroupBox("▼ DETECTION METRICS")
        stats_layout = QGridLayout()
        
        self.fps_label = QLabel("--")
        self.fps_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: #00ff00;")
        
        self.persons_label = QLabel("0")
        self.persons_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: #00ff00;")
        
        self.phones_label = QLabel("0")
        self.phones_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: #ffaa00;")
        
        self.critical_label = QLabel("0")
        self.critical_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: #ff0000;")
        
        self.high_label = QLabel("0")
        self.high_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: #ff8800;")
        
        stats_layout.addWidget(QLabel("FPS:"), 0, 0)
        stats_layout.addWidget(self.fps_label, 0, 1)
        
        stats_layout.addWidget(QLabel("PERSONS:"), 1, 0)
        stats_layout.addWidget(self.persons_label, 1, 1)
        
        stats_layout.addWidget(QLabel("PHONES:"), 2, 0)
        stats_layout.addWidget(self.phones_label, 2, 1)
        
        stats_layout.addWidget(QLabel("CRITICAL:"), 3, 0)
        stats_layout.addWidget(self.critical_label, 3, 1)
        
        stats_layout.addWidget(QLabel("HIGH:"), 4, 0)
        stats_layout.addWidget(self.high_label, 4, 1)
        
        stats_group.setLayout(stats_layout)
        right.addWidget(stats_group)
        
        # Settings
        settings_group = QGroupBox("▼ DETECTION OPTIONS")
        settings_layout = QVBoxLayout()
        
        self.detect_persons_cb = QCheckBox("☑ PERSON TRACKING")
        self.detect_persons_cb.setChecked(True)
        settings_layout.addWidget(self.detect_persons_cb)
        
        self.detect_phones_cb = QCheckBox("☑ PHONE DETECTION")
        self.detect_phones_cb.setChecked(True)
        settings_layout.addWidget(self.detect_phones_cb)
        
        settings_group.setLayout(settings_layout)
        right.addWidget(settings_group)
        
        # Activity log
        log_group = QGroupBox("▼ SYSTEM LOG")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(300)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        right.addWidget(log_group)
        
        right.addStretch()
        
        # Footer
        footer = QLabel("═══════════════════════════════════════\n"
                       " © 2026 Nanda NW – All rights reserved v3.0"
                       "═══════════════════════════════════════")
        footer.setStyleSheet("""
            color: #00aa00;
            font-size: 9pt;
            font-family: 'Courier New', monospace;
            padding: 10px;
            background-color: rgba(0, 40, 0, 0.4);
            border: 1px solid #00aa00;
        """)
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(footer)
        
        main_layout.addLayout(right, 1)
        
        # Initial log
        self.log(">>> SYSTEM INITIALIZED")
        self.log(">>> AWAITING OPERATOR COMMANDS")
        self.check_system()
    
    def check_system(self):
        """Check system capabilities"""
        try:
            # GPU check
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                cuda = torch.version.cuda
                
                status_text = (
                    f"GPU OPERATIONAL\n\n"
                    f"DEVICE: {gpu}\n"
                    f"VRAM: {vram:.1f} GB\n"
                    f"CUDA: {cuda}\n\n"
                    f"STATUS: ✓ READY"
                )
                self.status_label.setStyleSheet("color: #00ff00; font-weight: bold; padding: 10px; font-size: 10pt;")
                self.status_label.setText(status_text)
                self.log(f">>> GPU DETECTED: {gpu}")
                self.log(f">>> VRAM: {vram:.1f} GB | CUDA: {cuda}")
            else:
                status_text = (
                    "NO GPU DETECTED\n\n"
                    "MODE: CPU FALLBACK\n"
                    "PERFORMANCE: REDUCED\n\n"
                    "STATUS: ⚠ LIMITED"
                )
                self.status_label.setStyleSheet("color: #ffaa00; padding: 10px; font-size: 10pt;")
                self.status_label.setText(status_text)
                self.log(">>> WARNING: CPU MODE ACTIVE")
            
            # Check YouTube support
            if YOUTUBE_SUPPORT:
                self.log(">>> YOUTUBE/TWITCH SUPPORT: ENABLED")
            else:
                self.log(">>> WARNING: YOUTUBE SUPPORT DISABLED")
                self.log(">>> INSTALL: pip install yt-dlp")
            
        except Exception as e:
            self.log(f">>> ERROR: {e}")
    
    def log(self, msg):
        """Add to system log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {msg}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def start_detection(self):
        """Start detection"""
        url = self.url_input.text().strip()
        
        if not url:
            QMessageBox.warning(self, "INPUT REQUIRED", "Enter stream URL!")
            return
        
        if not YOLO_AVAILABLE:
            QMessageBox.critical(self, "ERROR", "Ultralytics not installed!\n\nRun: pip install ultralytics")
            return
        
        # Update placeholder based on stream type
        stream_type = self.stream_type.currentText()
        if "YouTube" in stream_type:
            if not YOUTUBE_SUPPORT:
                QMessageBox.warning(self, "WARNING", "YouTube support not installed!\n\nInstall: pip install yt-dlp")
                return
        
        self.log("=" * 50)
        self.log(">>> INITIATING DETECTION SEQUENCE")
        self.log(f">>> TARGET: {url}")
        
        # Create thread
        self.detection_thread = DetectionThread()
        self.detection_thread.set_stream_url(url)
        
        # Set model size
        model_idx = self.model_selector.currentIndex()
        model_sizes = ['n', 's', 'm', 'l']
        self.detection_thread.model_size = model_sizes[model_idx]
        
        # Connect signals
        self.detection_thread.frame_ready.connect(self.update_frame)
        self.detection_thread.stats_update.connect(self.update_stats)
        self.detection_thread.error_signal.connect(self.handle_error)
        self.detection_thread.status_signal.connect(self.log)
        
        # Apply settings
        self.detection_thread.detect_persons = self.detect_persons_cb.isChecked()
        self.detection_thread.detect_phones = self.detect_phones_cb.isChecked()
        
        # Start
        self.detection_thread.start()
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.url_input.setEnabled(False)
        self.stream_type.setEnabled(False)
        self.model_selector.setEnabled(False)
    
    def stop_detection(self):
        """Stop detection"""
        if self.detection_thread:
            self.log(">>> TERMINATING DETECTION SEQUENCE")
            self.detection_thread.stop()
            self.detection_thread.wait(3000)
            if self.detection_thread.isRunning():
                self.detection_thread.terminate()
            self.detection_thread = None
        
        # Reset UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.url_input.setEnabled(True)
        self.stream_type.setEnabled(True)
        self.model_selector.setEnabled(True)
        
        self.video_label.clear()
        self.video_label.setText(">>> DETECTION TERMINATED <<<")
        
        # Reset stats
        self.fps_label.setText("--")
        self.persons_label.setText("0")
        self.phones_label.setText("0")
        self.critical_label.setText("0")
        self.high_label.setText("0")
        
        self.log(">>> SYSTEM READY")
    
    def update_frame(self, frame):
        """Update video display"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            scaled = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(scaled)
        except Exception as e:
            self.log(f">>> FRAME ERROR: {e}")
    
    def update_stats(self, stats):
        """Update statistics"""
        self.fps_label.setText(f"{stats['fps']:.1f}")
        self.persons_label.setText(str(stats['persons']))
        self.phones_label.setText(str(stats['phones']))
        self.critical_label.setText(str(stats['critical']))
        self.high_label.setText(str(stats['high']))
        
        # Log alerts
        if stats['critical'] > 0:
            self.log(f">>> !!! CRITICAL ALERT: {stats['critical']} ACTIVE USE !!!")
        elif stats['high'] > 0:
            self.log(f">>> !! HIGH ALERT: {stats['high']} SUSPICIOUS !!")
    
    def handle_error(self, msg):
        """Handle errors"""
        self.log(f">>> ERROR: {msg}")
        QMessageBox.critical(self, "SYSTEM ERROR", msg)
        self.stop_detection()
    
    def closeEvent(self, event):
        """Handle window close"""
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread.wait(2000)
            if self.detection_thread.isRunning():
                self.detection_thread.terminate()
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set app metadata
    app.setApplicationName("AI Phone Detector Pro - Analog Edition")
    app.setApplicationVersion("3.0.0")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
