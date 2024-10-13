import platform
import sys
from collections import deque
import psutil
import torch
from PySide6.QtCore import QThread, Signal, QTimer, QObject, QPoint
from PySide6.QtWidgets import QWidget, QGridLayout, QLabel, QVBoxLayout, QHBoxLayout
import subprocess
from PySide6.QtGui import QPainter, QColor, QPolygon
from PySide6.QtCore import Qt
from math import sin, cos, pi


def is_nvidia_gpu_available():
    try:
        output = subprocess.check_output(["nvidia-smi"])
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


if is_nvidia_gpu_available():
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
else:
    handle = None


class MetricsCollector(QObject):
    metrics_updated = Signal(tuple)

    def __init__(self):
        super().__init__()
        self.gpu_available = is_nvidia_gpu_available()
        self.timer = QTimer()
        self.timer.timeout.connect(self.collect_metrics)
        self.timer.start(100)  # 100 ms interval

    def collect_metrics(self):
        cpu_usage = collect_cpu_metrics()
        ram_usage_percent, _ = collect_ram_metrics()

        # Only collect GPU metrics if NVIDIA GPU is present
        if self.gpu_available:
            gpu_utilization, vram_usage_percent = collect_gpu_metrics(handle)
            power_usage_percent, power_limit_percent = collect_power_metrics(handle)
        else:
            gpu_utilization, vram_usage_percent = None, None
            power_usage_percent, power_limit_percent = None, None

        self.metrics_updated.emit((cpu_usage, ram_usage_percent, gpu_utilization, vram_usage_percent, power_usage_percent, power_limit_percent))


class Speedometer(QWidget):
    def __init__(self, min_value=0, max_value=100, colors=None):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.current_value = 0
        self.colors = colors or ["#00FF00", "#FFFF00", "#FF0000"]  # Green, Yellow, Red
        self.setFixedSize(65, 65)  # 75x75 pixel size

    def set_value(self, value):
        self.current_value = max(self.min_value, min(self.max_value, value))
        self.update()

    def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            width = self.width()
            height = self.height()
            center_x = width / 2
            center_y = height / 2
            radius = min(width, height) / 2 * 0.7  # Reduce from 0.8 to 0.7
            
            # colored background arc
            start_angle = 180 * 16
            span_angle = -180 * 16
            for i in range(180):
                color = self.get_color_at_angle(i)
                painter.setPen(color)
                painter.drawArc(center_x - radius, center_y - radius, 
                                radius * 2, radius * 2, 
                                start_angle - i * 16, -16)

            angle = 180 - (self.current_value - self.min_value) / (self.max_value - self.min_value) * 180
            
            needle_length = radius * 0.9  # needle length of 90% of radius
            needle_width = 5  # Adjust this value to change the thickness of the needle
            needle_angle = angle * (pi / 180)
            
            needle_tip_x = center_x + needle_length * cos(needle_angle)
            needle_tip_y = center_y - needle_length * sin(needle_angle)
            
            perpendicular_angle = needle_angle + pi/2
            half_width = needle_width / 2
            
            point1 = QPoint(center_x + half_width * cos(perpendicular_angle),
                            center_y - half_width * sin(perpendicular_angle))
            point2 = QPoint(center_x - half_width * cos(perpendicular_angle),
                            center_y + half_width * sin(perpendicular_angle))
            point3 = QPoint(needle_tip_x, needle_tip_y)
            
            needle = QPolygon([point1, point2, point3])
            
            painter.setPen(Qt.NoPen)
            painter.setBrush(Qt.white)
            painter.drawPolygon(needle)

    def get_color_at_angle(self, angle):
        t = angle / 180
        if t <= 0:
            return QColor(self.colors[0])
        elif t >= 1:
            return QColor(self.colors[-1])
        else:
            # Find the two colors to interpolate between
            segment = t * (len(self.colors) - 1)
            index = int(segment)
            t = segment - index
            index = min(index, len(self.colors) - 2)
            c1 = QColor(self.colors[index])
            c2 = QColor(self.colors[index + 1])
            # Interpolate
            r = int(c1.red() * (1-t) + c2.red() * t)
            g = int(c1.green() * (1-t) + c2.green() * t)
            b = int(c1.blue() * (1-t) + c2.blue() * t)
            return QColor(r, g, b)


class MetricsBar(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setup_metrics_buffers()
        self.start_metrics_collector()

    def determine_compute_device(self):
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def is_nvidia_gpu(self):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return "nvidia" in gpu_name.lower()
        return False

    def get_os_name(self):
        return platform.system().lower()

    def initUI(self):
        main_layout = QGridLayout(self)
        main_layout.setSpacing(1)  # spacing between items
        main_layout.setContentsMargins(5, 5, 5, 5)  # margins

        def create_speedometer_group(name):
            group = QVBoxLayout()
            group.setSpacing(2)  # spacing within the group
            speedometer = Speedometer(colors=["#00FF00", "#FFFF00", "#FF0000"])
            speedometer.setFixedSize(65, 65)  # speedometer pixel size
            group.addWidget(speedometer, alignment=Qt.AlignCenter)
            label = QLabel(f"{name} 0.0%")
            label.setAlignment(Qt.AlignCenter)
            group.addWidget(label, alignment=Qt.AlignCenter)
            return group, speedometer, label

        # CPU Layout
        cpu_group, self.cpu_speedometer, self.cpu_label = create_speedometer_group("CPU")
        main_layout.addLayout(cpu_group, 0, 0)

        # RAM Layout
        ram_group, self.ram_speedometer, self.ram_label = create_speedometer_group("RAM")
        main_layout.addLayout(ram_group, 0, 1)

        if self.is_nvidia_gpu():
            # GPU Layout
            gpu_group, self.gpu_speedometer, self.gpu_label = create_speedometer_group("GPU")
            main_layout.addLayout(gpu_group, 0, 2)

            # VRAM Layout
            vram_group, self.vram_speedometer, self.vram_label = create_speedometer_group("VRAM")
            main_layout.addLayout(vram_group, 0, 3)

            # Power Layout
            power_group, self.power_speedometer, self.power_label = create_speedometer_group("GPU Power")
            main_layout.addLayout(power_group, 0, 4)

        for i in range(main_layout.columnCount()):
            main_layout.setColumnStretch(i, 1)

    def update_metrics(self, metrics):
        cpu_usage, ram_usage_percent, gpu_utilization, vram_usage_percent, power_usage_percent, power_limit_percent = metrics

        # Add new values to buffers
        self.cpu_buffer.append(cpu_usage)
        self.ram_buffer.append(ram_usage_percent)

        # Calculate and update CPU and RAM
        avg_cpu = sum(self.cpu_buffer) / len(self.cpu_buffer)
        self.cpu_speedometer.set_value(avg_cpu)
        self.cpu_label.setText(f"CPU {avg_cpu:.1f}%")
        
        avg_ram = sum(self.ram_buffer) / len(self.ram_buffer)
        self.ram_speedometer.set_value(avg_ram)
        self.ram_label.setText(f"RAM {avg_ram:.1f}%")

        if self.is_nvidia_gpu():  # Update GPU metrics if Nvidia GPU is present
            if gpu_utilization is not None:
                self.gpu_buffer.append(gpu_utilization)
                avg_gpu = sum(self.gpu_buffer) / len(self.gpu_buffer)
                self.gpu_speedometer.set_value(avg_gpu)
                self.gpu_label.setText(f"GPU {avg_gpu:.1f}%")

            if vram_usage_percent is not None:
                self.vram_buffer.append(vram_usage_percent)
                avg_vram = sum(self.vram_buffer) / len(self.vram_buffer)
                self.vram_speedometer.set_value(avg_vram)
                self.vram_label.setText(f"VRAM {avg_vram:.1f}%")

            if power_usage_percent is not None:
                self.power_buffer.append(power_usage_percent)
                avg_power = sum(self.power_buffer) / len(self.power_buffer)
                self.power_speedometer.set_value(avg_power)
                self.power_label.setText(f"GPU Power {avg_power:.1f}%")

    # Sets number of samples in rolling average to display
    def setup_metrics_buffers(self):
        self.cpu_buffer = deque(maxlen=5)
        self.ram_buffer = deque(maxlen=5)
        self.gpu_buffer = deque(maxlen=5)
        self.vram_buffer = deque(maxlen=5)
        self.power_buffer = deque(maxlen=5)

    def start_metrics_collector(self):
        self.metrics_collector = MetricsCollector()
        self.metrics_collector.metrics_updated.connect(self.update_metrics)

    def stop_metrics_collector(self):
        if hasattr(self, 'metrics_collector'):
            self.metrics_collector.timer.stop()


def collect_cpu_metrics():
    percentages = psutil.cpu_percent(interval=None, percpu=True)
    return sum(percentages) / len(percentages)


def collect_ram_metrics():
    ram = psutil.virtual_memory()
    return ram.percent, ram.used


def collect_gpu_metrics(handle):
    if handle is None:
        return None, None
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    vram_usage_percent = (memory_info.used / memory_info.total) * 100 if memory_info.total > 0 else 0
    return gpu_utilization, vram_usage_percent


def collect_power_metrics(handle):
    if handle is None:
        return None, None

    try:
        power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    except pynvml.NVMLError as err:
        print(f"Error collecting power usage: {err}")
        return None, None

    try:
        power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
    except pynvml.NVMLError_NotSupported:
        try:
            power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
        except pynvml.NVMLError:
            print("Power management and enforced power limit not supported.")
            power_limit = None

    if power_limit is not None and power_limit > 0:
        power_percentage = (power_usage / power_limit) * 100
    else:
        power_percentage = 0

    return power_percentage, power_limit
