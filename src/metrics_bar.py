from dataclasses import dataclass
from typing import Optional, List, Callable
from datetime import datetime
import csv

import subprocess
import psutil
from PySide6.QtCore import (
    Qt,
    QObject,
    QPointF,
    QTimer,
)
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QProgressBar,
    QMenu,
)
from PySide6.QtGui import (
    QPainter,
    QColor,
    QPolygon,
    QPainterPath,
    QPen,
    QPixmap,
    QPixmapCache,
    QLinearGradient,
)
from math import sin, cos, pi
from collections import deque
from typing import Optional

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_usage: float
    ram_usage_percent: float
    gpu_utilization: Optional[float] = None
    vram_usage_percent: Optional[float] = None
    power_usage_percent: Optional[float] = None
    power_limit_percent: Optional[float] = None

class MetricsStore:
    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self.metrics_history: List[SystemMetrics] = []
        self._subscribers: List[Callable[[SystemMetrics], None]] = []
    
    def add_metrics(self, metrics: SystemMetrics) -> None:
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.buffer_size:
            self.metrics_history.pop(0)
        self._notify_subscribers(metrics)
    
    def subscribe(self, callback: Callable[[SystemMetrics], None]) -> None:
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[SystemMetrics], None]) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    def _notify_subscribers(self, metrics: SystemMetrics) -> None:
        for subscriber in self._subscribers:
            subscriber(metrics)

def is_nvidia_gpu_available():
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

HAS_NVIDIA_GPU = is_nvidia_gpu_available()

if HAS_NVIDIA_GPU:
    import pynvml
    pynvml.nvmlInit()
    HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
else:
    HANDLE = None


class BatchCSVLogger(QObject):
    def __init__(self, filepath: str, flush_interval: int = 5000):
        """
        Initialize the BatchCSVLogger.

        :param filepath: Path to the CSV file where metrics will be logged.
        :param flush_interval: Time interval (in milliseconds) between flushes to disk.
        """
        super().__init__()
        self.filepath = filepath
        self.flush_interval = flush_interval
        self.buffer = []

        # Open the CSV file and write the header row.
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'timestamp', 'cpu_usage', 'ram_usage_percent',
            'gpu_utilization', 'vram_usage_percent',
            'power_usage_percent'
        ])

        self.timer = QTimer(self)
        self.timer.setInterval(self.flush_interval)
        self.timer.timeout.connect(self.flush)
        self.timer.start()

    def log(self, metrics):
        self.buffer.append(metrics)

    def flush(self):
        if not self.buffer:
            return

        for metrics in self.buffer:
            self.writer.writerow([
                metrics.timestamp.isoformat(),
                metrics.cpu_usage,
                metrics.ram_usage_percent,
                metrics.gpu_utilization if metrics.gpu_utilization is not None else '',
                metrics.vram_usage_percent if metrics.vram_usage_percent is not None else '',
                metrics.power_usage_percent if metrics.power_usage_percent is not None else '',
            ])
        self.file.flush()
        self.buffer.clear()

    def close(self):

        self.timer.stop()
        self.flush()
        self.file.close()

    def __del__(self):

        try:
            self.close()
        except Exception:
            pass


def collect_cpu_metrics():
    cpu_times = psutil.cpu_times_percent(interval=None, percpu=True)

    cpu_percentages = []
    for cpu in cpu_times:

        total_active = sum(
            value for field, value in cpu._asdict().items() 
            if field not in ('idle', 'iowait')
        )
        cpu_percentages.append(total_active)

    return sum(cpu_percentages) / len(cpu_percentages)

def collect_ram_metrics():
    ram = psutil.virtual_memory()
    return ram.percent, ram.used

def collect_gpu_metrics(handle):
    if handle is None:
        return None, None
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    vram_usage_percent = (
        (memory_info.used / memory_info.total) * 100 if memory_info.total > 0 else 0
    )
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


class MetricsCollector(QObject):
    def __init__(self, store: MetricsStore):
        super().__init__()
        self.store = store
        self.timer = QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._collect_metrics)
        self.gpu_available = HAS_NVIDIA_GPU

    def start(self):
        self.timer.start()

    def stop(self):
        self.timer.stop()
    
    def _collect_metrics(self) -> None:
        from datetime import datetime
        
        cpu_usage = collect_cpu_metrics()
        ram_usage_percent, _ = collect_ram_metrics()
        
        if self.gpu_available:
            gpu_util, vram_usage = collect_gpu_metrics(HANDLE)
            power_usage, power_limit = collect_power_metrics(HANDLE)
        else:
            gpu_util = vram_usage = power_usage = power_limit = None
        
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            ram_usage_percent=ram_usage_percent,
            gpu_utilization=gpu_util,
            vram_usage_percent=vram_usage,
            power_usage_percent=power_usage,
            power_limit_percent=power_limit
        )
        
        self.store.add_metrics(metrics)

    def cleanup(self):
        self.stop()
        if self.gpu_available:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as e:
                print(f"Error shutting down NVML: {e}")


class BaseVisualization(QWidget):
    def __init__(self, metrics_store: MetricsStore):
        super().__init__()
        self.metrics_store = metrics_store
        self.metrics_store.subscribe(self.update_metrics)
        self.has_nvidia_gpu = HAS_NVIDIA_GPU
    
    def update_metrics(self, metrics: SystemMetrics):
        raise NotImplementedError("Visualization must implement update_metrics")
    
    def cleanup(self):
        self.metrics_store.unsubscribe(self.update_metrics)


class BarVisualization(BaseVisualization):
    def __init__(self, metrics_store: MetricsStore):
        super().__init__(metrics_store)
        self.initUI()

    def initUI(self):
        grid_layout = QGridLayout(self)
        grid_layout.setSpacing(0)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        self.cpu_bar, self.cpu_percent_label = self.add_metric_to_grid(
            "CPU Usage:", "#FF4136", grid_layout, 0
        )
        self.ram_bar, self.ram_percent_label = self.add_metric_to_grid(
            "RAM Usage:", "#B10DC9", grid_layout, 1
        )

        if self.has_nvidia_gpu:
            self.gpu_bar, self.gpu_percent_label = self.add_metric_to_grid(
                "GPU Usage:", "#0074D9", grid_layout, 2
            )
            self.vram_bar, self.vram_percent_label = self.add_metric_to_grid(
                "VRAM Usage:", "#2ECC40", grid_layout, 3
            )
            self.power_bar, self.power_percent_label = self.add_metric_to_grid(
                "GPU Power:", "#FFD700", grid_layout, 4
            )

    def add_metric_to_grid(self, label_text, color, grid_layout, row):
            label = QLabel(label_text)
            grid_layout.addWidget(label, row, 0)

            percent_label = QLabel("0%")
            grid_layout.addWidget(percent_label, row, 1)

            progress_bar = self.create_progress_bar(color)
            grid_layout.addWidget(progress_bar, row, 2)

            return progress_bar, percent_label

    def create_progress_bar(self, color):
        bar = QProgressBar()
        bar.setMaximum(100)
        bar.setMaximumHeight(11)
        bar.setStyleSheet(
            f"QProgressBar {{ background-color: #1e2126; border: none; }}"
            f"QProgressBar::chunk {{ background-color: {color}; }}"
        )
        bar.setTextVisible(False)
        return bar

    def update_metrics(self, metrics: SystemMetrics):
        self.cpu_bar.setValue(int(metrics.cpu_usage))
        self.cpu_percent_label.setText(f"{int(metrics.cpu_usage)}%")
        
        self.ram_bar.setValue(int(metrics.ram_usage_percent))
        self.ram_percent_label.setText(f"{int(metrics.ram_usage_percent)}%")

        if self.has_nvidia_gpu:
            if metrics.gpu_utilization is not None:
                self.gpu_bar.setValue(int(metrics.gpu_utilization))
                self.gpu_percent_label.setText(f"{int(metrics.gpu_utilization)}%")

            if metrics.vram_usage_percent is not None:
                self.vram_bar.setValue(int(metrics.vram_usage_percent))
                self.vram_percent_label.setText(f"{int(metrics.vram_usage_percent)}%")

            if metrics.power_usage_percent is not None:
                self.power_bar.setValue(int(metrics.power_usage_percent))
                self.power_percent_label.setText(f"{int(metrics.power_usage_percent)}%")


class Sparkline(QWidget):
    def __init__(self, max_values=125, color="#0074D9"):
        super().__init__()
        self.max_values = max_values
        self.values = deque(maxlen=max_values)
        self.setFixedSize(125, 65)
        self.color = QColor(color)
        self._gradient_key = f"sparkline_gradient_{color}"

    def add_value(self, value):
        self.values.append(value)
        self.update()

    def _create_gradient(self):
        pixmap = QPixmap(1, self.height())
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        gradient = QLinearGradient(0, 0, 0, self.height())
        
        fill_color = QColor(self.color)
        fill_color.setAlpha(60)
        gradient.setColorAt(0, fill_color)
        gradient.setColorAt(1, QColor(0, 0, 0, 0))
        
        painter.fillRect(pixmap.rect(), gradient)
        painter.end()
        
        QPixmapCache.insert(self._gradient_key, pixmap)
        return pixmap

    def paintEvent(self, event):
        if not self.values:
            return

        gradient_pixmap = QPixmap()
        if not QPixmapCache.find(self._gradient_key, gradient_pixmap):
            gradient_pixmap = self._create_gradient()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        margin = 5

        min_value = 0
        max_value = 100
        value_range = max_value - min_value

        path = QPainterPath()
        x_step = (width - 2 * margin) / (len(self.values) - 1) if len(self.values) > 1 else 0
        points = []

        for i, value in enumerate(self.values):
            x = margin + i * x_step
            y = height - margin - (value / value_range) * (height - 2 * margin)
            points.append(QPointF(x, y))
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        fill_path = QPainterPath(path)
        fill_path.lineTo(points[-1].x(), height - margin)
        fill_path.lineTo(points[0].x(), height - margin)
        fill_path.closeSubpath()

        painter.save()
        painter.setClipPath(fill_path)
        for x in range(0, width, gradient_pixmap.width()):
            painter.drawPixmap(x, 0, gradient_pixmap)
        painter.restore()

        painter.setPen(QPen(self.color, 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)


class SparklineVisualization(BaseVisualization):
    def __init__(self, metrics_store: MetricsStore):
        super().__init__(metrics_store)
        self.initUI()

    def initUI(self):
        main_layout = QGridLayout(self)
        main_layout.setSpacing(1)
        main_layout.setContentsMargins(1, 1, 1, 1)

        def create_sparkline_group(name, color):
            group_widget = QWidget()
            group_layout = QVBoxLayout(group_widget)
            group_layout.setSpacing(1)
            group_layout.setContentsMargins(0, 0, 0, 0)
            sparkline = Sparkline(color=color)
            group_layout.addWidget(sparkline, alignment=Qt.AlignCenter)
            label = QLabel(f"{name} 0.0%")
            label.setAlignment(Qt.AlignCenter)
            group_layout.addWidget(label, alignment=Qt.AlignCenter)
            return group_widget, sparkline, label

        cpu_group, self.cpu_sparkline, self.cpu_label = create_sparkline_group(
            "CPU", "#FF4136"
        )
        main_layout.addWidget(cpu_group, 0, 0)

        ram_group, self.ram_sparkline, self.ram_label = create_sparkline_group(
            "RAM", "#B10DC9"
        )
        main_layout.addWidget(ram_group, 0, 1)

        if self.has_nvidia_gpu:
            gpu_group, self.gpu_sparkline, self.gpu_label = create_sparkline_group(
                "GPU", "#0074D9"
            )
            main_layout.addWidget(gpu_group, 0, 2)

            vram_group, self.vram_sparkline, self.vram_label = create_sparkline_group(
                "VRAM", "#2ECC40"
            )
            main_layout.addWidget(vram_group, 0, 3)

            power_group, self.power_sparkline, self.power_label = create_sparkline_group(
                "GPU Power", "#FFD700"
            )
            main_layout.addWidget(power_group, 0, 4)

        for i in range(main_layout.columnCount()):
            main_layout.setColumnStretch(i, 1)

    def update_metrics(self, metrics: SystemMetrics):
        self.cpu_sparkline.add_value(metrics.cpu_usage)
        self.cpu_label.setText(f"CPU {metrics.cpu_usage:.1f}%")

        self.ram_sparkline.add_value(metrics.ram_usage_percent)
        self.ram_label.setText(f"RAM {metrics.ram_usage_percent:.1f}%")

        if self.has_nvidia_gpu:
            if metrics.gpu_utilization is not None:
                self.gpu_sparkline.add_value(metrics.gpu_utilization)
                self.gpu_label.setText(f"GPU {metrics.gpu_utilization:.1f}%")

            if metrics.vram_usage_percent is not None:
                self.vram_sparkline.add_value(metrics.vram_usage_percent)
                self.vram_label.setText(f"VRAM {metrics.vram_usage_percent:.1f}%")

            if metrics.power_usage_percent is not None:
                self.power_sparkline.add_value(metrics.power_usage_percent)
                self.power_label.setText(f"GPU Power {metrics.power_usage_percent:.1f}%")


class Speedometer(QWidget):
    def __init__(self, min_value=0, max_value=100, colors=None):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.current_value = 0
        self.colors = colors or ["#00FF00", "#FFFF00", "#FF0000"]
        self.setFixedSize(105, 105)

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
        radius = min(width, height) / 2 * 0.7

        start_angle = 180 * 16
        span_angle = -180 * 16
        for i in range(180):
            color = self.get_color_at_angle(i)
            painter.setPen(color)
            painter.drawArc(
                center_x - radius,
                center_y - radius,
                radius * 2,
                radius * 2,
                start_angle - i * 16,
                -16,
            )

        angle = 180 - (self.current_value - self.min_value) / (
            self.max_value - self.min_value
        ) * 180

        needle_length = radius * 0.9
        needle_width = 5
        needle_angle = angle * (pi / 180)

        needle_tip_x = center_x + needle_length * cos(needle_angle)
        needle_tip_y = center_y - needle_length * sin(needle_angle)

        perpendicular_angle = needle_angle + pi / 2
        half_width = needle_width / 2

        point1 = QPointF(
            center_x + half_width * cos(perpendicular_angle),
            center_y - half_width * sin(perpendicular_angle),
        )
        point2 = QPointF(
            center_x - half_width * cos(perpendicular_angle),
            center_y + half_width * sin(perpendicular_angle),
        )
        point3 = QPointF(needle_tip_x, needle_tip_y)

        needle = QPolygon([point1.toPoint(), point2.toPoint(), point3.toPoint()])

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
                segment = t * (len(self.colors) - 1)
                index = int(segment)
                t = segment - index
                index = min(index, len(self.colors) - 2)
                c1 = QColor(self.colors[index])
                c2 = QColor(self.colors[index + 1])
                r = int(c1.red() * (1 - t) + c2.red() * t)
                g = int(c1.green() * (1 - t) + c2.green() * t)
                b = int(c1.blue() * (1 - t) + c2.blue() * t)
                return QColor(r, g, b)


class SpeedometerVisualization(BaseVisualization):
    def __init__(self, metrics_store: MetricsStore):
        super().__init__(metrics_store)
        self.initUI()

    def initUI(self):
        main_layout = QGridLayout(self)
        main_layout.setSpacing(1)
        main_layout.setContentsMargins(1, 1, 1, 1)

        def create_speedometer_group(name):
            group = QVBoxLayout()
            group.setSpacing(2)
            speedometer = Speedometer(colors=["#00FF00", "#FFFF00", "#FF0000"])
            speedometer.setFixedSize(105, 105)
            group.addWidget(speedometer, alignment=Qt.AlignCenter)
            label = QLabel(f"{name} 0.0%")
            label.setAlignment(Qt.AlignCenter)
            group.addWidget(label, alignment=Qt.AlignCenter)
            return group, speedometer, label

        cpu_group, self.cpu_speedometer, self.cpu_label = create_speedometer_group("CPU")
        main_layout.addLayout(cpu_group, 0, 0)

        ram_group, self.ram_speedometer, self.ram_label = create_speedometer_group("RAM")
        main_layout.addLayout(ram_group, 0, 1)

        if self.has_nvidia_gpu:
            gpu_group, self.gpu_speedometer, self.gpu_label = create_speedometer_group(
                "GPU"
            )
            main_layout.addLayout(gpu_group, 0, 2)

            vram_group, self.vram_speedometer, self.vram_label = create_speedometer_group(
                "VRAM"
            )
            main_layout.addLayout(vram_group, 0, 3)

            power_group, self.power_speedometer, self.power_label = create_speedometer_group(
                "GPU Power"
            )
            main_layout.addLayout(power_group, 0, 4)

        for i in range(main_layout.columnCount()):
            main_layout.setColumnStretch(i, 1)

    def update_metrics(self, metrics: SystemMetrics):
            self.cpu_speedometer.set_value(metrics.cpu_usage)
            self.cpu_label.setText(f"CPU {metrics.cpu_usage:.1f}%")

            self.ram_speedometer.set_value(metrics.ram_usage_percent)
            self.ram_label.setText(f"RAM {metrics.ram_usage_percent:.1f}%")

            if self.has_nvidia_gpu:
                if metrics.gpu_utilization is not None:
                    self.gpu_speedometer.set_value(metrics.gpu_utilization)
                    self.gpu_label.setText(f"GPU {metrics.gpu_utilization:.1f}%")

                if metrics.vram_usage_percent is not None:
                    self.vram_speedometer.set_value(metrics.vram_usage_percent)
                    self.vram_label.setText(f"VRAM {metrics.vram_usage_percent:.1f}%")

                if metrics.power_usage_percent is not None:
                    self.power_speedometer.set_value(metrics.power_usage_percent)
                    self.power_label.setText(f"GPU Power {metrics.power_usage_percent:.1f}%")


class ArcGraph(QWidget):
    def __init__(self, color="#0074D9"):
        super().__init__()
        self.color = QColor(color)
        self.value = 0
        self.setFixedSize(100, 100)
        self._cache_key = f"arc_bg_{color}"
        self._background = None

    def set_value(self, value):
        self.value = min(100, max(0, value))
        self.update()

    def _create_background(self):
        background = QPixmap(self.size())
        background.fill(Qt.transparent)
        
        painter = QPainter(background)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        radius = min(width, height) / 2 - 10
        center = QPointF(width / 2, height / 2)
        
        painter.setPen(QPen(QColor("#1e2126"), 8))
        painter.drawArc(
            int(center.x() - radius), 
            int(center.y() - radius),
            int(radius * 2), 
            int(radius * 2),
            180 * 16, 
            -180 * 16
        )
        painter.end()

        QPixmapCache.insert(self._cache_key, background)
        return background

    def paintEvent(self, event):
        background = QPixmap()
        if not QPixmapCache.find(self._cache_key, background):
            background = self._create_background()

        painter = QPainter(self)
        painter.drawPixmap(0, 0, background)

        painter.setRenderHint(QPainter.Antialiasing)
        width = self.width()
        height = self.height()
        radius = min(width, height) / 2 - 10
        center = QPointF(width / 2, height / 2)
        
        painter.setPen(QPen(self.color, 8))
        span_angle = -(self.value / 100.0) * 180
        painter.drawArc(
            int(center.x() - radius), 
            int(center.y() - radius),
            int(radius * 2), 
            int(radius * 2),
            180 * 16, 
            span_angle * 16
        )

        painter.setPen(Qt.white)
        font = painter.font()
        font.setPointSize(14)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, f"{int(self.value)}%")


class ArcGraphVisualization(BaseVisualization):
    def __init__(self, metrics_store: MetricsStore):
        super().__init__(metrics_store)
        self.initUI()

    def initUI(self):
        main_layout = QGridLayout(self)
        main_layout.setSpacing(1)
        main_layout.setContentsMargins(1, 1, 1, 1)

        def create_arc_group(name, color):
            group = QVBoxLayout()
            group.setSpacing(2)
            arc = ArcGraph(color=color)
            group.addWidget(arc, alignment=Qt.AlignCenter)
            label = QLabel(name)
            label.setAlignment(Qt.AlignCenter)
            group.addWidget(label, alignment=Qt.AlignCenter)
            return group, arc, label

        cpu_group, self.cpu_arc, self.cpu_label = create_arc_group("CPU", "#FF4136")
        main_layout.addLayout(cpu_group, 0, 0)

        ram_group, self.ram_arc, self.ram_label = create_arc_group("RAM", "#B10DC9")
        main_layout.addLayout(ram_group, 0, 1)

        if self.has_nvidia_gpu:
            gpu_group, self.gpu_arc, self.gpu_label = create_arc_group("GPU", "#0074D9")
            main_layout.addLayout(gpu_group, 0, 2)

            vram_group, self.vram_arc, self.vram_label = create_arc_group("VRAM", "#2ECC40")
            main_layout.addLayout(vram_group, 0, 3)

            power_group, self.power_arc, self.power_label = create_arc_group("GPU Power", "#FFD700")
            main_layout.addLayout(power_group, 0, 4)

        for i in range(main_layout.columnCount()):
            main_layout.setColumnStretch(i, 1)

    def update_metrics(self, metrics: SystemMetrics):
        self.cpu_arc.set_value(metrics.cpu_usage)
        self.cpu_label.setText(f"CPU {metrics.cpu_usage:.1f}%")

        self.ram_arc.set_value(metrics.ram_usage_percent)
        self.ram_label.setText(f"RAM {metrics.ram_usage_percent:.1f}%")

        if self.has_nvidia_gpu:
            if metrics.gpu_utilization is not None:
                self.gpu_arc.set_value(metrics.gpu_utilization)
                self.gpu_label.setText(f"GPU {metrics.gpu_utilization:.1f}%")

            if metrics.vram_usage_percent is not None:
                self.vram_arc.set_value(metrics.vram_usage_percent)
                self.vram_label.setText(f"VRAM {metrics.vram_usage_percent:.1f}%")

            if metrics.power_usage_percent is not None:
                self.power_arc.set_value(metrics.power_usage_percent)
                self.power_label.setText(f"GPU Power {metrics.power_usage_percent:.1f}%")


class MetricsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        QPixmapCache.setCacheLimit(10 * 1024)
        
        self.metrics_store = MetricsStore(buffer_size=100)

        # DEBUG
        # csv_logger = BatchCSVLogger('metrics_log.csv')
        # self.metrics_store.subscribe(csv_logger.log)

        self.init_ui()
        self.current_visualization_type = 1
        self.setToolTip("Right click for display options")
        self.metrics_collector = MetricsCollector(self.metrics_store)
        self.start_metrics_collector()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.current_visualization = SparklineVisualization(self.metrics_store)
        self.layout.addWidget(self.current_visualization)

    def contextMenuEvent(self, event):
        menu = QMenu(self)

        visual_menu = menu.addMenu("Visualization")
        bar_action = visual_menu.addAction("Bar")
        sparkline_action = visual_menu.addAction("Sparkline")
        speedometer_action = visual_menu.addAction("Speedometer")
        arc_action = visual_menu.addAction("Arc")

        actions = [bar_action, sparkline_action, speedometer_action, arc_action]
        actions[self.current_visualization_type].setCheckable(True)
        actions[self.current_visualization_type].setChecked(True)

        menu.addSeparator()

        is_running = self.metrics_collector and self.metrics_collector.timer.isActive()
        control_action = menu.addAction("Stop Monitoring" if is_running else "Start Monitoring")

        action = menu.exec_(event.globalPos())

        if action == bar_action:
            self.change_visualization(0)
        elif action == sparkline_action:
            self.change_visualization(1)
        elif action == speedometer_action:
            self.change_visualization(2)
        elif action == arc_action:
            self.change_visualization(3)
        elif action == control_action:
            if is_running:
                self.stop_metrics_collector()
            else:
                self.start_metrics_collector()

    def change_visualization(self, index):
        if index == self.current_visualization_type:
            return

        self.current_visualization_type = index
        self.current_visualization.cleanup()
        self.layout.removeWidget(self.current_visualization)
        self.current_visualization.deleteLater()

        if index == 0:
            self.current_visualization = BarVisualization(self.metrics_store)
        elif index == 1:
            self.current_visualization = SparklineVisualization(self.metrics_store)
        elif index == 2:
            self.current_visualization = SpeedometerVisualization(self.metrics_store)
        elif index == 3:
            self.current_visualization = ArcGraphVisualization(self.metrics_store)

        self.current_visualization.setToolTip("Right click for display options")
        self.layout.addWidget(self.current_visualization)

    def start_metrics_collector(self):
        if self.metrics_collector is None:
            self.metrics_collector = MetricsCollector(self.metrics_store)
        self.metrics_collector.start()

    def stop_metrics_collector(self):
        if self.metrics_collector:
            self.metrics_collector.stop()

    def cleanup(self):
        if self.metrics_collector:
            self.metrics_collector.cleanup()
        self.current_visualization.cleanup()
        QPixmapCache.clear()
        super().closeEvent(event)