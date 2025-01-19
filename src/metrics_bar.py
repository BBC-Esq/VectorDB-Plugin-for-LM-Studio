import subprocess
from collections import deque
import psutil
from PySide6.QtCore import (
    Qt,
    QObject,
    Signal,
    QPointF,
    QThread,
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
)
from math import sin, cos, pi
from dataclasses import dataclass
from typing import Optional


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


@dataclass
class Metrics:
    cpu_usage: float
    ram_usage_percent: float
    gpu_utilization: Optional[float] = None
    vram_usage_percent: Optional[float] = None
    power_usage_percent: Optional[float] = None
    power_limit_percent: Optional[float] = None


class MetricsCollectorThread(QThread):
    metrics_updated = Signal(Metrics)

    def __init__(self):
        super().__init__()
        self._running = True
        self.interval = 150
        self.gpu_available = HAS_NVIDIA_GPU

    def stop(self):
        self._running = False
        self.quit()
        self.wait()

    def run(self):
        while self._running:
            if not self._running:
                break
                
            cpu_usage = collect_cpu_metrics()
            ram_usage_percent, _ = collect_ram_metrics()

            if self.gpu_available:
                gpu_utilization, vram_usage_percent = collect_gpu_metrics(HANDLE)
                power_usage_percent, power_limit_percent = collect_power_metrics(HANDLE)
            else:
                gpu_utilization, vram_usage_percent = None, None
                power_usage_percent, power_limit_percent = None, None

            metrics = Metrics(
                cpu_usage=cpu_usage,
                ram_usage_percent=ram_usage_percent,
                gpu_utilization=gpu_utilization,
                vram_usage_percent=vram_usage_percent,
                power_usage_percent=power_usage_percent,
                power_limit_percent=power_limit_percent,
            )

            self.metrics_updated.emit(metrics)
            self.msleep(self.interval)


class MetricsCollector(QObject):
    metrics_updated = Signal(Metrics)

    def __init__(self):
        super().__init__()
        self.thread = MetricsCollectorThread()
        self.thread.metrics_updated.connect(self.metrics_updated)
        self.thread.start()


class MetricsVisualization(QWidget):
    def __init__(self, buffer_size=10, metrics=None):
        super().__init__()
        self.has_nvidia_gpu = HAS_NVIDIA_GPU
        # base metrics
        base_metrics = ['cpu', 'ram']
        
        # Add GPU metrics if NVIDIA GPU is available
        if self.has_nvidia_gpu:
            base_metrics += ['gpu', 'vram', 'power']
        
        if metrics:
            base_metrics += metrics
        
        self.setup_metrics_buffers(buffer_size=buffer_size, metrics=base_metrics)

    def setup_metrics_buffers(self, buffer_size, metrics=None):
        metrics = metrics or ['cpu', 'ram']
        for metric in metrics:
            setattr(self, f"{metric}_buffer", deque(maxlen=buffer_size))


class BarVisualization(MetricsVisualization):
    def __init__(self):
        super().__init__()
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

    def update_metrics(self, metrics: Metrics):
        self.cpu_buffer.append(metrics.cpu_usage)
        self.ram_buffer.append(metrics.ram_usage_percent)

        if self.has_nvidia_gpu:
            self.gpu_buffer.append(metrics.gpu_utilization)
            self.vram_buffer.append(metrics.vram_usage_percent)
            self.power_buffer.append(metrics.power_usage_percent)

        self.update_progress_bar(self.cpu_bar, self.cpu_buffer, self.cpu_percent_label)
        self.update_progress_bar(self.ram_bar, self.ram_buffer, self.ram_percent_label)

        if self.has_nvidia_gpu:
            self.update_progress_bar(
                self.gpu_bar, self.gpu_buffer, self.gpu_percent_label
            )
            self.update_progress_bar(
                self.vram_bar, self.vram_buffer, self.vram_percent_label
            )
            self.update_progress_bar(
                self.power_bar, self.power_buffer, self.power_percent_label
            )

    def update_progress_bar(self, bar, buffer, label):
        if buffer:
            avg_value = sum(buffer) / len(buffer)
            bar_value = min(100, int(avg_value))
            bar.setValue(bar_value)
            label.setText(f"{int(avg_value)}%")


class Sparkline(QWidget):
    def __init__(self, max_values=125, color="#0074D9"):
        super().__init__()
        self.max_values = max_values
        self.values = deque(maxlen=max_values)
        self.setFixedSize(125, 65)
        self.color = QColor(color)

    def add_value(self, value):
        self.values.append(value)
        self.update()

    def paintEvent(self, event):
        if not self.values:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        margin = 5

        min_value = 0
        max_value = 100
        value_range = max_value - min_value

        path = QPainterPath()
        x_step = (
            (width - 2 * margin) / (len(self.values) - 1) if len(self.values) > 1 else 0
        )
        points = []

        for i, value in enumerate(self.values):
            x = margin + i * x_step
            y = height - margin - (value / value_range) * (height - 2 * margin)
            points.append(QPointF(x, y))
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        fill_path = QPainterPath()
        fill_path.moveTo(points[0].x(), height - margin)
        for point in points:
            fill_path.lineTo(point)
        fill_path.lineTo(points[-1].x(), height - margin)
        fill_path.closeSubpath()

        # Fill the area under the curve
        fill_color = QColor(self.color)
        fill_color.setAlpha(60)  # Semi-transparent
        painter.setPen(Qt.NoPen)
        painter.setBrush(fill_color)
        painter.drawPath(fill_path)

        painter.setPen(QPen(self.color, 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)


class SparklineVisualization(MetricsVisualization):
    def __init__(self):
        super().__init__()
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

    def update_metrics(self, metrics: Metrics):
            self.cpu_buffer.append(metrics.cpu_usage)
            self.ram_buffer.append(metrics.ram_usage_percent)
            self.cpu_sparkline.add_value(metrics.cpu_usage)
            self.ram_sparkline.add_value(metrics.ram_usage_percent)

            avg_cpu = sum(self.cpu_buffer) / len(self.cpu_buffer)
            self.cpu_label.setText(f"CPU {avg_cpu:.1f}%")

            avg_ram = sum(self.ram_buffer) / len(self.ram_buffer)
            self.ram_label.setText(f"RAM {avg_ram:.1f}%")

            if self.has_nvidia_gpu:
                if metrics.gpu_utilization is not None:
                    self.gpu_buffer.append(metrics.gpu_utilization)
                    self.gpu_sparkline.add_value(metrics.gpu_utilization)
                    avg_gpu = sum(self.gpu_buffer) / len(self.gpu_buffer)
                    self.gpu_label.setText(f"GPU {avg_gpu:.1f}%")

                if metrics.vram_usage_percent is not None:
                    self.vram_buffer.append(metrics.vram_usage_percent)
                    self.vram_sparkline.add_value(metrics.vram_usage_percent)
                    avg_vram = sum(self.vram_buffer) / len(self.vram_buffer)
                    self.vram_label.setText(f"VRAM {avg_vram:.1f}%")

                if metrics.power_usage_percent is not None:
                    self.power_buffer.append(metrics.power_usage_percent)
                    self.power_sparkline.add_value(metrics.power_usage_percent)
                    avg_power = sum(self.power_buffer) / len(self.power_buffer)
                    self.power_label.setText(f"GPU Power {avg_power:.1f}%")


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

        # Colored background arc
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


class SpeedometerVisualization(MetricsVisualization):
    def __init__(self):
        super().__init__()
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

    def update_metrics(self, metrics: Metrics):
        self.cpu_buffer.append(metrics.cpu_usage)
        self.ram_buffer.append(metrics.ram_usage_percent)

        avg_cpu = sum(self.cpu_buffer) / len(self.cpu_buffer)
        self.cpu_speedometer.set_value(avg_cpu)
        self.cpu_label.setText(f"CPU {avg_cpu:.1f}%")

        avg_ram = sum(self.ram_buffer) / len(self.ram_buffer)
        self.ram_speedometer.set_value(avg_ram)
        self.ram_label.setText(f"RAM {avg_ram:.1f}%")

        if self.has_nvidia_gpu:
            if metrics.gpu_utilization is not None:
                self.gpu_buffer.append(metrics.gpu_utilization)
                avg_gpu = sum(self.gpu_buffer) / len(self.gpu_buffer)
                self.gpu_speedometer.set_value(avg_gpu)
                self.gpu_label.setText(f"GPU {avg_gpu:.1f}%")

            if metrics.vram_usage_percent is not None:
                self.vram_buffer.append(metrics.vram_usage_percent)
                avg_vram = sum(self.vram_buffer) / len(self.vram_buffer)
                self.vram_speedometer.set_value(avg_vram)
                self.vram_label.setText(f"VRAM {avg_vram:.1f}%")

            if metrics.power_usage_percent is not None:
                self.power_buffer.append(metrics.power_usage_percent)
                avg_power = sum(self.power_buffer) / len(self.power_buffer)
                self.power_speedometer.set_value(avg_power)
                self.power_label.setText(f"GPU Power {avg_power:.1f}%")


class MetricsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.metrics_collector = None
        self.current_visualization_type = 1
        self.setToolTip("Right click for display options")
        self.start_metrics_collector()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.current_visualization = SparklineVisualization()
        self.layout.addWidget(self.current_visualization)

    def start_metrics_collector(self):
        if self.metrics_collector is None or not self.metrics_collector.thread.isRunning():
            self.metrics_collector = MetricsCollector()
            self.metrics_collector.metrics_updated.connect(self.update_visualization)

    def stop_metrics_collector(self):
        if self.metrics_collector and self.metrics_collector.thread.isRunning():
            self.metrics_collector.thread.stop()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        
        visual_menu = menu.addMenu("Visualization")
        bar_action = visual_menu.addAction("Bar")
        sparkline_action = visual_menu.addAction("Sparkline")
        speedometer_action = visual_menu.addAction("Speedometer")

        actions = [bar_action, sparkline_action, speedometer_action]
        actions[self.current_visualization_type].setCheckable(True)
        actions[self.current_visualization_type].setChecked(True)

        menu.addSeparator()

        is_running = self.metrics_collector and self.metrics_collector.thread.isRunning()
        control_action = menu.addAction("Stop Monitoring" if is_running else "Start Monitoring")

        action = menu.exec_(event.globalPos())

        if action == bar_action:
            self.change_visualization(0)
        elif action == sparkline_action:
            self.change_visualization(1)
        elif action == speedometer_action:
            self.change_visualization(2)
        elif action == control_action:
            if is_running:
                self.stop_metrics_collector()
            else:
                self.start_metrics_collector()

    def change_visualization(self, index):
        if index == self.current_visualization_type:
            return

        self.current_visualization_type = index
        self.layout.removeWidget(self.current_visualization)
        self.current_visualization.deleteLater()

        if index == 0:
            self.current_visualization = BarVisualization()
        elif index == 1:
            self.current_visualization = SparklineVisualization()
        elif index == 2:
            self.current_visualization = SpeedometerVisualization()

        self.current_visualization.setToolTip("Right click for display options")
        self.layout.addWidget(self.current_visualization)

    def update_visualization(self, metrics: Metrics):
        self.current_visualization.update_metrics(metrics)

    def closeEvent(self, event):
        self.stop_metrics_collector()
        super().closeEvent(event)