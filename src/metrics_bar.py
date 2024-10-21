import platform
import subprocess
from collections import deque
import psutil
import torch
from PySide6.QtCore import (
    Qt,
    QTimer,
    QObject,
    Signal,
    QPointF,
)
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QComboBox,
    QGridLayout,
    QLabel,
    QProgressBar,
    QMenu,
    QToolTip,
)
from PySide6.QtGui import (
    QPainter,
    QColor,
    QPolygon,
    QPainterPath,
    QPen,
)
from math import sin, cos, pi


def is_nvidia_gpu_available():
    try:
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
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


class MetricsCollector(QObject):
    metrics_updated = Signal(tuple)

    def __init__(self):
        super().__init__()
        self.gpu_available = HAS_NVIDIA_GPU
        self.timer = QTimer()
        self.timer.timeout.connect(self.collect_metrics)
        self.timer.start(100)  # 100 ms polling interval

    def collect_metrics(self):
        cpu_usage = collect_cpu_metrics()
        ram_usage_percent, _ = collect_ram_metrics()

        # Only collect GPU metrics if NVIDIA GPU is present
        if self.gpu_available:
            gpu_utilization, vram_usage_percent = collect_gpu_metrics(HANDLE)
            power_usage_percent, power_limit_percent = collect_power_metrics(HANDLE)
        else:
            gpu_utilization, vram_usage_percent = None, None
            power_usage_percent, power_limit_percent = None, None

        self.metrics_updated.emit(
            (
                cpu_usage,
                ram_usage_percent,
                gpu_utilization,
                vram_usage_percent,
                power_usage_percent,
                power_limit_percent,
            )
        )


class MetricsVisualization(QWidget):
    def update_metrics(self, metrics):
        raise NotImplementedError("Subclasses should implement this method.")


### Bar Visualization Class ###

class BarVisualization(MetricsVisualization):
    def __init__(self):
        super().__init__()
        self.has_nvidia_gpu = HAS_NVIDIA_GPU
        self.initUI()
        self.setup_metrics_buffers()

    def initUI(self):
        grid_layout = QGridLayout(self)
        grid_layout.setSpacing(0)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        # CPU & RAM metrics
        self.cpu_bar, self.cpu_percent_label = self.add_metric_to_grid(
            "CPU Usage:", "#FF4136", grid_layout, 0
        )
        self.ram_bar, self.ram_percent_label = self.add_metric_to_grid(
            "RAM Usage:", "#B10DC9", grid_layout, 1
        )

        if self.has_nvidia_gpu:  # display GPU metrics if Nvidia GPU
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

    def update_metrics(self, metrics):
        (
            cpu_usage,
            ram_usage_percent,
            gpu_utilization,
            vram_usage_percent,
            power_usage_percent,
            power_limit_percent,
        ) = metrics

        self.cpu_buffer.append(cpu_usage)
        self.ram_buffer.append(ram_usage_percent)

        if self.has_nvidia_gpu:
            self.gpu_buffer.append(gpu_utilization)
            self.vram_buffer.append(vram_usage_percent)
            self.power_buffer.append(power_usage_percent)

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

    def setup_metrics_buffers(self):
        self.cpu_buffer = deque(maxlen=10)
        self.ram_buffer = deque(maxlen=10)

        if self.has_nvidia_gpu:
            self.gpu_buffer = deque(maxlen=10)
            self.vram_buffer = deque(maxlen=10)
            self.power_buffer = deque(maxlen=10)


### Sparkline Visualization Class ###

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
        self.setup_metrics_buffers()

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

        # CPU Layout
        cpu_group, self.cpu_sparkline, self.cpu_label = create_sparkline_group(
            "CPU", "#FF4136"
        )
        main_layout.addWidget(cpu_group, 0, 0)

        # RAM Layout
        ram_group, self.ram_sparkline, self.ram_label = create_sparkline_group(
            "RAM", "#B10DC9"
        )
        main_layout.addWidget(ram_group, 0, 1)

        if HAS_NVIDIA_GPU:
            # GPU Layout
            gpu_group, self.gpu_sparkline, self.gpu_label = create_sparkline_group(
                "GPU", "#0074D9"
            )
            main_layout.addWidget(gpu_group, 0, 2)

            # VRAM Layout
            vram_group, self.vram_sparkline, self.vram_label = create_sparkline_group(
                "VRAM", "#2ECC40"
            )
            main_layout.addWidget(vram_group, 0, 3)

            # Power Layout
            power_group, self.power_sparkline, self.power_label = create_sparkline_group(
                "GPU Power", "#FFD700"
            )
            main_layout.addWidget(power_group, 0, 4)

        for i in range(main_layout.columnCount()):
            main_layout.setColumnStretch(i, 1)

    def update_metrics(self, metrics):
        (
            cpu_usage,
            ram_usage_percent,
            gpu_utilization,
            vram_usage_percent,
            power_usage_percent,
            power_limit_percent,
        ) = metrics

        self.cpu_buffer.append(cpu_usage)
        self.ram_buffer.append(ram_usage_percent)
        self.cpu_sparkline.add_value(cpu_usage)
        self.ram_sparkline.add_value(ram_usage_percent)

        avg_cpu = sum(self.cpu_buffer) / len(self.cpu_buffer)
        self.cpu_label.setText(f"CPU {avg_cpu:.1f}%")

        avg_ram = sum(self.ram_buffer) / len(self.ram_buffer)
        self.ram_label.setText(f"RAM {avg_ram:.1f}%")

        if HAS_NVIDIA_GPU:
            if gpu_utilization is not None:
                self.gpu_buffer.append(gpu_utilization)
                self.gpu_sparkline.add_value(gpu_utilization)
                avg_gpu = sum(self.gpu_buffer) / len(self.gpu_buffer)
                self.gpu_label.setText(f"GPU {avg_gpu:.1f}%")

            if vram_usage_percent is not None:
                self.vram_buffer.append(vram_usage_percent)
                self.vram_sparkline.add_value(vram_usage_percent)
                avg_vram = sum(self.vram_buffer) / len(self.vram_buffer)
                self.vram_label.setText(f"VRAM {avg_vram:.1f}%")

            if power_usage_percent is not None:
                self.power_buffer.append(power_usage_percent)
                self.power_sparkline.add_value(power_usage_percent)
                avg_power = sum(self.power_buffer) / len(self.power_buffer)
                self.power_label.setText(f"GPU Power {avg_power:.1f}%")

    # Sets number of samples in rolling average to display
    def setup_metrics_buffers(self):
        buffer_size = 5
        self.cpu_buffer = deque(maxlen=buffer_size)
        self.ram_buffer = deque(maxlen=buffer_size)
        self.gpu_buffer = deque(maxlen=buffer_size)
        self.vram_buffer = deque(maxlen=buffer_size)
        self.power_buffer = deque(maxlen=buffer_size)


### Speedometer Visualization Class ###

class Speedometer(QWidget):
    def __init__(self, min_value=0, max_value=100, colors=None):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.current_value = 0
        # To change colors, modify this list. Each color corresponds to a section of the speedometer.
        self.colors = colors or ["#00FF00", "#FFFF00", "#FF0000"]  # Green, Yellow, Red
        self.setFixedSize(105, 105)  # Adjust these values to change the overall size of the speedometer

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
        radius = min(width, height) / 2 * 0.7  # Adjust the 0.7 factor to change the size of the arc relative to the widget

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

        needle_length = radius * 0.9  # Adjust this factor to change the length of the needle
        needle_width = 5  # Adjust this value to change the thickness of the needle
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
        painter.setBrush(Qt.white)  # Change this color to modify the needle color
        painter.drawPolygon(needle)

    def get_color_at_angle(self, angle):
        # This method determines the color gradient of the speedometer.
        # Modify the color interpolation logic here to change the appearance.
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
        self.setup_metrics_buffers()

    def initUI(self):
        main_layout = QGridLayout(self)
        main_layout.setSpacing(1)  # Adjust this value to change the spacing between speedometers
        main_layout.setContentsMargins(1, 1, 1, 1)  # Adjust these values to change the margins around the speedometers

        def create_speedometer_group(name):
            group = QVBoxLayout()
            group.setSpacing(2)  # Adjust this value to change the spacing between the speedometer and its label
            speedometer = Speedometer(colors=["#00FF00", "#FFFF00", "#FF0000"])  # Modify these colors to change the speedometer appearance
            speedometer.setFixedSize(105, 105)  # Adjust these values to change the size of individual speedometers
            group.addWidget(speedometer, alignment=Qt.AlignCenter)
            label = QLabel(f"{name} 0.0%")
            label.setAlignment(Qt.AlignCenter)
            group.addWidget(label, alignment=Qt.AlignCenter)
            return group, speedometer, label

        # CPU Layout
        cpu_group, self.cpu_speedometer, self.cpu_label = create_speedometer_group(
            "CPU"
        )
        main_layout.addLayout(cpu_group, 0, 0)

        # RAM Layout
        ram_group, self.ram_speedometer, self.ram_label = create_speedometer_group(
            "RAM"
        )
        main_layout.addLayout(ram_group, 0, 1)

        if HAS_NVIDIA_GPU:
            # GPU Layout
            gpu_group, self.gpu_speedometer, self.gpu_label = create_speedometer_group(
                "GPU"
            )
            main_layout.addLayout(gpu_group, 0, 2)

            # VRAM Layout
            vram_group, self.vram_speedometer, self.vram_label = create_speedometer_group(
                "VRAM"
            )
            main_layout.addLayout(vram_group, 0, 3)

            # Power Layout
            power_group, self.power_speedometer, self.power_label = create_speedometer_group(
                "GPU Power"
            )
            main_layout.addLayout(power_group, 0, 4)

        for i in range(main_layout.columnCount()):
            main_layout.setColumnStretch(i, 1)

    def update_metrics(self, metrics):
        # This method updates the speedometer values
        # Modify the formatting of the label text here if needed
        (
            cpu_usage,
            ram_usage_percent,
            gpu_utilization,
            vram_usage_percent,
            power_usage_percent,
            power_limit_percent,
        ) = metrics

        self.cpu_buffer.append(cpu_usage)
        self.ram_buffer.append(ram_usage_percent)

        avg_cpu = sum(self.cpu_buffer) / len(self.cpu_buffer)
        self.cpu_speedometer.set_value(avg_cpu)
        self.cpu_label.setText(f"CPU {avg_cpu:.1f}%")

        avg_ram = sum(self.ram_buffer) / len(self.ram_buffer)
        self.ram_speedometer.set_value(avg_ram)
        self.ram_label.setText(f"RAM {avg_ram:.1f}%")

        if HAS_NVIDIA_GPU:
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
        buffer_size = 5  # Adjust this value to change the number of samples used for the rolling average
        self.cpu_buffer = deque(maxlen=buffer_size)
        self.ram_buffer = deque(maxlen=buffer_size)
        self.gpu_buffer = deque(maxlen=buffer_size)
        self.vram_buffer = deque(maxlen=buffer_size)
        self.power_buffer = deque(maxlen=buffer_size)


### MetricsWidget Class ###

class MetricsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.metrics_collector = MetricsCollector()
        self.metrics_collector.metrics_updated.connect(self.update_visualization)
        self.current_visualization_type = 1  # Default to SparklineVisualization
        
        self.setToolTip("Right click for display options")

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.current_visualization = SparklineVisualization()
        self.layout.addWidget(self.current_visualization)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        bar_action = menu.addAction("Bar")
        sparkline_action = menu.addAction("Sparkline")
        speedometer_action = menu.addAction("Speedometer")

        # Check the current visualization and set it as checked
        actions = [bar_action, sparkline_action, speedometer_action]
        actions[self.current_visualization_type].setCheckable(True)
        actions[self.current_visualization_type].setChecked(True)

        action = menu.exec_(event.globalPos())

        if action == bar_action:
            self.change_visualization(0)
        elif action == sparkline_action:
            self.change_visualization(1)
        elif action == speedometer_action:
            self.change_visualization(2)

    def change_visualization(self, index):
        if index == self.current_visualization_type:
            return  # Do nothing if the same visualization is selected

        self.current_visualization_type = index

        # Remove the old visualization
        self.layout.removeWidget(self.current_visualization)
        self.current_visualization.deleteLater()

        # Create the new visualization
        if index == 0:
            self.current_visualization = BarVisualization()
        elif index == 1:
            self.current_visualization = SparklineVisualization()
        elif index == 2:
            self.current_visualization = SpeedometerVisualization()

        self.current_visualization.setToolTip("Right click for display options")
        
        self.layout.addWidget(self.current_visualization)

    def update_visualization(self, metrics):
        self.current_visualization.update_metrics(metrics)

    def stop_metrics_collector(self):
        if hasattr(self.metrics_collector, 'timer'):
            self.metrics_collector.timer.stop()
