from PySide6.QtWidgets import QFrame, QLabel, QProgressBar, QGridLayout
from PySide6.QtCore import Qt
from metrics_monitor import CombinedMonitor
import torch
import platform

class MetricsBar(QFrame):
    def __init__(self):
        super().__init__()

        self.setFixedHeight(55)
        grid_layout = QGridLayout(self)
        grid_layout.setSpacing(0)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        # Determine if GPU monitoring is enabled
        self.compute_device = self.determine_compute_device()
        os_name = self.get_os_name()
        enable_gpu_monitoring = self.compute_device != "mps" and os_name == "windows" and self.is_nvidia_gpu()

        metrics = [
            ("CPU Usage", "#FF4136"),
            ("RAM Usage", "#B10DC9")
        ]

        # Add GPU metrics only if GPU monitoring is enabled
        if enable_gpu_monitoring:
            metrics.extend([
                ("VRAM Usage", "#0074D9"),
                ("GPU Usage", "#2ECC40")
            ])

        for row, (metric_name, color) in enumerate(metrics):
            label = QLabel(f"{metric_name}: N/A")
            progress_bar = self.create_progress_bar(color)

            grid_layout.addWidget(label, row, 0)
            grid_layout.addWidget(progress_bar, row, 1)

            setattr(self, f"{metric_name.lower().replace(' ', '_')}_label", label)
            setattr(self, f"{metric_name.lower().replace(' ', '_')}_bar", progress_bar)

        self.monitor = CombinedMonitor(
            getattr(self, 'cpu_usage_bar', None), getattr(self, 'cpu_usage_label', None), 
            getattr(self, 'ram_usage_bar', None), getattr(self, 'ram_usage_label', None),
            getattr(self, 'vram_usage_bar', None), getattr(self, 'vram_usage_label', None),
            getattr(self, 'gpu_usage_bar', None), getattr(self, 'gpu_usage_label', None),
            enable_gpu_monitoring
        )

    def create_progress_bar(self, color):
        bar = QProgressBar()
        bar.setMaximumHeight(10)
        bar.setStyleSheet(
            f"QProgressBar {{ background-color: #1e2126; border: none; }}"
            f"QProgressBar::chunk {{ background-color: {color}; }}"
        )
        bar.setTextVisible(False)
        return bar

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

    def stop_monitors(self):
        if self.monitor:
            self.monitor.stop_and_exit_monitor()

if __name__ == "__main__":
    print("This script is not meant to be run directly.")
