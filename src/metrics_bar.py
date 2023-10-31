from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel
from PySide6.QtCore import Qt
from metrics_gpu import GPU_Monitor
from metrics_system import SystemMonitor
from initialize import determine_compute_device, is_nvidia_gpu, get_os_name

class MetricsBar(QFrame):
    def __init__(self):
        super().__init__()

        self.setFixedHeight(28)
        metrics_layout = QHBoxLayout(self)

        metrics_labels = [
            ("VRAM: N/A", "gpu_vram_label"),
            ("GPU: N/A", "gpu_util_label"),
            ("CPU: N/A", "cpu_label"),
            ("RAM: N/A", "ram_label"),
            ("RAM Usage: N/A", "ram_usage_label")
        ]

        for text, attribute_name in metrics_labels:
            label = QLabel(text)
            setattr(self, attribute_name, label)
            metrics_layout.addWidget(label)

        # Initialize monitors
        self.compute_device = determine_compute_device()
        os_name = get_os_name()

        if self.compute_device != "mps" and os_name == "windows" and is_nvidia_gpu():
            self.gpu_monitor = GPU_Monitor(self.gpu_vram_label, self.gpu_util_label, self)
            self.system_monitor = SystemMonitor(self.cpu_label, self.ram_label, self.ram_usage_label, self)
        else:
            self.gpu_monitor = None
            self.system_monitor = None

    def stop_monitors(self):
        if self.gpu_monitor:
            self.gpu_monitor.stop_and_exit_gpu_monitor()
        if self.system_monitor:
            self.system_monitor.stop_and_exit_system_monitor()
