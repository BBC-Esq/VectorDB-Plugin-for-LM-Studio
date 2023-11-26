import psutil
import time
from multiprocessing import Process, Pipe, Event
from PySide6.QtCore import QTimer
import humanize
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates

# Define the time interval for averaging (in seconds)
AVERAGING_INTERVAL = 3

def monitor_system(pipe, stop_event):
    data_buffer = {'cpu_percent': [], 'ram_percent': [], 'ram_used': []}
    start_time = time.time()

    while not stop_event.is_set():
        cpu_percent = collect_cpu_metrics()
        ram_percent, ram_used = collect_ram_metrics()

        data_buffer['cpu_percent'].append(cpu_percent)
        data_buffer['ram_percent'].append(ram_percent)
        
        try:
            ram_used = int(ram_used)
        except ValueError:
            ram_used = 0

        data_buffer['ram_used'].append(ram_used)

        elapsed_time = time.time() - start_time

        if elapsed_time >= AVERAGING_INTERVAL:
            avg_cpu_percent = sum(data_buffer['cpu_percent']) / len(data_buffer['cpu_percent'])
            avg_ram_percent = sum(data_buffer['ram_percent']) / len(data_buffer['ram_percent'])
            avg_ram_used = humanize.naturalsize(sum(data_buffer['ram_used']), binary=True)

            data = ('system', avg_cpu_percent, avg_ram_percent, avg_ram_used)

            data_buffer = {'cpu_percent': [], 'ram_percent': [], 'ram_used': []}

            pipe.send(data)

            start_time = time.time()

        time.sleep(0.5)

def monitor_nvml(pipe, stop_event):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)

    while not stop_event.is_set():
        memory_info = nvmlDeviceGetMemoryInfo(handle)
        utilization_rates = nvmlDeviceGetUtilizationRates(handle)

        vram_usage_percent = (memory_info.used / memory_info.total) * 100 if memory_info.total > 0 else 0

        data = ('gpu', vram_usage_percent, utilization_rates.gpu)
        pipe.send(data)
        time.sleep(0.5)

    nvmlShutdown()

def collect_cpu_metrics():
    percentages = psutil.cpu_percent(interval=0.1, percpu=True)
    return round(sum(percentages) / len(percentages), 2)

def collect_ram_metrics():
    ram = psutil.virtual_memory()
    ram_used = humanize.naturalsize(ram.used, binary=True)
    return round(ram.percent, 2), ram_used

class CombinedMonitor:
    def __init__(self, cpu_usage_bar, cpu_usage_label, ram_usage_bar, ram_usage_label, vram_usage_bar, vram_usage_label, gpu_usage_bar, gpu_usage_label, enable_gpu_monitoring):
        self.cpu_usage_bar = cpu_usage_bar
        self.cpu_usage_label = cpu_usage_label
        self.ram_usage_bar = ram_usage_bar
        self.ram_usage_label = ram_usage_label
        self.vram_usage_bar = vram_usage_bar
        self.vram_usage_label = vram_usage_label
        self.gpu_usage_bar = gpu_usage_bar
        self.gpu_usage_label = gpu_usage_label
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.parent_conn, self.system_process, self.gpu_process, self.stop_event = self.start_monitoring()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_info)
        self.timer.start(500)

    def start_monitoring(self):
        stop_event = Event()
        parent_conn, child_conn = Pipe()
        system_process = Process(target=monitor_system, args=(child_conn, stop_event))
        system_process.start()

        gpu_process = None
        if self.enable_gpu_monitoring:
            gpu_process = Process(target=monitor_nvml, args=(child_conn, stop_event))
            gpu_process.start()

        return parent_conn, system_process, gpu_process, stop_event

    def update_info(self):
        while self.parent_conn.poll():
            source, *data = self.parent_conn.recv()
            if source == 'system':
                cpu_percent, ram_percent, ram_used = data
                self.cpu_usage_bar.setValue(cpu_percent)
                
                formatted_cpu_percent = f"{cpu_percent:.2f}%"
                self.cpu_usage_label.setText(f"CPU Usage: {formatted_cpu_percent}")
                
                self.ram_usage_bar.setValue(ram_percent)
                
                formatted_ram_percent = f"{ram_percent:.2f}%"
                self.ram_usage_label.setText(f"RAM Usage: {formatted_ram_percent}")
                
                self.ram_usage_label.setToolTip(f"RAM Usage: {ram_used}")
            elif source == 'gpu' and self.enable_gpu_monitoring:
                vram_usage_percent, gpu_utilization = data
                self.vram_usage_bar.setValue(vram_usage_percent)
                
                formatted_vram_percent = f"{vram_usage_percent:.2f}%"
                self.vram_usage_label.setText(f"VRAM Usage: {formatted_vram_percent}")
                
                self.gpu_usage_bar.setValue(gpu_utilization)
                
                formatted_gpu_utilization = f"{gpu_utilization:.2f}%"
                self.gpu_usage_label.setText(f"GPU Usage: {formatted_gpu_utilization}")

    def stop_and_exit_monitor(self):
        self.timer.stop()
        if self.system_process:
            self.stop_event.set()
            self.system_process.join()
        if self.gpu_process:
            self.stop_event.set()
            self.gpu_process.join()

if __name__ == "__main__":
    print("This script is not meant to be run directly.")
