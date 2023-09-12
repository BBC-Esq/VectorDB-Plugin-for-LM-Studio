from pynvml import *
from multiprocessing import Process, Pipe, Event
import time

def monitor_nvml(pipe, stop_event):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)

    while not stop_event.is_set():
        # Get the memory information and GPU utilization
        memory_info = nvmlDeviceGetMemoryInfo(handle)
        utilization_rates = nvmlDeviceGetUtilizationRates(handle)

        memory_used_str = f"{memory_info.used / (1024 * 1024):.2f} MiB"
        gpu_utilization = f"{utilization_rates.gpu}%"

        data = (memory_used_str, gpu_utilization)
        pipe.send(data)

        # Sleep for a bit to simulate a periodic update
        time.sleep(0.5)

    nvmlShutdown()

def start_monitoring():
    stop_event = Event()
    parent_conn, child_conn = Pipe()
    p = Process(target=monitor_nvml, args=(child_conn, stop_event))
    p.start()
    return parent_conn, p, stop_event

def stop_monitoring(p, stop_event):
    stop_event.set()
    p.join()

class CudaVramLogic:
    def __init__(self, vram_label, gpu_label, root):
        self.vram_label = vram_label
        self.gpu_label = gpu_label
        self.root = root
        self.parent_conn, self.process, self.stop_event = start_monitoring()
        self.update_info()

    def update_info(self):
        if self.parent_conn.poll():
            memory_used_str, gpu_utilization = self.parent_conn.recv()
            self.vram_label.config(text=f"VRAM: {memory_used_str}")
            self.gpu_label.config(text=f"GPU: {gpu_utilization}")
        self.root.after(500, self.update_info)

    def stop_and_exit(self):
        stop_monitoring(self.process, self.stop_event)
        self.root.quit()
