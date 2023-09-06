from pynvml import *
from multiprocessing import Process, Pipe, Event
import time
import tkinter as tk

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
    def __init__(self, label, root):
        self.cuda_info_label = label
        self.root = root
        self.parent_conn, self.process, self.stop_event = start_monitoring()
        self.update_cuda_info()

    def update_cuda_info(self):
        if self.parent_conn.poll():
            memory_used_str, gpu_utilization = self.parent_conn.recv()
            info_text = f"Memory Used: {memory_used_str} | GPU Utilization: {gpu_utilization}"
            self.cuda_info_label.config(text=info_text)
        self.root.after(500, self.update_cuda_info)

    def stop_and_exit(self):
        stop_monitoring(self.process, self.stop_event)
        self.root.quit()

# If the script is executed directly, it will just run without outputting any metrics.
if __name__ == "__main__":
    pass
