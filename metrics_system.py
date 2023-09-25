import psutil
import time
from multiprocessing import Process, Pipe, Event

def monitor_system(pipe, stop_event):
    while not stop_event.is_set():
        cpu_percent = collect_cpu_metrics()
        ram_percent, ram_used_mib = collect_ram_metrics()
        data = (cpu_percent, ram_percent, ram_used_mib)
        pipe.send(data)
        time.sleep(0.5)

def collect_cpu_metrics():
    percentages = psutil.cpu_percent(interval=0.1, percpu=True)
    return round(sum(percentages) / len(percentages), 2)

def collect_ram_metrics():
    ram = psutil.virtual_memory()
    ram_used_mib = round(ram.used / (1024 ** 2), 2)
    return round(ram.percent, 2), ram_used_mib

def start_monitoring_system():
    stop_event = Event()
    parent_conn, child_conn = Pipe()
    p = Process(target=monitor_system, args=(child_conn, stop_event))
    p.start()
    return parent_conn, p, stop_event

def stop_monitoring_system(p, stop_event):
    stop_event.set()
    p.join()

class SystemMonitor:
    def __init__(self, cpu_label, ram_label, ram_usage_label, root):
        self.cpu_label = cpu_label
        self.ram_label = ram_label
        self.ram_usage_label = ram_usage_label
        self.root = root
        self.parent_conn, self.process, self.stop_event = start_monitoring_system()
        self.update_system_info()

    def update_system_info(self):
        if self.parent_conn.poll():
            cpu_percent, ram_percent, ram_used_mib = self.parent_conn.recv()
            self.cpu_label.config(text=f"CPU: {cpu_percent}%")
            self.ram_label.config(text=f"RAM: {ram_percent}%")
            self.ram_usage_label.config(text=f"RAM: {ram_used_mib:.2f} MiB")
        self.root.after(500, self.update_system_info)

    def stop_and_exit_system_monitor(self):
        stop_monitoring_system(self.process, self.stop_event)
        self.root.quit()
