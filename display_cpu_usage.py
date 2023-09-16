import tkinter as tk
import psutil

cpu_probes = []

def update_cpu_usage():
    global cpu_probes
    cpu_usage = psutil.cpu_percent(interval=0.4, percpu=True)
    
    cpu_probes.append(cpu_usage)
    if len(cpu_probes) > 5:
        cpu_probes.pop(0)

    averaged_cpu_usages = [sum(core) / len(cpu_probes) for core in zip(*cpu_probes)]
    average_usage = sum(averaged_cpu_usages) / len(averaged_cpu_usages)

    ram = psutil.virtual_memory()
    ram_percent_usage = f"Total RAM Usage: {ram.percent:.2f}%"
    ram_mib_usage = f"Used RAM: {ram.used / (1024**2):.2f} MiB"
    
    cpu_usage_label.config(text=f"Average CPU Usage: {average_usage:.2f}%\n\n{ram_percent_usage}\n{ram_mib_usage}")

    root.after(400, update_cpu_usage)

root = tk.Tk()
root.title("CPU and RAM Usage Monitor")
cpu_usage_label = tk.Label(root, text="", font=("Helvetica", 14))
cpu_usage_label.pack(pady=20)
update_cpu_usage()
root.mainloop()
