import torch
import yaml
import platform

def determine_compute_device():
    available_devices = []

    if torch.cuda.is_available():
        compute_device = 'cuda'
        available_devices.append('cuda') 
    elif torch.backends.mps.is_available():
        compute_device = 'mps'
        available_devices.append('mps')
    else:
        compute_device = 'cpu'
    
    available_devices.append('cpu')

    with open('config.yaml', 'r') as stream:
        config_data = yaml.safe_load(stream)
        
    config_data['COMPUTE_DEVICE'] = compute_device

    if 'Compute_Device' not in config_data:
        config_data['Compute_Device'] = {}
    config_data['Compute_Device']['available'] = available_devices

    with open('config.yaml', 'w') as stream:
        yaml.safe_dump(config_data, stream)

    return compute_device

def is_nvidia_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return "nvidia" in gpu_name.lower()
    return False

def get_os_name():
    return platform.system().lower()
