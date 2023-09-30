import torch
import yaml
import platform

def determine_compute_device():
    if torch.cuda.is_available():
        COMPUTE_DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        COMPUTE_DEVICE = "mps"
    else:
        COMPUTE_DEVICE = "cpu"

    with open("config.yaml", 'r') as stream:
        config_data = yaml.safe_load(stream)

    config_data['COMPUTE_DEVICE'] = COMPUTE_DEVICE

    with open("config.yaml", 'w') as stream:
        yaml.safe_dump(config_data, stream)

    return COMPUTE_DEVICE

def is_nvidia_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return "nvidia" in gpu_name.lower()
    return False
