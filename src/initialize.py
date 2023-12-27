import torch
import yaml
import platform
import ctranslate2
from pathlib import Path

def get_compute_device_info():
    available_devices = ["cpu"]
    gpu_brand = None

    if torch.cuda.is_available():
        available_devices.append('cuda')
        if torch.version.hip:
            gpu_brand = "AMD"
        elif torch.version.cuda:
            gpu_brand = "NVIDIA"

    if torch.backends.mps.is_available():
        available_devices.append('mps')
        gpu_brand = "Apple"

    return {'available': available_devices, 'gpu_brand': gpu_brand}

def get_platform_info():
    os_name = platform.system().lower()
    return {'os': os_name}

def get_supported_quantizations(device_type):
    types = ctranslate2.get_supported_compute_types(device_type)
    filtered_types = [q for q in types if q != 'int16']

    desired_order = ['float32', 'float16', 'bfloat16', 'int8_float32', 'int8_float16', 'int8_bfloat16', 'int8']
    sorted_types = [q for q in desired_order if q in filtered_types]
    return sorted_types

def update_config_file(**system_info):
    full_config_path = os.path.abspath('config.yaml')
    
    with open(full_config_path, 'r') as stream:
        config_data = yaml.safe_load(stream)

    compute_device_info = system_info.get('Compute_Device', {})
    config_data['Compute_Device']['available'] = compute_device_info.get('available', ['cpu'])
    config_data['Compute_Device']['gpu_brand'] = compute_device_info.get('gpu_brand', '')

    valid_devices = ['cpu', 'cuda', 'mps']
    for key in ['database_creation', 'database_query']:
        if config_data['Compute_Device'].get(key, 'cpu') not in valid_devices:
            config_data['Compute_Device'][key] = 'cpu'

    config_data['Supported_CTranslate2_Quantizations'] = {
        'CPU': get_supported_quantizations('cpu'),
        'GPU': get_supported_quantizations('cuda') if torch.cuda.is_available() else []
    }

    for key, value in system_info.items():
        if key != 'Compute_Device' and key != 'Supported_CTranslate2_Quantizations':
            config_data[key] = value

    with open(full_config_path, 'w') as stream:
        yaml.safe_dump(config_data, stream)

def main():
    compute_device_info = get_compute_device_info()
    platform_info = get_platform_info()
    update_config_file(Compute_Device=compute_device_info, Platform_Info=platform_info)

if __name__ == "__main__":
    main()
