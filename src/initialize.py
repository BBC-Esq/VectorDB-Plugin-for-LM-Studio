import torch
import yaml
import platform
import ctranslate2

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

    # Define the desired order of quantizations
    desired_order = ['float32', 'float16', 'bfloat16', 'int8_float32', 'int8_float16', 'int8_bfloat16', 'int8']

    # Sort the filtered_types based on the desired order
    sorted_types = [q for q in desired_order if q in filtered_types]

    return sorted_types

def update_config_file(**system_info):
    with open('config.yaml', 'r') as stream:
        config_data = yaml.safe_load(stream)

    config_data.setdefault('Compute_Device', {})
    config_data['Compute_Device'].setdefault('database_creation', 'cpu')
    config_data['Compute_Device'].setdefault('database_query', 'cpu')

    # Add supported quantizations for CPU and GPU
    config_data['Supported_CTranslate2_Quantizations'] = {
        'CPU': get_supported_quantizations('cpu'),
        'GPU': get_supported_quantizations('cuda') if torch.cuda.is_available() else []
    }

    # Update other keys
    for key, value in system_info.items():
        if key != 'Compute_Device' and key != 'Supported_CTranslate2_Quantizations':
            config_data[key] = value

    with open('config.yaml', 'w') as stream:
        yaml.safe_dump(config_data, stream)

def main():
    compute_device_info = get_compute_device_info()
    platform_info = get_platform_info()

    update_config_file(Compute_Device=compute_device_info, Platform_Info=platform_info)

if __name__ == "__main__":
    main()
