import torch
import yaml
import platform
import ctranslate2
from pathlib import Path
import shutil

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
    full_config_path = Path('config.yaml').resolve()
    
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

def check_for_necessary_folders_and_files():
    required_folders = ["Docs_for_DB", "Images_for_DB", "Vector_DB_Backup", "Vector_DB"]
    for folder in required_folders:
        path = Path(folder)
        if not path.is_dir():
            path.mkdir()

def restore_vector_db_backup():
    backup_folder = Path('Vector_DB_Backup')
    destination_folder = Path('Vector_DB')

    # Check for two .parquet files in backup folder
    backup_files = list(backup_folder.glob('*.parquet'))
    if len(backup_files) != 2:
        return

    # Delete destination folder files
    for item in destination_folder.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    # Copy backup folder files to the destination folder
    for item in backup_folder.iterdir():
        dest_path = destination_folder / item.name
        if item.is_dir():
            shutil.copytree(item, dest_path)
        else:
            shutil.copy2(item, dest_path)

def delete_chat_history():
    script_dir = Path(__file__).resolve().parent
    chat_history_path = script_dir / 'chat_history.txt'

    if chat_history_path.exists():
        chat_history_path.unlink()

def main():
    compute_device_info = get_compute_device_info()
    platform_info = get_platform_info()
    update_config_file(Compute_Device=compute_device_info, Platform_Info=platform_info)
    check_for_necessary_folders_and_files()
    delete_chat_history()
    restore_vector_db_backup()

if __name__ == "__main__":
    main()
