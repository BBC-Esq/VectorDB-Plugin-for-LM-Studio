import os
import platform
import shutil
from pathlib import Path

import ctranslate2
import torch
import yaml


def get_compute_device_info():
    available_devices = ["cpu"]
    gpu_brand = None

    if torch.cuda.is_available():
        available_devices.append("cuda")
        if torch.version.hip:
            gpu_brand = "AMD"
        elif torch.version.cuda:
            gpu_brand = "NVIDIA"

    if torch.backends.mps.is_available():
        available_devices.append("mps")
        gpu_brand = "Apple"

    return {"available": available_devices, "gpu_brand": gpu_brand}


def get_platform_info():
    os_name = platform.system().lower()
    return {"os": os_name}


def get_supported_quantizations(device_type):
    types = ctranslate2.get_supported_compute_types(device_type)
    filtered_types = [q for q in types if q != "int16"]

    desired_order = [
        "float32",
        "float16",
        "bfloat16",
        "int8_float32",
        "int8_float16",
        "int8_bfloat16",
        "int8",
    ]
    sorted_types = [q for q in desired_order if q in filtered_types]
    return sorted_types


def update_config_file(**system_info):
    full_config_path = os.path.abspath("config.yaml")

    with open(full_config_path, "r") as stream:
        config_data = yaml.safe_load(stream)

    compute_device_info = system_info.get("Compute_Device", {})
    config_data["Compute_Device"]["available"] = compute_device_info.get(
        "available", ["cpu"]
    )
    config_data["Compute_Device"]["gpu_brand"] = compute_device_info.get(
        "gpu_brand", ""
    )

    valid_devices = ["cpu", "cuda", "mps"]
    for key in ["database_creation", "database_query"]:
        if config_data["Compute_Device"].get(key, "cpu") not in valid_devices:
            config_data["Compute_Device"][key] = "cpu"

    config_data["Supported_CTranslate2_Quantizations"] = {
        "CPU": get_supported_quantizations("cpu"),
        "GPU": get_supported_quantizations("cuda") if torch.cuda.is_available() else [],
    }

    for key, value in system_info.items():
        if key != "Compute_Device" and key != "Supported_CTranslate2_Quantizations":
            config_data[key] = value

    with open(full_config_path, "w") as stream:
        yaml.safe_dump(config_data, stream)


def find_all_target_directories_with_file(base_path, target_folder, target_file):
    found_directories = []

    for entry in base_path.rglob(target_folder):  # Search for all 'parsers' directories
        if entry.is_dir() and entry.name.lower() == target_folder.lower():
            # Search for the file in a case-insensitive manner
            for file in entry.iterdir():
                if file.is_file() and file.name.lower() == target_file.lower():
                    found_directories.append(entry)
                    break

    return found_directories


def get_directory_depth(directory, base_directory):
    return len(directory.relative_to(base_directory).parts)


def find_closest_directory(directories, base_directory):
    depths = [(dir, get_directory_depth(dir, base_directory)) for dir in directories]
    return min(depths, key=lambda x: x[1])[0]


def get_base_directory():
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent.parent  # Move up two levels
    return base_dir


def move_custom_pdf_loader():
    script_dir = Path(__file__).parent
    user_manual_pdf_path = script_dir / "User_Manual" / "PDF.py"

    base_dir = get_base_directory()
    target_folder = "parsers"
    target_file = "pdf.py"
    found_paths = find_all_target_directories_with_file(
        base_dir, target_folder, target_file
    )

    if len(found_paths) == 1:
        chosen_pdf_path = found_paths[0] / target_file
    elif len(found_paths) > 1:
        closest_parsers_path = find_closest_directory(found_paths, base_dir)
        print(f"Chosen 'parsers' directory based on path depth: {closest_parsers_path}")
        chosen_pdf_path = closest_parsers_path / target_file
    else:
        print("No suitable 'parsers' directory found.")
        return

    # Proceed with file size comparison and move operation
    user_manual_pdf_size = user_manual_pdf_path.stat().st_size
    chosen_pdf_size = chosen_pdf_path.stat().st_size

    if user_manual_pdf_size != chosen_pdf_size:
        print("Replacing the existing pdf.py with the new one...")
        shutil.copy(user_manual_pdf_path, chosen_pdf_path)
        print(f"PDF.py replaced at: {chosen_pdf_path}")
    else:
        print("No replacement needed. The files are of the same size.")


def main():
    compute_device_info = get_compute_device_info()
    platform_info = get_platform_info()
    update_config_file(Compute_Device=compute_device_info, Platform_Info=platform_info)
    move_custom_pdf_loader()


if __name__ == "__main__":
    main()
