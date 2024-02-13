import subprocess
import sys

if sys.version_info.major != 3 or sys.version_info.minor not in [10, 11]:
    print("Only Python 3.10 or 3.11 are supported.")
    sys.exit(1)
    
def is_package_installed(package_name):
    result = subprocess.run(['dpkg', '-l', package_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0

def install_system_package(package_name):
    if not is_package_installed(package_name):
        subprocess.run(['sudo', 'apt-get', 'install', package_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def check_gpu():
    try:
        nvidia_smi = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if nvidia_smi.returncode == 0:
            return 'nvidia'
    except FileNotFoundError:
        pass

    try:
        rocm_smi = subprocess.run(['rocminfo'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if rocm_smi.returncode == 0:
            return 'amd'
    except FileNotFoundError:
        pass

    return 'cpu'

def install_packages(gpu_type):
    python_version = f"{sys.version_info.major}{sys.version_info.minor}"
    base_cmd = "pip3 install"
    
    if gpu_type == 'nvidia':
        subprocess.run(f"{base_cmd} torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118", shell=True)
        if python_version == "310":
            subprocess.run(f"{base_cmd} -U https://download.pytorch.org/whl/cu118/xformers-0.0.23%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl#sha256=5cbda33632505f634aee52ae55832ebd4010e64fe656c45ca477cd0b55b26d8f", shell=True)
            subprocess.run(f"{base_cmd} https://download.pytorch.org/whl/triton-2.1.0-0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl#sha256=66439923a30d5d48399b08a9eae10370f6c261a5ec864a64983bae63152d39d7", shell=True)
        elif python_version == "311":
            subprocess.run(f"{base_cmd} -U https://download.pytorch.org/whl/cu118/xformers-0.0.23%2Bcu118-cp311-cp311-manylinux2014_x86_64.whl#sha256=cf614b8ea3ff4635440f82095b7cae6583a49fa4161e4b3e50a6ab5cc31771d5", shell=True)
            subprocess.run(f"{base_cmd} https://download.pytorch.org/whl/triton-2.1.0-0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl#sha256=919b06453f0033ea52c13eaf7833de0e57db3178d23d4e04f9fc71c4f2c32bf8", shell=True)
        subprocess.run(f"{base_cmd} nvidia-ml-py==12.535.108", shell=True)
        subprocess.run(f"{base_cmd} bitsandbytes==0.41.2.post2", shell=True)
    elif gpu_type == 'amd':
        subprocess.run(f"{base_cmd} torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/rocm5.6", shell=True)
        subprocess.run(f"{base_cmd} -U xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/rocm5.6", shell=True)
        if python_version == "310":
            subprocess.run(f"{base_cmd} https://download.pytorch.org/whl/pytorch_triton_rocm-2.1.0-cp310-cp310-linux_x86_64.whl#sha256=12fbf2ded4e5efcab0ff9ecc2de17f667dc4ef0a8a952ab9b549344ca4feb19e", shell=True)
        elif python_version == "311":
            subprocess.run(f"{base_cmd} https://download.pytorch.org/whl/pytorch_triton_rocm-2.1.0-cp311-cp311-linux_x86_64.whl#sha256=317686e3b0b72c0c4162fe7893cbcc8ba37c1ab6bee3d0830b547dcc97c208e1", shell=True)
        subprocess.run(f"{base_cmd} bitsandbytes==0.41.2.post2", shell=True)
    else: # cpu
        subprocess.run(f"{base_cmd} torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu", shell=True)

def install_python_requirements():
    subprocess.run(['pip3', 'install', '-r', 'requirements.txt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def run_python_script(script_name):
    subprocess.run(['python', script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if __name__ == "__main__":
    install_system_package('portaudio19-dev')
    install_system_package('python3-dev')
    install_system_package('libxcb-cursor0')
    gpu_type = check_gpu()
    install_packages(gpu_type)
    install_python_requirements()
    run_python_script('replace_pdf.py')