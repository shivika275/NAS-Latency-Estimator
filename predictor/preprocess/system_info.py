import psutil
import GPUtil
import pdb
import platform
from numba import cuda
import os
import json

def get_cpu_info(cpu_info):
    cpu_info['cores'] = psutil.cpu_count(logical=False)  # Physical CPU cores
    cpu_info['threads'] = psutil.cpu_count(logical=True)  # Total CPU threads
    cpu_info['freq'] = psutil.cpu_freq().current  # CPU frequency in MHz
    cpu_info['max_freq'] = psutil.cpu_freq().max  # CPU frequency in MHz
    cpu_info['min_freq'] = psutil.cpu_freq().min  # CPU frequency in MHz
    cpu_info['ram_memory'] = psutil.virtual_memory().total  # CPU frequency in MHz
    return cpu_info

def get_platform_info(platform_info):
    uname = platform.uname()
    platform_info['machine'] = uname.machine  # Physical CPU cores
    platform_info['processor'] = uname.processor  # Physical CPU cores


    return platform_info

def get_gpu_info(gpu_info):

    cc_cores_per_SM_dict = {
        (2,0) : 32,
        (2,1) : 48,
        (3,0) : 192,
        (3,5) : 192,
        (3,7) : 192,
        (5,0) : 128,
        (5,2) : 128,
        (6,0) : 64,
        (6,1) : 128,
        (7,0) : 64,
        (7,5) : 64,
        (8,0) : 64,
        (8,6) : 128,
        (8,9) : 128,
        (9,0) : 128
        }
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu_info['gpu_name'] = gpus[0].name  # GPU name
        gpu_info['gpu_memory_total'] = gpus[0].memoryTotal  # Total GPU memory in MB
        device = cuda.get_current_device()
        sms = getattr(device, 'MULTIPROCESSOR_COUNT')
        cc_major = device.COMPUTE_CAPABILITY_MAJOR
        cc_minor = device.COMPUTE_CAPABILITY_MINOR
        cores_per_sm = cc_cores_per_SM_dict.get((cc_major,cc_minor))
        gpu_info['total_cores'] = cores_per_sm*sms
    else:

        gpu_info['gpu_name'] = 'None'
        gpu_info['gpu_memory_total'] = 'None'
    return gpu_info

# pdb.set_trace()
hw_info = {}
hw_info = get_cpu_info(hw_info)
hw_info = get_gpu_info(hw_info)
hw_info = get_platform_info(hw_info)
hw_info = [hw_info]

if(os.path.exists('../system_info.json')):
    with open('../system_info.json', 'r') as file:
        existing_data = json.load(file)
    existing_data.extend(hw_info)
    with open('../system_info.json', 'w') as file:
        json.dump(existing_data, file, indent=4)
else:
    with open('../system_info.json', 'w') as file:
        json.dump(hw_info, file, indent=4)

