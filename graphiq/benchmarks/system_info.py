import platform
import psutil
import cpuinfo


def print_system_info():
    # Basic system information
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")

    # CPU information
    cpu_info = cpuinfo.get_cpu_info()
    print(f"CPU: {cpu_info['brand_raw']}")
    print(f"Number of physical cores: {psutil.cpu_count(logical=False)}")
    print(f"Number of logical cores: {psutil.cpu_count(logical=True)}")
    print(f"Max CPU frequency: {psutil.cpu_freq().max:.2f} Mhz")

    # Memory information
    svmem = psutil.virtual_memory()
    print(f"Total memory: {svmem.total / (1024 ** 3):.2f} GB")

    # Python version
    print(f"Python version: {platform.python_version()}")
