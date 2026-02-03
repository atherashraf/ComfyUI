# import subprocess
# import os
# import sys
# from pathlib import Path
#
# # --- CONFIGURATION ---
# KOHYA_ROOT = Path(r"D:\DevProjects\PhotoAI\kohya_ss")
#
# # !!! FIX: Use the SDXL specific script !!!
# KOHYA_SCRIPT = KOHYA_ROOT / "sd-scripts" / "sdxl_train_network.py"
#
# PYTHON_EXE = KOHYA_ROOT / "venv" / "Scripts" / "python.exe"
# ACCELERATE_CONFIG = KOHYA_ROOT / "accelerate_config.yaml"
#
#
# def create_accelerate_config():
#     """Creates the hardware config for accelerate."""
#     config_content = """
# compute_environment: LOCAL_MACHINE
# debug: false
# distributed_type: NO
# downcast_bf16: 'no'
# machine_rank: 0
# main_training_function: main
# mixed_precision: fp16
# num_machines: 1
# num_processes: 1
# rdzv_backend: static
# same_network: true
# tpu_env: []
# tpu_use_cluster: false
# tpu_use_sudo: false
# use_cpu: false
# """
#     try:
#         with open(ACCELERATE_CONFIG, "w", encoding='utf-8') as f:
#             f.write(config_content)
#     except Exception as e:
#         print(f"❌ Failed to create accelerate config: {e}")
#
#
# def train_sdxl_lora_auto(config_toml_path):
#     if not PYTHON_EXE.exists():
#         print(f"❌ Error: Python not found at {PYTHON_EXE}")
#         return
#
#     if not KOHYA_SCRIPT.exists():
#         print(f"❌ Error: SDXL Training script not found at {KOHYA_SCRIPT}")
#         return
#
#     create_accelerate_config()
#
#     print(f"\n🚀 Starting SDXL Training...")
#     print(f"   Script: {KOHYA_SCRIPT.name}")
#     print(f"   Config: {config_toml_path}")
#
#     cmd = [
#         str(PYTHON_EXE), "-m", "accelerate.commands.launch",
#         "--config_file", "accelerate_config.yaml",
#         "--num_cpu_threads_per_process=2",
#         str(KOHYA_SCRIPT),
#         "--config_file", str(config_toml_path)
#     ]
#
#     try:
#         subprocess.run(cmd, check=True, cwd=KOHYA_ROOT)
#         print("\n✅ Training completed successfully!")
#     except subprocess.CalledProcessError as e:
#         print(f"\n❌ Training failed! Error code: {e.returncode}")


import subprocess
import os
import sys
import toml
from pathlib import Path

# --- CONFIGURATION ---
# KOHYA_ROOT = Path(r"D:\DevProjects\PhotoAI\kohya_ss")
# KOHYA_SCRIPT = KOHYA_ROOT / "sd-scripts" / "sdxl_train_network.py"
# PYTHON_EXE = KOHYA_ROOT / "venv" / "Scripts" / "python.exe"
# ACCELERATE_CONFIG = KOHYA_ROOT / "accelerate_config.yaml"
#
#
# def create_accelerate_config():
#     """Creates the hardware config for accelerate."""
#     config_content = """
# compute_environment: LOCAL_MACHINE
# debug: false
# distributed_type: NO
# downcast_bf16: 'no'
# machine_rank: 0
# main_training_function: main
# mixed_precision: fp16
# num_machines: 1
# num_processes: 1
# rdzv_backend: static
# same_network: true
# tpu_env: []
# tpu_use_cluster: false
# tpu_use_sudo: false
# use_cpu: false
# """
#     try:
#         with open(ACCELERATE_CONFIG, "w", encoding='utf-8') as f:
#             f.write(config_content)
#     except Exception as e:
#         print(f"❌ Failed to create accelerate config: {e}")
#
#
# def get_latest_state_folder(config_path):
#     """Reads config to find output dir, then looks for state folders."""
#     try:
#         with open(config_path, 'r', encoding='utf-8') as f:
#             conf = toml.load(f)
#
#         output_dir = Path(conf['training_arguments']['output_dir'])
#         if not output_dir.exists():
#             return None
#
#         # State folders usually end in "-state" (e.g., "judi-at-step-500-state")
#         state_folders = sorted(list(output_dir.glob("*-state")), key=os.path.getmtime)
#
#         if state_folders:
#             return state_folders[-1]  # Return the newest one
#     except Exception as e:
#         print(f"Warning: Could not check for resume state: {e}")
#
#     return None
#
#
# def train_sdxl_lora_auto(config_toml_path):
#     if not PYTHON_EXE.exists():
#         print(f"❌ Error: Python not found at {PYTHON_EXE}")
#         return
#
#     if not KOHYA_SCRIPT.exists():
#         print(f"❌ Error: SDXL Training script not found at {KOHYA_SCRIPT}")
#         return
#
#     create_accelerate_config()
#
#     # --- Check for Resume ---
#     cmd = [
#         str(PYTHON_EXE), "-m", "accelerate.commands.launch",
#         "--config_file", "accelerate_config.yaml",
#         "--num_cpu_threads_per_process=2",
#         str(KOHYA_SCRIPT),
#         "--config_file", str(config_toml_path)
#     ]
#
#     resume_folder = get_latest_state_folder(config_toml_path)
#     if resume_folder:
#         print(f"\n🔄 Found previous training state: {resume_folder.name}")
#         print("   Resuming will continue exactly where you left off.")
#         choice = input("   Resume training? (y/n): ").lower()
#         if choice == 'y':
#             cmd.append(f"--resume={resume_folder}")
#             print("   Resuming enabled!")
#         else:
#             print("   Starting fresh training.")
#
#     print(f"\n🚀 Starting SDXL Training...")
#
#     try:
#         subprocess.run(cmd, check=True, cwd=KOHYA_ROOT)
#         print("\n✅ Training completed successfully!")
#     except subprocess.CalledProcessError as e:
#         print(f"\n❌ Training failed! Error code: {e.returncode}")


import subprocess
import os
import sys
import toml
import re  # Added for parsing step numbers
from pathlib import Path

# --- CONFIGURATION ---
KOHYA_ROOT = Path(r"D:\DevProjects\PhotoAI\kohya_ss")
KOHYA_SCRIPT = KOHYA_ROOT / "sd-scripts" / "sdxl_train_network.py"
PYTHON_EXE = KOHYA_ROOT / "venv" / "Scripts" / "python.exe"
ACCELERATE_CONFIG = KOHYA_ROOT / "accelerate_config.yaml"


def create_accelerate_config():
    """Creates the hardware config for accelerate."""
    config_content = """
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: NO
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""
    try:
        with open(ACCELERATE_CONFIG, "w", encoding='utf-8') as f:
            f.write(config_content)
    except Exception as e:
        print(f"❌ Failed to create accelerate config: {e}")


def get_latest_state_info(config_path):
    """
    Reads config to find output dir, then looks for state folders.
    Returns: (Path to state folder, Step number as int)
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            conf = toml.load(f)

        output_dir = Path(conf['training_arguments']['output_dir'])
        if not output_dir.exists():
            return None, 0

        # State folders usually end in "-state" (e.g., "judi-step00000500-state")
        # We sort by modification time to get the newest one
        state_folders = sorted(list(output_dir.glob("*-state")), key=os.path.getmtime)

        if state_folders:
            latest_folder = state_folders[-1]

            # Use regex to extract the step number from the folder name
            # Looks for "step" followed by digits
            match = re.search(r'step(\d+)', latest_folder.name)
            if match:
                return latest_folder, int(match.group(1))

            return latest_folder, 0  # Found folder but couldn't parse number

    except Exception as e:
        print(f"Warning: Could not check for resume state: {e}")

    return None, 0


def train_sdxl_lora_auto(config_toml_path):
    if not PYTHON_EXE.exists():
        print(f"❌ Error: Python not found at {PYTHON_EXE}")
        return

    if not KOHYA_SCRIPT.exists():
        print(f"❌ Error: SDXL Training script not found at {KOHYA_SCRIPT}")
        return

    create_accelerate_config()

    # --- Check for Resume ---
    cmd = [
        str(PYTHON_EXE), "-m", "accelerate.commands.launch",
        "--config_file", "accelerate_config.yaml",
        "--num_cpu_threads_per_process=2",
        str(KOHYA_SCRIPT),
        "--config_file", str(config_toml_path)
    ]

    resume_folder, resume_step = get_latest_state_info(config_toml_path)

    if resume_folder:
        print(f"\n🔄 Found previous training state: {resume_folder.name}")
        # Show the user exactly where they are starting
        print(f"   Last saved at step: {resume_step}")
        print("   Resuming will continue exactly where you left off.")

        choice = input("   Resume training? (y/n): ").lower()
        if choice == 'y':
            cmd.append(f"--resume={resume_folder}")
            print(f"   Resuming enabled from step {resume_step}!")
        else:
            print("   Starting fresh training.")

    print(f"\n🚀 Starting SDXL Training...")

    try:
        # We allow standard output (console) so you see Kohya's native progress bar
        subprocess.run(cmd, check=True, cwd=KOHYA_ROOT)
        print("\n✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed! Error code: {e.returncode}")