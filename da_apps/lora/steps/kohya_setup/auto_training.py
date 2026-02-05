from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import toml


@dataclass
class KohyaRuntime:
    kohya_root: Path = Path(r"D:\DevProjects\PhotoAI\kohya_ss")
    python_exe: Path = Path(r"D:\DevProjects\PhotoAI\kohya_ss\venv\Scripts\python.exe")
    sdxl_script: Path = Path(r"D:\DevProjects\PhotoAI\kohya_ss\sd-scripts\sdxl_train_network.py")
    accelerate_config: Path = Path(r"D:\DevProjects\PhotoAI\kohya_ss\accelerate_config.yaml")


ACCELERATE_YAML = """\
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


def create_accelerate_config(rt: KohyaRuntime) -> None:
    rt.accelerate_config.write_text(ACCELERATE_YAML, encoding="utf-8")


def get_latest_state_info(config_path: Path) -> Tuple[Optional[Path], int]:
    try:
        conf = toml.loads(Path(config_path).read_text(encoding="utf-8"))
        output_dir = Path(conf["training_arguments"]["output_dir"])
        if not output_dir.exists():
            return None, 0

        states = sorted(output_dir.glob("*-state"), key=os.path.getmtime)
        if not states:
            return None, 0

        latest = states[-1]
        m = re.search(r"step(\d+)", latest.name)
        return latest, int(m.group(1)) if m else 0
    except Exception as e:
        print(f"Warning: Could not check for resume state: {e}")
        return None, 0


def train_sdxl_lora_auto(config_toml_path: Path, rt: KohyaRuntime = KohyaRuntime()) -> None:
    config_toml_path = Path(config_toml_path)

    if not rt.python_exe.exists():
        raise FileNotFoundError(f"Python not found at {rt.python_exe}")
    if not rt.sdxl_script.exists():
        raise FileNotFoundError(f"SDXL training script not found at {rt.sdxl_script}")

    create_accelerate_config(rt)

    cmd = [
        str(rt.python_exe),
        "-m", "accelerate.commands.launch",
        "--config_file", "accelerate_config.yaml",
        "--num_cpu_threads_per_process=2",
        str(rt.sdxl_script),
        "--config_file", str(config_toml_path),
    ]

    resume_folder, resume_step = get_latest_state_info(config_toml_path)
    if resume_folder:
        print(f"\n🔄 Found previous training state: {resume_folder.name}")
        print(f"   Last saved at step: {resume_step}")
        choice = input("   Resume training? (y/n): ").strip().lower()
        if choice == "y":
            cmd.append(f"--resume={resume_folder}")
            print(f"   Resuming enabled from step {resume_step}!")

    print("\n🚀 Starting SDXL Training...")
    subprocess.run(cmd, check=True, cwd=rt.kohya_root)
    print("\n✅ Training completed successfully!")
