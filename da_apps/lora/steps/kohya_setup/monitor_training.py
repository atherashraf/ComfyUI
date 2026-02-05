from __future__ import annotations

from pathlib import Path
from typing import Optional


def monitor_training(log_dir: str = r"D:\lora_training\logs", work_dir: str = r"D:\lora_training") -> None:
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Log directory not found: {log_dir}")
        return

    tb_logs = list(log_path.glob("*/events.out.tfevents.*"))
    if tb_logs:
        print("TensorBoard logs found. Run:")
        print(f"  tensorboard --logdir \"{log_dir}\"")
    else:
        text_logs = sorted(log_path.glob("*.log"), key=lambda p: p.stat().st_mtime)
        for lf in text_logs[-3:]:
            print(f"\n--- {lf.name} ---")
            try:
                lines = lf.read_text(encoding="utf-8", errors="ignore").splitlines()[-30:]
                for line in lines:
                    print(line)
            except Exception as e:
                print(f"(Could not read {lf.name}: {e})")

    work = Path(work_dir)
    outs = sorted([p for p in work.glob("output_*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if not outs:
        return

    newest = outs[-1]
    ckpts = list(newest.glob("*.safetensors"))
    print(f"\nLatest output: {newest}")
    print(f"Found {len(ckpts)} LoRA checkpoints")
    for p in sorted(ckpts, key=lambda x: x.stat().st_mtime)[-5:]:
        print(f"  - {p.name}")


if __name__ == "__main__":
    monitor_training()
