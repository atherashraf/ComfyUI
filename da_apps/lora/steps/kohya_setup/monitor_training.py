# Monitor training logs
def monitor_training(log_dir="D:/lora_training/logs"):
    """Monitor training progress."""
    import json
    from pathlib import Path

    log_path = Path(log_dir)

    if not log_path.exists():
        print(f"Log directory not found: {log_dir}")
        return

    # Check for tensorboard logs
    tb_logs = list(log_path.glob("*/events.out.tfevents.*"))

    if tb_logs:
        print("TensorBoard logs found. Run:")
        print(f"tensorboard --logdir {log_dir}")
    else:
        # Check text logs
        text_logs = list(log_path.glob("*.log"))
        for log_file in text_logs[-3:]:  # Last 3 logs
            print(f"\n--- {log_file.name} ---")
            with open(log_file, 'r') as f:
                lines = f.readlines()[-20:]  # Last 20 lines
                for line in lines:
                    print(line.strip())

    # Check output folder for checkpoints
    output_path = Path("D:/lora_training/output")
    if output_path.exists():
        lora_files = list(output_path.glob("*.safetensors"))
        print(f"\nFound {len(lora_files)} LoRA checkpoints")
        for lora in lora_files[-5:]:  # Last 5 checkpoints
            print(f"  - {lora.name}")


if __name__ == "__main__":
    # Run monitoring
    monitor_training()