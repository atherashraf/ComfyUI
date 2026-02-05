
import toml
# from pathlib import Path


def generate_kohya_config(output_file, train_data_dir, output_dir, logging_dir, model_path,concept_name):
    """
    Generates TOML config for SDXL training (Low VRAM + Resume Support).
    """

    model_path_str = str(model_path).replace('\\', '/')

    config = {
        "model_arguments": {
            "pretrained_model_name_or_path": model_path_str,
        },
        "additional_network_arguments": {
            "unet_lr": 0.0004,
            "text_encoder_lr": 0.0004,
            "network_module": "networks.lora",
            "network_dim": 64,
            "network_alpha": 32,
            "network_train_unet_only": True,  # Keep True for Low VRAM
            "network_train_text_encoder_only": False
        },
        "optimizer_arguments": {
            "optimizer_type": "AdamW8bit",
            "learning_rate": 0.0004,
            "max_grad_norm": 1.0,
            "lr_scheduler": "cosine_with_restarts",
            "lr_warmup_steps": 0,
        },
        "dataset_arguments": {
            "cache_latents": True,
            "debug_dataset": False,
            "resolution": "1024,1024",
            "train_data_dir": str(train_data_dir),
            "caption_extension": ".txt",
        },
        "training_arguments": {
            "output_dir": str(output_dir),
            "output_name": f"{concept_name}_sdxl_lora",
            "save_precision": "fp16",
            "save_every_n_epochs": 5,

            # --- RESUME SETTINGS ---
            "save_state": True,  # Saves restart data
            "save_last_n_epochs_state": 1,  # Keep only the most recent state to save space
            # -----------------------

            "train_batch_size": 1,
            "max_token_length": 225,
            "mem_eff_attn": False,
            "xformers": True,
            "max_train_epochs": 30,
            "max_data_loader_n_workers": 0,
            "persistent_data_loader_workers": False,
            "gradient_checkpointing": True,
            "mixed_precision": "fp16",
            "logging_dir": str(logging_dir),
            "noise_offset": 0.1,
        },
        "sdxl_arguments": {
            "cache_text_encoder_outputs": True,
            "cache_text_encoder_outputs_to_disk": True,
            "no_half_vae": True
        }
    }

    if not output_file.lower().endswith('.toml'):
        output_file = output_file + '.toml'

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            toml.dump(config, f)
        print(f"✓ Training config saved to: {output_file}")
    except Exception as e:
        print(f"Error saving config: {e}")