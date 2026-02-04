# scripts/smart_model_cache.py
from pathlib import Path
import os
import sys
import torch

# Torch performance settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ComfyUI setup
COMFY_PATH = os.path.abspath("../ComfyUI")
sys.path.append(COMFY_PATH)


# Import after adding to path
from nodes import CheckpointLoaderSimple


class SmartModelCache:
    def __init__(self, max_cache_size=3):
        self.cache = {}
        self.usage_order = []
        self.max_cache_size = max_cache_size
        self.checkpoints_dir = Path(COMFY_PATH) / "models" / "checkpoints"

        # Ensure checkpoints directory exists
        if not self.checkpoints_dir.exists():
            print(f"⚠️ Checkpoints directory not found: {self.checkpoints_dir}")
            print(f"   Creating directory...")
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        print(f"[CACHE] Initialized with max {max_cache_size} models")
        print(f"[CACHE] Checkpoints dir: {self.checkpoints_dir}")

    def get_checkpoint_path(self, checkpoint_file: str) -> Path:
        """Get full path to checkpoint file with fallbacks"""
        # Try direct filename
        path = self.checkpoints_dir / checkpoint_file
        if path.exists():
            return path

        # Try with .safetensors extension if not provided
        if not checkpoint_file.endswith(('.safetensors', '.ckpt', '.pt', '.pth')):
            for ext in ['.safetensors', '.ckpt', '.pt']:
                path_with_ext = self.checkpoints_dir / f"{checkpoint_file}{ext}"
                if path_with_ext.exists():
                    return path_with_ext

        # List available checkpoints for error message
        available = list(self.checkpoints_dir.glob("*.safetensors")) + \
                    list(self.checkpoints_dir.glob("*.ckpt")) + \
                    list(self.checkpoints_dir.glob("*.pt")) + \
                    list(self.checkpoints_dir.glob("*.pth"))

        available_names = [f.name for f in available]
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_file}' not found in {self.checkpoints_dir}.\n"
            f"Available checkpoints: {available_names}"
        )

    def load_checkpoint(self, checkpoint_file: str):
        """Load checkpoint with LRU caching"""
        # If already cached, move to front (most recently used)
        if checkpoint_file in self.cache:
            if checkpoint_file in self.usage_order:
                self.usage_order.remove(checkpoint_file)
            self.usage_order.insert(0, checkpoint_file)
            print(f"[CACHE] Using cached: {checkpoint_file}")
            return self.cache[checkpoint_file]

        # Load new checkpoint
        print(f"[CACHE] Loading: {checkpoint_file}")
        checkpoint_path = self.get_checkpoint_path(checkpoint_file)

        print(f"[CACHE] Loading from: {checkpoint_path}")
        ckpt_loader = CheckpointLoaderSimple()
        model, clip, vae = ckpt_loader.load_checkpoint(
            ckpt_name=checkpoint_path.name
        )

        # Cache it
        self.cache[checkpoint_file] = (model, clip, vae)
        self.usage_order.insert(0, checkpoint_file)

        # If cache full, remove least recently used
        if len(self.cache) > self.max_cache_size:
            lru = self.usage_order.pop()
            if lru in self.cache:
                del self.cache[lru]
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"[CACHE] Evicted: {lru}")

        return model, clip, vae

    def clear_cache(self):
        """Clear model cache to free VRAM"""
        self.cache.clear()
        self.usage_order.clear()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        print("[CACHE] Cleared all cached models")

    def get_cache_info(self):
        """Get information about cached models"""
        return {
            "cached_models": list(self.cache.keys()),
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "usage_order": self.usage_order
        }

    def list_available_checkpoints(self):
        """List all available checkpoints in directory"""
        if not self.checkpoints_dir.exists():
            return []

        checkpoints = []
        extensions = ['*.safetensors', '*.ckpt', '*.pt', '*.pth']
        for ext in extensions:
            for file in self.checkpoints_dir.glob(ext):
                size_mb = file.stat().st_size / (1024 * 1024)
                checkpoints.append({
                    "file": file.name,
                    "size_mb": round(size_mb, 2),
                    "type": ext.replace('*.', '')
                })

        checkpoints.sort(key=lambda x: x["file"].lower())
        return checkpoints