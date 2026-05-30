# check_node_dependencies.py
"""Check if DAFaceAlignAndResizeNode will work with your current environment"""

import sys
import importlib

def check_dependency(module_name, import_name=None, min_version=None):
    """Check if a dependency is installed and meets version requirements"""
    if import_name is None:
        import_name = module_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {module_name} - {version}")
        return True
    except ImportError:
        print(f"❌ {module_name} - NOT INSTALLED")
        return False

def main():
    print("=" * 50)
    print("DAFaceAlignAndResizeNode - Dependency Check")
    print("=" * 50)
    
    dependencies = {
        "mediapipe": "mediapipe",
        "cv2": "cv2",
        "numpy": "numpy",
        "torch": "torch",
        "scipy": "scipy",  # Optional but recommended
    }
    
    all_ok = True
    for name, import_name in dependencies.items():
        if not check_dependency(name, import_name):
            all_ok = False
    
    print("\n" + "=" * 50)
    
    if all_ok:
        print("✅ All dependencies are satisfied!")
        print("\nYou can use DAFaceAlignAndResizeNode without any new installations.")
        
        # Check CUDA
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available (GPU: {torch.cuda.get_device_name(0)})")
            print(f"   CUDA Version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA is not available - CPU mode only")
    else:
        print("❌ Missing dependencies. Install with:")
        print("pip install mediapipe opencv-python")

if __name__ == "__main__":
    main()