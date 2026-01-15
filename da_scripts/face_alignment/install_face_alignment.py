import sys
import subprocess
import pkg_resources

def check_and_install():
    required_packages = [
        'face-alignment',
        'scipy',
        'opencv-python'
    ]
    
    installed = {pkg.key for pkg in pkg_resources.working_set}
    
    for package in required_packages:
        if package.replace('-', '_') not in installed:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                package
            ])
        else:
            print(f"✓ {package} already installed")
    
    print("\nInstallation complete!")

if __name__ == "__main__":
    check_and_install()