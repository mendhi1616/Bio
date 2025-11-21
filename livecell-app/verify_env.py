import subprocess
import sys
import os
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

def bootstrap_env():
    try:
        subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"Erreur lors de la mise à jour de pip/setuptools : {e}")

def ensure_torch():
    try:
        import torch
    except ImportError:
        print("Installation de PyTorch (détection automatique CPU/GPU)...")
        has_cuda = False
        try:
            import torch.cuda
            has_cuda = torch.cuda.is_available()
        except Exception:
            pass

        if has_cuda:
            subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 "torch==2.5.1+cu121", "torchvision==0.20.1+cu121",
                 "--index-url", "https://download.pytorch.org/whl/cu121"]
            )
        else:
            subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 "torch==2.5.1+cpu", "torchvision==0.20.1+cpu",
                 "--index-url", "https://download.pytorch.org/whl/cpu"]
            )

REQUIRED = {
    "flet": ">=0.22.0",
    "numpy": ">=1.26.0",
    "pandas": ">=2.2.0",
    "scikit-image": ">=0.24.0",
    "matplotlib": ">=3.9.0",
    "tifffile": ">=2024.5.22",
    "scipy": ">=1.13.0",
    "tqdm": ">=4.66.0",
    "cellpose": ">=4.0.7",
    "opencv-python": ">=4.10.0",
    "Pillow": ">=10.4.0",
    "requests": ">=2.32.0",
    "flask": ">=3.0.3",
    "Werkzeug": ">=3.0.3",
    "gunicorn": ">=22.0.0",
    "certifi": ">=2024.6.2",
    "psutil": ">=6.0.0",
    "platformdirs": ">=4.3.2",
    "uuid": ">=1.30.0",
    "jsonschema": ">=4.22.0",
    "pathlib": ">=1.0.1",
    "torch": ">=2.5.1",
    "torchvision": ">=0.20.1",
    "torchaudio": ">=2.5.1",
    "jinja2": ">=3.1.0",
    "fastapi": ">=0.115.0",
    "h5py": ">=3.11.0",
    "imageio": ">=2.34.0",
    "typing-extensions": ">=4.8.0",
    "plotly": ">=5.0.0",
    "napari": ">=0.5.0",
}


_verified_flag = os.path.join(os.path.dirname(__file__), ".env_verified")

def check_and_install():
    if os.path.exists(_verified_flag):
        return  

    print("Vérification de l'environnement LiveCell...")
    bootstrap_env()
    ensure_torch()

    import pkg_resources
    missing = []

    for pkg, version in REQUIRED.items():
        try:
            pkg_resources.require(f"{pkg}{version}")
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            missing.append(f"{pkg}{version}")

    if missing:
        print("Installation des dépendances manquantes...")
        subprocess.run([sys.executable, "-m", "pip", "install", *missing])
        print("Mise à jour de l'environnement terminée.")
    else:
        print("Tous les modules sont déjà à jour.")

    with open(_verified_flag, "w") as f:
        f.write("checked")

if __name__ == "__main__":
    check_and_install()
