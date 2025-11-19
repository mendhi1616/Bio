import os
import sys
import subprocess
import urllib.request
import tempfile
import platform

if platform.system() == "Darwin":
    print("macOS détecté — Vérification de Python 3.12...")
    try:
        out = subprocess.check_output(["python3", "-V"]).decode()
        if "3.12" in out:
            print("Python 3.12 déjà présent.")
        else:
            print("Version différente détectée :", out)
            print("Installez via Homebrew : brew install python@3.12")
    except Exception:
        print("Python introuvable — installez-le avec : brew install python@3.12")
else:
    pass


TARGET_VERSION = (3, 12)
PYTHON_INSTALLER_URL = "https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe"

def is_python312():
    return sys.version_info[:2] == TARGET_VERSION

def download_python_installer():
    print("⬇Téléchargement de l’installeur Python 3.12.0 depuis python.org ...")
    tmp_path = os.path.join(tempfile.gettempdir(), "python312_installer.exe")
    urllib.request.urlretrieve(PYTHON_INSTALLER_URL, tmp_path)
    print("Téléchargement terminé :", tmp_path)
    return tmp_path

def install_python312(installer_path):
    print("Installation silencieuse de Python 3.12 (cela peut prendre 1-2 minutes)...")
    cmd = f'"{installer_path}" /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1 Include_test=0'
    subprocess.run(cmd, shell=True)
    print("Installation terminée.")

def relaunch_with_python312():
    possible_paths = [
        r"C:\Program Files\Python312\python.exe",
        r"C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe",
    ]
    for path in possible_paths:
        expanded = os.path.expandvars(path)
        if os.path.exists(expanded):
            print(f"Relance de Live-Cell avec {expanded}")
            os.execv(expanded, [expanded] + sys.argv)
    print("Python 3.12 introuvable après installation. Relance manuelle requise.")

def main():
    print(f"Version actuelle : Python {platform.python_version()}")
    if not is_python312():
        print("Version incompatible — installation de Python 3.12 requise.")
        installer = download_python_installer()
        install_python312(installer)
        relaunch_with_python312()
    else:
        print("Version compatible : Python 3.12 détecté. Lancement de Live-Cell...")

if __name__ == "__main__":
    main()
