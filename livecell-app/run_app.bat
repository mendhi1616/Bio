@echo off
setlocal

title Live-Cell - Initialisation

echo Vérification du chemin Python...
set "PYTHON_PATH=%LocalAppData%\Programs\Python\Python312\python.exe"

if not exist "%PYTHON_PATH%" (
    echo Python 3.12 introuvable. Téléchargement et installation...
    python ensure_python312.py
)

echo Vérification et installation des dépendances...
"%PYTHON_PATH%" verify_env.py

echo Lancement de Live-Cell...
"%PYTHON_PATH%" app_flet.py

if %errorlevel% neq 0 (
    echo.
    echo Une erreur s'est produite pendant l'exécution.
    echo Consulte les messages ci-dessus.
    echo.
    pause
) else (
    echo.
    echo Fermeture normale.
    echo.
    pause
)

endlocal
