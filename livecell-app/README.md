# Live-Cell — Project Explanation

Live-Cell is a standalone application designed to automate the analysis of cell microscopy images using artificial intelligence. The main goal of the project is to provide biologists and students with a simple, reliable, and fast tool for detecting, segmenting, and analyzing live cells without requiring programming experience.

## 1. Context & Motivation

Modern microscopes produce hundreds to thousands of images per experiment. Manually analyzing these images is:
* time-consuming
* subjective
* error-prone
* difficult for beginners

Existing tools such as ImageJ or CellProfiler are powerful but often complex to configure, especially for students or small research teams.
Live-Cell solves this by offering a one-click solution powered by AI.

***

## 2. Main Objective

The objective of Live-Cell is to provide an application capable of:
* Loading microscope images or image sequences
* Automatically detecting and segmenting cells using an AI model (Cellpose)
* **Tracking cell movement across frames and analyzing key metrics (velocity, distance, morphology)**
* Exporting numerical data and annotated images
* Running smoothly without manual installation, even on non-technical machines

***

## 3. Key Features

### Automatic AI Segmentation
* Uses **Cellpose** (deep learning model specialized for cell morphology)
* Works on wild-type, knockout, and multiple experimental conditions
* Produces clean overlay images with segmented contours
* Includes **"Mode Rapide" (Fast Mode)** to optimize performance by processing 1 frame sur 2.

### Cell Tracking and Quantitative Analysis (Nouveautés)
* **Advanced Tracking:** Implémente le suivi complet des cellules à travers les séquences, utilisant l'algorithme LAP (Linear Assignment Problem) avec gestion des écarts (gap closing).
* **Calcul des Métriques:** Calcule automatiquement des métriques de mouvement (vitesse en $\mu m/s$, distance cumulée, rectitude) et de morphologie (aire en $\mu m^2$, circularité, allongement).
* **Détection des Mitoses:** Identifie les événements de division cellulaire.
* **Visualisation Avancée:** Interface dédiée avec un `TrackViewer` pour revoir les trajectoires et contours des cellules image par image.

### Reporting et Flux de Travail
* **Rapports Statistiques:** Génération de courbes interactives de **Prolifération** et de **Survie**.
* **Outil de Comparaison:** Fonctionnalité intégrée pour comparer les métriques clés entre deux conditions/fichiers.
* **Export Complet:** Exportation des données de tracking au format **CSV** et génération d'une **vidéo MP4** annotée (superposition du tracking).
* **Intégration Napari:** Permet d'ouvrir le tracking et les masques dans **Napari** (viewer externe) pour la correction manuelle et d'importer les masques corrigés pour un nouveau calcul de métriques.

### User-friendly interface (Flet)
* Intuitive, clean, easy to use
* Organizes experiments by condition (KO, WT, outputs…)
* No coding required

### Automatic environment management
The app includes:
* Automatic dependency installation
* GPU/CPU autodetection
* Auto-installation of Python 3.12 when needed
* Torch CUDA installation if GPU available
* Environment cleanup at exit

This makes Live-Cell portable, especially for students.

### Security & Access Control
To share the software safely, Live-Cell includes:
* A remote whitelist system
* Device registration with nickname
* Logs for admin (who added/removed which device and when)
* Ability to block/unblock machines remotely

***

## 4. Architecture Overview

The project is built around four major components:

1.  **app_flet.py**
    The user interface. Handles:
    * file loading
    * UI logic
    * segmentation preview
    * experiment navigation

2.  **pipeline.py**
    The analysis engine:
    * preprocessing
    * segmentation
    * overlay generation
    * **cell tracking and metric calculation**
    * exporting outputs

3.  **verify_env.py**
    Ensures the environment is correct:
    * installs missing packages
    * updates pip
    * installs PyTorch CUDA or CPU version
    * suppresses logs for user comfort

4.  **ensure_python312.py**
    If Python 3.12 is not installed:
    * downloads installer
    * installs silently
    * relaunches the app automatically

***

## 5. Results

The application successfully performs:
* clean segmentation of cells
* visualization of boundaries on raw images
* classification of experimental conditions
* **tracking of cell movement and calculation of quantitative metrics**
* ready-to-analyze numerical outputs and **statistical reports** (courbes de prolifération/survie, comparaison de groupes) for biology experiments.

Your own dataset images (wild-type & knockout) show high-quality contour detection, indicating that the chosen AI model is suitable for your cell type.

***

## 6. Impact

Live-Cell is useful for:
* students learning microscopy analysis
* researchers needing fast qualitative segmentation
* small labs without access to complex software
* teaching and demonstrations

It lowers the barrier between biology and computational analysis while maintaining high scientific reliability.

***

## 7. Future Improvements

Planned or possible enhancements include:
* improved models for specific cell types
* real-time live microscopy analysis
* cloud synchronization for multiple users
* Statistical reports integrated in the app.
* Intégration d'outils d'ajustement automatique des paramètres de segmentation pour différents contrastes d'images.

***

## 8. Conclusion

Live-Cell is a lightweight, intelligent, and accessible tool that brings AI-powered cell segmentation **and quantitative tracking** into the hands of students and biologists. It combines deep learning, automation, and simple interface design to solve a real problem in biological image analysis, while remaining easy to distribute and secure.