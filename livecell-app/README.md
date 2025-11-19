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

## 2. Main Objective

The objective of Live-Cell is to provide an application capable of:
* Loading microscope images or image sequences
* Automatically detecting and segmenting cells using an AI model (Cellpose)
* Highlighting contours directly on the images
* (Future) Tracking cell movement across frames
* Exporting numerical data and annotated images
* Running smoothly without manual installation, even on non-technical machines

## 3. Key Features

### Automatic AI segmentation
* Uses Cellpose (deep learning model specialized for cell morphology)
* Works on wild-type, knockout, and multiple experimental conditions
* Produces clean overlay images with segmented contours

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

## 4. Architecture Overview

The project is built around four major components:

1. **app_flet.py**
   The user interface. Handles:
   * file loading
   * UI logic
   * segmentation preview
   * experiment navigation

2. **pipeline.py**
   The analysis engine:
   * preprocessing
   * segmentation
   * overlay generation
   * exporting outputs

3. **verify_env.py**
   Ensures the environment is correct:
   * installs missing packages
   * updates pip
   * installs PyTorch CUDA or CPU version
   * suppresses logs for user comfort

4. **ensure_python312.py**
   If Python 3.12 is not installed:
   * downloads installer
   * installs silently
   * relaunches the app automatically

## 5. Results

The application successfully performs:
* clean segmentation of cells
* visualization of boundaries on raw images
* classification of experimental conditions
* ready-to-analyze outputs for biology experiments

Your own dataset images (wild-type & knockout) show high-quality contour detection, indicating that the chosen AI model is suitable for your cell type.

## 6. Impact

Live-Cell is useful for:
* students learning microscopy analysis
* researchers needing fast qualitative segmentation
* small labs without access to complex software
* teaching and demonstrations

It lowers the barrier between biology and computational analysis while maintaining high scientific reliability.

## 7. Future Improvements

Planned or possible enhancements include:
* full cell tracking (movement, trajectory, velocity)
* improved models for specific cell types
* real-time live microscopy analysis
* cloud synchronization for multiple users
* statistical reports integrated in the app

## 8. Conclusion

Live-Cell is a lightweight, intelligent, and accessible tool that brings AI-powered cell segmentation into the hands of students and biologists. It combines deep learning, automation, and simple interface design to solve a real problem in biological image analysis, while remaining easy to distribute and secure.
