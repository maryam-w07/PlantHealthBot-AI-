# PlantHealthBot-AI-

**An AI-Powered Cable-Driven Robotic System for Plant Health Assessment in Vertical Farming**

## Overview
This project implements a cost-effective, AI-powered cable-driven robotic system for monitoring plant health in vertical farming environments. It features:
- A custom CNN (TensorFlow/Keras) that classifies plant images as healthy or unhealthy.
- An image preprocessing pipeline using OpenCV.
- Real-time video monitoring with automatic email alerts upon detecting unhealthy plants.
- Integration with a motorized pulley and live-streaming camera for autonomous scanning.


## Environment Setup

To recreate the conda environment used for this project, run:
```bash
conda env create -f environment.yml

## Dataset
This project uses a custom image dataset of lettuce plants, organized into two folders based on health condition:
healthy/: contains images of healthy lettuce plants.
unhealthy/: contains images of lettuce plants showing signs of disease or stress.
Note: The dataset was self-collected and is not publicly shared due to privacy or usage constraints. If you're replicating this project or experimenting with the code, you can replace our dataset with a similar plant health image dataset (or your own images) structured in the same way (i.e. with healthy/ and unhealthy/ subfolders)
