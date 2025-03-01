## Forest Detection Model

### This folder contains a deep learning model for detecting forest cover in satellite imagery. The model is part of a larger system for verifying carbon credit claims and detecting potential fraud in environmental reporting.

## Overview

### The forest detection model uses a U-Net architecture with a pre-trained MobileNetV2 backbone to segment forested areas in multispectral satellite imagery. It takes 64x64 images with 5 channels (RGB + NDVI + Land Cover) as input and outputs binary segmentation masks indicating forest presence.