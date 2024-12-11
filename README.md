# Marine Debris Detection using YOLOv8-Segmentation

## Overview
This repository contains the implementation of a novel approach for detecting marine debris in underwater environments. The project integrates advanced preprocessing techniques and a YOLOv8-segmentation model to enhance detection accuracy for both bounding box and segmentation tasks. 

## Table of Contents
- [Project Motivation](#project-motivation)
- [Features](#features)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Results](#results)

## Project Motivation
The accumulation of marine debris poses a significant threat to marine ecosystems. This project is designed to address this issue by creating a scalable detection system that:

* Identifies underwater debris
* Reduces false positives by improving image preprocessing
* Enables future research in marine environment conservation

## Features
- **YOLOv8-Segmentation Model**: Utilized for both object detection and mask segmentation, ensuring high precision and robust performance.
- **Data Preprocessing**:
  - **Masking and Inpainting**: Removes unwanted text from underwater images to prevent the model from focusing on irrelevant features.
  - **Canny Edge Detection**: Enhances the edges of objects in murky underwater environments, improving feature clarity for the model.
- **Performance Metrics**: Evaluated using mAP@0.5 and mAP@[0.5:0.95] for both bounding box and segmentation tasks.
- **Dataset**:
  - Combined TrashCan dataset and additional images from Kaggle (3,433 marine life images).
  - Augmented data for improved variability and robustness.

## Dataset
- **TrashCan**: Underwater debris images, including trash, remotely operated vehicles (ROVs), and marine flora and fauna.
- **Kaggle**: Additional 3,433 marine life images to reduce false positives and enhance the model's ability to differentiate debris from marine organisms.

## Preprocessing

1. **Masking and Inpainting**:
   - Removes text and other irrelevant elements from images to ensure the model focuses only on meaningful features.
2. **Canny Edge Detection**:
   - Extracts clear edges of objects to improve detection in murky and low-light underwater environments.


## Model Architecture

- **YOLOv8-Segmentation**:
  - Combines the object detection capability of YOLO with segmentation features, ensuring precise bounding box and mask generation for detected objects.
  - Trained on the preprocessed dataset for optimal performance.

## Performance Metrics

- **mAP@0.5**: 91.4% overall, with the highest performance for the "trash_plastic" class (97.7%).
- **mAP@[0.5:0.95]**: 64.2%, reflecting the model's robustness across varying IoU thresholds.
- Bounding box and mask segmentation metrics were evaluated to ensure comprehensive performance analysis.

## Results

- **High Precision and Recall**: The model achieves superior accuracy in distinguishing debris from marine life.
- **Robust Segmentation**: Accurate mask generation for various underwater debris and biological entities.

