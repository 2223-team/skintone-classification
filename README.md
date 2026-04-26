# Skintone Classification

This repository contains the documentation and codebase for an end-to-end computer vision pipeline designed for skin tone classification. 

## Dataset
This project utilizes the [Monk Skin Tone (MST) Scale](https://skintone.google/) dataset, classifying inputs into 10 distinct skin tone categories. 

## End-to-End Pipeline
The model leverages an EfficientNet-B0 architecture with a robust data processing and training pipeline:

* **Skin Patch Extraction:** Prioritizes extraction from the cheek, cascading to the forehead, and uses a blob fallback method if necessary.
* **Data Organization:** Extracted patches are routed to their respective class folders (`MST_1/` through `MST_10/`).
* **Dataset Splitting:** Implements a stratified random split of 70% Training / 15% Validation / 15% Testing. *(Note: Identity features are safely stripped prior to training).*
* **Model Architecture:** Fine-tuned EfficientNet-B0 (Blocks 6, 7, and 8 are unfrozen).
* **Optimization:** Trained using `CrossEntropyLoss`, the `AdamW` optimizer, and a `CosineAnnealingLR` learning rate scheduler.
* **Imbalance Handling:** Employs a `WeightedRandomSampler` combined with rare class augmentation to ensure balanced learning across all MST classes.
* **Evaluation & Export:** Evaluated using Test-Time Augmentation (TTA) and confusion matrix plotting, followed by exporting the final model to ONNX format for deployment.

## Repository Structure

```text
├── example/                 # Example images used for inference testing
├── model/                   # Saved model weights (.pth, .onnx, .onnx.data)
├── result/                  # Test/Validation accuracy, loss plots, and confusion matrix
├── face_landmarker.task     # MediaPipe/Landmarker task file for facial feature extraction
├── inference.ipynb          # Notebook demonstrating model inference
└── train.ipynb              # Notebook containing the complete training pipeline
