# MS Lesion Segmentation using 3D CNN

This repository contains an implementation of a 3D Convolutional Neural Network (3D CNN) for Multiple Sclerosis (MS) white matter lesion segmentation from multi-modal MRI scans. The model is based on voxel-wise classification using 3D patches extracted from MRI volumes.

## Overview

Multiple Sclerosis is a neurological disorder characterized by the formation of white matter lesions in the brain. Accurate lesion segmentation from MRI is essential for disease monitoring and clinical assessment.

This project implements a baseline deep learning approach for MS lesion segmentation using a 3D CNN trained on FLAIR and T1-weighted MRI images. The model performs voxel-wise classification by extracting local 3D patches and predicting lesion probability for each voxel.

## Method

The implemented model is based on the approach proposed by Valverde et al. (2017), with architectural modifications to improve feature representation capacity.

### Key characteristics:
- Input: Multi-modal MRI (FLAIR + T1-weighted)
- Patch-based input: 3D patches of size 7×7×7
- Architecture:
  - Two 3D convolutional blocks (32 and 64 filters)
  - Max-pooling layers for downsampling
  - Fully connected layers (256 → 128 → 64)
  - Dropout for regularization
  - Softmax output for voxel classification
- Output: Probabilistic lesion map (thresholded to obtain binary segmentation)

The model is trained using balanced sampling of lesion and non-lesion voxels.

## Dataset

The model is evaluated on publicly available MS MRI datasets provided by Lesjak et al.:

- Cross-sectional dataset: used for training and testing the 3D CNN
- Includes:
  - T1-weighted MRI
  - FLAIR MRI
  - Brain masks
  - Consensus-based lesion segmentations

Data is assumed to be preprocessed (bias corrected, co-registered, skull stripped).

## Training Strategy

- Patch-based training using voxel-centered 3D patches
- Balanced sampling between lesion and healthy tissue
- Optimization using stochastic gradient descent / backpropagation
- Loss function: categorical cross-entropy
- Regularization: dropout in fully connected layers

## Evaluation Metrics

Model performance is evaluated using:

- Dice Similarity Coefficient (DSC)
- Precision
- Recall
- IoU (Intersection over Union)
- F1-score
- Balanced Accuracy

## Results

The 3D CNN achieves high recall but relatively low precision:

- Dice Score: ~0.37
- Recall: ~0.91
- Precision: ~0.26

This indicates that the model is highly sensitive to lesion regions but tends to over-segment and produce false positives.

