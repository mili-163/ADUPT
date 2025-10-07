# Missing Modality Prompt Learning Framework

A comprehensive multimodal learning framework with GPU acceleration support for training and testing, designed to handle missing modality problems in multimodal data.

## Project Overview

This project implements an advanced multimodal learning system specifically designed to handle missing modality problems common in real-world scenarios. The system employs prompt learning, hierarchical distillation, and contrastive learning techniques to maintain high performance even when some modalities are missing.

## Key Features

- **70% Missing Rate Support**: Handles extreme missing modality scenarios
- **Multiple Datasets**: Supports MM-IMDb, Food101, Hateful Memes, and CMU-MOSEI
- **Proper Evaluation Metrics**: 
  - MM-IMDb: Macro-F1 
  - Food101: Accuracy 
  - Hateful Memes: AUROC 
  - CMU-MOSEI: Accuracy 
- **GPU Acceleration**: Seamless CPU/GPU switching with MPS support
- **Three-Stage Training**: Teacher → Inference → Runtime Calibration (CPA)

## Core Architecture

### 1. Modality Encoders (`pag_mpd/core/`)
- **Text, Image, Audio**: Three-modality support
- **Frozen Encoders**: Pre-trained backbone networks
- **Null Token Generation**: Handles missing modalities

### 2. Prompt Systems
- **Modality-Level Prompts** (`modality_prompts_v2.py`): Context-aware prompts based on presence masks
- **Instance-Level Prompts** (`dual_prompts_v3.py`): Specialization and generalization dual-branch design

### 3. Knowledge Distillation (`hierarchical_distillation.py`)
- **Teacher-Student Framework**: Complete data → Incomplete data knowledge transfer
- **Multi-Level Distillation**: Feature and prediction level alignment
- **Mask Adapter**: FiLM-style adaptation for different missing patterns

### 4. Contrastive Learning (`hard_negative_contrastive.py`)
- **Hard Negative Mining**: InfoNCE loss with class prototypes
- **Specialization Branch**: Targeted contrastive supervision
- **Class Boundary Sharpening**: Improved discrimination

### 5. Runtime Adaptation (`calibrated_prompt_adaptation.py`)
- **Single-Step CPA**: Fast inference-time adaptation
- **Entropy Regularization**: Confident prediction promotion
- **Distribution Alignment**: Teacher statistics matching

### 6. GPU Interface (`gpu_interface.py`)
- **Device Manager**: Automatic CUDA/MPS/CPU detection
- **Model Wrapper**: Seamless device switching
- **DataLoader Wrapper**: Efficient batch processing

## Installation and Usage

### Environment Requirements

```bash
# Core dependencies
torch>=1.9.0
torchvision>=0.10.0
transformers>=4.2.1
scikit-learn>=0.24.0
opencv-python>=4.5.0
Pillow>=9.3.0
numpy>=1.19.5

# Optional for advanced features
pytorch_lightning>=1.1.4
pyarrow>=5.0.0
```

### Quick Start

```bash
# Install dependencies
pip install -r pag_mpd/requirements.txt

cd pag_mpd
python main.py --dataset xxx --missing_rate xxx
