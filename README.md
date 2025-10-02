# 多模态缺失学习框架

一个完整的多模态缺失学习框架，支持GPU加速训练和测试，能够处理多模态数据中的缺失模态问题。

## 项目概述

本项目实现了一个先进的多模态学习系统，专门设计用于处理现实世界中常见的缺失模态问题。系统采用提示学习、分层蒸馏和对比学习等技术，能够在部分模态缺失的情况下保持高性能。


## 核心模块

### 1. 模态编码器 (`incomplete_multimodal_learning.py`)
- 支持文本、图像、音频三种模态
- 冻结的预训练编码器
- 空标记生成器处理缺失模态

### 2. 提示生成器
- **模态级提示** (`modality_prompts_v2.py`): 基于存在掩码生成上下文提示
- **实例级提示** (`dual_prompts_v3.py`): 专业化和泛化双分支设计

### 3. 知识蒸馏 (`hierarchical_distillation.py`)
- 教师模型和推理网络
- 多层级特征蒸馏
- 软标签预测蒸馏

### 4. 对比学习 (`hard_negative_contrastive.py`)
- 硬负样本挖掘
- InfoNCE损失实现
- 类别原型学习

### 5. 自适应调整 (`calibrated_prompt_adaptation.py`)
- 单步提示适应
- 熵正则化
- 分布对齐

### 6. GPU接口 (`gpu_interface.py`)
- 设备管理器
- 模型包装器
- 数据加载器包装

## 安装和使用

### 环境要求

```bash
torch>=1.9.0
torchvision>=0.10.0
pytorch_lightning>=1.1.4
transformers>=4.2.1
opencv-python
Pillow>=9.3.0
numpy>=1.19.5
```