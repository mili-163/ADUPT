"""
单步校准提示适应

实现Single-step Calibrated Prompt Adaptation (CPA)
CPA机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math
import numpy as np


class WeakAugmentation(nn.Module):
    """
    弱增强模块
    
    从可用模态生成弱增强样本集合A
    """
    
    def __init__(self, augmentation_strength: float = 0.1):
        super().__init__()
        
        self.augmentation_strength = augmentation_strength
    
    def augment_text(self, text_tokens: torch.Tensor) -> List[torch.Tensor]:
        """
        文本弱增强
        
        Args:
            text_tokens: 文本标记 (B, L, d)
            
        Returns:
            List[torch.Tensor]: 增强后的文本标记列表
        """
        augmentations = []
        
        # 原始样本
        augmentations.append(text_tokens)
        
        # 添加噪声
        noise = torch.randn_like(text_tokens) * self.augmentation_strength
        augmentations.append(text_tokens + noise)
        
        # 随机掩码部分标记
        masked_tokens = text_tokens.clone()
        mask_prob = 0.1
        mask = torch.rand_like(text_tokens[:, :, 0]) < mask_prob
        masked_tokens[mask.unsqueeze(-1).expand_as(masked_tokens)] = 0
        augmentations.append(masked_tokens)
        
        return augmentations
    
    def augment_image(self, image_tokens: torch.Tensor) -> List[torch.Tensor]:
        """
        图像弱增强
        
        Args:
            image_tokens: 图像标记 (B, L, d)
            
        Returns:
            List[torch.Tensor]: 增强后的图像标记列表
        """
        augmentations = []
        
        # 原始样本
        augmentations.append(image_tokens)
        
        # 添加噪声
        noise = torch.randn_like(image_tokens) * self.augmentation_strength
        augmentations.append(image_tokens + noise)
        
        # 随机掩码部分标记
        masked_tokens = image_tokens.clone()
        mask_prob = 0.1
        mask = torch.rand_like(image_tokens[:, :, 0]) < mask_prob
        masked_tokens[mask.unsqueeze(-1).expand_as(masked_tokens)] = 0
        augmentations.append(masked_tokens)
        
        return augmentations
    
    def augment_audio(self, audio_tokens: torch.Tensor) -> List[torch.Tensor]:
        """
        音频弱增强
        
        Args:
            audio_tokens: 音频标记 (B, L, d)
            
        Returns:
            List[torch.Tensor]: 增强后的音频标记列表
        """
        augmentations = []
        
        # 原始样本
        augmentations.append(audio_tokens)
        
        # 添加噪声
        noise = torch.randn_like(audio_tokens) * self.augmentation_strength
        augmentations.append(audio_tokens + noise)
        
        # 随机掩码部分标记
        masked_tokens = audio_tokens.clone()
        mask_prob = 0.1
        mask = torch.rand_like(audio_tokens[:, :, 0]) < mask_prob
        masked_tokens[mask.unsqueeze(-1).expand_as(masked_tokens)] = 0
        augmentations.append(masked_tokens)
        
        return augmentations
    
    def augment_modalities(self, x: Dict[str, torch.Tensor], 
                          m: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        对可用模态进行弱增强
        
        Args:
            x: 原始输入 {modality: x^(k)}
            m: 存在掩码 (M,)
            
        Returns:
            List[Dict[str, torch.Tensor]]: 增强样本列表
        """
        augmentations = []
        
        # 为每个可用模态生成增强
        for i, modality in enumerate(['text', 'image', 'audio']):
            if modality in x and m[i].item() == 1:
                if modality == 'text':
                    aug_list = self.augment_text(x[modality])
                elif modality == 'image':
                    aug_list = self.augment_image(x[modality])
                elif modality == 'audio':
                    aug_list = self.augment_audio(x[modality])
                
                # 为每个增强创建样本
                for aug_tokens in aug_list:
                    aug_sample = x.copy()
                    aug_sample[modality] = aug_tokens
                    augmentations.append(aug_sample)
        
        return augmentations


class TeacherStatistics(nn.Module):
    """
    教师统计量计算器
    
    在完整数据D_c上预计算教师统计量(μ_T, σ²_T)
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.register_buffer('teacher_mean', torch.zeros(hidden_size))
        self.register_buffer('teacher_var', torch.ones(hidden_size))
        self.register_buffer('is_initialized', torch.tensor(False))
    
    def update_statistics(self, features: torch.Tensor):
        """
        更新教师统计量
        
        Args:
            features: 教师特征 (B, d)
        """
        if not self.is_initialized:
            # 初始化
            self.teacher_mean = features.mean(dim=0)
            self.teacher_var = features.var(dim=0)
            self.is_initialized = torch.tensor(True)
        else:
            # 在线更新
            batch_mean = features.mean(dim=0)
            batch_var = features.var(dim=0)
            
            # 简单的移动平均更新
            alpha = 0.1
            self.teacher_mean = (1 - alpha) * self.teacher_mean + alpha * batch_mean
            self.teacher_var = (1 - alpha) * self.teacher_var + alpha * batch_var
    
    def get_statistics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取教师统计量
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (μ_T, σ²_T)
        """
        return self.teacher_mean, self.teacher_var


class CalibratedPromptAdaptation(nn.Module):
    """
    单步校准提示适应 (CPA)
    
    实现论文公式：
    K_cpa = E_{v~U} H(Softmax(s^I(x_v; P̃^ins))) + 
            η ||μ(x) - μ_T||_1 + η ||σ²(x) - σ²_T||_1
    
    其中：
    - H(·) 是Shannon熵
    - μ(x), σ²(x) 是增强样本的均值和方差
    - μ_T, σ²_T 是教师统计量
    - η 是分布对齐权重
    """
    
    def __init__(self, hidden_size: int, num_classes: int, 
                 eta: float = 1.0, gamma: float = 0.01):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.eta = eta  # 分布对齐权重
        self.gamma = gamma  # 学习率
        
        # 弱增强模块
        self.weak_augmentation = WeakAugmentation()
        
        # 教师统计量
        self.teacher_stats = TeacherStatistics(hidden_size)
    
    def compute_entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        计算熵损失
        
        H(Softmax(s^I(x_v; P̃^ins)))
        
        Args:
            logits: 模型logits (B, num_classes)
            
        Returns:
            torch.Tensor: 熵损失
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy.mean()
    
    def compute_distribution_alignment_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算分布对齐损失
        
        η ||μ(x) - μ_T||_1 + η ||σ²(x) - σ²_T||_1
        
        Args:
            features: 融合特征 (B, d)
            
        Returns:
            torch.Tensor: 分布对齐损失
        """
        # 计算当前样本的统计量
        current_mean = features.mean(dim=0)  # (d,)
        current_var = features.var(dim=0)    # (d,)
        
        # 获取教师统计量
        teacher_mean, teacher_var = self.teacher_stats.get_statistics()
        
        # 计算L1损失
        mean_loss = F.l1_loss(current_mean, teacher_mean)
        var_loss = F.l1_loss(current_var, teacher_var)
        
        return self.eta * (mean_loss + var_loss)
    
    def compute_cpa_loss(self, x: Dict[str, torch.Tensor], m: torch.Tensor,
                        model, instance_prompts: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        计算CPA损失
        
        K_cpa = E_{v~U} H(Softmax(s^I(x_v; P̃^ins))) + 
                η ||μ(x) - μ_T||_1 + η ||σ²(x) - σ²_T||_1
        
        Args:
            x: 原始输入 {modality: x^(k)}
            m: 存在掩码 (M,)
            model: 推理模型
            instance_prompts: 实例级提示 {layer_idx: (B, L_p, d)}
            
        Returns:
            torch.Tensor: CPA损失
        """
        # 生成弱增强样本
        augmentations = self.weak_augmentation.augment_modalities(x, m)
        
        if not augmentations:
            return torch.tensor(0.0, device=next(iter(x.values())).device)
        
        # 计算每个增强样本的损失
        entropy_losses = []
        features_list = []
        
        for aug_x in augmentations:
            # 前向传播
            with torch.no_grad():
                # 编码各模态
                modality_tokens = model.base_iml.encode_modalities(aug_x, m)
                
                # 生成模态级提示
                modality_prompts = model.modality_prompts(m.unsqueeze(0))
                
                # 注入提示
                processed_tokens = model.prompt_injector.inject_prompts(
                    modality_tokens, modality_prompts, m.unsqueeze(0)
                )
                
                # 创建掩码
                modality_masks = {}
                for i, modality in enumerate(model.base_iml.modality_names):
                    if m[i].item() == 1:
                        total_length = processed_tokens[modality].shape[1]
                        modality_masks[modality] = torch.ones(
                            processed_tokens[modality].shape[0], 
                            total_length,
                            device=processed_tokens[modality].device
                        )
                    else:
                        total_length = processed_tokens[modality].shape[1]
                        modality_masks[modality] = torch.zeros(
                            processed_tokens[modality].shape[0], 
                            total_length,
                            device=processed_tokens[modality].device
                        )
                
                # 通过融合transformer
                z = model.base_iml.fusion_transformer(processed_tokens, modality_masks)
                
                # 任务预测
                s = model.base_iml.task_head(z)
                
                # 计算熵损失
                entropy_loss = self.compute_entropy_loss(s)
                entropy_losses.append(entropy_loss)
                
                # 收集特征用于分布对齐
                features_list.append(z)
        
        # 平均熵损失
        avg_entropy_loss = torch.stack(entropy_losses).mean()
        
        # 分布对齐损失
        if features_list:
            all_features = torch.cat(features_list, dim=0)
            dist_alignment_loss = self.compute_distribution_alignment_loss(all_features)
        else:
            dist_alignment_loss = torch.tensor(0.0, device=next(iter(x.values())).device)
        
        # 总CPA损失
        cpa_loss = avg_entropy_loss + dist_alignment_loss
        
        return cpa_loss
    
    def adapt_prompts(self, x: Dict[str, torch.Tensor], m: torch.Tensor,
                     model, instance_prompts: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        单步校准提示适应
        
        P̃^ins ← P̃^ins - γ ∇_{P̃^ins} K_cpa
        
        Args:
            x: 原始输入 {modality: x^(k)}
            m: 存在掩码 (M,)
            model: 推理模型
            instance_prompts: 实例级提示 {layer_idx: (B, L_p, d)}
            
        Returns:
            Dict[int, torch.Tensor]: 适应后的提示
        """
        # 确保提示需要梯度
        adapted_prompts = {}
        for layer_idx, prompt in instance_prompts.items():
            adapted_prompts[layer_idx] = prompt.clone().detach().requires_grad_(True)
        
        # 计算CPA损失
        cpa_loss = self.compute_cpa_loss(x, m, model, adapted_prompts)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            cpa_loss, 
            list(adapted_prompts.values()),
            create_graph=False,
            retain_graph=False
        )
        
        # 单步更新
        updated_prompts = {}
        for i, (layer_idx, prompt) in enumerate(adapted_prompts.items()):
            if gradients[i] is not None:
                updated_prompt = prompt - self.gamma * gradients[i]
                updated_prompts[layer_idx] = updated_prompt.detach()
            else:
                updated_prompts[layer_idx] = prompt.detach()
        
        return updated_prompts
    
    def update_teacher_statistics(self, teacher_features: torch.Tensor):
        """
        更新教师统计量
        
        Args:
            teacher_features: 教师特征 (B, d)
        """
        self.teacher_stats.update_statistics(teacher_features)
    
    def forward(self, x: Dict[str, torch.Tensor], m: torch.Tensor,
                model, instance_prompts: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 原始输入 {modality: x^(k)}
            m: 存在掩码 (M,)
            model: 推理模型
            instance_prompts: 实例级提示 {layer_idx: (B, L_p, d)}
            
        Returns:
            Dict[int, torch.Tensor]: 适应后的提示
        """
        return self.adapt_prompts(x, m, model, instance_prompts)
