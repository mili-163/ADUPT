"""
硬负样本对比正则化

实现Hard-negative Contrastive Regularization
InfoNCE损失和硬负样本挖掘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class HardNegativeMiner(nn.Module):
    """
    硬负样本挖掘器
    
    基于教师模型的混淆选择top-K非y类别作为硬负样本
    """
    
    def __init__(self, num_classes: int, top_k: int = 5):
        super().__init__()
        
        self.num_classes = num_classes
        self.top_k = top_k
    
    def mine_hard_negatives(self, teacher_logits: torch.Tensor, 
                           labels: torch.Tensor) -> List[int]:
        """
        挖掘硬负样本
        
        基于教师logits选择top-K非y类别
        
        Args:
            teacher_logits: 教师logits (B, num_classes)
            labels: 真实标签 (B,)
            
        Returns:
            List[int]: 硬负样本类别索引列表
        """
        batch_size = teacher_logits.shape[0]
        hard_negatives = []
        
        for i in range(batch_size):
            # 获取当前样本的logits和标签
            sample_logits = teacher_logits[i]  # (num_classes,)
            true_label = labels[i].item()
            
            # 排除真实标签，获取其他类别的logits
            mask = torch.ones(self.num_classes, dtype=torch.bool, device=teacher_logits.device)
            mask[true_label] = False
            negative_logits = sample_logits[mask]
            negative_indices = torch.where(mask)[0]
            
            # 选择top-K硬负样本
            top_k_values, top_k_indices = torch.topk(negative_logits, 
                                                   min(self.top_k, len(negative_logits)))
            
            # 获取原始类别索引
            hard_negative_indices = negative_indices[top_k_indices].tolist()
            hard_negatives.append(hard_negative_indices)
        
        return hard_negatives


class ClassPrototype(nn.Module):
    """
    类别原型
    
    可以是分类器权重向量W_y或文本原型
    """
    
    def __init__(self, hidden_size: int, num_classes: int, prototype_type: str = "weight"):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.prototype_type = prototype_type
        
        if prototype_type == "weight":
            # 使用分类器权重作为原型
            self.prototypes = nn.Parameter(torch.randn(num_classes, hidden_size))
        elif prototype_type == "learnable":
            # 可学习的类别原型
            self.prototypes = nn.Parameter(torch.randn(num_classes, hidden_size))
        else:
            raise ValueError(f"Unknown prototype type: {prototype_type}")
        
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化原型参数"""
        nn.init.normal_(self.prototypes, std=0.02)
    
    def get_prototype(self, class_id: int) -> torch.Tensor:
        """
        获取指定类别的原型
        
        Args:
            class_id: 类别ID
            
        Returns:
            torch.Tensor: 类别原型 (hidden_size,)
        """
        return self.prototypes[class_id]
    
    def get_prototypes(self, class_ids: List[int]) -> torch.Tensor:
        """
        获取多个类别的原型
        
        Args:
            class_ids: 类别ID列表
            
        Returns:
            torch.Tensor: 类别原型 (len(class_ids), hidden_size)
        """
        return self.prototypes[class_ids]


class HardNegativeContrastive(nn.Module):
    """
    硬负样本对比正则化
    
    实现论文公式：
    L_hno = InfoNCE(z_x, c_y, {c_n}_{n∈N}) + InfoNCE(c_y, z_x, {z_n}_{n∈N})
    
    其中：
    - z_x: 推理网络I在P^sp驱动下产生的融合特征
    - c_y: 标签y的类别原型
    - N: 基于教师混淆挖掘的硬负样本集合
    """
    
    def __init__(self, hidden_size: int, num_classes: int, 
                 temperature: float = 0.07, top_k: int = 5):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.temperature = temperature
        self.top_k = top_k
        
        # 硬负样本挖掘器
        self.hard_negative_miner = HardNegativeMiner(num_classes, top_k)
        
        # 类别原型
        self.class_prototype = ClassPrototype(hidden_size, num_classes, "learnable")
        
        # 特征投影层（用于对比学习）
        self.feature_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化参数"""
        for layer in self.feature_projector:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
    
    def compute_infonce_loss(self, query: torch.Tensor, positive: torch.Tensor, 
                            negatives: torch.Tensor) -> torch.Tensor:
        """
        计算InfoNCE损失
        
        Args:
            query: 查询向量 (B, d)
            positive: 正样本向量 (B, d)
            negatives: 负样本向量 (B, K, d)
            
        Returns:
            torch.Tensor: InfoNCE损失
        """
        batch_size = query.shape[0]
        device = query.device
        
        # 计算查询-正样本相似度
        pos_sim = torch.sum(query * positive, dim=-1) / self.temperature  # (B,)
        
        # 计算查询-负样本相似度
        neg_sim = torch.bmm(
            query.unsqueeze(1),  # (B, 1, d)
            negatives.transpose(-2, -1)  # (B, d, K)
        ).squeeze(1) / self.temperature  # (B, K)
        
        # 拼接正样本和负样本
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+K)
        
        # 计算InfoNCE损失
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)  # 正样本在位置0
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def forward(self, specialization_features: torch.Tensor, 
                labels: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        计算硬负样本对比正则化损失
        
        Args:
            specialization_features: 专业化分支特征 z_x (B, d)
            labels: 真实标签 (B,)
            teacher_logits: 教师logits (B, num_classes)
            
        Returns:
            torch.Tensor: 硬负样本对比损失
        """
        batch_size = specialization_features.shape[0]
        device = specialization_features.device
        
        # 投影特征
        projected_features = self.feature_projector(specialization_features)  # (B, d)
        
        # 挖掘硬负样本
        hard_negatives = self.hard_negative_miner.mine_hard_negatives(teacher_logits, labels)
        
        # 获取类别原型
        class_prototypes = []
        for i in range(batch_size):
            # 正样本原型
            pos_prototype = self.class_prototype.get_prototype(labels[i].item())  # (d,)
            class_prototypes.append(pos_prototype)
        
        class_prototypes = torch.stack(class_prototypes, dim=0)  # (B, d)
        
        # 获取硬负样本原型
        hard_negative_prototypes = []
        for i in range(batch_size):
            neg_prototypes = self.class_prototype.get_prototypes(hard_negatives[i])  # (K, d)
            hard_negative_prototypes.append(neg_prototypes)
        
        # 填充到相同长度
        max_k = max(len(negs) for negs in hard_negatives)
        padded_negatives = []
        for i in range(batch_size):
            neg_prototypes = hard_negative_prototypes[i]  # (K_i, d)
            if len(neg_prototypes) < max_k:
                # 填充
                padding = torch.zeros(max_k - len(neg_prototypes), self.hidden_size, device=device)
                neg_prototypes = torch.cat([neg_prototypes, padding], dim=0)
            padded_negatives.append(neg_prototypes)
        
        hard_negative_prototypes = torch.stack(padded_negatives, dim=0)  # (B, max_k, d)
        
        # 计算对称InfoNCE损失
        # InfoNCE(z_x, c_y, {c_n}_{n∈N})
        loss1 = self.compute_infonce_loss(
            projected_features, class_prototypes, hard_negative_prototypes
        )
        
        # InfoNCE(c_y, z_x, {z_n}_{n∈N})
        # 这里简化处理，使用相同的负样本
        loss2 = self.compute_infonce_loss(
            class_prototypes, projected_features, hard_negative_prototypes
        )
        
        # 总损失
        total_loss = loss1 + loss2
        
        return total_loss


class SpecializationContrastiveRegularizer(nn.Module):
    """
    专业化分支对比正则化器
    
    只对专业化分支P^sp应用对比约束，保持泛化分支P^gn不变
    """
    
    def __init__(self, hidden_size: int, num_classes: int, 
                 temperature: float = 0.07, top_k: int = 5):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # 硬负样本对比学习
        self.hard_negative_contrastive = HardNegativeContrastive(
            hidden_size, num_classes, temperature, top_k
        )
    
    def forward(self, specialization_features: torch.Tensor,
                labels: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        只对专业化分支应用对比正则化
        
        Args:
            specialization_features: 专业化分支特征 (B, d)
            labels: 真实标签 (B,)
            teacher_logits: 教师logits (B, num_classes)
            
        Returns:
            torch.Tensor: 对比正则化损失
        """
        return self.hard_negative_contrastive(
            specialization_features, labels, teacher_logits
        )
    
    def apply_gradient_stop_to_generalization(self, dual_prompts):
        """
        停止泛化分支和门控的梯度流
        
        保持泛化分支P^gn和门控ω的可迁移性
        """
        # 停止泛化分支梯度
        for param in dual_prompts.generalization_branch.parameters():
            param.requires_grad = False
        
        # 停止门控梯度
        for param in dual_prompts.gate_network.parameters():
            param.requires_grad = False
    
    def apply_gradient_stop_to_teacher(self, teacher_model):
        """
        停止教师模型梯度流
        
        防止教师模型被对比学习影响
        """
        for param in teacher_model.parameters():
            param.requires_grad = False
