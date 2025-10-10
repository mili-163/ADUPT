"""
分层蒸馏机制

实现Teacher→Inference with Hierarchical Distillation
特征级和预测级蒸馏
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class MaskAdapter(nn.Module):
    """
    掩码适配器 Π_m
    
    实现论文公式：
    Π_m(h) = γ(m) ⊙ h + β(m)
    γ(m), β(m) = MLP(Hash(m))
    
    这是一个轻量级的FiLM风格适配器，将教师特征适配到学生的掩码条件
    """
    
    def __init__(self, hidden_size: int, num_modalities: int):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        
        # 掩码哈希函数 Hash(m)
        self.mask_hash = nn.Linear(num_modalities, hidden_size // 2)
        
        # MLP生成γ(m)和β(m)
        self.mask_mlp = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * 2)  # 输出γ和β
        )
        
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化参数"""
        nn.init.normal_(self.mask_hash.weight, std=0.02)
        nn.init.zeros_(self.mask_hash.bias)
        
        for layer in self.mask_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
    
    def forward(self, teacher_features: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Π_m(h) = γ(m) ⊙ h + β(m)
        
        Args:
            teacher_features: 教师特征 (B, d)
            missing_mask: 缺失掩码 (B, M)
            
        Returns:
            torch.Tensor: 适配后的特征 (B, d)
        """
        # 掩码哈希 Hash(m)
        mask_hash = self.mask_hash(missing_mask.float())  # (B, d//2)
        
        # 生成γ(m)和β(m)
        gamma_beta = self.mask_mlp(mask_hash)  # (B, 2d)
        gamma = gamma_beta[:, :self.hidden_size]  # (B, d)
        beta = gamma_beta[:, self.hidden_size:]  # (B, d)
        
        # FiLM变换：Π_m(h) = γ(m) ⊙ h + β(m)
        adapted_features = gamma * teacher_features + beta
        
        return adapted_features


class HierarchicalDistillation(nn.Module):
    """
    分层蒸馏机制
    
    实现论文中的特征级和预测级蒸馏：
    1. 特征级蒸馏：L_feat = (1/|S|) ∑_{ℓ∈S} ||Norm(Z^I_ℓ) - Norm(Ẑ^T_ℓ)||_2^2
    2. 预测级蒸馏：L_pred = τ² KL(Softmax(s^T/τ) || Softmax(s^I/τ))
    """
    
    def __init__(self, hidden_size: int, num_modalities: int, 
                 selected_layers: List[int], temperature: float = 3.0):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        self.selected_layers = selected_layers
        self.temperature = temperature
        
        # 掩码适配器
        self.mask_adapter = MaskAdapter(hidden_size, num_modalities)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def compute_feature_distillation_loss(self, 
                                        teacher_features: Dict[int, torch.Tensor],
                                        student_features: Dict[int, torch.Tensor],
                                        missing_mask: torch.Tensor) -> torch.Tensor:
        # L_feat = (1/|S|) ∑_{ℓ∈S} ||Norm(Z^I_ℓ) - Norm(Ẑ^T_ℓ)||_2^2
        # where Ẑ^T_ℓ = Π_m(Z^T_ℓ)
        total_loss = 0.0
        valid_layers = 0
        
        for layer_idx in self.selected_layers:
            if layer_idx in teacher_features and layer_idx in student_features:
                # 获取教师和学生特征
                teacher_feat = teacher_features[layer_idx]  # (B, d)
                student_feat = student_features[layer_idx]  # (B, d)
                
                # 适配教师特征到学生条件
                adapted_teacher_feat = self.mask_adapter(teacher_feat, missing_mask)  # (B, d)
                
                # 层归一化
                norm_student = self.layer_norm(student_feat)
                norm_teacher = self.layer_norm(adapted_teacher_feat)
                
                # 计算L2损失
                layer_loss = F.mse_loss(norm_student, norm_teacher)
                total_loss += layer_loss
                valid_layers += 1
        
        if valid_layers > 0:
            return total_loss / valid_layers
        else:
            return torch.tensor(0.0, device=next(iter(teacher_features.values())).device)
    
    def compute_prediction_distillation_loss(self, 
                                           teacher_logits: torch.Tensor,
                                           student_logits: torch.Tensor) -> torch.Tensor:
        # L_pred = τ² KL(Softmax(s^T/τ) || Softmax(s^I/τ))
        # 温度缩放
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # KL散度
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        
        # 温度平方缩放
        return self.temperature ** 2 * kl_loss
    
    def forward(self, teacher_outputs: Dict[str, torch.Tensor],
                student_outputs: Dict[str, torch.Tensor],
                missing_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            teacher_outputs: 教师输出
            student_outputs: 学生输出
            missing_mask: 缺失掩码 (B, M)
            
        Returns:
            Dict[str, torch.Tensor]: 蒸馏损失
        """
        losses = {}
        
        # 特征级蒸馏损失
        if "features" in teacher_outputs and "features" in student_outputs:
            # 简化处理：只使用最终特征
            teacher_features = {0: teacher_outputs["features"]}
            student_features = {0: student_outputs["features"]}
            
            feat_loss = self.compute_feature_distillation_loss(
                teacher_features, student_features, missing_mask
            )
            losses["feat_distill"] = feat_loss
        
        # 预测级蒸馏损失
        if "logits" in teacher_outputs and "logits" in student_outputs:
            pred_loss = self.compute_prediction_distillation_loss(
                teacher_outputs["logits"], student_outputs["logits"]
            )
            losses["pred_distill"] = pred_loss
        
        return losses


class TeacherModel(nn.Module):
    """
    教师模型 T
    
    在完整数据D_c上训练，所有backbone冻结
    """
    
    def __init__(self, base_iml, hidden_size: int, num_classes: int, parent_model=None):
        super().__init__()
        
        self.base_iml = base_iml
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.parent_model = parent_model  # 主模型引用
        
        # 冻结所有backbone参数
        self._freeze_backbones()
    
    def _freeze_backbones(self):
        """冻结所有backbone参数"""
        for param in self.base_iml.parameters():
            param.requires_grad = False
    
    def forward(self, x: Dict[str, torch.Tensor], m: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        教师模型前向传播
        
        只在完整数据上训练，m = 1
        
        Args:
            x: 原始输入 {modality: x^(k)}
            m: 存在掩码 (M,)，教师只处理m=1的情况
            
        Returns:
            Dict[str, torch.Tensor]: 教师输出
        """
        # 编码各模态（教师只处理完整数据）
        if self.parent_model and hasattr(self.parent_model, 'num_modalities'):
            if self.parent_model.num_modalities == 3:
                modality_tokens = self.parent_model._encode_trimodal(x, m)
            else:
                modality_tokens = self.parent_model._encode_bimodal(x, m)
        else:
            # 回退到基础编码
            modality_tokens = self.base_iml.encode_modalities(x, m)
        
        # 创建掩码
        modality_masks = {}
        for i, modality in enumerate(self.base_iml.modality_names):
            if m[i].item() == 1:
                modality_masks[modality] = torch.ones(
                    modality_tokens[modality].shape[0], 
                    modality_tokens[modality].shape[1],
                    device=modality_tokens[modality].device
                )
            else:
                modality_masks[modality] = torch.zeros(
                    modality_tokens[modality].shape[0], 
                    modality_tokens[modality].shape[1],
                    device=modality_tokens[modality].device
                )
        
        # 通过融合transformer
        z = self.base_iml.fusion_transformer(modality_tokens, modality_masks)
        
        # 任务预测
        s = self.base_iml.task_head(z)
        
        return {
            "features": z,
            "logits": s,
            "probs": F.softmax(s, dim=-1)
        }


class InferenceModel(nn.Module):
    """
    推理模型 I
    
    在D_s∪D_c上训练，使用提示机制处理缺失模态
    """
    
    def __init__(self, base_iml, modality_prompts, dual_prompts, 
                 prompt_injector, hidden_size: int, num_classes: int, parent_model=None):
        super().__init__()
        
        self.base_iml = base_iml
        self.modality_prompts = modality_prompts
        self.dual_prompts = dual_prompts
        self.prompt_injector = prompt_injector
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.parent_model = parent_model  # 主模型引用
    
    def forward(self, x: Dict[str, torch.Tensor], m: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        推理模型前向传播
        
        在D_s∪D_c上训练，使用提示机制处理缺失模态
        
        Args:
            x: 原始输入 {modality: x^(k)}
            m: 存在掩码 (M,)
            
        Returns:
            Dict[str, torch.Tensor]: 推理输出
        """
        # 1. 编码各模态
        if self.parent_model and hasattr(self.parent_model, 'num_modalities'):
            if self.parent_model.num_modalities == 3:
                modality_tokens = self.parent_model._encode_trimodal(x, m)
            else:
                modality_tokens = self.parent_model._encode_bimodal(x, m)
        else:
            # 回退到基础编码
            modality_tokens = self.base_iml.encode_modalities(x, m)
        
        # 2. 生成模态级提示 P^mod(m)
        modality_prompts = self.modality_prompts(m.unsqueeze(0))
        
        # 3. 生成实例级双提示 P^sp, P^gn
        instance_prompts = self.dual_prompts(modality_tokens, modality_prompts)
        
        # 4. 注入提示到模态标记
        processed_tokens = self.prompt_injector.inject_prompts(
            modality_tokens, modality_prompts, m.unsqueeze(0)
        )
        
        # 5. 创建掩码（包含提示部分）
        modality_masks = {}
        for i, modality in enumerate(self.base_iml.modality_names):
            if m[i].item() == 1:
                # 模态存在：包含提示和原始标记
                total_length = processed_tokens[modality].shape[1]
                modality_masks[modality] = torch.ones(
                    processed_tokens[modality].shape[0], 
                    total_length,
                    device=processed_tokens[modality].device
                )
            else:
                # 模态缺失：只有空标记
                total_length = processed_tokens[modality].shape[1]
                modality_masks[modality] = torch.zeros(
                    processed_tokens[modality].shape[0], 
                    total_length,
                    device=processed_tokens[modality].device
                )
        
        # 6. 通过融合transformer
        z = self.base_iml.fusion_transformer(processed_tokens, modality_masks)
        
        # 7. 任务预测
        s = self.base_iml.task_head(z)
        
        return {
            "features": z,
            "logits": s,
            "probs": F.softmax(s, dim=-1),
            "modality_prompts": modality_prompts,
            "instance_prompts": instance_prompts
        }
