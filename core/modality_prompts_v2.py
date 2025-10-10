"""
模态级提示模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class AbsenceAwareModalityPrompts(nn.Module):
    
    def __init__(self, hidden_size: int, num_modalities: int, prompt_length: int, num_layers: int):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        self.prompt_length = prompt_length
        self.num_layers = num_layers
        
        self.presence_embeddings = nn.Parameter(
            torch.randn(num_modalities, hidden_size)
        )
        self.absence_embeddings = nn.Parameter(
            torch.randn(num_modalities, hidden_size)
        )
        
        self.prompt_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
        )
        
        self.layer_prompts = nn.ModuleList([
            nn.Linear(hidden_size, prompt_length * hidden_size)
            for _ in range(num_layers)
        ])
        self.mask_reconstruction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_modalities),
            nn.Sigmoid()
        )
        
        
        self.mask_decoder = nn.Sequential(
            nn.Linear(prompt_length * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self._init_parameters()
    
    def _init_parameters(self):
        
        # 初始化存在和缺失嵌入
        nn.init.normal_(self.presence_embeddings, std=0.02)
        nn.init.normal_(self.absence_embeddings, std=0.02)
        
        # 初始化MLP参数
        for layer in self.prompt_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
        
        # 初始化层提示参数
        for layer_prompt in self.layer_prompts:
            nn.init.normal_(layer_prompt.weight, std=0.02)
            nn.init.zeros_(layer_prompt.bias)
    
    def forward(self, missing_mask: torch.Tensor) -> Dict[int, torch.Tensor]:
        batch_size = missing_mask.shape[0]
        device = missing_mask.device
        
        # Equation (1): P^mod(m) = φ([∑_k m_k E^(+)_k || ∑_k (1-m_k) E^(-)_k])
        presence_sum = torch.matmul(missing_mask.float(), self.presence_embeddings)  # (B, d)
        absence_sum = torch.matmul((1 - missing_mask.float()), self.absence_embeddings)  # (B, d)
        
        # Concatenation [presence_sum || absence_sum]
        combined = torch.cat([presence_sum, absence_sum], dim=-1)  # (B, 2d)
        
        # Apply MLP φ
        prompt_features = self.prompt_mlp(combined)  # (B, d)
        
        # 为每个层生成提示 P^mod(m) ∈ R^(L_p×d)
        layer_prompts = {}
        for layer_idx, layer_prompt in enumerate(self.layer_prompts):
            # 生成该层的提示
            layer_prompt_tokens = layer_prompt(prompt_features)  # (B, L_p * d)
            layer_prompt_tokens = layer_prompt_tokens.view(
                batch_size, self.prompt_length, self.hidden_size
            )  # (B, L_p, d)
            layer_prompts[layer_idx] = layer_prompt_tokens
        
        return layer_prompts
    
    def reconstruct_mask(self, layer_prompts: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        从提示重构缺失掩码 - 论文公式(2)
        
        L_abs = BCE(h_mask(Φ(P^mod(m))), m)
        
        Args:
            layer_prompts: 每层的提示 {layer_idx: (B, L_p, d)}
            
        Returns:
            torch.Tensor: 重构的缺失掩码 (B, M)
        """
        # 使用第一层的提示进行重构
        first_layer_prompt = layer_prompts[0]  # (B, L_p, d)
        
        # 通过掩码解码器 Φ
        decoded_features = self.mask_decoder(
            first_layer_prompt.view(first_layer_prompt.shape[0], -1)
        )  # (B, d)
        
        # 通过掩码重构头 h_mask
        reconstructed_mask = self.mask_reconstruction_head(decoded_features)  # (B, M)
        
        return reconstructed_mask
    
    def compute_mask_reconstruction_loss(self, missing_mask: torch.Tensor) -> torch.Tensor:
        """
        计算掩码重构损失 - 论文公式(2)
        
        L_abs = BCE(h_mask(Φ(P^mod(m))), m)
        
        Args:
            missing_mask: 真实缺失掩码 (B, M)
            
        Returns:
            torch.Tensor: 掩码重构损失
        """
        # 生成提示
        layer_prompts = self.forward(missing_mask)
        
        # 重构掩码
        reconstructed_mask = self.reconstruct_mask(layer_prompts)
        
        # 计算BCE损失
        loss = F.binary_cross_entropy(reconstructed_mask, missing_mask.float())
        
        return loss


class ModalityPromptInjector(nn.Module):
    
    def __init__(self, hidden_size: int, prompt_length: int, num_layers: int):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.prompt_length = prompt_length
        self.num_layers = num_layers
        
        # 学习的空标记 t̄^(k) ∈ R^(L_k × d)
        # 为每个模态创建不同的空标记
        self.null_tokens = nn.Parameter(
            torch.randn(1, prompt_length, hidden_size)
        )
        
        self._init_parameters()
    
    def _init_parameters(self):
        
        nn.init.normal_(self.null_tokens, std=0.02)
    
    def inject_prompts(self, modality_tokens: Dict[str, torch.Tensor], 
                      modality_prompts: Dict[int, torch.Tensor],
                      missing_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
       
        batch_size = missing_mask.shape[0]
        device = missing_mask.device
        
        # 为每个模态处理
        processed_tokens = {}
        modality_names = list(modality_tokens.keys())
        
        for i, modality in enumerate(modality_names):
            tokens = modality_tokens[modality]  # (B, L_k, d)
            
            if missing_mask[0, i].item() == 1:
                # 模态存在：前置P^mod(m)到标记流
                # 使用第一层的提示进行前置
                prompt = modality_prompts[0]  # (B, L_p, d)
                # 前置提示到标记流
                processed_tokens[modality] = torch.cat([prompt, tokens], dim=1)
            else:
                # 模态缺失：使用空标记t̄^(k)
                null_tokens = self.null_tokens.expand(batch_size, -1, -1)
                processed_tokens[modality] = null_tokens
        
        return processed_tokens


class ContentAggregator(nn.Module):
    """
    内容聚合器 ψ
    
    实现论文公式：
    u = ψ(x) = LN([Pool(t^(1)) || ... || Pool(t^(M))])
    
    其中：
    - Pool(t^(k)) 聚合模态k的标记
    - 对于存在模态使用真实标记，对于缺失模态使用学习的空标记
    - u 捕获存在模态的信息性
    """
    
    def __init__(self, hidden_size: int, num_modalities: int):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size * num_modalities)
        
        # 投影到隐藏维度
        self.projection = nn.Linear(hidden_size * num_modalities, hidden_size)
    
    def forward(self, modality_tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        聚合内容信息
        
        u = ψ(x) = LN([Pool(t^(1)) || ... || Pool(t^(M))])
        
        Args:
            modality_tokens: 各模态标记 {modality: (B, L_k, d)}
            
        Returns:
            torch.Tensor: 内容摘要 u (B, d)
        """
        pooled_tokens = []
        
        # 池化各模态标记
        for modality, tokens in modality_tokens.items():
            # 平均池化 Pool(t^(k))
            pooled = tokens.mean(dim=1)  # (B, d)
            pooled_tokens.append(pooled)
        
        # 拼接所有模态 [Pool(t^(1)) || ... || Pool(t^(M))]
        if pooled_tokens:
            combined = torch.cat(pooled_tokens, dim=-1)  # (B, M*d)
        else:
            batch_size = next(iter(modality_tokens.values())).shape[0]
            combined = torch.zeros(batch_size, self.hidden_size * self.num_modalities,
                                 device=next(iter(modality_tokens.values())).device)
        
        # 层归一化 LN([...])
        normalized = self.layer_norm(combined)
        
        # 投影到隐藏维度
        content_summary = self.projection(normalized)
        
        return content_summary
