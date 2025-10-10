import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class TrainingObjective(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.loss_weights = config.get("loss_weights", {
            "alpha": 0.1,
            "lambda_feat": 0.5,
            "lambda_pred": 0.5,
            "beta": 0.3
        })
    
    def forward(self, L_cls, L_abs, L_feat, L_pred, L_hno):
        alpha = self.loss_weights["alpha"]
        lambda_feat = self.loss_weights["lambda_feat"]
        lambda_pred = self.loss_weights["lambda_pred"]
        beta = self.loss_weights["beta"]
        
        L_total = L_cls + alpha * L_abs + lambda_feat * L_feat + lambda_pred * L_pred + beta * L_hno
        
        return {
            "L_total": L_total,
            "L_cls": L_cls,
            "L_abs": L_abs,
            "L_feat": L_feat,
            "L_pred": L_pred,
            "L_hno": L_hno
        }


class ParameterEfficiencyManager(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
    
    def get_learnable_parameters(self, model):
        learnable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                learnable_params.append(param)
        return learnable_params
    
    def count_parameters(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params