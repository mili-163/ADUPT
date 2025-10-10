"""
三模态ViLT适配器

将原本的双模态ViLT (文本+图像) 扩展为三模态 (文本+图像+音频)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union


class TrimodalViLTAdapter(nn.Module):
    """
    三模态ViLT适配器
    
    将ViLT从双模态扩展为三模态：
    1. 保持原有的文本和图像处理
    2. 添加音频模态处理
    3. 扩展融合机制
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.hidden_size = config["hidden_size"]
        
        # 导入原始ViLT模块
        try:
            from ..modules.vilt_module import ViLTransformerSS
            self.base_vilt = ViLTransformerSS(config)
        except ImportError:
            # 如果无法导入，创建模拟版本
            self.base_vilt = self._create_mock_vilt(config)
        
        # 音频模态处理
        self.audio_processor = self._create_audio_processor(config)
        
        # 音频token type embedding
        self.audio_token_type_embedding = nn.Embedding(1, self.hidden_size)
        
        # 位置编码扩展
        self.audio_position_embeddings = nn.Embedding(
            config.get("max_audio_len", 100), self.hidden_size
        )
        
        # 三模态融合层
        self.trimodal_fusion = self._create_trimodal_fusion()
        
    def _create_audio_processor(self, config):
        """创建音频处理器"""
        audio_config = config.get("audio_processing", {})
        
        class AudioProcessor(nn.Module):
            def __init__(self, input_dim, hidden_size):
                super().__init__()
                self.projection = nn.Linear(input_dim, hidden_size)
                self.layer_norm = nn.LayerNorm(hidden_size)
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, audio_features):
                # audio_features: (B, seq_len, feature_dim) 或 (B, feature_dim)
                if audio_features.dim() == 2:
                    # 如果是2D，添加序列维度
                    audio_features = audio_features.unsqueeze(1)
                
                # 投影到hidden_size
                projected = self.projection(audio_features)
                projected = self.layer_norm(projected)
                projected = self.dropout(projected)
                
                return projected
        
        audio_dim = audio_config.get("feature_dim", 74)
        return AudioProcessor(audio_dim, self.hidden_size)
    
    def _create_trimodal_fusion(self):
        """创建三模态融合层"""
        class TrimodalFusion(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=12,
                    dropout=0.1,
                    batch_first=True
                )
                self.norm = nn.LayerNorm(hidden_size)
                
            def forward(self, combined_embeds, combined_masks):
                # Self-attention across all modalities
                attn_output, _ = self.attention(
                    combined_embeds, combined_embeds, combined_embeds,
                    key_padding_mask=~combined_masks.bool()
                )
                return self.norm(attn_output + combined_embeds)
        
        return TrimodalFusion(self.hidden_size)
    
    def _create_mock_vilt(self, config):
        """创建模拟ViLT模块"""
        class MockViLT(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size
                self.pooler = nn.Linear(hidden_size, hidden_size)
                
            def infer(self, batch):
                # 简单的模拟实现
                batch_size = batch.get("text_embeds", torch.tensor([[]])).shape[0]
                if batch_size == 0:
                    batch_size = 1
                
                cls_feats = torch.randn(batch_size, self.hidden_size)
                return {
                    "cls_feats": cls_feats,
                    "text_feats": batch.get("text_embeds", torch.empty(0)),
                    "image_feats": batch.get("image_embeds", torch.empty(0))
                }
        
        return MockViLT(self.hidden_size)
    
    def process_audio_modality(self, audio_data: torch.Tensor, audio_masks: torch.Tensor = None):
        """
        处理音频模态
        
        Args:
            audio_data: 音频特征 (B, seq_len, feature_dim) 或 (B, feature_dim)
            audio_masks: 音频掩码 (B, seq_len)
            
        Returns:
            Dict: 处理后的音频嵌入和掩码
        """
        batch_size = audio_data.shape[0]
        
        # 处理音频特征
        audio_embeds = self.audio_processor(audio_data)  # (B, seq_len, hidden_size)
        seq_len = audio_embeds.shape[1]
        
        # 添加位置编码
        position_ids = torch.arange(seq_len, device=audio_data.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.audio_position_embeddings(position_ids)
        audio_embeds = audio_embeds + position_embeds
        
        # 添加token type embedding (音频类型为2)
        audio_token_type_ids = torch.full((batch_size, seq_len), 2, device=audio_data.device)
        audio_type_embeds = self.audio_token_type_embedding(torch.zeros_like(audio_token_type_ids))
        audio_embeds = audio_embeds + audio_type_embeds
        
        # 处理掩码
        if audio_masks is None:
            audio_masks = torch.ones(batch_size, seq_len, device=audio_data.device)
        
        return {
            "audio_embeds": audio_embeds,
            "audio_masks": audio_masks
        }
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        三模态前向传播
        
        Args:
            batch: 包含文本、图像、音频数据的批次
            
        Returns:
            Dict: 融合后的特征
        """
        # 1. 处理文本和图像 (使用原始ViLT)
        vilt_batch = {
            "text_embeds": batch.get("text_embeds"),
            "text_masks": batch.get("text_masks"),
            "image_embeds": batch.get("image_embeds"), 
            "image_masks": batch.get("image_masks")
        }
        
        # 过滤None值
        vilt_batch = {k: v for k, v in vilt_batch.items() if v is not None}
        
        if len(vilt_batch) > 0:
            vilt_output = self.base_vilt.infer(vilt_batch)
        else:
            # 如果没有文本或图像数据，创建空输出
            batch_size = 1
            vilt_output = {
                "cls_feats": torch.zeros(batch_size, self.hidden_size),
                "text_feats": torch.empty(0),
                "image_feats": torch.empty(0)
            }
        
        # 2. 处理音频模态
        audio_output = None
        if "audio_data" in batch and batch["audio_data"] is not None:
            audio_output = self.process_audio_modality(
                batch["audio_data"], 
                batch.get("audio_masks")
            )
        
        # 3. 三模态融合
        if audio_output is not None:
            # 拼接所有模态
            all_embeds = []
            all_masks = []
            
            # 添加文本嵌入
            if "text_embeds" in batch and batch["text_embeds"] is not None:
                all_embeds.append(batch["text_embeds"])
                all_masks.append(batch["text_masks"])
            
            # 添加图像嵌入
            if "image_embeds" in batch and batch["image_embeds"] is not None:
                all_embeds.append(batch["image_embeds"])
                all_masks.append(batch["image_masks"])
            
            # 添加音频嵌入
            all_embeds.append(audio_output["audio_embeds"])
            all_masks.append(audio_output["audio_masks"])
            
            # 拼接
            if len(all_embeds) > 0:
                combined_embeds = torch.cat(all_embeds, dim=1)
                combined_masks = torch.cat(all_masks, dim=1)
                
                # 三模态融合
                fused_embeds = self.trimodal_fusion(combined_embeds, combined_masks)
                
                # 使用CLS token作为最终表示
                cls_feats = fused_embeds[:, 0]  # 假设第一个token是CLS
                
                vilt_output["cls_feats"] = cls_feats
                vilt_output["audio_feats"] = audio_output["audio_embeds"]
        
        return vilt_output


class TrimodalMissingAwarePrompts(nn.Module):
    """
    三模态缺失感知提示
    
    扩展原有的缺失感知提示以支持音频模态
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.prompt_length = config.get("prompt_length", 16)
        
        # 三模态缺失模式 (8种组合: 000, 001, 010, 011, 100, 101, 110, 111)
        self.missing_patterns = {
            (0, 0, 0): "all_missing",      # 不应该出现
            (0, 0, 1): "audio_only",       # 只有音频
            (0, 1, 0): "image_only",       # 只有图像
            (0, 1, 1): "image_audio",      # 图像+音频
            (1, 0, 0): "text_only",        # 只有文本
            (1, 0, 1): "text_audio",       # 文本+音频
            (1, 1, 0): "text_image",       # 文本+图像 (原ViLT)
            (1, 1, 1): "complete"          # 完整三模态
        }
        
        # 为每种缺失模式创建提示
        self.pattern_prompts = nn.ParameterDict()
        for pattern_key, pattern_name in self.missing_patterns.items():
            if pattern_key != (0, 0, 0):  # 排除全缺失情况
                self.pattern_prompts[pattern_name] = nn.Parameter(
                    torch.randn(1, self.prompt_length, self.hidden_size)
                )
    
    def get_missing_pattern(self, missing_mask: torch.Tensor) -> str:
        """
        根据缺失掩码获取缺失模式
        
        Args:
            missing_mask: (B, 3) 缺失掩码 [text, image, audio]
            
        Returns:
            str: 缺失模式名称
        """
        # 取第一个样本的模式（假设批次内模式相同）
        pattern = tuple(missing_mask[0].int().tolist())
        return self.missing_patterns.get(pattern, "complete")
    
    def forward(self, missing_mask: torch.Tensor) -> torch.Tensor:
        """
        根据缺失模式生成提示
        
        Args:
            missing_mask: (B, 3) 缺失掩码
            
        Returns:
            torch.Tensor: 提示嵌入 (B, prompt_length, hidden_size)
        """
        batch_size = missing_mask.shape[0]
        pattern_name = self.get_missing_pattern(missing_mask)
        
        if pattern_name in self.pattern_prompts:
            prompt = self.pattern_prompts[pattern_name]
            return prompt.expand(batch_size, -1, -1)
        else:
            # 默认使用完整模式提示
            prompt = self.pattern_prompts["complete"]
            return prompt.expand(batch_size, -1, -1)
