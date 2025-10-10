import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import math


class ModalityEncoder(nn.Module):
    """
    模态编码器 f_k
    
    对于每个模态k，冻结的编码器f_k将原始输入映射为标记序列：
    t^(k) = f_k(x^(k)) ∈ R^(L_k × d)
    """
    
    def __init__(self, modality_type: str, hidden_size: int, config: Dict):
        super().__init__()
        
        self.modality_type = modality_type
        self.hidden_size = hidden_size
        self.config = config
        
        if modality_type == "text":
            self._init_text_encoder(config)
        elif modality_type == "image":
            self._init_image_encoder(config)
        elif modality_type == "audio":
            self._init_audio_encoder(config)
        else:
            raise ValueError(f"Unsupported modality type: {modality_type}")
        
        # 冻结编码器参数
        self._freeze_parameters()
    
    def _init_text_encoder(self, config):
        """初始化文本编码器 (BERT)"""
        from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
        
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=self.hidden_size,
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=self.hidden_size * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )
        
        self.encoder = BertEmbeddings(bert_config)
        self.token_length = config["max_text_len"]
    
    def _init_image_encoder(self, config):
        """初始化图像编码器 (ViT)"""
        try:
            from ..modules.vision_transformer_prompts import VisionTransformer
            self.encoder = VisionTransformer(pretrained=True, config=config)
        except ImportError:
            # 如果无法导入，创建一个简单的模拟编码器
            class MockVisionTransformer(nn.Module):
                def __init__(self, hidden_size, token_length):
                    super().__init__()
                    self.hidden_size = hidden_size
                    self.token_length = token_length
                    self.linear = nn.Linear(3 * 224 * 224, hidden_size * token_length)
                
                def visual_embed(self, x, max_image_len=None, mask_it=False):
                    batch_size = x.shape[0]
                    x_flat = x.view(batch_size, -1)
                    tokens = self.linear(x_flat).view(batch_size, self.token_length, self.hidden_size)
                    masks = torch.ones(batch_size, self.token_length, device=x.device)
                    return tokens, masks, None, None
            
            self.encoder = MockVisionTransformer(self.hidden_size, config["max_image_len"])
        
        self.token_length = config["max_image_len"]
    
    def _init_audio_encoder(self, config):
        """初始化音频编码器"""
        # 这里可以实现音频编码器，如Wav2Vec2或COVAREP
        self.encoder = nn.Linear(config["audio_feature_dim"], self.hidden_size)
        self.token_length = config["audio_token_length"]
    
    def _freeze_parameters(self):
        """冻结编码器参数"""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 原始输入 x^(k)
            
        Returns:
            torch.Tensor: 标记序列 t^(k) ∈ R^(L_k × d)
        """
        if self.modality_type == "text":
            return self.encoder(x)  # (B, L_k, d)
        elif self.modality_type == "image":
            # 对于图像，需要特殊处理
            if hasattr(self.encoder, 'visual_embed'):
                embeds, masks, patch_index, image_labels = self.encoder.visual_embed(
                    x, max_image_len=self.token_length, mask_it=False
                )
                return embeds  # (B, L_k, d)
            else:
                return self.encoder(x)
        elif self.modality_type == "audio":
            return self.encoder(x)  # (B, L_k, d)
        else:
            raise ValueError(f"Unsupported modality type: {self.modality_type}")


class NullTokenGenerator(nn.Module):
    """
    空标记生成器
    
    为缺失模态生成学习的空标记序列 t̄^(k) ∈ R^(L_k × d)
    """
    
    def __init__(self, num_modalities: int, token_lengths: List[int], hidden_size: int):
        super().__init__()
        
        self.num_modalities = num_modalities
        self.token_lengths = token_lengths
        self.hidden_size = hidden_size
        
        # 为每个模态创建空标记
        self.null_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(token_lengths[i], hidden_size))
            for i in range(num_modalities)
        ])
        
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化空标记参数"""
        for null_token in self.null_tokens:
            nn.init.normal_(null_token, std=0.02)
    
    def get_null_tokens(self, modality_idx: int, batch_size: int) -> torch.Tensor:
        """
        获取指定模态的空标记
        
        Args:
            modality_idx: 模态索引
            batch_size: 批次大小
            
        Returns:
            torch.Tensor: 空标记序列 (B, L_k, d)
        """
        return self.null_tokens[modality_idx].unsqueeze(0).expand(batch_size, -1, -1)


class CrossModalFusion(nn.Module):
    """
    跨模态融合器 F
    
    冻结的跨模态transformer F消费各模态的标记流并产生融合表示：
    z = F({t̃^(k)}_k=1^M) ∈ R^d
    """
    
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, config: Dict):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # 使用ViLT作为跨模态融合器
        try:
            from ..modules.vilt_module import ViLTransformerSS
            self.fusion_transformer = ViLTransformerSS(config)
        except ImportError:
            # 如果无法导入，创建一个简单的模拟融合器
            class MockFusionTransformer(nn.Module):
                def __init__(self, hidden_size):
                    super().__init__()
                    self.hidden_size = hidden_size
                    self.pooler = nn.Linear(hidden_size, hidden_size)
                
                def infer(self, batch):
                    # 模拟ViLT的infer方法
                    text_embeds = batch.get("text_embeds", torch.empty(0, 0, self.hidden_size))
                    image_embeds = batch.get("image_embeds", torch.empty(0, 0, self.hidden_size))
                    
                    if text_embeds.numel() > 0 and image_embeds.numel() > 0:
                        # 拼接文本和图像嵌入
                        combined = torch.cat([text_embeds, image_embeds], dim=1)
                        # 使用CLS token
                        cls_feats = self.pooler(combined[:, 0])
                    elif text_embeds.numel() > 0:
                        cls_feats = self.pooler(text_embeds[:, 0])
                    elif image_embeds.numel() > 0:
                        cls_feats = self.pooler(image_embeds[:, 0])
                    else:
                        batch_size = 1
                        cls_feats = torch.zeros(batch_size, self.hidden_size)
                    
                    return {"cls_feats": cls_feats}
            
            self.fusion_transformer = MockFusionTransformer(hidden_size)
        
        # 冻结融合器参数
        self._freeze_parameters()
    
    def _freeze_parameters(self):
        """冻结融合器参数"""
        for param in self.fusion_transformer.parameters():
            param.requires_grad = False
    
    def forward(self, modality_tokens: Dict[str, torch.Tensor], 
                modality_masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        跨模态融合
        
        Args:
            modality_tokens: 各模态标记 {modality: (B, L_k, d)}
            modality_masks: 各模态掩码 {modality: (B, L_k)}
            
        Returns:
            torch.Tensor: 融合表示 z ∈ R^d
        """
        # 拼接所有模态的标记和掩码
        all_tokens = []
        all_masks = []
        
        for modality, tokens in modality_tokens.items():
            # 确保所有tokens都是3维的 (B, L_k, d)
            if tokens.dim() == 4:
                tokens = tokens.squeeze(0)  # 移除多余的维度
            all_tokens.append(tokens)
            all_masks.append(modality_masks[modality])
        
        # 拼接
        combined_tokens = torch.cat(all_tokens, dim=1)  # (B, sum(L_k), d)
        combined_masks = torch.cat(all_masks, dim=1)    # (B, sum(L_k))
        
        # 通过融合transformer
        # 这里需要根据实际的ViLT实现进行调整
        fused_features = self.fusion_transformer.infer({
            "text_embeds": modality_tokens.get("text", torch.empty(0, 0, self.hidden_size)),
            "text_masks": modality_masks.get("text", torch.empty(0, 0)),
            "image_embeds": modality_tokens.get("image", torch.empty(0, 0, self.hidden_size)),
            "image_masks": modality_masks.get("image", torch.empty(0, 0)),
        })
        
        return fused_features["cls_feats"]  # (B, d)


class TaskHead(nn.Module):
    """
    任务头 W
    
    浅层任务头 W ∈ R^(|Y| × d) 产生分类logits：
    s = W z + b
    """
    
    def __init__(self, hidden_size: int, num_classes: int, task_type: str = "classification"):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.task_type = task_type
        
        if task_type == "classification":
            self.head = nn.Linear(hidden_size, num_classes)
        elif task_type == "regression":
            self.head = nn.Linear(hidden_size, 1)
        elif task_type == "multilabel":
            self.head = nn.Linear(hidden_size, num_classes)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化任务头参数"""
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z: 融合表示 (B, d)
            
        Returns:
            torch.Tensor: 任务logits (B, num_classes) 或 (B, 1)
        """
        return self.head(z)


class IncompleteMultimodalLearning(nn.Module):
    """
    不完整多模态学习 (IML) 基础模型
    
    实现论文Preliminaries部分的基础设置：
    - M个异构模态 X = {x^(1), ..., x^(M)}
    - 二进制存在掩码 m ∈ {0,1}^M
    - 完整子集 D_c = {(x, 1, y)}
    - 不完整子集 D_s = {(x, m, y): m ≠ 1}
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.num_modalities = config["num_modalities"]
        self.modality_names = config["modality_names"]  # ["text", "image", "audio"]
        self.num_classes = config["num_classes"]
        self.task_type = config.get("task_type", "classification")
        
        # 初始化模态编码器
        self.modality_encoders = nn.ModuleDict()
        self.token_lengths = []
        
        for i, modality in enumerate(self.modality_names):
            encoder = ModalityEncoder(modality, self.hidden_size, config)
            self.modality_encoders[modality] = encoder
            self.token_lengths.append(encoder.token_length)
        
        # 初始化空标记生成器
        self.null_token_generator = NullTokenGenerator(
            self.num_modalities, self.token_lengths, self.hidden_size
        )
        
        # 初始化跨模态融合器
        self.fusion_transformer = CrossModalFusion(
            self.hidden_size, config["num_layers"], config["num_heads"], config
        )
        
        # 初始化任务头
        self.task_head = TaskHead(self.hidden_size, self.num_classes, self.task_type)
        
        # 模态索引映射
        self.modality_to_idx = {modality: i for i, modality in enumerate(self.modality_names)}
    
    def get_present_absent_sets(self, m: torch.Tensor) -> Tuple[List[int], List[int]]:
        """
        获取存在和缺失的索引集合
        
        Args:
            m: 存在掩码 (B, M)
            
        Returns:
            Tuple[List[int], List[int]]: (P(m), A(m))
        """
        # P(m) = {k: m_k = 1}
        # A(m) = {k: m_k = 0}
        present_indices = []
        absent_indices = []
        
        for i in range(self.num_modalities):
            if m[i] == 1:
                present_indices.append(i)
            else:
                absent_indices.append(i)
        
        return present_indices, absent_indices
    
    def encode_modalities(self, x: Dict[str, torch.Tensor], m: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        编码各模态
        
        Args:
            x: 原始输入 {modality: x^(k)}
            m: 存在掩码 (M,)
            
        Returns:
            Dict[str, torch.Tensor]: 编码后的标记 {modality: t^(k)}
        """
        batch_size = next(iter(x.values())).shape[0]
        encoded_tokens = {}
        
        for i, modality in enumerate(self.modality_names):
            if modality in x:
                # 模态存在
                if m[i].item() == 1:
                    # 使用真实编码器
                    tokens = self.modality_encoders[modality](x[modality])
                else:
                    # 使用空标记
                    tokens = self.null_token_generator.get_null_tokens(i, batch_size)
            else:
                # 模态不存在，使用空标记
                tokens = self.null_token_generator.get_null_tokens(i, batch_size)
            
            encoded_tokens[modality] = tokens
        
        return encoded_tokens
    
    def forward(self, x: Dict[str, torch.Tensor], m: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 原始输入 {modality: x^(k)}
            m: 存在掩码 (M,)
            
        Returns:
            torch.Tensor: 任务logits s
        """
        # 编码各模态
        modality_tokens = self.encode_modalities(x, m)
        
        # 创建掩码
        modality_masks = {}
        for i, modality in enumerate(self.modality_names):
            if m[i] == 1:
                # 模态存在，创建全1掩码
                modality_masks[modality] = torch.ones(
                    modality_tokens[modality].shape[0], 
                    modality_tokens[modality].shape[1],
                    device=modality_tokens[modality].device
                )
            else:
                # 模态缺失，创建全0掩码
                modality_masks[modality] = torch.zeros(
                    modality_tokens[modality].shape[0], 
                    modality_tokens[modality].shape[1],
                    device=modality_tokens[modality].device
                )
        
        # 跨模态融合
        z = self.fusion_transformer(modality_tokens, modality_masks)
        
        # 任务预测
        s = self.task_head(z)
        
        return s
    
    def compute_task_loss(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算任务损失
        
        Args:
            s: 预测logits
            y: 真实标签
            
        Returns:
            torch.Tensor: 任务损失
        """
        if self.task_type == "classification":
            return F.cross_entropy(s, y)
        elif self.task_type == "regression":
            return F.mse_loss(s.squeeze(), y.float())
        elif self.task_type == "multilabel":
            return F.binary_cross_entropy_with_logits(s, y.float())
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    
    def get_learnable_parameters(self) -> List[torch.Tensor]:
        """
        获取可学习参数
        
        只有轻量级提示参数、小适配器和任务头是可学习的；
        所有编码器{f_k}和F保持冻结。
        
        Returns:
            List[torch.Tensor]: 可学习参数列表
        """
        learnable_params = []
        
        # 任务头参数
        learnable_params.extend(list(self.task_head.parameters()))
        
        # 空标记参数
        learnable_params.extend(list(self.null_token_generator.parameters()))
        
        return learnable_params
