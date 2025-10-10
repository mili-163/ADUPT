"""
集成GPU接口的主模型

基于method_model_implementation.py，集成GPU接口，支持CPU/GPU无缝切换
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import math

from .incomplete_multimodal_learning import IncompleteMultimodalLearning
from .modality_prompts_v2 import AbsenceAwareModalityPrompts, ModalityPromptInjector
from .dual_prompts_v3 import DualInstancePromptsV3, ContentAggregator
from .hierarchical_distillation import HierarchicalDistillation, TeacherModel, InferenceModel
from .hard_negative_contrastive import SpecializationContrastiveRegularizer
from .calibrated_prompt_adaptation import CalibratedPromptAdaptation
from .trimodal_vilt_adapter import TrimodalViLTAdapter, TrimodalMissingAwarePrompts
from .training_objective import TrainingObjective, ParameterEfficiencyManager

from .modality_processors import ModalityProcessorManager
from .experimental_protocol import ExperimentalProtocol, MissingnessGenerator
from .gpu_interface import DeviceManager, GPUModelWrapper, create_device_manager

try:
    import pytorch_lightning as pl
    _BaseLM = pl.LightningModule
except Exception:
    class _LightningModuleShim(nn.Module):
        def __init__(self):
            super().__init__()
        def save_hyperparameters(self):
            pass
    pl = None
    _BaseLM = _LightningModuleShim


class MethodModelGPU(_BaseLM):
    """
    集成GPU接口的主模型
    
    支持CPU/GPU无缝切换，自动设备管理
    """
    
    def __init__(self, config: Dict, device: Optional[str] = None, auto_detect_device: bool = True):
        super().__init__()
        
        # 保存超参数
        try:
            self.save_hyperparameters()
        except Exception:
            pass
        
        if not hasattr(self, 'hparams'):
            class _HP: pass
            self.hparams = _HP()
            self.hparams.config = dict(config)
        
        # 初始化设备管理器
        self.device_manager = create_device_manager(device, auto_detect_device)
        self.device_manager.print_device_info()
        
        # 基础配置
        self.hidden_size = config["hidden_size"]
        self.num_modalities = config["num_modalities"]
        self.modality_names = config["modality_names"]
        self.num_classes = config["num_classes"]
        
        
        self.implementation_config = config.get("implementation_details", {})
        
        # 初始化模态处理器
        self._init_modality_processors()
        
        # 初始化基础IML模型
        self.base_iml = IncompleteMultimodalLearning(config)
        
        # 初始化三模态ViLT适配器
        self.trimodal_adapter = TrimodalViLTAdapter(config)
        
        # 初始化三模态缺失感知提示
        self.trimodal_prompts = TrimodalMissingAwarePrompts(config)
        
        # 初始化两种提示族 - 
        self._init_prompt_families()
        
        # 初始化教师和推理网络
        self._init_teacher_inference_networks()
        
        # 初始化分层蒸馏
        self._init_hierarchical_distillation()
        
        # 初始化硬负样本对比正则化
        self._init_hard_negative_contrastive()
        
        # 初始化单步校准提示适应
        self._init_calibrated_prompt_adaptation()
        
        # 初始化训练目标
        self._init_training_objective()
        
        # 初始化实验协议
        self._init_experimental_protocol()
        
        # 初始化参数效率管理器
        self.param_manager = ParameterEfficiencyManager(self)
        
        # 将模型移动到指定设备
        self.to(self.device_manager.get_device())
        
        # 设置训练状态
        self.teacher_trained = False
        self.inference_trained = False
        self.cpa_enabled = False
    
    def _init_modality_processors(self):
        
        self.modality_processor = ModalityProcessorManager(
            self.implementation_config
        )
    
    def _init_prompt_families(self):
        
        prompt_config = self.implementation_config.get("prompt_config", {})
        
        # 模态级提示 P^mod(m) - 缺失感知上下文
        self.modality_prompts = AbsenceAwareModalityPrompts(
            hidden_size=self.hidden_size,
            num_modalities=self.num_modalities,
            prompt_length=prompt_config.get("prompt_length", 16),
            num_layers=prompt_config.get("injection_layers", 5)
        )
        
        # 模态提示注入器
        self.prompt_injector = ModalityPromptInjector(
            hidden_size=self.hidden_size,
            prompt_length=prompt_config.get("prompt_length", 16),
            num_layers=prompt_config.get("injection_layers", 5)
        )
        
        # 实例级双提示 P^sp, P^gn - 专业化和泛化
        self.dual_instance_prompts = DualInstancePromptsV3(
            hidden_size=self.hidden_size,
            num_modalities=self.num_modalities,
            prompt_length=prompt_config.get("prompt_length", 16),
            num_layers=prompt_config.get("injection_layers", 5),
            config=prompt_config
        )
    
    def _init_teacher_inference_networks(self):
        
        # 教师网络T - 在D_c上训练
        self.teacher_T = TeacherModel(
            base_iml=self.base_iml,
            hidden_size=self.hidden_size,
            num_classes=self.num_classes,
            parent_model=self  # 传递主模型引用以访问编码方法
        )
        
        # 推理网络I - 在D_s∪D_c上训练
        self.inference_I = InferenceModel(
            base_iml=self.base_iml,
            modality_prompts=self.modality_prompts,
            dual_prompts=self.dual_instance_prompts,
            prompt_injector=self.prompt_injector,
            hidden_size=self.hidden_size,
            num_classes=self.num_classes,
            parent_model=self  # 传递主模型引用以访问编码方法
        )
    
    def _init_hierarchical_distillation(self):
        
        loss_weights = self.implementation_config.get("loss_weights", {})
        
        self.hierarchical_distillation = HierarchicalDistillation(
            hidden_size=self.hidden_size,
            num_modalities=self.num_modalities,
            selected_layers=[0, 1, 2, 3, 4],  # 前5层
            temperature=loss_weights.get("tau", 2)
        )
    
    def _init_hard_negative_contrastive(self):
        
        loss_weights = self.implementation_config.get("loss_weights", {})
        
        self.contrastive_regularizer = SpecializationContrastiveRegularizer(
            hidden_size=self.hidden_size,
            num_classes=self.num_classes,
            temperature=0.07,  # 默认温度
            top_k=loss_weights.get("K", 10)
        )
    
    def _init_calibrated_prompt_adaptation(self):
        
        cpa_config = self.implementation_config.get("cpa_config", {})
        
        self.calibrated_prompt_adaptation = CalibratedPromptAdaptation(
            hidden_size=self.hidden_size,
            num_classes=self.num_classes,
            eta=cpa_config.get("statistic_weight", 0.5),
            gamma=cpa_config.get("learning_rate", 1e-2)
        )
    
    def _init_training_objective(self):
        
        loss_weights = self.implementation_config.get("loss_weights", {})
        

    
    def _init_experimental_protocol(self):
        
        self.experimental_protocol = ExperimentalProtocol(self.implementation_config)
    
    def process_input_data(self, raw_data: Dict[str, any], 
                          missing_mask: torch.Tensor) -> Dict[str, any]:
        """
        处理输入数据 - 严格按照Implementation Details
        
        Args:
            raw_data: 原始数据 {modality: data}
            missing_mask: 缺失掩码 (M,)
            
        Returns:
            Dict[str, any]: 处理后的数据
        """
        # 确保缺失掩码在正确设备上
        missing_mask = self.device_manager.to_device(missing_mask)
        
        # 处理数据
        processed_data = self.modality_processor.process_batch(raw_data, missing_mask)
        
        # 确保所有数据在正确设备上
        return self.device_manager.to_device(processed_data)
    
    def teacher_forward(self, x: Dict[str, torch.Tensor], m: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        教师网络T前向传播
        
        在完整数据D_c上训练，m = 1
        """
        # 确保输入在正确设备上
        x = self.device_manager.to_device(x)
        m = self.device_manager.to_device(m)
        
        return self.teacher_T(x, m)
    
    def _encode_bimodal(self, x: Dict[str, torch.Tensor], m: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        二模态编码：使用原始ViLT处理文本+图像
        
        Args:
            x: 原始输入 {modality: x^(k)}
            m: 存在掩码 (M,)
            
        Returns:
            Dict[str, torch.Tensor]: 编码后的模态标记
        """
        return self.base_iml.encode_modalities(x, m)
    
    def _encode_trimodal(self, x: Dict[str, torch.Tensor], m: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        三模态编码：使用TrimodalViLTAdapter处理文本+图像+音频
        
        Args:
            x: 原始输入 {modality: x^(k)}
            m: 存在掩码 (M,)
            
        Returns:
            Dict[str, torch.Tensor]: 编码后的模态标记
        """
        # 构造三模态批次
        batch = {
            "text_embeds": x.get("text"),
            "text_masks": torch.ones_like(x.get("text", torch.zeros(1, 40))) if "text" in x else None,
            "image_embeds": x.get("image"),
            "image_masks": torch.ones_like(x.get("image", torch.zeros(1, 197))) if "image" in x else None,
            "audio_data": x.get("audio"),
            "audio_masks": torch.ones_like(x.get("audio", torch.zeros(1, 100))) if "audio" in x else None,
        }
        
        # 使用三模态适配器
        trimodal_output = self.trimodal_adapter(batch)
        
        # 转换为标准格式
        modality_tokens = {}
        if "text_feats" in trimodal_output:
            modality_tokens["text"] = trimodal_output["text_feats"]
        if "image_feats" in trimodal_output:
            modality_tokens["image"] = trimodal_output["image_feats"]
        if "audio_feats" in trimodal_output:
            modality_tokens["audio"] = trimodal_output["audio_feats"]
            
        return modality_tokens
    
    def inference_forward(self, x: Dict[str, torch.Tensor], m: torch.Tensor, 
                         use_cpa: bool = False) -> Dict[str, torch.Tensor]:
        """
        推理网络I前向传播
        
        在D_s∪D_c上训练，使用提示机制处理缺失模态
        
        Args:
            x: 原始输入 {modality: x^(k)}
            m: 存在掩码 (M,)
            use_cpa: 是否使用单步校准提示适应
        """
        # 确保输入在正确设备上
        x = self.device_manager.to_device(x)
        m = self.device_manager.to_device(m)
        
        if use_cpa and self.cpa_enabled:
            return self._inference_forward_with_cpa(x, m)
        else:
            return self.inference_I(x, m)
    
    def _inference_forward_with_cpa(self, x: Dict[str, torch.Tensor], 
                                   m: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        带CPA的推理前向传播
        
        Args:
            x: 原始输入 {modality: x^(k)}
            m: 存在掩码 (M,)
            
        Returns:
            Dict[str, torch.Tensor]: 推理输出
        """
        # 1. 编码各模态 - 根据模态数量选择合适的编码器
        if self.num_modalities == 3:
            # 三模态数据集：使用TrimodalViLTAdapter
            modality_tokens = self._encode_trimodal(x, m)
        else:
            # 二模态数据集：使用原始ViLT
            modality_tokens = self._encode_bimodal(x, m)
        
        # 2. 生成模态级提示 P^mod(m)
        modality_prompts = self.modality_prompts(m.unsqueeze(0))
        
        # 3. 生成实例级双提示 P^sp, P^gn
        dual_outputs = self.dual_instance_prompts(
            modality_tokens, modality_prompts, return_specialization_features=True
        )
        instance_prompts = dual_outputs["mixed_prompts"]
        specialization_features = dual_outputs.get("specialization_features")
        
        # 4. 单步校准提示适应
        adapted_prompts = self.calibrated_prompt_adaptation(
            x, m, self, instance_prompts
        )
        
        # 5. 注入适应后的提示到模态标记
        processed_tokens = self.prompt_injector.inject_prompts(
            modality_tokens, adapted_prompts, m.unsqueeze(0)
        )
        
        # 6. 创建掩码（包含提示部分）
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
        
        # 7. 通过融合transformer
        z = self.base_iml.fusion_transformer(processed_tokens, modality_masks)
        
        # 8. 任务预测
        s = self.base_iml.task_head(z)
        
        result = {
            "features": z,
            "logits": s,
            "probs": F.softmax(s, dim=-1),
            "modality_prompts": modality_prompts,
            "instance_prompts": adapted_prompts
        }
        
        if specialization_features is not None:
            result["specialization_features"] = specialization_features
        
        return result
    
    def forward(self, raw_data: Dict[str, any], missing_mask: torch.Tensor, 
                y: Optional[torch.Tensor] = None, use_cpa: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            raw_data: 原始数据 {modality: data}
            missing_mask: 存在掩码 (M,)
            y: 真实标签（可选）
            use_cpa: 是否使用单步校准提示适应
            
        Returns:
            Dict[str, torch.Tensor]: 模型输出
        """
        # 处理输入数据
        x = self.process_input_data(raw_data, missing_mask)
        
        ret = {}
        
        if self.training:
            # 训练模式
            if not self.teacher_trained:
                # 阶段1：教师训练
                teacher_outputs = self.teacher_forward(x, missing_mask)
                ret.update(teacher_outputs)
                
                if y is not None:
                    y = self.device_manager.to_device(y)
                    
                    losses = self.training_objective.compute_total_loss(
                        logits=teacher_outputs["logits"],
                        labels=y,
                        modality_prompts=self.modality_prompts,
                        hierarchical_distillation=self.hierarchical_distillation,
                        contrastive_regularizer=self.contrastive_regularizer,
                        teacher_outputs=None,  # 教师训练阶段没有教师输出
                        student_outputs=teacher_outputs,
                        missing_mask=missing_mask.unsqueeze(0)
                    )
                    ret.update(losses)
                    
            elif not self.inference_trained:
                # 阶段2：推理训练（带分层蒸馏和硬负样本对比）
                teacher_outputs = self.teacher_forward(x, missing_mask)
                inference_outputs = self.inference_forward(x, missing_mask, use_cpa=False)
                
                ret.update(inference_outputs)
                
                if y is not None:
                    y = self.device_manager.to_device(y)
                    
                    losses = self.training_objective.compute_total_loss(
                        logits=inference_outputs["logits"],
                        labels=y,
                        modality_prompts=self.modality_prompts,
                        hierarchical_distillation=self.hierarchical_distillation,
                        contrastive_regularizer=self.contrastive_regularizer,
                        teacher_outputs=teacher_outputs,
                        student_outputs=inference_outputs,
                        missing_mask=missing_mask.unsqueeze(0)
                    )
                    ret.update(losses)
                
            else:
                # 阶段3：推理模式
                inference_outputs = self.inference_forward(x, missing_mask, use_cpa=use_cpa)
                ret.update(inference_outputs)
        else:
            # 推理模式
            inference_outputs = self.inference_forward(x, missing_mask, use_cpa=use_cpa)
            ret.update(inference_outputs)
        
        return ret
    
    def set_teacher_trained(self, trained: bool = True):
        
        self.teacher_trained = trained
        if trained:
            # 冻结教师网络
            for param in self.teacher_T.parameters():
                param.requires_grad = False
    
    def set_inference_trained(self, trained: bool = True):
        
        self.inference_trained = trained
    
    def enable_cpa(self, enabled: bool = True):
        
        self.cpa_enabled = enabled
    
    def update_teacher_statistics(self, teacher_features: torch.Tensor):
        
        teacher_features = self.device_manager.to_device(teacher_features)
        self.calibrated_prompt_adaptation.update_teacher_statistics(teacher_features)
    
    def get_learnable_parameters(self) -> List[torch.Tensor]:
        
        return self.param_manager.get_learnable_parameters()
    
    def print_parameter_efficiency(self):
        
        self.param_manager.print_parameter_efficiency()
    
    def get_parameter_count(self) -> Dict[str, int]:
        
        return self.param_manager.get_parameter_count()
    
    def get_device_info(self) -> Dict[str, Any]:
        
        return self.device_manager.get_device_info()
    
    def run_experiment(self, experimental_data: Dict) -> Dict:
        """
        运行实验
        
        Args:
            experimental_data: 实验数据
            
        Returns:
            Dict: 实验结果
        """
        return self.experimental_protocol.run_experiment(self, experimental_data)
    
    def benchmark_performance(self, input_shape: Tuple[int, ...], num_runs: int = 10) -> Dict[str, float]:
        """
        基准测试性能
        
        Args:
            input_shape: 输入形状
            num_runs: 运行次数
            
        Returns:
            Dict[str, float]: 性能指标
        """
        from .gpu_interface import benchmark_device_performance
        return benchmark_device_performance(self, self.device_manager, input_shape, num_runs)
    
    def get_optimal_batch_size(self, input_shape: Tuple[int, ...], max_batch_size: int = 64) -> int:
        """
        获取最优批次大小
        
        Args:
            input_shape: 输入形状
            max_batch_size: 最大批次大小
            
        Returns:
            int: 最优批次大小
        """
        from .gpu_interface import get_optimal_batch_size
        return get_optimal_batch_size(self, self.device_manager, input_shape, max_batch_size)
    
    def clear_memory(self):
        
        from .gpu_interface import clear_gpu_memory
        clear_gpu_memory(self.device_manager)

