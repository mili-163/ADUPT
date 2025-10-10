import torch
import torchvision.transforms as transforms
from transformers import BertTokenizer
import numpy as np

IMPLEMENTATION_CONFIG = {
    "image_processing": {
        "resize_short_side": 384,
        "max_long_side": 640,
        "maintain_aspect_ratio": True,
        "video_frame_processing": {
            "process_individually": True,
            "fusion_method": "average",
            "temporal_fusion": "average_across_frames"
        },
        "missing_image_replacement": {
            "type": "virtual_input",
            "pixel_value": 1.0,
            "shape": (3, 384, 384)
        },
        
        # 图像变换
        "transforms": {
            "resize": transforms.Resize((384, 384)),
            "normalize": transforms.Normalize(
                mean=[0.5, 0.5, 0.5], 
                std=[0.5, 0.5, 0.5]
            ),
            "to_tensor": transforms.ToTensor()
        }
    },
    
    # ==================== 文本模态处理 ====================
    "text_processing": {
        # 使用bert-base-uncased tokenizer
        "tokenizer_name": "bert-base-uncased",
        "tokenizer_config": {
            "do_lower_case": True,
            "add_special_tokens": True,
            "padding": True,
            "truncation": True
        },
        
        # 不同数据集的最大序列长度
        "max_sequence_lengths": {
            "MM-IMDb": 1024,
            "UPMC_Food-101": 512,
            "Hateful_Memes": 128,
            "CMU-MOSEI": 128
        },
        
        # 缺失文本处理：空字符串
        "missing_text_replacement": "",
        
        # 默认最大长度
        "default_max_length": 128
    },
    
    # ==================== 音频模态处理 ====================
    "audio_processing": {
        # COVAREP特征：74维向量
        "feature_dim": 74,
        "feature_extractor": "COVAREP",
        
        # 缺失音频处理：零向量
        "missing_audio_replacement": {
            "type": "zero_vector",
            "dim": 74
        },
        
        # 直接输入ViLT融合块
        "direct_to_fusion": True,
        "no_separate_encoder": True
    },
    
    # ==================== ViLT Backbone配置 ====================
    "vilt_backbone": {
        "model_name": "dandelin/vilt-b32-mlm",
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "drop_rate": 0.1,
        
        # 冻结所有预训练参数
        "frozen_parameters": True,
        "frozen_modules": [
            "embeddings",
            "encoder",
            "pooler"
        ],
        
        # 只更新轻量级参数
        "learnable_modules": [
            "prompts",
            "gating_units", 
            "mask_adapters",
            "classification_heads"
        ]
    },
    
    # ==================== 提示配置 ====================
    "prompt_config": {
        # 模态级提示：小MLP从存在/缺失嵌入生成
        "modality_prompts": {
            "generation_method": "small_mlp",
            "input_embeddings": "presence_missing_embeddings",
            "mlp_hidden_size": 256
        },
        
        # 实例级提示：泛化分支+专业化分支
        "instance_prompts": {
            "generalization_branch": True,
            "specialization_branch": True,
            "gating_unit": True,
            "gating_inputs": ["masks", "instance_content"]
        },
        
        # 提示长度和注入层数
        "prompt_length": 16,
        "injection_layers": 5,  # 前5层
        "injection_layers_list": [0, 1, 2, 3, 4]
    },
    
    # ==================== 损失权重配置 ====================
    "loss_weights": {
        # 缺失模态重构损失权重
        "alpha": 0.1,  # 掩码重构权重
        
        # 特征蒸馏权重
        "lambda_f": 5e-4,  # 特征蒸馏权重
        
        # 预测蒸馏权重
        "lambda_p": 1e-2,  # 预测蒸馏权重
        
        # 蒸馏温度
        "tau": 2,  # 蒸馏温度
        
        # 硬负样本对比损失权重
        "beta": 0.1,  # 专业化分支对比权重
        "K": 10  # 负样本数量
    },
    
    # ==================== CPA配置 ====================
    "cpa_config": {
        "learning_rate": 1e-2,  # γ
        "statistic_weight": 0.5,  # η
        "weak_views": 2,  # V
        "single_step_updates": True,
        "update_only_instance_prompts": True
    },
    
    # ==================== 优化器配置 ====================
    "optimizer": {
        "type": "Adam",
        "beta1": 0.9,
        "beta2": 0.999,
        "learning_rate": 1e-2,
        "weight_decay": 2e-2,
        "batch_size": 32
    },
    
    # ==================== 训练配置 ====================
    "training": {
        "teacher_epochs": {
            "min": 10,
            "max": 15,
            "default": 12
        },
        "student_epochs": {
            "min": 30,
            "max": 50,
            "default": 40
        },
        "teacher_data": "complete_only",
        "student_data": "complete_and_incomplete"
    },
    
    # ==================== 硬件配置 ====================
    "hardware": {
        "cpu": "Intel(R) Xeon(R) Platinum 8462Y",
        "gpu": "2 × NVIDIA A800-SXM4-80GB",
        "ram": "1TB",
        "repetitions": 3  # 每个实验重复3次
    }
}

# 缺失模式配置
MISSINGNESS_REGIMES = {
    "asymmetric": {
        "description": "缺失集中在单个模态",
        "missing_concentration": "single_modality",
        "other_modalities": "fully_observed"
    },
    "symmetric": {
        "description": "缺失均匀分布在模态子集",
        "missing_distribution": "even_across_subset",
        "min_affected_modalities": 2,
        "other_modalities": "fully_observed"
    },
    "uncertain": {
        "description": "训练和测试使用不同缺失配置",
        "train_test_different": True,
        "emulate_unseen_patterns": True,
        "vary_regime_and_affected_set": True
    }
}

# 教师-学生训练计划
TEACHER_STUDENT_SCHEDULE = {
    "teacher": {
        "data": "complete_subset_only",
        "cache_statistics": True,
        "statistics_to_cache": [
            "feature_means",
            "feature_variances", 
            "class_prototypes"  # 可选
        ]
    },
    "student": {
        "data": "incomplete_and_complete",
        "hierarchical_distillation": True,
        "feature_level_distillation": {
            "method": "mask_conditioned_adapter",
            "adapter_type": "FiLM"
        },
        "prediction_level_distillation": {
            "method": "softened_KL",
            "temperature": 2
        }
    }
}

def get_implementation_config(dataset_name: str = "CMU-MOSEI") -> dict:
    config = IMPLEMENTATION_CONFIG.copy()
    
    # 根据数据集设置最大序列长度
    if dataset_name in config["text_processing"]["max_sequence_lengths"]:
        config["text_processing"]["max_length"] = config["text_processing"]["max_sequence_lengths"][dataset_name]
    else:
        config["text_processing"]["max_length"] = config["text_processing"]["default_max_length"]
    
    return config

def get_missingness_config(regime: str = "asymmetric", missing_rate: float = 0.5) -> dict:
    """
    获取缺失模式配置
    
    Args:
        regime: 缺失模式 ("asymmetric", "symmetric", "uncertain")
        missing_rate: 缺失率 ξ ∈ [0, 1]
        
    Returns:
        dict: 缺失模式配置
    """
    config = MISSINGNESS_REGIMES[regime].copy()
    config["missing_rate"] = missing_rate
    config["missing_rate_percentage"] = missing_rate * 100
    
    return config

def get_teacher_student_schedule() -> dict:
    """
    获取教师-学生训练计划
    
    Returns:
        dict: 训练计划配置
    """
    return TEACHER_STUDENT_SCHEDULE.copy()
