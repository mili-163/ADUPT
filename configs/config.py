import torch


BASE_CONFIG = {
    "num_modalities": 3,
    "modality_names": ["text", "image", "audio"],
    "mask_dim": 3,
    "complete_subset_ratio": 0.1,
    "incomplete_subset_ratio": 0.9,
    
    # 基础模型配置
    "hidden_size": 768,
    "num_layers": 12,
    "num_heads": 12,
    "mlp_ratio": 4,
    "drop_rate": 0.1,
    "vocab_size": 30522,
    "max_text_len": 40,
    "max_image_len": 197,
    "audio_feature_dim": 74,      # COVAREP feature dimension
    "audio_token_length": 100,    # Audio sequence length
    "num_classes": 2,
    "task_type": "classification",
    
    "backbone_config": {
        "text_encoder": {
            "type": "bert",
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "vocab_size": 30522,
            "max_text_len": 40,
            "frozen": True
        },
        "image_encoder": {
            "type": "vit",
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "image_size": 224,
            "patch_size": 16,
            "max_image_len": 197,
            "frozen": True
        },
        "audio_encoder": {
            "type": "wav2vec2",
            "hidden_size": 768,
            "audio_feature_dim": 74,      # COVAREP feature dimension
            "audio_token_length": 100,    # Audio sequence length
            "frozen": True  # 冻结 f_audio
        }
    },
    
    
    "fusion_config": {
        "type": "vilt",
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "frozen": True  # 冻结 F
    },
    
    
    "task_head_config": {
        "hidden_size": 768,
        "num_classes": 2,  # 根据具体任务调整
        "task_type": "classification"
    },
    
    # 设备设置
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


PROMPT_CONFIG = {
    # 教师网络T设置 - 在D_c上训练
    "teacher_config": {
        "hidden_size": 768,
        "num_classes": 2,
        "training_data": "D_c",  # 只在完整数据上训练
        "frozen_backbones": True
    },
    
    # 推理网络I设置 - 在D_s∪D_c上训练
    "inference_config": {
        "hidden_size": 768,
        "num_classes": 2,
        "training_data": "D_s_union_D_c",  # 在完整+不完整数据上训练
        "frozen_backbones": True
    },
    
    # 两种提示族设置
    "prompt_families": {
        # 模态级提示 P^mod(m) - 缺失感知上下文
        "modality_prompts": {
            "hidden_size": 768,
            "num_modalities": 3,
            "prompt_length": 4,
            "num_layers": 4,
            "presence_embedding_dim": 768,
            "absence_embedding_dim": 768
        },
        
        # 实例级双提示 P^sp, P^gn - 专业化和泛化
        "dual_instance_prompts": {
            "hidden_size": 768,
            "num_modalities": 3,
            "prompt_length": 4,
            "num_layers": 4,
            "specialization_branch": True,
            "generalization_branch": True,
            "mask_conditioned_gate": True
        }
    },
    
    # 分层蒸馏设置
    "hierarchical_distillation": {
        "hidden_size": 768,
        "selected_layers": [4, 8, 12],  # 选择进行特征对齐的层
        "distillation_temperature": 4.0,
        "feature_alignment": True,
        "prediction_alignment": True,
        "mask_adapter": True
    },
    
    # 硬负样本对比设置
    "hard_negative_contrastive": {
        "hidden_size": 768,
        "num_classes": 2,
        "contrastive_temperature": 0.07,
        "top_k_negatives": 5,
        "apply_to_specialization_only": True  # 仅应用于专业化分支
    },
    
    # 校准提示适应设置
    "calibrated_prompt_adaptation": {
        "hidden_size": 768,
        "adaptation_lr": 0.01,
        "entropy_weight": 1.0,
        "alignment_weight": 0.1,
        "num_augmentations": 3,
        "teacher_statistics": True
    }
}

# 训练设置
OPTIMIZER_CONFIG = {
    # 三阶段训练
    "stages": {
        "stage1_teacher_training": {
            "description": "教师T在D_c上训练",
            "data": "D_c",
            "epochs": 50,
            "learning_rate": 1e-4,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR"
        },
        "stage2_inference_training": {
            "description": "推理网络I在D_s∪D_c上训练",
            "data": "D_s_union_D_c",
            "epochs": 100,
            "learning_rate": 1e-4,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "teacher_frozen": True
        },
        "stage3_runtime_calibration": {
            "description": "运行时校准提示适应",
            "data": "test_data",
            "use_cpa": True,
            "single_step": True
        }
    },
    
    # 损失权重设置
    "loss_weights": {
        "lambda_task": 1.0,        # 任务损失
        "lambda_mask_recon": 0.1,  # 掩码重构损失
        "lambda_feat": 1.0,        # 特征蒸馏损失
        "lambda_pred": 1.0,        # 预测蒸馏损失
        "lambda_contrastive": 0.5  # 硬负样本对比损失
    }
}

# 数据集特定设置
DATASET_CONFIG = {
    "hatememes": {
        "num_classes": 2,
        "class_names": ["normal", "hateful"],
        "modalities": ["text", "image"],
        "missing_patterns": [0, 1, 2],  # 0: 完整, 1: 缺失文本, 2: 缺失图像
        "text_max_length": 40,
        "image_size": (224, 224)
    },
    "food101": {
        "num_classes": 101,
        "class_names": None,  # 动态加载
        "modalities": ["text", "image"],
        "missing_patterns": [0, 1, 2],
        "text_max_length": 40,
        "image_size": (224, 224)
    },
    "mmimdb": {
        "num_classes": 23,
        "class_names": None,  # 动态加载
        "modalities": ["text", "image"],
        "missing_patterns": [0, 1, 2],
        "text_max_length": 40,
        "image_size": (224, 224)
    },
    "cmu-mosei": {
        "num_classes": 4,
        "class_names": ["negative", "neutral", "positive", "very positive"],
        "modalities": ["text", "image", "audio"],
        "missing_patterns": [0, 1, 2, 3, 4, 5, 6, 7],
        "text_max_length": 40,
        "image_size": (224, 224),
        "audio_length": 100
    }
}

# 评估设置
EVALUATION_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1", "auc"],
    "missing_patterns": {
        "seen_during_training": True,
        "unseen_during_training": True  # 测试未见过的缺失模式
    },
    "use_cpa": True,
    "cpa_temperature": 0.07
}

# 完整配置生成函数
def get_config(dataset_name: str = "hatememes") -> dict:
    config = BASE_CONFIG.copy()
    config.update(PROMPT_CONFIG)
    config.update(OPTIMIZER_CONFIG)
    
    # 添加数据集特定配置
    if dataset_name in DATASET_CONFIG:
        dataset_config = DATASET_CONFIG[dataset_name]
        config.update(dataset_config)
        
        # 更新模态设置
        config["num_modalities"] = len(dataset_config["modalities"])
        config["modality_names"] = dataset_config["modalities"]
        config["mask_dim"] = len(dataset_config["modalities"])
        
        # 更新任务头配置
        config["task_head_config"]["num_classes"] = dataset_config["num_classes"]
    
    # 添加评估配置
    config.update(EVALUATION_CONFIG)
    
    return config

# 验证配置函数
def validate_config(config: dict) -> bool:
    """
    验证配置是否符合论文描述
    
    Args:
        config: 配置字典
        
    Returns:
        bool: 是否有效
    """
    required_keys = [
        "num_modalities", "modality_names", "mask_dim",
        "backbone_config", "fusion_config", "task_head_config",
        "teacher_config", "inference_config", "prompt_families"
    ]
    
    for key in required_keys:
        if key not in config:
            print(f"Missing required key: {key}")
            return False
    
    # 验证模态数量
    if config["num_modalities"] != len(config["modality_names"]):
        print("num_modalities must equal length of modality_names")
        return False
    
    # 验证掩码维度
    if config["mask_dim"] != config["num_modalities"]:
        print("mask_dim must equal num_modalities")
        return False
    
    return True

# 示例使用
if __name__ == "__main__":
    # 获取HatefulMemes配置
    config = get_config("hatememes")
    
    # 验证配置
    if validate_config(config):
        print("配置验证通过！")
        print(f"模态数量: {config['num_modalities']}")
        print(f"模态名称: {config['modality_names']}")
        print(f"类别数量: {config['task_head_config']['num_classes']}")
    else:
        print("配置验证失败！")
