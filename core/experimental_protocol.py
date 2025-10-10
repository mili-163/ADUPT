import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import random
from dataclasses import dataclass


@dataclass
class MissingnessConfig:
    """缺失模式配置"""
    regime: str  # "asymmetric", "symmetric", "uncertain"
    missing_rate: float  # ξ ∈ [0, 1]
    affected_modalities: List[int]  # 受影响的模态索引
    train_config: Optional[Dict] = None  # 训练时配置
    test_config: Optional[Dict] = None  # 测试时配置


class MissingnessGenerator:
    """
    缺失模式生成器
    
    三种缺失模式：
    1. Asymmetric: 缺失集中在单个模态
    2. Symmetric: 缺失均匀分布在模态子集
    3. Uncertain: 训练和测试使用不同缺失配置
    """
    
    def __init__(self, num_modalities: int = 3):
        self.num_modalities = num_modalities
        self.modality_names = ["text", "image", "audio"]
    
    def generate_asymmetric_missingness(self, 
                                      missing_rate: float, 
                                      target_modality: int,
                                      num_samples: int) -> torch.Tensor:
        """
        生成非对称缺失模式
        
        缺失集中在单个模态k，其他模态完全观察
        
        Args:
            missing_rate: 缺失率 ξ ∈ [0, 1]
            target_modality: 目标模态索引 k
            num_samples: 样本数量
            
        Returns:
            torch.Tensor: 缺失掩码 (num_samples, num_modalities)
        """
        masks = torch.ones(num_samples, self.num_modalities)
        
        # 选择ξ%的样本，将目标模态设为0
        num_missing = int(missing_rate * num_samples)
        missing_indices = random.sample(range(num_samples), num_missing)
        
        masks[missing_indices, target_modality] = 0
        
        return masks
    
    def generate_symmetric_missingness(self, 
                                     missing_rate: float,
                                     affected_modalities: List[int],
                                     num_samples: int) -> torch.Tensor:
        """
        生成对称缺失模式
        
        缺失均匀分布在模态子集M，每个模态缺失ξ/|M|%的样本
        
        Args:
            missing_rate: 缺失率 ξ ∈ [0, 1]
            affected_modalities: 受影响的模态子集 M
            num_samples: 样本数量
            
        Returns:
            torch.Tensor: 缺失掩码 (num_samples, num_modalities)
        """
        masks = torch.ones(num_samples, self.num_modalities)
        
        # 每个受影响的模态缺失ξ/|M|%的样本
        per_modality_rate = missing_rate / len(affected_modalities)
        per_modality_missing = int(per_modality_rate * num_samples)
        
        for modality in affected_modalities:
            # 为每个模态选择缺失样本（不重叠）
            available_indices = torch.where(masks[:, modality] == 1)[0].tolist()
            if len(available_indices) >= per_modality_missing:
                missing_indices = random.sample(available_indices, per_modality_missing)
                masks[missing_indices, modality] = 0
        
        return masks
    
    def generate_uncertain_missingness(self, 
                                     missing_rate: float,
                                     train_config: Dict,
                                     test_config: Dict,
                                     num_train_samples: int,
                                     num_test_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成不确定缺失模式
        
        训练和测试使用不同的缺失配置
        
        Args:
            missing_rate: 全局缺失率 ξ
            train_config: 训练时配置
            test_config: 测试时配置
            num_train_samples: 训练样本数量
            num_test_samples: 测试样本数量
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (训练掩码, 测试掩码)
        """
        # 生成训练掩码
        if train_config["regime"] == "asymmetric":
            train_masks = self.generate_asymmetric_missingness(
                missing_rate, train_config["target_modality"], num_train_samples
            )
        elif train_config["regime"] == "symmetric":
            train_masks = self.generate_symmetric_missingness(
                missing_rate, train_config["affected_modalities"], num_train_samples
            )
        else:
            raise ValueError(f"Unknown train regime: {train_config['regime']}")
        
        # 生成测试掩码
        if test_config["regime"] == "asymmetric":
            test_masks = self.generate_asymmetric_missingness(
                missing_rate, test_config["target_modality"], num_test_samples
            )
        elif test_config["regime"] == "symmetric":
            test_masks = self.generate_symmetric_missingness(
                missing_rate, test_config["affected_modalities"], num_test_samples
            )
        else:
            raise ValueError(f"Unknown test regime: {test_config['regime']}")
        
        return train_masks, test_masks
    
    def generate_missingness(self, config: MissingnessConfig, 
                           num_samples: int) -> torch.Tensor:
        """
        根据配置生成缺失模式
        
        Args:
            config: 缺失模式配置
            num_samples: 样本数量
            
        Returns:
            torch.Tensor: 缺失掩码
        """
        if config.regime == "asymmetric":
            return self.generate_asymmetric_missingness(
                config.missing_rate, config.affected_modalities[0], num_samples
            )
        elif config.regime == "symmetric":
            return self.generate_symmetric_missingness(
                config.missing_rate, config.affected_modalities, num_samples
            )
        elif config.regime == "uncertain":
            # 不确定模式需要分别处理训练和测试
            raise ValueError("Uncertain regime requires separate train/test generation")
        else:
            raise ValueError(f"Unknown regime: {config.regime}")


class TeacherStudentSchedule:
    """
    教师-学生训练计划
    
    教师-学生训练策略
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.teacher_epochs = config["teacher_epochs"]
        self.student_epochs = config["student_epochs"]
        
        # 教师统计量缓存
        self.teacher_statistics = {
            "feature_means": None,
            "feature_variances": None,
            "class_prototypes": None
        }
    
    def get_teacher_schedule(self) -> Dict:
        """
        获取教师训练计划
        
        Returns:
            Dict: 教师训练配置
        """
        return {
            "data": "complete_subset_only",
            "epochs": self.teacher_epochs,
            "cache_statistics": True,
            "statistics_to_cache": [
                "feature_means",
                "feature_variances",
                "class_prototypes"
            ]
        }
    
    def get_student_schedule(self) -> Dict:
        """
        获取学生训练计划
        
        Returns:
            Dict: 学生训练配置
        """
        return {
            "data": "incomplete_and_complete",
            "epochs": self.student_epochs,
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
    
    def cache_teacher_statistics(self, features: torch.Tensor, 
                               labels: torch.Tensor = None):
        """
        缓存教师统计量
        
        Args:
            features: 教师特征 (B, d)
            labels: 标签 (B,) - 可选，用于类别原型
        """
        # 特征均值和方差
        self.teacher_statistics["feature_means"] = features.mean(dim=0)
        self.teacher_statistics["feature_variances"] = features.var(dim=0)
        
        # 类别原型（如果提供标签）
        if labels is not None:
            unique_labels = torch.unique(labels)
            class_prototypes = {}
            for label in unique_labels:
                label_mask = labels == label
                class_features = features[label_mask]
                class_prototypes[label.item()] = class_features.mean(dim=0)
            self.teacher_statistics["class_prototypes"] = class_prototypes
    
    def get_cached_statistics(self) -> Dict:
        """
        获取缓存的教师统计量
        
        Returns:
            Dict: 缓存的统计量
        """
        return self.teacher_statistics


class ExperimentalProtocol:
    """
    实验协议管理器
    
    统一管理实验配置、缺失模式生成和训练计划
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.num_modalities = config.get("num_modalities", 3)
        
        # 初始化组件
        self.missingness_generator = MissingnessGenerator(self.num_modalities)
        self.teacher_student_schedule = TeacherStudentSchedule(config["training"])
        
        # 实验配置
        self.hardware_config = config["hardware"]
        self.repetitions = self.hardware_config["repetitions"]
    
    def create_missingness_config(self, regime: str, missing_rate: float, 
                                 affected_modalities: List[int],
                                 train_config: Dict = None,
                                 test_config: Dict = None) -> MissingnessConfig:
        """
        创建缺失模式配置
        
        Args:
            regime: 缺失模式
            missing_rate: 缺失率
            affected_modalities: 受影响的模态
            train_config: 训练配置（用于uncertain模式）
            test_config: 测试配置（用于uncertain模式）
            
        Returns:
            MissingnessConfig: 缺失模式配置
        """
        return MissingnessConfig(
            regime=regime,
            missing_rate=missing_rate,
            affected_modalities=affected_modalities,
            train_config=train_config,
            test_config=test_config
        )
    
    def generate_experimental_data(self, 
                                 train_data: Dict,
                                 test_data: Dict,
                                 missingness_config: MissingnessConfig) -> Dict:
        """
        生成实验数据
        
        Args:
            train_data: 训练数据
            test_data: 测试数据
            missingness_config: 缺失模式配置
            
        Returns:
            Dict: 包含缺失掩码的实验数据
        """
        num_train = len(train_data["labels"])
        num_test = len(test_data["labels"])
        
        if missingness_config.regime == "uncertain":
            # 不确定模式：训练和测试使用不同配置
            train_masks, test_masks = self.missingness_generator.generate_uncertain_missingness(
                missingness_config.missing_rate,
                missingness_config.train_config,
                missingness_config.test_config,
                num_train,
                num_test
            )
        else:
            # 对称或非对称模式
            train_masks = self.missingness_generator.generate_missingness(
                missingness_config, num_train
            )
            test_masks = self.missingness_generator.generate_missingness(
                missingness_config, num_test
            )
        
        return {
            "train_data": {**train_data, "missing_masks": train_masks},
            "test_data": {**test_data, "missing_masks": test_masks},
            "missingness_config": missingness_config
        }
    
    def run_experiment(self, model, experimental_data: Dict) -> Dict:
        """
        运行实验
        
        Args:
            model: 模型
            experimental_data: 实验数据
            
        Returns:
            Dict: 实验结果
        """
        results = []
        
        # 重复实验
        for rep in range(self.repetitions):
            print(f"Running experiment repetition {rep + 1}/{self.repetitions}")
            
            # 教师训练
            teacher_schedule = self.teacher_student_schedule.get_teacher_schedule()
            teacher_results = self._run_teacher_training(
                model, experimental_data["train_data"], teacher_schedule
            )
            
            # 学生训练
            student_schedule = self.teacher_student_schedule.get_student_schedule()
            student_results = self._run_student_training(
                model, experimental_data, student_schedule
            )
            
            # 测试
            test_results = self._run_testing(
                model, experimental_data["test_data"]
            )
            
            results.append({
                "repetition": rep + 1,
                "teacher_results": teacher_results,
                "student_results": student_results,
                "test_results": test_results
            })
        
        # 计算均值和标准差
        return self._compute_statistics(results)
    
    def _run_teacher_training(self, model, train_data: Dict, schedule: Dict) -> Dict:
        """运行教师训练"""
        # 实现教师训练逻辑
        pass
    
    def _run_student_training(self, model, experimental_data: Dict, schedule: Dict) -> Dict:
        """运行学生训练"""
        # 实现学生训练逻辑
        pass
    
    def _run_testing(self, model, test_data: Dict) -> Dict:
        """运行测试"""
        # 实现测试逻辑
        pass
    
    def _compute_statistics(self, results: List[Dict]) -> Dict:
        """计算统计量"""
        # 计算均值和标准差
        pass
