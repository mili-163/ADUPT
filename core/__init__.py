"""
Core modules for missing modality prompt learning
"""

from .method_model_gpu import MethodModelGPU
from .incomplete_multimodal_learning import IncompleteMultimodalLearning
from .modality_prompts_v2 import AbsenceAwareModalityPrompts, ModalityPromptInjector
from .dual_prompts_v3 import DualInstancePromptsV3, ContentAggregator
from .hierarchical_distillation import HierarchicalDistillation, TeacherModel, InferenceModel
from .hard_negative_contrastive import SpecializationContrastiveRegularizer
from .calibrated_prompt_adaptation import CalibratedPromptAdaptation
from .training_objective import TrainingObjective, ParameterEfficiencyManager
from .modality_processors import ModalityProcessorManager
from .experimental_protocol import ExperimentalProtocol, MissingnessGenerator
from .gpu_interface import DeviceManager, GPUModelWrapper, create_device_manager

__all__ = [
    'MethodModelGPU',
    'IncompleteMultimodalLearning',
    'AbsenceAwareModalityPrompts',
    'ModalityPromptInjector',
    'DualInstancePromptsV3',
    'ContentAggregator',
    'HierarchicalDistillation',
    'TeacherModel',
    'InferenceModel',
    'SpecializationContrastiveRegularizer',
    'CalibratedPromptAdaptation',
    'TrainingObjective',
    'ParameterEfficiencyManager',
    'ModalityProcessorManager',
    'ExperimentalProtocol',
    'MissingnessGenerator',
    'DeviceManager',
    'GPUModelWrapper',
    'create_device_manager'
]


