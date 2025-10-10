"""
GPU接口模块
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings


class DeviceManager:
    """
    设备管理器
    
    统一管理GPU/CPU设备，提供自动设备检测和切换功能
    """
    
    def __init__(self, device: Optional[str] = None, auto_detect: bool = True):
        """
        初始化设备管理器
        
        Args:
            device: 指定设备 ("cuda", "cpu", "mps"等)
            auto_detect: 是否自动检测最佳设备
        """
        self.device = self._setup_device(device, auto_detect)
        self.device_type = self.device.type
        self.device_index = self.device.index if self.device.index is not None else 0
        
        print(f"设备管理器初始化: {self.device}")
    
    def _setup_device(self, device: Optional[str], auto_detect: bool) -> torch.device:
        
        if device is not None:
            # 用户指定设备
            try:
                return torch.device(device)
            except Exception as e:
                warnings.warn(f"无法使用指定设备 {device}: {e}")
                return self._auto_detect_device()
        
        if auto_detect:
            return self._auto_detect_device()
        else:
            return torch.device("cpu")
    
    def _auto_detect_device(self) -> torch.device:
        
        # 优先级：CUDA > MPS > CPU
        
        # 检查CUDA
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"检测到 {device_count} 个CUDA设备")
            for i in range(device_count):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return torch.device("cuda:0")
        
        # 检查MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("检测到Apple Silicon MPS设备")
            return torch.device("mps")
        
        # 默认CPU
        print("使用CPU设备")
        return torch.device("cpu")
    
    def get_device(self) -> torch.device:
        
        return self.device
    
    def to_device(self, data: Any) -> Any:
        
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.to_device(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self.to_device(item) for item in data)
        else:
            return data
    
    def to_cpu(self, data: Any) -> Any:
        
        if isinstance(data, torch.Tensor):
            return data.cpu()
        elif isinstance(data, dict):
            return {k: self.to_cpu(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.to_cpu(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self.to_cpu(item) for item in data)
        else:
            return data
    
    def get_device_info(self) -> Dict[str, Any]:
        
        info = {
            "device": str(self.device),
            "device_type": self.device_type,
            "device_index": self.device_index
        }
        
        if self.device_type == "cuda":
            info.update({
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(self.device_index),
                "cuda_memory_allocated": torch.cuda.memory_allocated(self.device_index),
                "cuda_memory_reserved": torch.cuda.memory_reserved(self.device_index)
            })
        elif self.device_type == "mps":
            info.update({
                "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            })
        
        return info
    
    def print_device_info(self):
        
        info = self.get_device_info()
        print("=" * 50)
        print("设备信息")
        print("=" * 50)
        for key, value in info.items():
            print(f"{key}: {value}")
        print("=" * 50)


class GPUModelWrapper(nn.Module):
    """
    GPU模型包装器
    
    自动处理模型和数据的GPU/CPU转换
    """
    
    def __init__(self, model: nn.Module, device_manager: DeviceManager):
        super().__init__()
        self.model = model
        self.device_manager = device_manager
        
        # 将模型移动到指定设备
        self.model = self.model.to(self.device_manager.get_device())
    
    def forward(self, *args, **kwargs):
        
        # 将输入数据移动到模型设备
        args = self.device_manager.to_device(args)
        kwargs = self.device_manager.to_device(kwargs)
        
        # 执行前向传播
        return self.model(*args, **kwargs)
    
    def train(self, mode: bool = True):
        
        self.model.train(mode)
        return self
    
    def eval(self):
        
        self.model.eval()
        return self
    
    def parameters(self):
        
        return self.model.parameters()
    
    def named_parameters(self):
        
        return self.model.named_parameters()
    
    def state_dict(self):
        
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        
        return self.model.load_state_dict(state_dict)


class GPUDataLoader:
    """
    GPU数据加载器
    
    自动处理批次数据的GPU转换
    """
    
    def __init__(self, dataloader, device_manager: DeviceManager):
        self.dataloader = dataloader
        self.device_manager = device_manager
    
    def __iter__(self):
        
        for batch in self.dataloader:
            yield self.device_manager.to_device(batch)
    
    def __len__(self):
        
        return len(self.dataloader)
    
    def __getitem__(self, idx):
        
        item = self.dataloader[idx]
        return self.device_manager.to_device(item)


def create_device_manager(device: Optional[str] = None, auto_detect: bool = True) -> DeviceManager:
    """
    创建设备管理器
    
    Args:
        device: 指定设备
        auto_detect: 是否自动检测
        
    Returns:
        DeviceManager: 设备管理器实例
    """
    return DeviceManager(device, auto_detect)


def wrap_model_for_gpu(model: nn.Module, device_manager: DeviceManager) -> GPUModelWrapper:
    """
    为GPU包装模型
    
    Args:
        model: 要包装的模型
        device_manager: 设备管理器
        
    Returns:
        GPUModelWrapper: GPU包装的模型
    """
    return GPUModelWrapper(model, device_manager)


def wrap_dataloader_for_gpu(dataloader, device_manager: DeviceManager) -> GPUDataLoader:
    """
    为GPU包装数据加载器
    
    Args:
        dataloader: 要包装的数据加载器
        device_manager: 设备管理器
        
    Returns:
        GPUDataLoader: GPU包装的数据加载器
    """
    return GPUDataLoader(dataloader, device_manager)


def get_optimal_batch_size(model: nn.Module, device_manager: DeviceManager, 
                          input_shape: Tuple[int, ...], max_batch_size: int = 64) -> int:
    """
    获取最优批次大小
    
    Args:
        model: 模型
        device_manager: 设备管理器
        input_shape: 输入形状
        max_batch_size: 最大批次大小
        
    Returns:
        int: 最优批次大小
    """
    device = device_manager.get_device()
    model = model.to(device)
    
    # 二分搜索最优批次大小
    left, right = 1, max_batch_size
    optimal_batch_size = 1
    
    while left <= right:
        mid = (left + right) // 2
        try:
            # 创建测试输入
            test_input = torch.randn(mid, *input_shape).to(device)
            
            # 测试前向传播
            with torch.no_grad():
                _ = model(test_input)
            
            optimal_batch_size = mid
            left = mid + 1
        except RuntimeError as e:
            if "out of memory" in str(e):
                right = mid - 1
                # 清理GPU内存
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise e
    
    return optimal_batch_size


def clear_gpu_memory(device_manager: DeviceManager):
    
    if device_manager.device_type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device_manager.device_type == "mps":
        # MPS没有显式的内存清理方法
        pass


def benchmark_device_performance(model: nn.Module, device_manager: DeviceManager,
                               input_shape: Tuple[int, ...], num_runs: int = 10) -> Dict[str, float]:
    """
    基准测试设备性能
    
    Args:
        model: 模型
        device_manager: 设备管理器
        input_shape: 输入形状
        num_runs: 运行次数
        
    Returns:
        Dict[str, float]: 性能指标
    """
    device = device_manager.get_device()
    model = model.to(device)
    model.eval()
    
    # 创建测试输入
    test_input = torch.randn(1, *input_shape).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(3):
            _ = model(test_input)
    
    # 基准测试
    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        
        if device.type == "cuda":
            start_time.record()
        
        with torch.no_grad():
            _ = model(test_input)
        
        if device.type == "cuda":
            end_time.record()
            torch.cuda.synchronize()
            times.append(start_time.elapsed_time(end_time) / 1000.0)  # 转换为秒
        else:
            import time
            start = time.time()
            _ = model(test_input)
            times.append(time.time() - start)
    
    return {
        "mean_time": sum(times) / len(times),
        "std_time": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5,
        "min_time": min(times),
        "max_time": max(times)
    }

