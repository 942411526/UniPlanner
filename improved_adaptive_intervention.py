import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import random


class ImprovedAdaptiveLayerIntervention(nn.Module):
    """
    改进的自适应单层干预模块
    针对CR环境优化，同时保持NCR环境的良好表现
    """
    
    def __init__(
        self,
        dim: int = 128,
        layer_idx: int = 0,
        total_layers: int = 4,
        base_intervention_prob: float = 0.3,
        base_noise_scale: float = 0.1,
        # 新增参数
        adaptive_threshold: bool = True,
        min_intervention_prob: float = 0.1,
        max_intervention_prob: float = 0.5,
    ):
        super().__init__()
        
        self.dim = dim
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.adaptive_threshold = adaptive_threshold
        self.min_intervention_prob = min_intervention_prob
        self.max_intervention_prob = max_intervention_prob
        
        # 计算该层的干预强度 - 改进策略
        depth_ratio = (total_layers - layer_idx) / total_layers
        
        # 使用register_buffer存储tensor版本
        self.register_buffer('_depth_ratio', torch.tensor(depth_ratio, dtype=torch.float32))
        self.register_buffer('_intervention_prob', torch.tensor(base_intervention_prob * depth_ratio, dtype=torch.float32))
        self.register_buffer('_noise_scale', torch.tensor(base_noise_scale * depth_ratio, dtype=torch.float32))
        
        # Python float版本
        self.intervention_prob_value = float(base_intervention_prob * depth_ratio)
        self.noise_scale_value = float(base_noise_scale * depth_ratio)
        self.depth_ratio_value = float(depth_ratio)
        
        # 统计跟踪
        self.register_buffer('feature_mean', torch.zeros(dim))
        self.register_buffer('feature_var', torch.ones(dim))
        self.register_buffer('update_count', torch.tensor(0))
        
        # 可学习的层级特定参数
        self.layer_scale = nn.Parameter(torch.ones(1) * depth_ratio)
        
        # 新增：自适应异常检测
        if adaptive_threshold:
            self.anomaly_threshold = nn.Parameter(torch.tensor(2.0))  # 可学习的阈值
            self.strong_anomaly_threshold = nn.Parameter(torch.tensor(0.3))
        else:
            self.register_buffer('anomaly_threshold', torch.tensor(2.0))
            self.register_buffer('strong_anomaly_threshold', torch.tensor(0.3))
        
        # 新增：复杂度检测器
        self.complexity_detector = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 新增：干预强度调节器
        self.intensity_adapter = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        改进的前向传播
        """
        
        if not self.training:
            return x, None
        
        batch_size, seq_len, dim = x.shape
        
        # 更新running statistics
        if self.training:
            with torch.no_grad():
                valid_x = x
                if key_padding_mask is not None:
                    valid_mask = ~key_padding_mask.unsqueeze(-1)
                    valid_x = x * valid_mask
                    
                batch_mean = valid_x.mean(dim=(0, 1))
                batch_var = valid_x.var(dim=(0, 1))
                
                momentum = 0.1
                self.feature_mean = (1 - momentum) * self.feature_mean + momentum * batch_mean
                self.feature_var = (1 - momentum) * self.feature_var + momentum * batch_var
                self.update_count += 1
        
        # 检测特征复杂度
        with torch.no_grad():
            feature_complexity = self.complexity_detector(x.mean(dim=1))  # [B, 1]
            avg_complexity = feature_complexity.mean().item()
        
        # 自适应调整干预概率
        adaptive_prob = self.intervention_prob_value
        if avg_complexity > 0.7:  # 高复杂度环境（类似CR）
            adaptive_prob = min(self.max_intervention_prob, adaptive_prob * 1.5)
        elif avg_complexity < 0.3:  # 低复杂度环境（类似NCR）
            adaptive_prob = max(self.min_intervention_prob, adaptive_prob * 0.7)
        
        # 使用自适应概率
        if random.random() > adaptive_prob:
            return x, {'anomaly_ratio': 0.0, 'complexity': avg_complexity, 'adaptive_prob': adaptive_prob}
        
        # 改进的异常检测
        with torch.no_grad():
            z_scores = (x - self.feature_mean.unsqueeze(0).unsqueeze(0)) / \
                      (self.feature_var.sqrt().unsqueeze(0).unsqueeze(0) + 1e-6)
            
            # 使用自适应阈值
            if self.adaptive_threshold:
                threshold = self.anomaly_threshold.item()
                strong_threshold = self.strong_anomaly_threshold.item()
            else:
                threshold = 2.0
                strong_threshold = 0.3
            
            is_anomaly = z_scores.abs() > threshold
            anomaly_score = is_anomaly.float().mean(dim=-1)
            strong_anomaly = anomaly_score > strong_threshold
        
        # 改进的干预策略
        if strong_anomaly.any():
            # 计算自适应干预强度
            intensity_factor = self.intensity_adapter(x.mean(dim=1))  # [B, 1]
            avg_intensity = intensity_factor.mean().item()
            
            # 根据复杂度和强度调整噪声
            base_noise = self.noise_scale_value
            if avg_complexity > 0.7:  # 高复杂度环境需要更强干预
                noise_scale = base_noise * (1.0 + avg_intensity)
            else:  # 低复杂度环境使用温和干预
                noise_scale = base_noise * (0.5 + avg_intensity * 0.5)
            
            noise = torch.randn_like(x) * noise_scale
            
            # 只在异常位置添加噪声
            intervention_mask = strong_anomaly.unsqueeze(-1).float()
            
            # 应用层级特定的缩放
            x = x + noise * intervention_mask * self.layer_scale
            
            return x, {
                'anomaly_ratio': anomaly_score.mean().item(),
                'complexity': avg_complexity,
                'adaptive_prob': adaptive_prob,
                'intensity_factor': avg_intensity,
                'noise_scale_used': noise_scale,
                'layer_idx': self.layer_idx
            }
        
        return x, {
            'anomaly_ratio': anomaly_score.mean().item(),
            'complexity': avg_complexity,
            'adaptive_prob': adaptive_prob,
            'layer_idx': self.layer_idx
        }


class ImprovedMultiLevelCausalIntervention(nn.Module):
    """
    改进的多层级因果干预框架
    针对CR环境优化，同时保持NCR环境的良好表现
    """
    
    def __init__(
        self,
        dim: int = 128,
        encoder_depth: int = 4,
        base_intervention_prob: float = 0.3,
        base_noise_scale: float = 0.1,
        enable_layer_interventions: Optional[list] = None,
        # 新增参数
        adaptive_threshold: bool = True,
        min_intervention_prob: float = 0.1,
        max_intervention_prob: float = 0.5,
    ):
        super().__init__()
        
        self.dim = dim
        self.encoder_depth = encoder_depth
        self.base_intervention_prob = base_intervention_prob
        self.base_noise_scale = base_noise_scale
        self.adaptive_threshold = adaptive_threshold
        self.min_intervention_prob = min_intervention_prob
        self.max_intervention_prob = max_intervention_prob
        
        if enable_layer_interventions is None:
            enable_layer_interventions = [True] * (encoder_depth - 1) + [False]
        
        self.enable_layer_interventions = enable_layer_interventions
        
        # 为每层创建改进的干预模块
        self.layer_interventions = nn.ModuleList()
        self.layer_configs = []
        
        for i in range(encoder_depth):
            if enable_layer_interventions[i]:
                depth_ratio = (encoder_depth - i) / encoder_depth
                layer_prob = base_intervention_prob * depth_ratio
                layer_noise = base_noise_scale * depth_ratio
                
                self.layer_configs.append({
                    'layer_idx': i,
                    'depth_ratio': float(depth_ratio),
                    'intervention_prob': float(layer_prob),
                    'noise_scale': float(layer_noise),
                    'enabled': True,
                })
                
                # 创建改进的干预模块
                intervention = ImprovedAdaptiveLayerIntervention(
                    dim=dim,
                    layer_idx=i,
                    total_layers=encoder_depth,
                    base_intervention_prob=base_intervention_prob,
                    base_noise_scale=base_noise_scale,
                    adaptive_threshold=adaptive_threshold,
                    min_intervention_prob=min_intervention_prob,
                    max_intervention_prob=max_intervention_prob,
                )
                self.layer_interventions.append(intervention)
            else:
                self.layer_configs.append({
                    'layer_idx': i,
                    'depth_ratio': 0.0,
                    'intervention_prob': 0.0,
                    'noise_scale': 0.0,
                    'enabled': False,
                })
                self.layer_interventions.append(None)
    
    def apply_intervention_after_layer(
        self,
        x: torch.Tensor,
        layer_idx: int,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        在指定的encoder层后应用干预
        """
        
        if layer_idx >= len(self.layer_interventions):
            return x, None
        
        intervention_module = self.layer_interventions[layer_idx]
        
        if intervention_module is None:
            return x, None
        
        x, metrics = intervention_module(x, key_padding_mask)
        
        return x, metrics
    
    def get_all_metrics(self) -> Dict:
        """
        获取所有层的干预统计信息
        """
        metrics = {}
        
        for config in self.layer_configs:
            layer_idx = config['layer_idx']
            prefix = f'layer_{layer_idx}'
            
            metrics[f'{prefix}_prob'] = config['intervention_prob']
            metrics[f'{prefix}_noise_scale'] = config['noise_scale']
            
            if config['enabled'] and layer_idx < len(self.layer_interventions):
                module = self.layer_interventions[layer_idx]
                if module is not None and hasattr(module, 'layer_scale'):
                    try:
                        scale_val = float(module.layer_scale.item())
                        metrics[f'{prefix}_scale'] = scale_val
                    except Exception as e:
                        print(f"Warning: Failed to get scale for layer {layer_idx}: {e}")
                        metrics[f'{prefix}_scale'] = config['depth_ratio']
                else:
                    metrics[f'{prefix}_scale'] = 0.0
            else:
                metrics[f'{prefix}_scale'] = 0.0
        
        return metrics


# 使用示例
def create_improved_intervention_for_cr():
    """
    创建针对CR环境优化的干预框架
    """
    return ImprovedMultiLevelCausalIntervention(
        dim=128,
        encoder_depth=4,
        base_intervention_prob=0.35,  # 提高基础概率
        base_noise_scale=0.12,        # 提高基础噪声
        adaptive_threshold=True,       # 启用自适应阈值
        min_intervention_prob=0.15,    # 提高最小概率
        max_intervention_prob=0.6,     # 提高最大概率
    )

def create_balanced_intervention():
    """
    创建平衡的干预框架，在CR和NCR环境都能表现良好
    """
    return ImprovedMultiLevelCausalIntervention(
        dim=128,
        encoder_depth=4,
        base_intervention_prob=0.3,
        base_noise_scale=0.1,
        adaptive_threshold=True,
        min_intervention_prob=0.1,
        max_intervention_prob=0.5,
    )