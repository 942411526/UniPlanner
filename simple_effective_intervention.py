import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import random


class SimpleEffectiveAdaptiveLayerIntervention(nn.Module):
    """
    简单有效的自适应单层干预模块
    基于特征方差和异常程度进行自适应调整
    """
    
    def __init__(
        self,
        dim: int = 128,
        layer_idx: int = 0,
        total_layers: int = 4,
        base_intervention_prob: float = 0.3,
        base_noise_scale: float = 0.1,
    ):
        super().__init__()
        
        self.dim = dim
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        
        # 计算该层的干预强度
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
        
        # 自适应异常检测阈值
        self.anomaly_threshold = nn.Parameter(torch.tensor(2.0))
        self.strong_anomaly_threshold = nn.Parameter(torch.tensor(0.3))
        
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        简单有效的自适应前向传播
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
        
        # 计算特征复杂度（基于方差）
        with torch.no_grad():
            feature_variance = x.var(dim=1).mean(dim=1)  # [B] 每个样本的特征方差
            avg_variance = feature_variance.mean().item()
            
            # 基于方差判断环境复杂度
            is_complex_env = avg_variance > 1.0  # 高方差表示复杂环境（类似CR）
        
        # 根据环境复杂度调整干预策略
        if is_complex_env:
            # 复杂环境（类似CR）：使用更强干预
            adaptive_prob = min(0.6, self.intervention_prob_value * 2.0)
            threshold = max(1.2, self.anomaly_threshold.item() * 0.6)  # 更敏感
            strong_threshold = max(0.15, self.strong_anomaly_threshold.item() * 0.5)
            noise_multiplier = 1.8
        else:
            # 简单环境（类似NCR）：使用温和干预
            adaptive_prob = max(0.05, self.intervention_prob_value * 0.4)
            threshold = min(3.5, self.anomaly_threshold.item() * 1.5)  # 更保守
            strong_threshold = min(0.6, self.strong_anomaly_threshold.item() * 1.5)
            noise_multiplier = 0.3
        
        # 使用自适应概率
        if random.random() > adaptive_prob:
            return x, {
                'anomaly_ratio': 0.0, 
                'variance': avg_variance, 
                'adaptive_prob': adaptive_prob,
                'env_type': 'Complex' if is_complex_env else 'Simple',
                'threshold_used': threshold
            }
        
        # 异常检测
        with torch.no_grad():
            z_scores = (x - self.feature_mean.unsqueeze(0).unsqueeze(0)) / \
                      (self.feature_var.sqrt().unsqueeze(0).unsqueeze(0) + 1e-6)
            
            is_anomaly = z_scores.abs() > threshold
            anomaly_score = is_anomaly.float().mean(dim=-1)
            strong_anomaly = anomaly_score > strong_threshold
        
        # 应用干预
        if strong_anomaly.any():
            # 根据环境类型调整噪声强度
            noise_scale = self.noise_scale_value * noise_multiplier
            noise = torch.randn_like(x) * noise_scale
            
            # 只在异常位置添加噪声
            intervention_mask = strong_anomaly.unsqueeze(-1).float()
            
            # 应用层级特定的缩放
            x = x + noise * intervention_mask * self.layer_scale
            
            return x, {
                'anomaly_ratio': anomaly_score.mean().item(),
                'variance': avg_variance,
                'adaptive_prob': adaptive_prob,
                'noise_scale_used': noise_scale,
                'env_type': 'Complex' if is_complex_env else 'Simple',
                'threshold_used': threshold,
                'layer_idx': self.layer_idx
            }
        
        return x, {
            'anomaly_ratio': anomaly_score.mean().item(),
            'variance': avg_variance,
            'adaptive_prob': adaptive_prob,
            'env_type': 'Complex' if is_complex_env else 'Simple',
            'threshold_used': threshold,
            'layer_idx': self.layer_idx
        }


class SimpleEffectiveMultiLevelCausalIntervention(nn.Module):
    """
    简单有效的多层级因果干预框架
    """
    
    def __init__(
        self,
        dim: int = 128,
        encoder_depth: int = 4,
        base_intervention_prob: float = 0.3,
        base_noise_scale: float = 0.1,
        enable_layer_interventions: Optional[list] = None,
    ):
        super().__init__()
        
        self.dim = dim
        self.encoder_depth = encoder_depth
        self.base_intervention_prob = base_intervention_prob
        self.base_noise_scale = base_noise_scale
        
        if enable_layer_interventions is None:
            enable_layer_interventions = [True] * (encoder_depth - 1) + [False]
        
        self.enable_layer_interventions = enable_layer_interventions
        
        # 为每层创建干预模块
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
                
                # 创建干预模块
                intervention = SimpleEffectiveAdaptiveLayerIntervention(
                    dim=dim,
                    layer_idx=i,
                    total_layers=encoder_depth,
                    base_intervention_prob=base_intervention_prob,
                    base_noise_scale=base_noise_scale,
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
def create_simple_effective_intervention():
    """
    创建简单有效的干预框架
    """
    return SimpleEffectiveMultiLevelCausalIntervention(
        dim=128,
        encoder_depth=4,
        base_intervention_prob=0.3,
        base_noise_scale=0.1,
    )

def create_cr_focused_intervention():
    """
    创建针对CR环境优化的干预框架
    """
    return SimpleEffectiveMultiLevelCausalIntervention(
        dim=128,
        encoder_depth=4,
        base_intervention_prob=0.4,  # 提高基础概率
        base_noise_scale=0.15,       # 提高基础噪声
    )