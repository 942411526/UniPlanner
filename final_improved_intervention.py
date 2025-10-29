import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import random


class FinalImprovedAdaptiveLayerIntervention(nn.Module):
    """
    最终改进的自适应单层干预模块
    专门针对CR环境优化，同时保持NCR环境的良好表现
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
        
        # 计算该层的干预强度 - 改进策略：深层也需要足够强的干预
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
        
        # 环境复杂度检测器
        self.complexity_detector = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 干预强度调节器
        self.intensity_adapter = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, 1),
            nn.Sigmoid()
        )
        
        # 环境类型检测器
        self.env_detector = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 2),  # 2个环境类型：CR和NCR
            nn.Softmax(dim=-1)
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
        
        # 检测环境类型和复杂度
        with torch.no_grad():
            # 环境类型检测
            env_logits = self.env_detector(x.mean(dim=1))  # [B, 2]
            env_type = env_logits.mean(dim=0)  # [2]
            is_cr_env = env_type[0] > env_type[1]  # CR环境概率
            
            # 复杂度检测
            feature_complexity = self.complexity_detector(x.mean(dim=1))  # [B, 1]
            avg_complexity = feature_complexity.mean().item()
        
        # 根据环境类型和复杂度调整干预策略
        if is_cr_env or avg_complexity > 0.6:  # CR环境或高复杂度
            # 使用更强的干预策略
            adaptive_prob = min(0.6, self.intervention_prob_value * 1.8)
            threshold = max(1.5, self.anomaly_threshold.item() * 0.8)  # 更敏感的阈值
            strong_threshold = max(0.2, self.strong_anomaly_threshold.item() * 0.7)
        else:  # NCR环境或低复杂度
            # 使用温和的干预策略
            adaptive_prob = max(0.1, self.intervention_prob_value * 0.6)
            threshold = min(3.0, self.anomaly_threshold.item() * 1.2)  # 更保守的阈值
            strong_threshold = min(0.5, self.strong_anomaly_threshold.item() * 1.3)
        
        # 使用自适应概率
        if random.random() > adaptive_prob:
            return x, {
                'anomaly_ratio': 0.0, 
                'complexity': avg_complexity, 
                'adaptive_prob': adaptive_prob,
                'env_type': 'CR' if is_cr_env else 'NCR',
                'threshold_used': threshold
            }
        
        # 改进的异常检测
        with torch.no_grad():
            z_scores = (x - self.feature_mean.unsqueeze(0).unsqueeze(0)) / \
                      (self.feature_var.sqrt().unsqueeze(0).unsqueeze(0) + 1e-6)
            
            is_anomaly = z_scores.abs() > threshold
            anomaly_score = is_anomaly.float().mean(dim=-1)
            strong_anomaly = anomaly_score > strong_threshold
        
        # 改进的干预策略
        if strong_anomaly.any():
            # 计算自适应干预强度
            intensity_factor = self.intensity_adapter(x.mean(dim=1))  # [B, 1]
            avg_intensity = intensity_factor.mean().item()
            
            # 根据环境类型调整噪声强度
            if is_cr_env or avg_complexity > 0.6:
                # CR环境：使用更强的噪声
                noise_scale = self.noise_scale_value * (1.5 + avg_intensity)
            else:
                # NCR环境：使用温和的噪声
                noise_scale = self.noise_scale_value * (0.3 + avg_intensity * 0.4)
            
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
                'env_type': 'CR' if is_cr_env else 'NCR',
                'threshold_used': threshold,
                'layer_idx': self.layer_idx
            }
        
        return x, {
            'anomaly_ratio': anomaly_score.mean().item(),
            'complexity': avg_complexity,
            'adaptive_prob': adaptive_prob,
            'env_type': 'CR' if is_cr_env else 'NCR',
            'threshold_used': threshold,
            'layer_idx': self.layer_idx
        }


class FinalImprovedMultiLevelCausalIntervention(nn.Module):
    """
    最终改进的多层级因果干预框架
    专门针对CR环境优化，同时保持NCR环境的良好表现
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
                intervention = FinalImprovedAdaptiveLayerIntervention(
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
def create_cr_optimized_intervention():
    """
    创建针对CR环境优化的干预框架
    """
    return FinalImprovedMultiLevelCausalIntervention(
        dim=128,
        encoder_depth=4,
        base_intervention_prob=0.4,  # 提高基础概率
        base_noise_scale=0.15,       # 提高基础噪声
    )

def create_balanced_intervention():
    """
    创建平衡的干预框架
    """
    return FinalImprovedMultiLevelCausalIntervention(
        dim=128,
        encoder_depth=4,
        base_intervention_prob=0.3,
        base_noise_scale=0.1,
    )