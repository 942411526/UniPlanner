import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import random


class ConservativeAnomalyDetector(nn.Module):
    """
    保守的异常检测器 - 减少误报，避免过度干预
    """
    def __init__(
        self,
        dim: int = 128,
        intervention_prob: float = 0.2,  # 降低默认概率
        noise_scale: float = 0.05,  # 降低默认噪声强度
        anomaly_threshold: float = 3.0,  # 提高异常阈值
        strong_anomaly_threshold: float = 0.5,  # 提高强异常阈值
    ):
        super().__init__()
        
        self.dim = dim
        self.intervention_prob = intervention_prob
        self.noise_scale = noise_scale
        self.anomaly_threshold = anomaly_threshold
        self.strong_anomaly_threshold = strong_anomaly_threshold
        
        # 用running statistics代替学习网络
        self.register_buffer('feature_mean', torch.zeros(dim))
        self.register_buffer('feature_var', torch.ones(dim))
        self.register_buffer('update_count', torch.tensor(0, dtype=torch.long))  # 确保是tensor类型
        
    def forward(self, x, noise_scale, key_padding_mask=None):
        """
        保守的异常检测和干预
        """
        if not self.training:
            return x, None
            
        if random.random() > self.intervention_prob:
            return x, {'anomaly_ratio': 0.0}
        
        batch_size, seq_len, dim = x.shape
        
        # 更新running statistics（非常轻量）
        if self.training:
            with torch.no_grad():
                valid_x = x
                if key_padding_mask is not None:
                    valid_mask = ~key_padding_mask.unsqueeze(-1)
                    valid_x = x * valid_mask
                    
                batch_mean = valid_x.mean(dim=(0, 1))
                batch_var = valid_x.var(dim=(0, 1))
                
                # 指数移动平均更新
                momentum = 0.05  # 降低更新速度，更保守
                self.feature_mean = (1 - momentum) * self.feature_mean + momentum * batch_mean
                self.feature_var = (1 - momentum) * self.feature_var + momentum * batch_var
                self.update_count += 1  # 正确的buffer更新方式
        
        # 更保守的异常检测
        with torch.no_grad():
            # 计算z-score
            z_scores = (x - self.feature_mean.unsqueeze(0).unsqueeze(0)) / (self.feature_var.sqrt().unsqueeze(0).unsqueeze(0) + 1e-6)
            
            # 使用更高的阈值，减少误报
            is_anomaly = z_scores.abs() > self.anomaly_threshold  # 3.0
            anomaly_score = is_anomaly.float().mean(dim=-1)  # [B, L]
            
            # 要求更严格的异常条件
            strong_anomaly = anomaly_score > self.strong_anomaly_threshold  # 0.5
        
        # 只对确实异常的位置进行干预
        if strong_anomaly.any():
            noise = torch.randn_like(x) * noise_scale * 0.5  # 减少噪声强度
            intervention_mask = strong_anomaly.unsqueeze(-1).float()
            x = x + noise * intervention_mask
            
        return x, {'anomaly_ratio': anomaly_score.mean().item()}


# 如果你需要替换现有文件中的ConservativeAnomalyDetector，请使用以下代码片段：

"""
# 在你的AdaptiveLayerIntervention2_2.py文件中，将ConservativeAnomalyDetector的forward方法替换为：

def forward(self, x, noise_scale, key_padding_mask=None):
    if not self.training:
        return x, None
        
    if random.random() > self.intervention_prob:
        return x, {'anomaly_ratio': 0.0}
    
    batch_size, seq_len, dim = x.shape
    
    # 更新running statistics（非常轻量）
    if self.training:
        with torch.no_grad():
            valid_x = x
            if key_padding_mask is not None:
                valid_mask = ~key_padding_mask.unsqueeze(-1)
                valid_x = x * valid_mask
                
            batch_mean = valid_x.mean(dim=(0, 1))
            batch_var = valid_x.var(dim=(0, 1))
            
            # 指数移动平均更新
            momentum = 0.05  # 降低更新速度，更保守
            self.feature_mean = (1 - momentum) * self.feature_mean + momentum * batch_mean
            self.feature_var = (1 - momentum) * self.feature_var + momentum * batch_var
            self.update_count += 1  # 关键修复：使用 += 而不是 =
    
    # 更保守的异常检测
    with torch.no_grad():
        # 计算z-score
        z_scores = (x - self.feature_mean.unsqueeze(0).unsqueeze(0)) / (self.feature_var.sqrt().unsqueeze(0).unsqueeze(0) + 1e-6)
        
        # 使用更高的阈值，减少误报
        is_anomaly = z_scores.abs() > self.anomaly_threshold  # 3.0
        anomaly_score = is_anomaly.float().mean(dim=-1)  # [B, L]
        
        # 要求更严格的异常条件
        strong_anomaly = anomaly_score > self.strong_anomaly_threshold  # 0.5
    
    # 只对确实异常的位置进行干预
    if strong_anomaly.any():
        noise = torch.randn_like(x) * noise_scale * 0.5  # 减少噪声强度
        intervention_mask = strong_anomaly.unsqueeze(-1).float()
        x = x + noise * intervention_mask
        
    return x, {'anomaly_ratio': anomaly_score.mean().item()}
"""