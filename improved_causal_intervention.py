import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
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
        self.register_buffer('update_count', torch.tensor(0))
        
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
                self.update_count += 1
        
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


class SmartFeatureDecorrelation(nn.Module):
    """
    智能特征去相关模块 - 基于相似性的去相关
    """
    def __init__(
        self,
        dim: int = 128,
        decorrelate_prob: float = 0.15,  # 降低概率
        mix_ratio: float = 0.05,  # 降低混合比例
        similarity_threshold: float = 0.8,  # 相似性阈值
    ):
        super().__init__()
        
        self.dim = dim
        self.decorrelate_prob = decorrelate_prob
        self.mix_ratio = mix_ratio
        self.similarity_threshold = similarity_threshold
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or random.random() > self.decorrelate_prob:
            return x
            
        batch_size, seq_len, dim = x.shape
        
        # 计算特征相似性矩阵
        x_norm = F.normalize(x, p=2, dim=-1)  # [B, L, D]
        similarity = torch.bmm(x_norm, x_norm.transpose(1, 2))  # [B, L, L]
        
        # 选择相似度高的位置进行去相关
        high_sim_mask = similarity > self.similarity_threshold
        
        # 限制去相关的数量，避免过度干预
        num_pairs = min(seq_len // 8, 5)  # 进一步限制数量
        
        for _ in range(num_pairs):
            # 随机选择两个位置
            i = random.randint(0, seq_len - 1)
            j = random.randint(0, seq_len - 1)
            
            if i != j and high_sim_mask[:, i, j].any():
                # 在batch维度shuffle其中一个位置
                perm = torch.randperm(batch_size, device=x.device)
                
                # 软混合而不是硬替换
                x[:, j] = (1 - self.mix_ratio) * x[:, j] + self.mix_ratio * x[perm, j]
                
        return x


class ImprovedCausalGate(nn.Module):
    """
    改进的因果门控机制 - 添加注意力机制
    """
    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 8,
        gate_threshold: float = 0.5
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.gate_threshold = gate_threshold
        
        # 使用自注意力计算门控权重
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
            nn.Sigmoid()
        )
        
        # 初始化门控接近1（默认不阻挡）
        nn.init.constant_(self.gate[2].weight, 0.1)
        
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        batch_size, seq_len, dim = x.shape
        
        # 使用自注意力计算门控权重
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        gate_weights = self.gate(attn_out)
        
        # 应用门控（软选择）
        x = x * gate_weights
        
        return x


class ConservativeIntervention(nn.Module):
    """
    保守干预策略 - 用于浅层
    """
    def __init__(self, dim, prob=0.1):
        super().__init__()
        self.dim = dim
        self.prob = prob
        self.detector = ConservativeAnomalyDetector(dim, intervention_prob=prob)
        
    def forward(self, x, key_padding_mask=None):
        return self.detector(x, 0.02, key_padding_mask)  # 很小的噪声


class BalancedIntervention(nn.Module):
    """
    平衡干预策略 - 用于中层
    """
    def __init__(self, dim, prob=0.3):
        super().__init__()
        self.dim = dim
        self.prob = prob
        self.detector = ConservativeAnomalyDetector(dim, intervention_prob=prob)
        
    def forward(self, x, key_padding_mask=None):
        return self.detector(x, 0.05, key_padding_mask)


class AggressiveIntervention(nn.Module):
    """
    积极干预策略 - 用于深层
    """
    def __init__(self, dim, prob=0.5):
        super().__init__()
        self.dim = dim
        self.prob = prob
        self.detector = ConservativeAnomalyDetector(dim, intervention_prob=prob)
        
    def forward(self, x, key_padding_mask=None):
        return self.detector(x, 0.1, key_padding_mask)


class AdaptiveIntervention(nn.Module):
    def __init__(self, dim, base_prob=0.3):
        super().__init__()
        self.dim = dim
        self.base_prob = base_prob
        
        # 学习干预强度调节器
        self.intensity_controller = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, key_padding_mask=None):
        if not self.training:
            return x, None
            
        # 根据特征复杂度动态调整干预强度
        feature_complexity = self.intensity_controller(x.mean(dim=1))  # [B, 1]
        adaptive_prob = self.base_prob * feature_complexity
        
        # 只在复杂度高时进行强干预
        if random.random() < adaptive_prob.mean():
            # 进行干预
            pass


class LayeredInterventionStrategy(nn.Module):
    def __init__(self, dim, total_layers):
        super().__init__()
        self.dim = dim
        self.total_layers = total_layers
        
        # 为不同层设置不同的干预策略
        self.layer_strategies = nn.ModuleList([
            self._create_strategy_for_layer(i, total_layers) 
            for i in range(total_layers)
        ])
    
    def _create_strategy_for_layer(self, layer_idx, total_layers):
        depth_ratio = (total_layers - layer_idx) / total_layers
        
        if depth_ratio > 0.7:  # 浅层：保守干预
            return ConservativeIntervention(self.dim, prob=0.1)
        elif depth_ratio > 0.3:  # 中层：平衡干预
            return BalancedIntervention(self.dim, prob=0.3)
        else:  # 深层：积极干预
            return AggressiveIntervention(self.dim, prob=0.5)


class FeatureProtection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 学习特征重要性
        self.importance_estimator = nn.Linear(dim, 1)
        
    def forward(self, x, intervention_mask):
        # 估计每个特征的重要性
        importance = torch.sigmoid(self.importance_estimator(x))  # [B, L, 1]
        
        # 保护重要特征
        protection_mask = (importance > 0.7).float()
        intervention_mask = intervention_mask * (1 - protection_mask)
        
        return intervention_mask


class GradualIntervention(nn.Module):
    """
    渐进式干预 - 避免突然的强干预
    """
    def __init__(self, dim, base_noise_scale=0.1):
        super().__init__()
        self.dim = dim
        self.base_noise_scale = base_noise_scale
        
    def forward(self, x, anomaly_score, noise_scale):
        # 根据异常程度渐进式干预
        intervention_strength = torch.sigmoid(anomaly_score - 0.5)  # 软阈值
        
        # 渐进式噪声强度
        noise = torch.randn_like(x) * noise_scale * intervention_strength.unsqueeze(-1)
        
        # 渐进式应用
        x = x + noise * intervention_strength.unsqueeze(-1)
        
        return x


class ImprovedUltraLightLayerIntervention(nn.Module):
    def __init__(self, dim, layer_idx, total_layers, base_intervention_prob=0.3):
        super().__init__()
        self.dim = dim
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        
        # 计算层级特定的干预强度
        depth_ratio = (total_layers - layer_idx) / total_layers
        
        # 浅层使用更保守的策略
        if depth_ratio > 0.7:
            self.intervention_prob = base_intervention_prob * 0.5
            self.noise_scale = 0.05
        else:
            self.intervention_prob = base_intervention_prob * depth_ratio
            self.noise_scale = 0.1 * depth_ratio
        
        # 使用改进的模块
        self.anomaly_detector = ConservativeAnomalyDetector(
            dim, 
            intervention_prob=self.intervention_prob,
            noise_scale=self.noise_scale
        )
        self.decorrelation = SmartFeatureDecorrelation(dim)
        self.gating = ImprovedCausalGate(dim)
        self.feature_protection = FeatureProtection(dim)
        self.gradual_intervention = GradualIntervention(dim)
        
    def forward(self, x, key_padding_mask=None):
        if not self.training:
            return x, None
        
        # 1. 保守的异常检测
        x, anom_metrics = self.anomaly_detector(x, self.noise_scale, key_padding_mask)
        
        # 2. 智能特征去相关
        x = self.decorrelation(x)
        
        # 3. 特征保护
        if hasattr(self, 'last_intervention_mask') and anom_metrics:
            anomaly_score = torch.tensor(anom_metrics.get('anomaly_ratio', 0.0))
            protected_mask = self.feature_protection(x, anomaly_score.unsqueeze(0).unsqueeze(0))
            x = x * protected_mask + x * (1 - protected_mask)
        
        # 4. 改进的门控
        x = self.gating(x, key_padding_mask)
        
        return x, anom_metrics


class MultiLevelCausalIntervention2_2(nn.Module):
    """
    多层级因果干预框架 - 自适应版本（版本3）
    每层自动感知场景动态性并调整干预策略
    """
    
    def __init__(
        self,
        dim: int = 128,
        encoder_depth: int = 4,
        base_intervention_prob: float = 0.25,
        base_noise_scale: float = 0.08,
        enable_layer_interventions: Optional[List[bool]] = None,
        enable_dynamic_gating: bool = True,
        dynamic_threshold: float = 0.3,
    ):
        super().__init__()
        
        self.dim = dim
        self.encoder_depth = encoder_depth
        self.base_intervention_prob = base_intervention_prob
        self.base_noise_scale = base_noise_scale
        
        if enable_layer_interventions is None:
            enable_layer_interventions = [True] * (encoder_depth - 1) + [False]
        
        self.enable_layer_interventions = enable_layer_interventions
        
        # 为每层创建自适应干预模块
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
                
                # 创建自适应干预模块
                intervention = ImprovedUltraLightLayerIntervention(
                    dim=dim,
                    layer_idx=i,
                    total_layers=encoder_depth,
                    base_intervention_prob=base_intervention_prob
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
        
        Args:
            x: [B, L, D] 特征张量
            layer_idx: 层索引
            key_padding_mask: [B, L] padding mask
        
        Returns:
            (x_out, metrics): 干预后的特征和指标
        """
        if layer_idx >= len(self.layer_interventions):
            return x, None
        
        intervention_module = self.layer_interventions[layer_idx]
        
        if intervention_module is None:
            return x, None
        
        # 调用干预模块
        x_out, metrics = intervention_module(x, key_padding_mask)
        
        return x_out, metrics
    
    def get_all_metrics(self) -> Dict:
        """获取所有层的干预统计信息"""
        metrics = {}
        
        for config in self.layer_configs:
            layer_idx = config['layer_idx']
            prefix = f'layer_{layer_idx}'
            
            metrics[f'{prefix}_prob'] = config['intervention_prob']
            metrics[f'{prefix}_noise_scale'] = config['noise_scale']
            metrics[f'{prefix}_enabled'] = config['enabled']
            
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