#!/usr/bin/env python3
"""
改进的因果干预框架使用示例
展示如何在规划任务中使用这个框架来缓解因果混淆
"""

import torch
import torch.nn as nn
from improved_causal_intervention import MultiLevelCausalIntervention2_2

class PlanningModelWithIntervention(nn.Module):
    """
    带因果干预的规划模型示例
    """
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=64, num_layers=4):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 编码器层
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # 因果干预框架
        self.causal_intervention = MultiLevelCausalIntervention2_2(
            dim=hidden_dim,
            encoder_depth=num_layers,
            base_intervention_prob=0.25,  # 降低基础干预概率
            base_noise_scale=0.08,        # 降低基础噪声强度
            enable_layer_interventions=[True] * (num_layers - 1) + [False]  # 最后一层不干预
        )
        
    def forward(self, x, key_padding_mask=None):
        """
        前向传播
        
        Args:
            x: [B, L, input_dim] 输入特征
            key_padding_mask: [B, L] padding mask
            
        Returns:
            output: [B, L, output_dim] 输出预测
            intervention_metrics: dict 干预统计信息
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)  # [B, L, hidden_dim]
        
        # 通过编码器层，每层后应用干预
        intervention_metrics = {}
        
        for layer_idx, encoder_layer in enumerate(self.encoder_layers):
            # 通过编码器层
            x = encoder_layer(x, src_key_padding_mask=key_padding_mask)
            
            # 应用因果干预（除了最后一层）
            x, layer_metrics = self.causal_intervention.apply_intervention_after_layer(
                x, layer_idx, key_padding_mask
            )
            
            # 收集干预指标
            if layer_metrics:
                intervention_metrics[f'layer_{layer_idx}'] = layer_metrics
        
        # 输出投影
        output = self.output_projection(x)
        
        # 获取所有干预统计信息
        all_metrics = self.causal_intervention.get_all_metrics()
        intervention_metrics.update(all_metrics)
        
        return output, intervention_metrics

def demonstrate_usage():
    """演示框架的使用"""
    print("演示改进的因果干预框架在规划任务中的应用")
    print("=" * 60)
    
    # 创建模型
    model = PlanningModelWithIntervention(
        input_dim=128,
        hidden_dim=256,
        output_dim=64,
        num_layers=4
    )
    
    # 创建示例数据
    batch_size, seq_len = 4, 20
    input_dim = 128
    
    # 模拟规划任务的输入特征（例如：车辆状态、道路信息、其他车辆等）
    x = torch.randn(batch_size, seq_len, input_dim)
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    
    print(f"输入形状: {x.shape}")
    print(f"Padding mask形状: {key_padding_mask.shape}")
    
    # 训练模式
    model.train()
    
    # 前向传播
    with torch.no_grad():
        output, metrics = model(x, key_padding_mask)
    
    print(f"输出形状: {output.shape}")
    print(f"干预指标数量: {len(metrics)}")
    
    # 显示一些关键指标
    print("\n关键干预指标:")
    for key, value in metrics.items():
        if 'anomaly_ratio' in key or 'prob' in key:
            print(f"  {key}: {value:.4f}")
    
    # 测试梯度流
    print("\n测试梯度流...")
    model.train()
    x.requires_grad_(True)
    
    output, _ = model(x, key_padding_mask)
    loss = output.sum()
    loss.backward()
    
    print(f"输入梯度形状: {x.grad.shape}")
    print(f"梯度范数: {x.grad.norm().item():.6f}")
    
    # 测试不同场景
    print("\n测试不同场景的适应性...")
    
    # 场景1：简单场景（低复杂度）
    simple_x = torch.randn(batch_size, seq_len, input_dim) * 0.1  # 小方差
    simple_output, simple_metrics = model(simple_x, key_padding_mask)
    
    # 场景2：复杂场景（高复杂度）
    complex_x = torch.randn(batch_size, seq_len, input_dim) * 2.0  # 大方差
    complex_output, complex_metrics = model(complex_x, key_padding_mask)
    
    print("简单场景 vs 复杂场景的干预强度对比:")
    for layer in range(4):
        simple_anomaly = simple_metrics.get(f'layer_{layer}_anomaly_ratio', 0)
        complex_anomaly = complex_metrics.get(f'layer_{layer}_anomaly_ratio', 0)
        print(f"  第{layer}层: {simple_anomaly:.4f} vs {complex_anomaly:.4f}")
    
    print("\n✅ 框架使用演示完成！")
    print("\n主要改进点:")
    print("1. 保守的异常检测 - 减少误报，避免过度干预")
    print("2. 智能特征去相关 - 基于相似性的去相关策略")
    print("3. 改进的因果门控 - 使用自注意力机制")
    print("4. 特征保护机制 - 保护重要特征不被破坏")
    print("5. 分层干预策略 - 不同层使用不同的干预强度")
    print("6. 自适应干预强度 - 根据特征复杂度动态调整")

if __name__ == "__main__":
    demonstrate_usage()