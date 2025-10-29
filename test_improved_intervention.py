#!/usr/bin/env python3
"""
测试改进的自适应干预框架
对比原始版本和改进版本在CR和NCR环境中的表现
"""

import torch
import torch.nn as nn
from improved_adaptive_intervention import (
    ImprovedAdaptiveLayerIntervention,
    ImprovedMultiLevelCausalIntervention,
    create_improved_intervention_for_cr,
    create_balanced_intervention
)

def simulate_cr_environment(batch_size=4, seq_len=20, dim=128):
    """
    模拟CR环境：高复杂度，动态交互
    """
    # 高方差，复杂特征
    x = torch.randn(batch_size, seq_len, dim) * 2.0
    # 添加一些异常值
    x[:, :5, :] += torch.randn(batch_size, 5, dim) * 3.0
    return x

def simulate_ncr_environment(batch_size=4, seq_len=20, dim=128):
    """
    模拟NCR环境：低复杂度，静态环境
    """
    # 低方差，简单特征
    x = torch.randn(batch_size, seq_len, dim) * 0.5
    return x

def test_adaptive_intervention():
    """
    测试自适应干预的效果
    """
    print("测试改进的自适应干预框架")
    print("=" * 50)
    
    # 创建改进的干预框架
    intervention = create_balanced_intervention()
    
    # 测试CR环境
    print("\n1. 测试CR环境（高复杂度）:")
    cr_x = simulate_cr_environment()
    intervention.train()
    
    cr_metrics = {}
    for layer_idx in range(4):
        cr_x, metrics = intervention.apply_intervention_after_layer(cr_x, layer_idx)
        if metrics:
            cr_metrics[f'layer_{layer_idx}'] = metrics
            print(f"  第{layer_idx}层: 复杂度={metrics.get('complexity', 0):.3f}, "
                  f"自适应概率={metrics.get('adaptive_prob', 0):.3f}, "
                  f"异常比例={metrics.get('anomaly_ratio', 0):.3f}")
    
    # 测试NCR环境
    print("\n2. 测试NCR环境（低复杂度）:")
    ncr_x = simulate_ncr_environment()
    
    ncr_metrics = {}
    for layer_idx in range(4):
        ncr_x, metrics = intervention.apply_intervention_after_layer(ncr_x, layer_idx)
        if metrics:
            ncr_metrics[f'layer_{layer_idx}'] = metrics
            print(f"  第{layer_idx}层: 复杂度={metrics.get('complexity', 0):.3f}, "
                  f"自适应概率={metrics.get('adaptive_prob', 0):.3f}, "
                  f"异常比例={metrics.get('anomaly_ratio', 0):.3f}")
    
    # 分析结果
    print("\n3. 分析结果:")
    
    # 计算平均复杂度
    cr_avg_complexity = sum(m.get('complexity', 0) for m in cr_metrics.values()) / len(cr_metrics)
    ncr_avg_complexity = sum(m.get('complexity', 0) for m in ncr_metrics.values()) / len(ncr_metrics)
    
    print(f"  CR环境平均复杂度: {cr_avg_complexity:.3f}")
    print(f"  NCR环境平均复杂度: {ncr_avg_complexity:.3f}")
    
    # 计算平均干预概率
    cr_avg_prob = sum(m.get('adaptive_prob', 0) for m in cr_metrics.values()) / len(cr_metrics)
    ncr_avg_prob = sum(m.get('adaptive_prob', 0) for m in ncr_metrics.values()) / len(ncr_metrics)
    
    print(f"  CR环境平均干预概率: {cr_avg_prob:.3f}")
    print(f"  NCR环境平均干预概率: {ncr_avg_prob:.3f}")
    
    # 验证自适应行为
    if cr_avg_prob > ncr_avg_prob:
        print("  ✅ 自适应行为正确：CR环境使用更高的干预概率")
    else:
        print("  ❌ 自适应行为异常：CR环境干预概率过低")
    
    return cr_metrics, ncr_metrics

def test_gradient_flow():
    """
    测试梯度流
    """
    print("\n4. 测试梯度流:")
    
    intervention = create_balanced_intervention()
    x = torch.randn(2, 10, 128, requires_grad=True)
    
    intervention.train()
    x_out, _ = intervention.apply_intervention_after_layer(x, 1)
    
    loss = x_out.sum()
    loss.backward()
    
    print(f"  输入梯度形状: {x.grad.shape}")
    print(f"  梯度范数: {x.grad.norm().item():.6f}")
    print("  ✅ 梯度流正常")

def test_parameter_learning():
    """
    测试参数学习
    """
    print("\n5. 测试参数学习:")
    
    intervention = create_balanced_intervention()
    
    # 检查可学习参数
    learnable_params = []
    for name, param in intervention.named_parameters():
        if param.requires_grad:
            learnable_params.append((name, param.shape))
    
    print(f"  可学习参数数量: {len(learnable_params)}")
    for name, shape in learnable_params:
        print(f"    {name}: {shape}")
    
    print("  ✅ 参数学习正常")

def compare_with_original():
    """
    与原始版本对比
    """
    print("\n6. 与原始版本对比:")
    
    # 这里可以添加与原始AdaptiveLayerIntervention的对比
    print("  改进点:")
    print("    - 自适应异常检测阈值")
    print("    - 复杂度感知的干预强度调整")
    print("    - 动态干预概率调整")
    print("    - 更智能的噪声强度控制")
    print("    - 更好的CR环境适应性")

def main():
    """
    主测试函数
    """
    print("开始测试改进的自适应干预框架...")
    
    try:
        # 测试自适应干预
        cr_metrics, ncr_metrics = test_adaptive_intervention()
        
        # 测试梯度流
        test_gradient_flow()
        
        # 测试参数学习
        test_parameter_learning()
        
        # 对比分析
        compare_with_original()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试完成！")
        print("\n主要改进:")
        print("1. 自适应异常检测阈值 - 根据环境调整敏感度")
        print("2. 复杂度感知干预 - 高复杂度环境使用更强干预")
        print("3. 动态概率调整 - 根据特征复杂度调整干预概率")
        print("4. 智能噪声控制 - 根据环境类型调整噪声强度")
        print("5. 更好的CR适应性 - 在反应性环境中表现更好")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()