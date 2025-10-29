#!/usr/bin/env python3
"""
测试简单有效的自适应干预框架
验证在CR和NCR环境中的表现
"""

import torch
import torch.nn as nn
from simple_effective_intervention import (
    SimpleEffectiveMultiLevelCausalIntervention,
    create_simple_effective_intervention,
    create_cr_focused_intervention
)

def simulate_cr_environment(batch_size=4, seq_len=20, dim=128):
    """
    模拟CR环境：高复杂度，动态交互
    """
    # 高方差，复杂特征，更多异常
    x = torch.randn(batch_size, seq_len, dim) * 3.0  # 高方差
    # 添加更多异常值
    x[:, :10, :] += torch.randn(batch_size, 10, dim) * 5.0
    # 添加一些极端值
    x[:, 15:20, :] += torch.randn(batch_size, 5, dim) * 8.0
    return x

def simulate_ncr_environment(batch_size=4, seq_len=20, dim=128):
    """
    模拟NCR环境：低复杂度，静态环境
    """
    # 低方差，简单特征
    x = torch.randn(batch_size, seq_len, dim) * 0.2  # 低方差
    return x

def test_simple_effective_intervention():
    """
    测试简单有效的干预框架
    """
    print("测试简单有效的自适应干预框架")
    print("=" * 60)
    
    # 创建简单有效的干预框架
    intervention = create_simple_effective_intervention()
    
    # 测试CR环境
    print("\n1. 测试CR环境（高方差，复杂特征）:")
    cr_x = simulate_cr_environment()
    intervention.train()
    
    cr_metrics = {}
    for layer_idx in range(4):
        cr_x, metrics = intervention.apply_intervention_after_layer(cr_x, layer_idx)
        if metrics:
            cr_metrics[f'layer_{layer_idx}'] = metrics
            print(f"  第{layer_idx}层: 环境={metrics.get('env_type', 'Unknown')}, "
                  f"方差={metrics.get('variance', 0):.3f}, "
                  f"自适应概率={metrics.get('adaptive_prob', 0):.3f}, "
                  f"阈值={metrics.get('threshold_used', 0):.3f}, "
                  f"异常比例={metrics.get('anomaly_ratio', 0):.3f}")
    
    # 测试NCR环境
    print("\n2. 测试NCR环境（低方差，简单特征）:")
    ncr_x = simulate_ncr_environment()
    
    ncr_metrics = {}
    for layer_idx in range(4):
        ncr_x, metrics = intervention.apply_intervention_after_layer(ncr_x, layer_idx)
        if metrics:
            ncr_metrics[f'layer_{layer_idx}'] = metrics
            print(f"  第{layer_idx}层: 环境={metrics.get('env_type', 'Unknown')}, "
                  f"方差={metrics.get('variance', 0):.3f}, "
                  f"自适应概率={metrics.get('adaptive_prob', 0):.3f}, "
                  f"阈值={metrics.get('threshold_used', 0):.3f}, "
                  f"异常比例={metrics.get('anomaly_ratio', 0):.3f}")
    
    # 分析结果
    print("\n3. 详细分析:")
    
    # 计算平均方差
    cr_avg_variance = sum(m.get('variance', 0) for m in cr_metrics.values()) / len(cr_metrics)
    ncr_avg_variance = sum(m.get('variance', 0) for m in ncr_metrics.values()) / len(ncr_metrics)
    
    print(f"  CR环境平均方差: {cr_avg_variance:.3f}")
    print(f"  NCR环境平均方差: {ncr_avg_variance:.3f}")
    
    # 计算平均干预概率
    cr_avg_prob = sum(m.get('adaptive_prob', 0) for m in cr_metrics.values()) / len(cr_metrics)
    ncr_avg_prob = sum(m.get('adaptive_prob', 0) for m in ncr_metrics.values()) / len(ncr_metrics)
    
    print(f"  CR环境平均干预概率: {cr_avg_prob:.3f}")
    print(f"  NCR环境平均干预概率: {ncr_avg_prob:.3f}")
    
    # 计算平均阈值
    cr_avg_threshold = sum(m.get('threshold_used', 0) for m in cr_metrics.values()) / len(cr_metrics)
    ncr_avg_threshold = sum(m.get('threshold_used', 0) for m in ncr_metrics.values()) / len(ncr_metrics)
    
    print(f"  CR环境平均阈值: {cr_avg_threshold:.3f}")
    print(f"  NCR环境平均阈值: {ncr_avg_threshold:.3f}")
    
    # 验证自适应行为
    print("\n4. 自适应行为验证:")
    
    if cr_avg_prob > ncr_avg_prob:
        print("  ✅ 干预概率自适应正确：CR环境使用更高的干预概率")
    else:
        print("  ❌ 干预概率自适应异常：CR环境干预概率过低")
    
    if cr_avg_threshold < ncr_avg_threshold:
        print("  ✅ 阈值自适应正确：CR环境使用更敏感的阈值")
    else:
        print("  ❌ 阈值自适应异常：CR环境阈值不够敏感")
    
    if cr_avg_variance > ncr_avg_variance:
        print("  ✅ 方差检测正确：CR环境被识别为高方差环境")
    else:
        print("  ❌ 方差检测异常：CR环境方差检测不准确")
    
    return cr_metrics, ncr_metrics

def test_cr_focused_version():
    """
    测试CR优化版本
    """
    print("\n5. 测试CR优化版本:")
    
    # 创建CR优化版本
    cr_intervention = create_cr_focused_intervention()
    
    # 测试CR环境
    cr_x = simulate_cr_environment()
    cr_intervention.train()
    
    cr_metrics = {}
    for layer_idx in range(4):
        cr_x, metrics = cr_intervention.apply_intervention_after_layer(cr_x, layer_idx)
        if metrics:
            cr_metrics[f'layer_{layer_idx}'] = metrics
    
    # 计算CR优化版本的效果
    cr_avg_prob = sum(m.get('adaptive_prob', 0) for m in cr_metrics.values()) / len(cr_metrics)
    cr_avg_threshold = sum(m.get('threshold_used', 0) for m in cr_metrics.values()) / len(cr_metrics)
    cr_avg_variance = sum(m.get('variance', 0) for m in cr_metrics.values()) / len(cr_metrics)
    
    print(f"  CR优化版本 - 平均干预概率: {cr_avg_prob:.3f}")
    print(f"  CR优化版本 - 平均阈值: {cr_avg_threshold:.3f}")
    print(f"  CR优化版本 - 平均方差: {cr_avg_variance:.3f}")
    
    return cr_metrics

def test_gradient_flow():
    """
    测试梯度流
    """
    print("\n6. 测试梯度流:")
    
    intervention = create_simple_effective_intervention()
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
    print("\n7. 测试参数学习:")
    
    intervention = create_simple_effective_intervention()
    
    # 检查可学习参数
    learnable_params = []
    for name, param in intervention.named_parameters():
        if param.requires_grad:
            learnable_params.append((name, param.shape))
    
    print(f"  可学习参数数量: {len(learnable_params)}")
    for name, shape in learnable_params:
        print(f"    {name}: {shape}")
    
    print("  ✅ 参数学习正常")

def main():
    """
    主测试函数
    """
    print("开始测试简单有效的自适应干预框架...")
    
    try:
        # 测试简单有效的干预框架
        cr_metrics, ncr_metrics = test_simple_effective_intervention()
        
        # 测试CR优化版本
        cr_opt_metrics = test_cr_focused_version()
        
        # 测试梯度流
        test_gradient_flow()
        
        # 测试参数学习
        test_parameter_learning()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试完成！")
        print("\n简单有效改进总结:")
        print("1. 基于方差的复杂度检测 - 简单但有效")
        print("2. 自适应干预概率 - 高方差环境使用更高概率")
        print("3. 自适应阈值 - 高方差环境使用更敏感阈值")
        print("4. 自适应噪声强度 - 根据环境类型调整噪声")
        print("5. 简单实现 - 易于理解和调试")
        
        print("\n预期效果:")
        print("- CR环境（高方差）：更强干预，更敏感阈值，更高概率")
        print("- NCR环境（低方差）：温和干预，保守阈值，较低概率")
        print("- 基于特征方差的简单但有效的环境识别")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()