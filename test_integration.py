#!/usr/bin/env python3
"""
测试改进的因果干预框架的集成和兼容性
"""

import torch
import torch.nn as nn
from improved_causal_intervention import (
    ConservativeAnomalyDetector,
    SmartFeatureDecorrelation,
    ImprovedCausalGate,
    FeatureProtection,
    GradualIntervention,
    ImprovedUltraLightLayerIntervention,
    MultiLevelCausalIntervention2_2,
    ConservativeIntervention,
    BalancedIntervention,
    AggressiveIntervention
)

def test_conservative_anomaly_detector():
    """测试保守异常检测器"""
    print("测试 ConservativeAnomalyDetector...")
    
    detector = ConservativeAnomalyDetector(dim=128)
    
    # 创建测试数据
    batch_size, seq_len, dim = 2, 10, 128
    x = torch.randn(batch_size, seq_len, dim)
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    
    # 训练模式测试
    detector.train()
    x_out, metrics = detector(x, noise_scale=0.05, key_padding_mask=key_padding_mask)
    
    assert x_out.shape == x.shape, f"输出形状不匹配: {x_out.shape} vs {x.shape}"
    assert metrics is not None, "应该返回指标"
    assert 'anomaly_ratio' in metrics, "应该包含异常比例指标"
    
    print("✓ ConservativeAnomalyDetector 测试通过")
    return True

def test_smart_feature_decorrelation():
    """测试智能特征去相关"""
    print("测试 SmartFeatureDecorrelation...")
    
    decorrelation = SmartFeatureDecorrelation(dim=128)
    
    # 创建测试数据
    batch_size, seq_len, dim = 2, 10, 128
    x = torch.randn(batch_size, seq_len, dim)
    
    # 训练模式测试
    decorrelation.train()
    x_out = decorrelation(x)
    
    assert x_out.shape == x.shape, f"输出形状不匹配: {x_out.shape} vs {x.shape}"
    
    print("✓ SmartFeatureDecorrelation 测试通过")
    return True

def test_improved_causal_gate():
    """测试改进的因果门控"""
    print("测试 ImprovedCausalGate...")
    
    gate = ImprovedCausalGate(dim=128)
    
    # 创建测试数据
    batch_size, seq_len, dim = 2, 10, 128
    x = torch.randn(batch_size, seq_len, dim)
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    
    x_out = gate(x, key_padding_mask)
    
    assert x_out.shape == x.shape, f"输出形状不匹配: {x_out.shape} vs {x.shape}"
    
    print("✓ ImprovedCausalGate 测试通过")
    return True

def test_feature_protection():
    """测试特征保护"""
    print("测试 FeatureProtection...")
    
    protection = FeatureProtection(dim=128)
    
    # 创建测试数据
    batch_size, seq_len, dim = 2, 10, 128
    x = torch.randn(batch_size, seq_len, dim)
    intervention_mask = torch.ones(batch_size, seq_len, 1)
    
    protected_mask = protection(x, intervention_mask)
    
    assert protected_mask.shape == intervention_mask.shape, f"输出形状不匹配: {protected_mask.shape} vs {intervention_mask.shape}"
    
    print("✓ FeatureProtection 测试通过")
    return True

def test_gradual_intervention():
    """测试渐进式干预"""
    print("测试 GradualIntervention...")
    
    intervention = GradualIntervention(dim=128)
    
    # 创建测试数据
    batch_size, seq_len, dim = 2, 10, 128
    x = torch.randn(batch_size, seq_len, dim)
    anomaly_score = torch.rand(batch_size, seq_len)
    noise_scale = 0.1
    
    x_out = intervention(x, anomaly_score, noise_scale)
    
    assert x_out.shape == x.shape, f"输出形状不匹配: {x_out.shape} vs {x.shape}"
    
    print("✓ GradualIntervention 测试通过")
    return True

def test_improved_ultra_light_layer():
    """测试改进的超轻量级层干预"""
    print("测试 ImprovedUltraLightLayerIntervention...")
    
    layer_intervention = ImprovedUltraLightLayerIntervention(
        dim=128,
        layer_idx=1,
        total_layers=4,
        base_intervention_prob=0.3
    )
    
    # 创建测试数据
    batch_size, seq_len, dim = 2, 10, 128
    x = torch.randn(batch_size, seq_len, dim)
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    
    # 训练模式测试
    layer_intervention.train()
    x_out, metrics = layer_intervention(x, key_padding_mask)
    
    assert x_out.shape == x.shape, f"输出形状不匹配: {x_out.shape} vs {x.shape}"
    
    print("✓ ImprovedUltraLightLayerIntervention 测试通过")
    return True

def test_multi_level_intervention():
    """测试多层级干预框架"""
    print("测试 MultiLevelCausalIntervention2_2...")
    
    multi_intervention = MultiLevelCausalIntervention2_2(
        dim=128,
        encoder_depth=4,
        base_intervention_prob=0.25,
        base_noise_scale=0.08
    )
    
    # 创建测试数据
    batch_size, seq_len, dim = 2, 10, 128
    x = torch.randn(batch_size, seq_len, dim)
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    
    # 训练模式测试
    multi_intervention.train()
    
    # 测试每层的干预
    for layer_idx in range(4):
        x_out, metrics = multi_intervention.apply_intervention_after_layer(
            x, layer_idx, key_padding_mask
        )
        assert x_out.shape == x.shape, f"第{layer_idx}层输出形状不匹配: {x_out.shape} vs {x.shape}"
    
    # 测试获取指标
    all_metrics = multi_intervention.get_all_metrics()
    assert isinstance(all_metrics, dict), "应该返回字典类型的指标"
    
    print("✓ MultiLevelCausalIntervention2_2 测试通过")
    return True

def test_intervention_strategies():
    """测试不同干预策略"""
    print("测试干预策略...")
    
    dim = 128
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, dim)
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    
    # 测试保守干预
    conservative = ConservativeIntervention(dim, prob=0.1)
    conservative.train()
    x_out, _ = conservative(x, key_padding_mask)
    assert x_out.shape == x.shape
    
    # 测试平衡干预
    balanced = BalancedIntervention(dim, prob=0.3)
    balanced.train()
    x_out, _ = balanced(x, key_padding_mask)
    assert x_out.shape == x.shape
    
    # 测试积极干预
    aggressive = AggressiveIntervention(dim, prob=0.5)
    aggressive.train()
    x_out, _ = aggressive(x, key_padding_mask)
    assert x_out.shape == x.shape
    
    print("✓ 干预策略测试通过")
    return True

def test_gradient_flow():
    """测试梯度流"""
    print("测试梯度流...")
    
    multi_intervention = MultiLevelCausalIntervention2_2(
        dim=128,
        encoder_depth=4,
        base_intervention_prob=0.25,
        base_noise_scale=0.08
    )
    
    # 创建测试数据
    batch_size, seq_len, dim = 2, 10, 128
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    
    # 训练模式
    multi_intervention.train()
    
    # 前向传播
    x_out, _ = multi_intervention.apply_intervention_after_layer(x, 1, key_padding_mask)
    
    # 计算损失
    loss = x_out.sum()
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    assert x.grad is not None, "输入应该有梯度"
    assert not torch.isnan(x.grad).any(), "梯度不应该包含NaN"
    
    print("✓ 梯度流测试通过")
    return True

def main():
    """运行所有测试"""
    print("开始测试改进的因果干预框架...")
    print("=" * 50)
    
    tests = [
        test_conservative_anomaly_detector,
        test_smart_feature_decorrelation,
        test_improved_causal_gate,
        test_feature_protection,
        test_gradual_intervention,
        test_improved_ultra_light_layer,
        test_multi_level_intervention,
        test_intervention_strategies,
        test_gradient_flow,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} 失败: {e}")
    
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！框架集成成功。")
    else:
        print("❌ 部分测试失败，需要修复。")
    
    return passed == total

if __name__ == "__main__":
    main()