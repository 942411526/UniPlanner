#!/usr/bin/env python3
"""
修复AdaptiveLayerIntervention2_2.py中的buffer更新问题
"""

import re

def fix_buffer_update_issue(file_path):
    """
    修复文件中的buffer更新问题
    将 self.update_count = 1 改为 self.update_count += 1
    """
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找并替换问题行
    # 查找 self.update_count = 1 或类似的赋值
    pattern = r'self\.update_count\s*=\s*1'
    replacement = 'self.update_count += 1'
    
    # 执行替换
    new_content = re.sub(pattern, replacement, content)
    
    # 检查是否有变化
    if new_content != content:
        # 备份原文件
        backup_path = file_path + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"已创建备份文件: {backup_path}")
        
        # 写入修复后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"已修复文件: {file_path}")
        print("修复内容: 将 'self.update_count = 1' 改为 'self.update_count += 1'")
    else:
        print("未找到需要修复的内容")
    
    return new_content != content

def fix_register_buffer_issue(file_path):
    """
    修复register_buffer的问题
    确保update_count被正确注册为tensor类型
    """
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找register_buffer('update_count', ...) 的行
    pattern = r"self\.register_buffer\('update_count',\s*torch\.tensor\(0\)\)"
    replacement = "self.register_buffer('update_count', torch.tensor(0, dtype=torch.long))"
    
    # 执行替换
    new_content = re.sub(pattern, replacement, content)
    
    # 检查是否有变化
    if new_content != content:
        # 备份原文件
        backup_path = file_path + '.backup2'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"已创建备份文件: {backup_path}")
        
        # 写入修复后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"已修复文件: {file_path}")
        print("修复内容: 确保update_count被正确注册为tensor类型")
    else:
        print("未找到需要修复的register_buffer内容")
    
    return new_content != content

if __name__ == "__main__":
    # 你的文件路径
    file_path = "/home/yx/paper3/PlanScope/src/models/pluto/modules/AdaptiveLayerIntervention2_2.py"
    
    print("开始修复AdaptiveLayerIntervention2_2.py...")
    
    # 修复buffer更新问题
    fixed1 = fix_buffer_update_issue(file_path)
    
    # 修复register_buffer问题
    fixed2 = fix_register_buffer_issue(file_path)
    
    if fixed1 or fixed2:
        print("\n✅ 修复完成！")
        print("\n修复内容总结:")
        print("1. 将 'self.update_count = 1' 改为 'self.update_count += 1'")
        print("2. 确保 'update_count' 被正确注册为tensor类型")
        print("\n现在可以重新运行你的代码了。")
    else:
        print("\n❌ 未找到需要修复的内容，可能文件已经是正确的。")