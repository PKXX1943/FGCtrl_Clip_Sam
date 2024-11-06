import glob
import os
import random

# 1. 找到所有名为 annotation.txt 的文件
file_paths = glob.glob('**/annotation.txt', recursive=True)

# 2. 针对每个 annotation.txt 文件分别进行划分
for file_path in file_paths:
    # 获取当前 annotation 文件的父目录名称作为区分
    parent_dir = os.path.dirname(file_path)
    
    # 读取 annotation.txt 的内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 随机打乱数据
    random.shuffle(lines)
    
    # 按照 8:1:1 比例划分为 train, val, test 数据
    total_count = len(lines)
    train_count = int(total_count * 0.8)
    val_count = int(total_count * 0.1)
    
    train_data = lines[:train_count]
    val_data = lines[train_count:train_count + val_count]
    test_data = lines[train_count + val_count:]
    
    # 在当前目录下为每个 annotation 生成对应的 train.txt, val.txt 和 test.txt
    train_file_path = os.path.join(parent_dir, 'train.txt')
    val_file_path = os.path.join(parent_dir, 'val.txt')
    test_file_path = os.path.join(parent_dir, 'test.txt')
    
    with open(train_file_path, 'w') as train_file:
        train_file.writelines(train_data)
    
    with open(val_file_path, 'w') as val_file:
        val_file.writelines(val_data)
    
    with open(test_file_path, 'w') as test_file:
        test_file.writelines(test_data)

    print(f"文件 {file_path} 划分完成：训练集 {len(train_data)} 条，验证集 {len(val_data)} 条，测试集 {len(test_data)} 条。")
