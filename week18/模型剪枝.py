import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
import time
import os
import numpy as np
from collections import OrderedDict

# ============================================================================
# 设备检查函数
# ============================================================================

def get_device():
    """
    设备检查函数：优先使用CUDA，如果没有则使用CPU
    返回：torch.device对象和设备信息字符串
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
        print(f"✓ 检测到CUDA设备: {device_info}")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        # 显示GPU内存信息
        if torch.cuda.is_available():
            print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        device_info = "CPU"
        print(f"✓ 未检测到CUDA设备，使用CPU")
    return device, device_info


# ============================================================================
# 第一部分：模型定义
# ============================================================================

class SimpleCNN(nn.Module):
    """简单的CNN模型用于MNIST分类"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 全连接层
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # 第一层卷积 + ReLU + 池化
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14
        
        # 第二层卷积 + ReLU + 池化
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 14x14 -> 7x7
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class SimpleNet(nn.Module):
    """简单全连接网络用于演示"""
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================================================================
# 第二部分：工具函数
# ============================================================================

def count_parameters(model):
    """计算模型参数总数"""
    return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model):
    """计算非零参数数量（考虑剪枝掩码）"""
    total_nonzero = 0
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 检查是否有剪枝掩码
            if hasattr(module, 'weight_mask'):
                total_nonzero += (module.weight_mask != 0).sum().item()
            else:
                total_nonzero += (module.weight != 0).sum().item()
    return total_nonzero


def get_model_size(model, filepath="temp_model.pth"):
    """计算模型大小（MB）"""
    torch.save(model.state_dict(), filepath)
    size = os.path.getsize(filepath) / (1024 * 1024)
    os.remove(filepath)
    return size


def evaluate_model(model, test_loader, device):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def benchmark_inference_speed(model, test_loader, device, num_runs=100):
    """测试模型推理速度"""
    model.eval()
    # 准备一个批次的数据
    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    images = images.to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(images)
    
    # 正式测试
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(images)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # 转换为毫秒
    return avg_time


def train_model(model, train_loader, device, epochs=3):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    print(f"\n开始训练模型（{epochs}个epoch）...")
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
    
    print("训练完成！\n")


# ============================================================================
# 第三部分：剪枝函数
# ============================================================================

def unstructured_pruning(model, amount=0.3):
    """
    非结构化剪枝（L1 Unstructured Pruning）
    移除权重中L1范数最小的参数
    """
    print(f"\n执行非结构化剪枝（剪枝比例: {amount*100:.1f}%）...")
    
    # 对卷积层和全连接层进行剪枝
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            prune.l1_unstructured(module, name="weight", amount=amount)
    
    print("非结构化剪枝完成！")


def structured_pruning(model, amount=0.3):
    """
    结构化剪枝（移除整个通道/滤波器）
    注意：结构化剪枝需要特殊处理，这里演示移除整个通道
    """
    print(f"\n执行结构化剪枝（剪枝比例: {amount*100:.1f}%）...")
    print("注意：结构化剪枝会移除整个通道，可能影响模型结构")
    
    # 对卷积层进行结构化剪枝（移除通道）
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 计算要移除的通道数
            num_channels = module.out_channels
            num_prune = int(num_channels * amount)
            if num_prune > 0 and num_prune < num_channels:
                prune.ln_structured(module, name="weight", amount=num_prune, n=2, dim=0)
                print(f"  对 {name} 剪枝了 {num_prune}/{num_channels} 个通道")
    
    print("结构化剪枝完成！")


def global_pruning(model, amount=0.3):
    """
    全局剪枝（考虑所有层的权重，统一剪枝）
    """
    print(f"\n执行全局剪枝（剪枝比例: {amount*100:.1f}%）...")
    
    # 收集所有需要剪枝的参数
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            parameters_to_prune.append((module, 'weight'))
    
    # 全局剪枝
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    print("全局剪枝完成！")


def remove_pruning_mask(model):
    """移除剪枝掩码，永久删除被剪枝的参数"""
    print("\n移除剪枝掩码，永久删除被剪枝的参数...")
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            prune.remove(module, 'weight')
    print("掩码移除完成！")


# ============================================================================
# 第四部分：完整演示
# ============================================================================

def create_fresh_model(model_class, device):
    """创建新的模型实例（用于恢复原始状态）"""
    model = model_class().to(device)
    return model


def comprehensive_pruning_demo():
    """完整的剪枝演示：训练、剪枝、对比"""
    
    print("=" * 80)
    print("模型剪枝完整演示")
    print("=" * 80)
    
    # 设备检查
    print("\n【设备检查】")
    device, device_info = get_device()
    print(f"\n使用设备: {device} ({device_info})")
    
    # 准备数据
    print("\n准备数据...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化
    ])
    
    # 使用较小的数据集用于快速演示
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # 使用部分数据加快演示速度
    train_indices = torch.randperm(len(train_dataset))[:5000]
    test_indices = torch.randperm(len(test_dataset))[:1000]
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=64, shuffle=False
    )
    
    print(f"训练集大小: {len(train_subset)}")
    print(f"测试集大小: {len(test_subset)}")
    
    # 创建模型
    print("\n创建模型...")
    model = SimpleCNN().to(device)
    
    # 训练模型
    train_model(model, train_loader, device, epochs=3)
    
    # ========================================================================
    # 剪枝前评估
    # ========================================================================
    print("\n" + "=" * 80)
    print("剪枝前模型评估")
    print("=" * 80)
    
    # 保存原始模型
    original_model_state = model.state_dict().copy()
    
    # 评估指标
    print("\n正在评估模型...")
    accuracy_before = evaluate_model(model, test_loader, device)
    size_before = get_model_size(model)
    params_before = count_parameters(model)
    nonzero_before = count_nonzero_parameters(model)
    speed_before = benchmark_inference_speed(model, test_loader, device)
    
    print(f"\n【剪枝前指标】")
    print(f"  准确率: {accuracy_before:.2f}%")
    print(f"  模型大小: {size_before:.4f} MB")
    print(f"  总参数数: {params_before:,}")
    print(f"  非零参数数: {nonzero_before:,}")
    print(f"  平均推理时间: {speed_before:.2f} ms")
    print(f"  稀疏度: 0.00%")
    
    # ========================================================================
    # 方法1：非结构化剪枝
    # ========================================================================
    print("\n" + "=" * 80)
    print("方法1：非结构化剪枝（L1 Unstructured Pruning）")
    print("=" * 80)
    
    # 恢复原始模型：创建新实例并加载状态
    model = create_fresh_model(SimpleCNN, device)
    model.load_state_dict(original_model_state, strict=True)
    
    # 执行非结构化剪枝（30%）
    unstructured_pruning(model, amount=0.3)
    
    # 评估剪枝后模型
    print("\n正在评估剪枝后的模型...")
    accuracy_after_unstructured = evaluate_model(model, test_loader, device)
    size_after_unstructured = get_model_size(model)
    params_after_unstructured = count_parameters(model)
    nonzero_after_unstructured = count_nonzero_parameters(model)
    speed_after_unstructured = benchmark_inference_speed(model, test_loader, device)
    sparsity_unstructured = (1 - nonzero_after_unstructured / params_before) * 100
    
    print(f"\n【非结构化剪枝后指标】")
    print(f"  准确率: {accuracy_after_unstructured:.2f}% (变化: {accuracy_after_unstructured - accuracy_before:+.2f}%)")
    print(f"  模型大小: {size_after_unstructured:.4f} MB (压缩比: {size_before/size_after_unstructured:.2f}x)")
    print(f"  总参数数: {params_after_unstructured:,}")
    print(f"  非零参数数: {nonzero_after_unstructured:,}")
    print(f"  平均推理时间: {speed_after_unstructured:.2f} ms (加速比: {speed_before/speed_after_unstructured:.2f}x)")
    print(f"  稀疏度: {sparsity_unstructured:.2f}%")
    
    # ========================================================================
    # 方法2：全局剪枝
    # ========================================================================
    print("\n" + "=" * 80)
    print("方法2：全局剪枝（Global Pruning）")
    print("=" * 80)
    
    # 恢复原始模型：创建新实例并加载状态
    model = create_fresh_model(SimpleCNN, device)
    model.load_state_dict(original_model_state, strict=True)
    
    # 执行全局剪枝（30%）
    global_pruning(model, amount=0.3)
    
    # 评估剪枝后模型
    print("\n正在评估剪枝后的模型...")
    accuracy_after_global = evaluate_model(model, test_loader, device)
    size_after_global = get_model_size(model)
    params_after_global = count_parameters(model)
    nonzero_after_global = count_nonzero_parameters(model)
    speed_after_global = benchmark_inference_speed(model, test_loader, device)
    sparsity_global = (1 - nonzero_after_global / params_before) * 100
    
    print(f"\n【全局剪枝后指标】")
    print(f"  准确率: {accuracy_after_global:.2f}% (变化: {accuracy_after_global - accuracy_before:+.2f}%)")
    print(f"  模型大小: {size_after_global:.4f} MB (压缩比: {size_before/size_after_global:.2f}x)")
    print(f"  总参数数: {params_after_global:,}")
    print(f"  非零参数数: {nonzero_after_global:,}")
    print(f"  平均推理时间: {speed_after_global:.2f} ms (加速比: {speed_before/speed_after_global:.2f}x)")
    print(f"  稀疏度: {sparsity_global:.2f}%")
    
    # ========================================================================
    # 方法3：渐进式剪枝（Iterative Pruning）
    # ========================================================================
    print("\n" + "=" * 80)
    print("方法3：渐进式剪枝（Iterative Pruning）")
    print("=" * 80)
    
    # 恢复原始模型：创建新实例并加载状态
    model = create_fresh_model(SimpleCNN, device)
    model.load_state_dict(original_model_state, strict=True)
    
    # 渐进式剪枝：分多次剪枝，每次剪枝后重新训练
    pruning_amounts = [0.1, 0.1, 0.1]  # 分3次，每次10%
    total_pruning = sum(pruning_amounts)
    
    print(f"\n执行渐进式剪枝（总共剪枝 {total_pruning*100:.0f}%）...")
    
    for i, amount in enumerate(pruning_amounts):
        print(f"\n第 {i+1} 轮剪枝（{amount*100:.0f}%）...")
        unstructured_pruning(model, amount=amount)
        
        # 重新训练（fine-tuning）
        print(f"重新训练模型...")
        train_model(model, train_loader, device, epochs=1)
        
        # 评估当前状态
        current_accuracy = evaluate_model(model, test_loader, device)
        print(f"当前准确率: {current_accuracy:.2f}%")
    
    # 最终评估
    print("\n正在评估渐进式剪枝后的模型...")
    accuracy_after_iterative = evaluate_model(model, test_loader, device)
    size_after_iterative = get_model_size(model)
    params_after_iterative = count_parameters(model)
    nonzero_after_iterative = count_nonzero_parameters(model)
    speed_after_iterative = benchmark_inference_speed(model, test_loader, device)
    sparsity_iterative = (1 - nonzero_after_iterative / params_before) * 100
    
    print(f"\n【渐进式剪枝后指标】")
    print(f"  准确率: {accuracy_after_iterative:.2f}% (变化: {accuracy_after_iterative - accuracy_before:+.2f}%)")
    print(f"  模型大小: {size_after_iterative:.4f} MB (压缩比: {size_before/size_after_iterative:.2f}x)")
    print(f"  总参数数: {params_after_iterative:,}")
    print(f"  非零参数数: {nonzero_after_iterative:,}")
    print(f"  平均推理时间: {speed_after_iterative:.2f} ms (加速比: {speed_before/speed_after_iterative:.2f}x)")
    print(f"  稀疏度: {sparsity_iterative:.2f}%")
    
    # ========================================================================
    # 对比总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("剪枝方法对比总结")
    print("=" * 80)
    
    print(f"\n{'指标':<20} {'原始模型':<15} {'非结构化':<15} {'全局剪枝':<15} {'渐进式':<15}")
    print("-" * 80)
    print(f"{'准确率 (%)':<20} {accuracy_before:<15.2f} {accuracy_after_unstructured:<15.2f} {accuracy_after_global:<15.2f} {accuracy_after_iterative:<15.2f}")
    print(f"{'模型大小 (MB)':<20} {size_before:<15.4f} {size_after_unstructured:<15.4f} {size_after_global:<15.4f} {size_after_iterative:<15.4f}")
    print(f"{'非零参数数':<20} {nonzero_before:<15,} {nonzero_after_unstructured:<15,} {nonzero_after_global:<15,} {nonzero_after_iterative:<15,}")
    print(f"{'推理时间 (ms)':<20} {speed_before:<15.2f} {speed_after_unstructured:<15.2f} {speed_after_global:<15.2f} {speed_after_iterative:<15.2f}")
    print(f"{'稀疏度 (%)':<20} {'0.00':<15} {sparsity_unstructured:<15.2f} {sparsity_global:<15.2f} {sparsity_iterative:<15.2f}")
    
    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)
    
    # 说明
    print("\n【技术说明】")
    print("1. 非结构化剪枝：移除权重中L1范数最小的参数，不改变模型结构")
    print("2. 全局剪枝：考虑所有层的权重统一剪枝，可能获得更好的压缩效果")
    print("3. 渐进式剪枝：分多次剪枝并重新训练，通常能保持更好的准确率")
    print("\n注意：实际应用中，剪枝后通常需要重新训练（fine-tuning）以恢复性能")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    comprehensive_pruning_demo()

