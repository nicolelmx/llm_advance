# -*- coding: utf-8 -*-
"""
AI模型优化与部署 - 简化教学脚本
"""

import torch
import torch.nn as nn
import torch.quantization
import torch.nn.utils.prune as prune
import time
import os
import numpy as np

# ============================================================================
# 第一部分：模型量化
# ============================================================================

class SimpleCNN(nn.Module):
    """简单的CNN模型用于演示量化"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        # 使用自适应池化避免固定尺寸问题
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(32 * 4 * 4, 10)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def get_model_size(model):
    """计算模型大小（MB）"""
    torch.save(model.state_dict(), "temp_model.pth")
    size = os.path.getsize("temp_model.pth") / (1024 * 1024)
    os.remove("temp_model.pth")
    return size

def benchmark_model(model, input_data, num_runs=100):
    """性能测试函数"""
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_data)
    end_time = time.time()
    return (end_time - start_time) / num_runs * 1000  # ms

def demo_dynamic_quantization():
    """演示动态量化"""
    print("=" * 50)
    print("1. PyTorch动态量化演示")
    print("=" * 50)
    
    # 创建模型实例
    model = SimpleCNN()
    model.eval()
    
    # 准备示例输入
    example_input = torch.randn(1, 3, 32, 32)
    
    # 动态量化（只量化Linear层）
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    
    # 比较模型大小
    original_size = get_model_size(model)
    quantized_size = get_model_size(quantized_model)
    
    print(f"原始模型大小: {original_size:.4f} MB")
    print(f"量化后模型大小: {quantized_size:.4f} MB")
    if quantized_size > 0:
        print(f"压缩比: {original_size/quantized_size:.2f}x")
    
    # 推理速度对比
    original_time = benchmark_model(model, example_input, num_runs=50)
    quantized_time = benchmark_model(quantized_model, example_input, num_runs=50)
    
    print(f"原始模型推理时间: {original_time:.2f} ms")
    print(f"量化模型推理时间: {quantized_time:.2f} ms")
    if quantized_time > 0:
        print(f"加速比: {original_time/quantized_time:.2f}x")

def demo_static_quantization():
    """演示静态量化（简化版，跳过需要fbgemm的部分）"""
    print("=" * 50)
    print("2. PyTorch静态量化演示（需要fbgemm支持）")
    print("=" * 50)
    print("注意：静态量化需要fbgemm后端，在某些环境下可能不可用")
    print("这里仅展示代码结构，实际运行可能需要额外配置")

# ============================================================================
# 第二部分：模型剪枝
# ============================================================================

class SimpleNet(nn.Module):
    """简单全连接网络用于演示剪枝"""
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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

def demo_unstructured_pruning():
    """演示非结构化剪枝"""
    print("=" * 50)
    print("3. 非结构化剪枝演示")
    print("=" * 50)
    
    # 创建模型
    model = SimpleNet()
    
    total_params = count_parameters(model)
    nonzero_before = count_nonzero_parameters(model)
    
    print(f"剪枝前总参数数: {total_params:,}")
    print(f"剪枝前非零参数数: {nonzero_before:,}")
    
    # 基于权重大小的剪枝（L1 Unstructured Pruning）
    prune.l1_unstructured(model.fc1, name="weight", amount=0.3)  # 剪枝30%
    prune.l1_unstructured(model.fc2, name="weight", amount=0.3)
    prune.l1_unstructured(model.fc3, name="weight", amount=0.3)
    
    nonzero_after = count_nonzero_parameters(model)
    sparsity = (1 - nonzero_after / total_params) * 100
    
    print(f"\n剪枝后非零参数数: {nonzero_after:,}")
    print(f"稀疏度: {sparsity:.2f}%")

# ============================================================================
# 第三部分：ONNX模型导出
# ============================================================================

def export_simple_model_to_onnx():
    """导出简单模型为ONNX格式"""
    print("=" * 50)
    print("4. ONNX模型导出演示")
    print("=" * 50)
    
    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = nn.Linear(10, 5)
        
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleModel()
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(1, 10)
    
    # 导出为ONNX
    onnx_path = "simple_model.onnx"
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"模型已导出到: {onnx_path}")
        
        # 验证ONNX模型（可选）
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX模型验证通过！")
        except ImportError:
            print("提示: 安装onnx可以验证模型: pip install onnx")
    except Exception as e:
        print(f"导出失败: {str(e)}")

# ONNX推理（需要onnxruntime）
try:
    import onnxruntime as ort
    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    ONNX_RUNTIME_AVAILABLE = False

def demo_onnx_inference(onnx_model_path="simple_model.onnx"):
    """演示ONNX Runtime推理"""
    if not ONNX_RUNTIME_AVAILABLE:
        print("提示: 安装onnxruntime可以运行推理: pip install onnxruntime")
        return
    
    print("=" * 50)
    print("5. ONNX Runtime推理演示")
    print("=" * 50)
    
    if not os.path.exists(onnx_model_path):
        print(f"模型文件不存在: {onnx_model_path}")
        return
    
    try:
        # 加载ONNX模型
        session = ort.InferenceSession(onnx_model_path)
        
        # 获取输入输出信息
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"输入名称: {input_name}")
        print(f"输入形状: {input_shape}")
        print(f"输出名称: {output_name}")
        
        # 准备输入数据
        dummy_input = np.random.randn(1, 10).astype(np.float32)
        
        # 推理
        outputs = session.run([output_name], {input_name: dummy_input})
        predictions = outputs[0]
        
        print(f"输出形状: {predictions.shape}")
        print(f"推理成功！")
    except Exception as e:
        print(f"推理失败: {str(e)}")


# ============================================================================
# 主函数：演示所有功能
# ============================================================================

def main():
    """主函数：演示所有功能"""
    print("=" * 70)
    print("AI模型优化与部署 - 简化教学脚本")
    print("=" * 70)
    
    # 1. 量化演示
    print("\n")
    try:
        demo_dynamic_quantization()
    except Exception as e:
        print(f"量化演示失败: {str(e)}")
    
    # 2. 静态量化说明
    print("\n")
    demo_static_quantization()
    
    # 3. 剪枝演示
    print("\n")
    try:
        demo_unstructured_pruning()
    except Exception as e:
        print(f"剪枝演示失败: {str(e)}")
    
    # 4. ONNX导出
    print("\n")
    try:
        export_simple_model_to_onnx()
    except Exception as e:
        print(f"ONNX导出失败: {str(e)}")
    
    # 5. ONNX推理
    print("\n")
    if os.path.exists("simple_model.onnx"):
        try:
            demo_onnx_inference("simple_model.onnx")
        except Exception as e:
            print(f"ONNX推理失败: {str(e)}")
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()