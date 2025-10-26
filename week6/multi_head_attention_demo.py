import torch  # 提供张量操作和自动微分
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib

# 设置matplotlib后端，避免兼容性问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# 完整的多头注意力机制
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制的 PyTorch 实现。
    这比纯 NumPy 实现更贴近实际应用，并封装在 nn.Module 中。
    """

    def __init__(self, d_model, num_heads):
        """
        初始化函数。
        
        参数:
        d_model (int): 模型的总维度，必须能被 num_heads 整除。
        num_heads (int): 注意力头的数量。
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"  # 断言检查

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 线性变换层 W_q,W_k,W_v,W_o
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        核心的缩放点积注意力计算。矩阵运算
        """
        # TODO
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)  # 防止梯度消失的问题
        # 做一个数据泄露的问题扼制
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def split_heads(self, x):
        """
        将输入张量分割成多个头。
        输入 x 形状: (batch_size, seq_len, d_model)
        输出形状: (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        将多个头的输出合并。
        输入 x 形状: (batch_size, num_heads, seq_len, d_k)
        输出形状: (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        前向传播。
        """
        """
        1. 线性变换, QKV
        2. 分割成多头
        3. 计算缩放点积注意力
        4. 合并多头
        5. 最终进行线性变换，output
        """
        Q, K, V = self.W_q(Q), self.W_k(K), self.W_v(V)
        Q, K, V = self.split_heads(Q), self.split_heads(K), self.split_heads(V)
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.combine_heads(output)
        output = self.W_o(output)

        return output, attention_weights


def visualize_attention(attention_weights, tokens):  # 用于显示注意力权重
    """
    可视化注意力权重。
    """
    assert len(tokens) == attention_weights.shape[-1]

    num_heads = attention_weights.shape[1]
    fig, axes = plt.subplots(1, num_heads, figsize=(num_heads * 4, 4), dpi=120)
    if num_heads == 1:
        axes = [axes]

    for i in range(num_heads):
        ax = axes[i]
        sns.heatmap(attention_weights[0, i].detach().numpy(),
                    xticklabels=tokens, yticklabels=tokens, ax=ax, cmap='viridis', cbar=False)
        ax.set_title(f"Attention Head {i + 1}")
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()

    # 保存图片而不是显示，避免兼容性问题
    filename = 'attention_visualization.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"注意力可视化已保存为 '{filename}'")
    plt.close()  # 关闭图形以释放内存


def analyze_attention_patterns(attention_weights, tokens):
    """
    分析注意力模式，提供统计信息。
    """
    attention_np = attention_weights[0].detach().numpy()
    num_heads = attention_np.shape[0]
    seq_len = attention_np.shape[1]

    print("\n--- 注意力模式统计 ---")

    for head_idx in range(num_heads):
        head_attention = attention_np[head_idx]

        # 计算自注意力强度（对角线元素）
        self_attention = np.diag(head_attention)
        avg_self_attention = np.mean(self_attention)

        # 计算注意力集中度（最大值的平均值）
        max_attention_per_token = np.max(head_attention, axis=1)
        avg_max_attention = np.mean(max_attention_per_token)

        # 计算注意力分布的熵（衡量分布的均匀性）
        attention_entropy = -np.sum(head_attention * np.log(head_attention + 1e-10), axis=1)
        avg_entropy = np.mean(attention_entropy)

        print(f"\n注意力头 {head_idx + 1}:")
        print(f"  平均自注意力强度: {avg_self_attention:.3f}")
        print(f"  平均最大注意力: {avg_max_attention:.3f}")
        print(f"  平均注意力熵: {avg_entropy:.3f}")

        # 判断注意力模式类型
        if avg_self_attention > 0.5:
            pattern_type = "自注意力型"
        elif avg_entropy < 1.0:
            pattern_type = "集中型"
        else:
            pattern_type = "分散型"

        print(f"  模式类型: {pattern_type}")


if __name__ == '__main__':
    # --- 教学演示 ---
    # 定义模型超参数
    d_model = 128
    num_heads = 4

    # 准备输入数据
    # 使用一个具体的句子
    sentence = "The cat sat on the mat"
    tokens = sentence.split()
    seq_len = len(tokens)
    batch_size = 1

    # 创建词汇表和简单的词嵌入
    vocab = {word: i for i, word in enumerate(tokens)}
    embedding = nn.Embedding(len(vocab), d_model)

    # 将 token 转换为索引
    token_ids = torch.tensor([vocab[word] for word in tokens]).unsqueeze(0)  # (batch_size, seq_len)

    # 获取词嵌入
    # 在自注意力中, Q, K, V 都来自同一个输入序列
    input_embeddings = embedding(token_ids)  # (batch_size, seq_len, d_model)

    print("--- 输入 ---")
    print(f"句子: '{sentence}'")
    print(f"输入嵌入的形状: {input_embeddings.shape} (Batch, SeqLen, d_model)")
    print("-" * 20)

    # 初始化多头注意力模块
    multi_head_attention = MultiHeadAttention(d_model, num_heads)

    # 前向传播
    # 在自注意力中，Q, K, V 是相同的
    output, attention_weights = multi_head_attention(input_embeddings, input_embeddings, input_embeddings)

    print("\n--- 输出 ---")
    print(f"输出张量的形状: {output.shape} (Batch, SeqLen, d_model)")
    print(f"注意力权重形状: {attention_weights.shape} (Batch, NumHeads, SeqLen, SeqLen)")
    print("\n解释: 输出的每个词向量现在都融合了句子中其他词的信息（根据注意力权重）。")
    print("-" * 20)

    print("\n--- 注意力可视化 ---")
    print("下面的热力图展示了每个注意力头学到的不同关注模式。")
    print("每个单元格的颜色深浅代表了Y轴的词（Query）对X轴的词（Key）的关注程度。")
    visualize_attention(attention_weights, tokens)

    print("\n--- 可视化解读 ---")
    print("观察不同头的热力图：")
    print(" - 有的头可能主要关注对角线，即每个词更关注自己。")
    print(" - 有的头可能关注相邻的词，形成一种类似n-gram的模式。")
    print(" - 有的头可能会学习到语法关系，例如 'sat' 可能会同时关注 'cat' 和 'mat'。")
    print("这就是多头注意力的优势：它能从不同子空间捕捉多样化的依赖关系。")

    # 添加详细的注意力分析
    print("\n--- 详细注意力分析 ---")
    attention_np = attention_weights[0].detach().numpy()

    for head_idx in range(num_heads):
        print(f"\n注意力头 {head_idx + 1} 的分析:")
        head_attention = attention_np[head_idx]

        # 找到每个词最关注的词
        for i, token in enumerate(tokens):
            max_attention_idx = np.argmax(head_attention[i])
            max_attention_val = head_attention[i][max_attention_idx]
            print(f"  '{token}' 最关注 '{tokens[max_attention_idx]}' (权重: {max_attention_val:.3f})")

        # 计算平均注意力分布
        avg_attention = np.mean(head_attention, axis=0)
        print(f"  平均注意力分布: {dict(zip(tokens, [f'{val:.3f}' for val in avg_attention]))}")

    print(f"\n可视化图片已保存为 'attention_visualization.png'，请查看该文件以观察注意力热力图。")

    # 调用注意力模式分析函数
    analyze_attention_patterns(attention_weights, tokens)

    print("\n--- 多头注意力机制深度解析 ---")
    print("1. 为什么需要多头注意力？")
    print("   - 单一注意力机制只能学习一种依赖关系模式")
    print("   - 多头机制允许模型同时关注不同的特征子空间")
    print("   - 每个头可以学习不同的语法、语义或位置关系")

    print("\n2. 多头注意力的数学原理：")
    print("   - 将输入向量分割成h个子空间（h个头）")
    print("   - 每个子空间独立计算注意力")
    print("   - 最后将所有头的输出拼接并投影")
    print(f"   - 在我们的例子中：{d_model}维 → {num_heads}个{d_model // num_heads}维子空间")

    print("\n3. 实际应用中的优势：")
    print("   - 并行计算：所有头可以同时计算，提高效率")
    print("   - 特征多样性：不同头学习不同的特征模式")
    print("   - 鲁棒性：单个头的失效不会严重影响整体性能")
    print("   - 可解释性：可以分析每个头的专门功能")

    print("\n4. 训练建议：")
    print("   - 头数通常选择为d_model的因子（如8、16等）")
    print("   - 可以通过可视化分析每个头的功能")
    print("   - 某些头可能会被训练成专门关注特定模式")
    print("   - 可以通过头剪枝技术移除不重要的头")
