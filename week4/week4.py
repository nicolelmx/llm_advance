"""
词向量训练脚本 (Word2Vec/FastText/GloVe)
功能：训练词向量模型并可视化结果
"""
import matplotlib

matplotlib.use('MacOSX')
import os
import gensim
from gensim.models import Word2Vec, FastText, KeyedVectors
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = [
    'PingFang SC',       # macOS 苹方字体
    'Hiragino Sans GB',  # macOS 冬青黑体
    'STHeiti',           # macOS 华文黑体
    'Arial Unicode MS',  # macOS 包含的中文字体
    'DejaVu Sans',       # 备用字体
    'sans-serif'         # 最终备用
]
plt.rcParams['axes.unicode_minus'] = False

# 1. 准备示例语料库
sentences = [
    ["natural", "language", "processing", "is", "fascinating"],
    ["word", "embeddings", "capture", "semantic", "meaning"],
    ["king", "queen", "man", "woman", "royalty"],
    ["similar", "words", "have", "close", "vectors"],
    ["machine", "learning", "models", "learn", "patterns"]
]

# 2. 训练Word2Vec模型
word2vec_model = Word2Vec(
    sentences=sentences,
    vector_size=100,  # 词向量维度
    window=5,  # 上下文窗口
    min_count=1,  # 最小词频
    workers=4,  # 并行线程
    epochs=50  # 训练轮次
)

# 3. 训练FastText模型（支持子词信息）
fasttext_model = FastText(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    epochs=50,
    min_n=3,  # 子词最小长度
    max_n=6  # 子词最大长度
)

# 4. 保存模型
word2vec_model.save("word2vec_model.bin")
fasttext_model.save("fasttext_model.bin")

# 打印模型信息
print(f"\nWord2Vec模型信息:")
print(f"- 词汇量: {len(word2vec_model.wv)}")
print(f"- 向量维度: {word2vec_model.vector_size}")
print(f"- 训练轮次: {word2vec_model.epochs}")
print(f"\nFastText模型信息:")
print(f"- 词汇量: {len(fasttext_model.wv)}")
print(f"- 向量维度: {fasttext_model.vector_size}")
print(f"- 子词长度范围: {fasttext_model.wv.min_n}-{fasttext_model.wv.max_n}")


# 5. 词向量可视化函数
def visualize_vectors(model, words, method='pca'):
    # 兼容处理完整模型对象和仅KeyedVectors对象
    vector_model = model.wv if hasattr(model, 'wv') else model
    vectors = [vector_model[word] for word in words]
    vectors = np.array(vectors)  # 转换为numpy数组

    # 降维到2D
    if method == 'pca':
        # PCA降维（线性，保持全局结构）
        reducer = PCA(n_components=2)
        title = 'Word Vector Visualization (PCA)'
    else:
        # t-SNE降维（非线性，保持局部结构）
        reducer = TSNE(n_components=2, perplexity=min(5, len(words) - 1),
                       learning_rate=200, random_state=42)
        title = 'Word Vector Visualization (t-SNE)'

    result = reducer.fit_transform(vectors)

    # 返回结果供合并图使用
    return result, title


# 创建单独的可视化函数
def plot_vectors(result, words, title):
    print(result, words, title)
    # 创建散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(result[:, 0], result[:, 1])

    # 添加标签
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

    plt.title(title)
    plt.show()


# 可视化示例词
test_words = ["king", "queen", "man", "woman", "royalty"]

# 单独可视化
# print('单独可视化')
# result_pca, title_pca = visualize_vectors(word2vec_model, test_words, method='pca')
# plot_vectors(result_pca, test_words, title_pca)
# result_tsne, title_tsne = visualize_vectors(word2vec_model, test_words, method='tsne')
# plot_vectors(result_tsne, test_words, title_tsne)

# 合并可视化（PCA和t-SNE对比）
print('合并可视化')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('词向量可视化对比: PCA vs t-SNE', fontsize=16)

# PCA降维
print('PCA降维')
result_pca, _ = visualize_vectors(word2vec_model, test_words, method='pca')
ax1.scatter(result_pca[:, 0], result_pca[:, 1])
for i, word in enumerate(test_words):
    ax1.annotate(word, xy=(result_pca[i, 0], result_pca[i, 1]))
ax1.set_title('PCA降维 (保持全局结构)')

# t-SNE降维
print('t-SNE降维')
result_tsne, _ = visualize_vectors(word2vec_model, test_words, method='tsne')
ax2.scatter(result_tsne[:, 0], result_tsne[:, 1])
for i, word in enumerate(test_words):
    ax2.annotate(word, xy=(result_tsne[i, 0], result_tsne[i, 1]))
ax2.set_title('t-SNE降维 (保持局部结构)')

plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局
plt.show(block=False)


# 6. 加载预训练GloVe模型（如果存在）
def load_glove_model(glove_file='glove.6B.100d.txt', convert=True):
    """加载GloVe模型，可选是否先转换为Word2Vec格式"""
    # 获取当前脚本的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构造GloVe文件的完整路径
    glove_path = os.path.join(script_dir, glove_file)
    print(glove_path)

    # 如果文件不存在，提供下载指南
    if not os.path.exists(glove_path):
        print(f"未找到GloVe模型文件: {glove_path}")
        print("下载指南: wget https://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip")
        return None

    # 转换为Word2Vec格式（如需）
    w2v_path = f"{glove_path}.w2v.txt"
    if convert and not os.path.exists(w2v_path):
        print(f"正在将GloVe转换为Word2Vec格式: {w2v_path}")
        from gensim.scripts.glove2word2vec import glove2word2vec
        glove2word2vec(glove_path, w2v_path)

    # 加载模型
    try:
        path_to_use = w2v_path if convert else glove_path
        print(f"正在加载GloVe模型: {path_to_use}")
        model = KeyedVectors.load_word2vec_format(path_to_use, binary=False)
        return model
    except Exception as e:
        print(f"加载GloVe模型失败: {e}")
        return None


# 7. 词类比测试
def word_analogy_test(model, a, b, c):
    """测试词类比: a:b :: c:?"""
    try:
        # 兼容处理完整模型对象和仅KeyedVectors对象
        vector_model = model.wv if hasattr(model, 'wv') else model
        result = vector_model.most_similar(positive=[b, c], negative=[a], topn=3)
        print(f"\n词类比测试: {a}:{b} :: {c}:?\n结果: {result}")
    except KeyError as e:
        print(f"词类比测试失败，可能是词汇表中缺少某些词: {e}")


# 测试词类比
word_analogy_test(word2vec_model, "man", "woman", "king")

# 尝试加载GloVe（如果文件存在）
glove_model = load_glove_model()
if glove_model:
    # 可视化GloVe词向量
    glove_words = [w for w in test_words if w in glove_model]
    if len(glove_words) >= 3:  # 确保有足够的词进行可视化
        # 合并可视化（GloVe: PCA和t-SNE对比）
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('GloVe词向量可视化对比: PCA vs t-SNE', fontsize=16)

        # PCA降维
        result_pca, _ = visualize_vectors(glove_model, glove_words, method='pca')
        ax1.scatter(result_pca[:, 0], result_pca[:, 1])
        for i, word in enumerate(glove_words):
            ax1.annotate(word, xy=(result_pca[i, 0], result_pca[i, 1]))
        ax1.set_title('GloVe - PCA降维')

        # t-SNE降维
        result_tsne, _ = visualize_vectors(glove_model, glove_words, method='tsne')
        ax2.scatter(result_tsne[:, 0], result_tsne[:, 1])
        for i, word in enumerate(glove_words):
            ax2.annotate(word, xy=(result_tsne[i, 0], result_tsne[i, 1]))
        ax2.set_title('GloVe - t-SNE降维')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        # 词类比测试
        try:
            word_analogy_test(glove_model, "man", "woman", "king")
        except:
            print("GloVe模型不支持词类比测试，需要完整模型")
