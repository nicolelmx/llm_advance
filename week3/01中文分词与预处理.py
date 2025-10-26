# 禁用jieba的警告信息
import warnings
warnings.filterwarnings('ignore')

# 设置jieba不显示加载信息
import jieba
jieba.setLogLevel(20)  # 设置日志级别，只显示错误信息

# 安装并导入jieba库
import jieba.posseg as pseg # 用于词性标注
import jieba.analyse # 用于关键词提取

# jieba分词的三种模式
sentence = "我来到北京清华大学"

# 1. 精确模式 (默认)
seg_list_exact = jieba.cut(sentence, cut_all=False)
print("精确模式:", "/ ".join(seg_list_exact))

# 2. 全模式
seg_list_all = jieba.cut(sentence, cut_all=True)
print("全模式:", "/ ".join(seg_list_all))

# 3. 搜索引擎模式
seg_list_search = jieba.cut_for_search(sentence)
print("搜索引擎模式:", "/ ".join(seg_list_search))

# 添加自定义词典
# 案例：不加自定义词典
text = "今天我试驾了问界M9，它的鸿蒙座舱体验真不错。"
print("默认分词:", "/ ".join(jieba.cut(text)))

# 添加自定义词
jieba.add_word("问界M9")
jieba.add_word("鸿蒙座舱")
print("添加自定义词后:", "/ ".join(jieba.cut(text)))

# 停用词过滤
# 准备一个简单的停用词列表
stopwords = {'的', '了', '我', '它', '是', '也', '都'}

def filter_stopwords(words, stopwords_set):
    return [word for word in words if word not in stopwords_set]

words = jieba.cut("我的天啊，这部电影的特效真是太棒了！")
filtered_words = filter_stopwords(words, stopwords)
print("过滤停用词后:", list(filtered_words))

# 关键词提取 
document = """
人工智能（AI）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。
AI是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
"""

# 基于TF-IDF算法的关键词提取
keywords_tfidf = jieba.analyse.extract_tags(document, topK=3, withWeight=True)
print("\n关键词 (TF-IDF):", keywords_tfidf)

# 基于TextRank算法的关键词提取
keywords_textrank = jieba.analyse.textrank(document, topK=3, withWeight=True)
print("关键词 (TextRank):", keywords_textrank)