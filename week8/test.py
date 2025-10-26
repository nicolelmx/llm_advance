import json

with open("data.json", "r", encoding="utf-8") as f:
    # json.load() 将 JSON 字符串转换为 Python 字典/列表
    training_data = json.load(f)
# 构建词表/字表 和 标签表
word_to_ix = {}
tag_to_ix = {}
for question in training_data["questions"]:
    for entity in question['entities']:
        for word in entity['text']:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        for tag in entity['label']:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

# 添加特殊token: PAD 用于填充, UNK 用于未知词
word_to_ix["<PAD>"] = len(word_to_ix)
word_to_ix["<UNK>"] = len(word_to_ix)
ix_to_tag = {v: k for k, v in tag_to_ix.items()}