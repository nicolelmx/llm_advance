import torch
from transformers import AutoTokenizer, BertModel


class MyBertGptCompare:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    def compare_load(self):
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        self.bert_model = BertModel.from_pretrained('bert-base-chinese')

    def compare_framework(self):
        pass

    def compare_function(self):
        pass