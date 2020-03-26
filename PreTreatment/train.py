#coding:utf-8
from pytorch_pretrained_bert import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
text = "Hey, Mary. you made it. how was your trip?"
token = tokenizer.tokenize(text)
inputs_id = tokenizer.convert_tokens_to_ids(token)

tokens = []
segment_ids = []

tokens.append("[CLS]")
for i in token:
    tokens.append(i)
    segment_ids.append(0)
tokens.append("[SEP]")
segment_ids.append(1)
print(tokens)
