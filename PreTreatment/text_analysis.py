import torch
import pandas as pd
import json
import collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


f = open("../Dataset/small_train_data.json", encoding='utf-8')
#设置以utf-8解码模式读取文件，encoding参数必须设置，否则默认以gbk模式读取文件，当文件中包含中文时，会报错

dataset = json.load(f)['data']
# 查看总案例数
size = len(dataset) #2000

'''
从结构体中分离出: domain context qas 它们在序列上一一对应
context: 案例内容
qas: 包含了五个小问题
domain: 案件类型
'''
domain, context, qas = [], [], []
context_len, answers_len, question_len, answer_type = [], [], [], []

for item in dataset:
    domain.append(item['domain'])
    paragraphs = item['paragraphs'][0]
    text = paragraphs['context']
    qa = paragraphs['qas']

    context_len.append(len(text))
    for q in qa:
        question_len.append(len(q['question']))
        if(q['is_impossible'] == 'false'):
            answers_len.append(len(q['answers'][0]['text']))
            if(q['answers'][0]['text'] == 'YES'):
                answer_type.append('YES')
            elif(q['answers'][0]['text'] == 'NO'):
                answer_type.append('NO')
            else:
                answer_type.append('ELSE')
        else:
            answer_type.append('FALSE')

    context.append(text)
    qas.append(qa)

count_domain = collections.Counter(domain) #Counter({'civil': 1000, 'criminal': 1000})
count_answer_type = collections.Counter(answer_type)
print(min(question_len), max(question_len))
print(min(answers_len), max(answers_len))
print(min(context_len), max(context_len))
# print(count_contexts_len) #把材料的长度分成区间展示
# c_most = count_contexts_len.most_common(5) #输出出现频率最高的前5名
# m = max(count_contexts_len, key=(lambda x:count_contexts_len[x]))
# t = min(count_contexts_len, key=(lambda x:count_contexts_len[x]))
# print(m, t) #999 265
# print(c_most)


#答案类型图标
answer_data = list(count_answer_type.values())
answer_names = list(count_answer_type.keys())
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots()
ax.barh(answer_names, answer_data)

# plt.show()
