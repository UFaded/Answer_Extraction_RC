#coding:utf-8
import torch
import jieba
import jieba.posseg as pseg
import jieba.analyse as anls

# ==========================PART ONE===================================

'''
将原词典的格式修正一下 并写入新的文件中

原词典 law.txt
更新词典 law_word.txt

最后载入新词典
'''
# file = ""
# with open("../Dataset/law.txt", "r") as f:
#     for line in f.readlines():
#         a = line.split()
#         file = file + a[0] + ' '
#         if(len(a)> 1):
#             file += a[1]
#         file += '\n'
#
# with open("../Dataset/law_word.txt", "w") as f:
#     f.write(file)

jieba.load_userdict("../Dataset/law_word.txt") #加载词典
# seg_list = jieba.cut(text)
# print("/".join(seg_list))

# ==========================PART ONE===================================

# ==========================PART TWO===================================

with open('../Dataset/data.txt') as f:
    t = f.read().encode('utf-8').decode('utf-8')
print(t)

# ==========================PART TWO===================================