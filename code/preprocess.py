# -*- coding:utf-8 -*-

import random
import pandas as pd

# # convert label
# all_data = pd.read_csv('../data/train/all_data.csv', encoding='utf-8')
# def txt2num(text):
#     if text == '个人隐私泄露':
#         return 0
#     elif text == '非法信息技术':
#         return 1
#     elif text == '买凶杀人':
#         return 2
#     elif text == '涉黄':
#         return 3
#     elif text == '暴恐':
#         return 4
#     elif text == '政治敏感':
#         return 5
#     elif text == '金融犯罪':
#         return 6
#     elif text == '涉赌':
#         return 7
#     elif text == '涉毒':
#         return 8
#     elif text == '涉枪':
#         return 9
#     elif text == '药品':
#         return 10
#     else:
#         return 11
#
# all_data['domain'] = all_data['domain'].map(lambda x: txt2num(x))
# all_data.to_csv('../data/train/all_data_labeled.csv', encoding='utf-8', index=False)
# group = all_data.groupby(['domain'])['id'].count().values
# test_list = []
# for i in range(12):
#     candidate_list = all_data[all_data['domain'] == i]['id'].values.tolist()
#     test_ids = random.sample(candidate_list, int(group[i] * 0.2))
#     test_list.extend(test_ids)
# test_data = all_data[all_data['id'].isin(test_list)]
# train_data = all_data[~all_data['id'].isin(test_list)]
# test_data.to_csv('../data/test/test_data.csv', encoding='utf-8', index=False)
# train_data.to_csv('../data/train/train_data.csv', encoding='utf-8', index=False)

# Preprocess text
import jieba
import re
punct = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',
             '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^', '\n','®', '`', '<', '→',
             '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
             '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',
             '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼',
             '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
             'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',
             '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√','【','】','￥','$']
def preprocess_text(texts):
    stopwords = open('../data/stopwords.txt', 'r', encoding='utf-8').readlines()
    stopwords = [x.strip() for x in stopwords]
    seed_words = open('../data/seed_words.txt', 'r', encoding='utf-8').readlines()
    seed_words = [x.strip().split(',') for x in seed_words]
    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text
    texts = texts.astype(str).apply(lambda x: x.lower())
    texts = texts.astype(str).apply(lambda x: clean_special_chars(x, punct))
    texts = texts.astype(str).apply(lambda x: re.sub('[a-z0-9]', '', x))
    texts = texts.astype(str).apply(lambda x: x.replace(' ',''))
    for words in seed_words:
        for w in words:
            jieba.add_word(w)
    return texts

# all_data = pd.read_csv('../data/train/all_data_labeled.csv',encoding='utf-8')
# all_data = all_data.fillna(' ')
# texts = preprocess_text(all_data['text'])
# all_data['text'] = texts
# all_data.to_csv('../data/train/all_data_preprocessed.csv', encoding='utf-8', index=False)
# all_data = pd.read_csv('../data/train/all_data_preprocessed.csv', encoding='utf-8')
# all_data = all_data.fillna(' ')
# avg_lens = all_data['text'].map(lambda x: len(x)).mean()
# print(avg_lens)

# test_data = pd.read_csv('../data/test/test_data.csv', encoding='utf-8')
# test_data = test_data.sort_values(by=['domain'],ascending=True)
# test_data.to_csv('../data/test/test_data.csv', encoding='utf-8', index=False)
# print(test_data.iloc[352])


