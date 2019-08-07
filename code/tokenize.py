# -*- coding:utf-8 -*-


import pandas as pd
import jieba
import re

from codes.word2vec import word2vec

r='[’!"#$%&\'()（）：【】？！、；《》～％￥%•#*+,-./:;<=>?@[\\]^_`……‘’“”，。{|}~]+'
def load_text_and_classification(filename):
    file = pd.read_csv(filename,usecols=['text','domain'])
    content = []
    classification = []
    for index,row in file.iterrows():
        t = str(row['text'])
        content.append(t)
        d = row['domain']
        classification.append(d)
    return content,classification

def preprocess_text(seed_word_path,texts,output_path):
    seed_words = open(seed_word_path,'r',encoding='utf-8').readlines()
    seed_words = [x.strip().split(',') for x in seed_words]
    data = []
    for words in seed_words:
        for w in words:
            jieba.add_word(w)
        for i,line in enumerate(texts):
            if i % 500 == 0:
                print(i,line)
            line = line.strip().replace('\t',' ').replace('\r',' ').replace('\n',' ')
            line = re.sub(r,' ',line)
            words = jieba.lcut(line)
            while ' ' in words:
                words.remove(' ')
            data.append(' '.join(words))
        with open(output_path ,'w',encoding='utf-8') as f:
            for d in data:
                f.write(d + '\n')
        return data
#
# texts,y = load_text_and_classification('../data/train/corpus_data.csv')
# corpus = preprocess_text(seed_word_path= '../data/seed_words.txt',texts = texts,output_path= '../data/models/corpus_cut.txt')
# from gensim.models import word2vec
# s = word2vec.Text8Corpus('../data/models/corpus_cut.txt')
# model=word2vec.Word2Vec(s,size=300, window=10, min_count=5,iter=30)
output1='../data/models/vec.model'
output2='../data/models/word2vec_format'
# model.save(output1)
# model.wv.save_word2vec_format(output2, binary=False)
# model = word2vec('../data/models/corpus_cut.txt')
# model.embed(output1, 300, min_count=5)
# print(len(model.E))
# print(model.w2i)
# w2i = lambda w: model.w2i[w] if w in model.w2i else model.w2i['<unk>']
# x = [[w2i(w) for w in s] for s in preprocess_text(seed_word_path= '../data/seed_words.txt',texts = texts,output_path= '../data/model/corpus_cut.txt')]