#!/usr/bin/env python
# coding: utf-8
# -*- coding: utf-8 -*-

import warnings

from code.gru.gru_model import GRU_Model
from code.tokenize import preprocess_text
from code.word2vec import word2vec
from code.utils.utils import Config

warnings.filterwarnings('ignore')
import gc
import numpy as np
import pandas as pd
import random
import time
import torch
from torch.nn import functional as F
import os

tags = '个人隐私泄露 非法信息技术 买凶杀人 涉黄 暴恐 政治敏感 金融犯罪 涉赌 涉毒 涉枪 药品 其他'
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def distinctConvert_np(c_list):
    '''
    1. Convert list data to numpy zero padded data, 2 distinct matrices for headlines and bodies
    2. Also outputs sequences lengths as np vector
    '''
    # Compute sequences lengths
    n_sentences = len(c_list)
    c_seqlen = []
    for i in range(n_sentences):
        c_seqlen.append(len(c_list[i]))

    c_max_len = 120

    # Convert to numpy
    count = 0
    c_np = np.zeros((n_sentences, c_max_len))
    for i in range(n_sentences):
        if (c_seqlen[i] == 0):
            c_seqlen[i] = 1
            count = count + 1
        elif c_seqlen[i] <= c_max_len:
            c_np[i, :c_seqlen[i]] = c_list[i]
        else:
            c_np[i, :] = c_list[i][:c_max_len]
            c_seqlen[i] = c_max_len

    return c_np, np.array(c_seqlen)


pre_start = time.time()
output1='../../data/models/vec.model'
output2='../../data/models/word2vec_format'
w2v = word2vec('../../data/corpus_cut.txt')
w2v.embed(output1, 300, min_count=5)
w2i = lambda w: w2v.w2i[w] if w in w2v.w2i else w2v.w2i['<unk>']
all_data = pd.read_csv('../../data/test/test_data.csv')
all_data = all_data.fillna(' ')
x = all_data['text'].values
x = [[w2i(w) for w in s] for s in preprocess_text(seed_word_path= '../../data/seed_words.txt',texts = x,output_path= 'corpus_cut.txt')]
x, x_len = distinctConvert_np(x)
x = torch.from_numpy(x).long()
x_len = torch.from_numpy(x_len).long()
print('Preprocess x done.')
gc.collect()
pre_end = time.time()
print('Preprocess done. Takes {:.2f}s'.format(pre_end - pre_start))

def test_model(model,x, x_len):
    x_test = x
    x_len_test = x_len
    N = model.config.batch_size
    model.eval()
    test_batches = int(len(x_test) // N)
    batch_start = 0
    batch_end = 0
    test_preds = np.zeros(len(x_test))
    for i in range(test_batches):
        batch_start = (i * N)
        batch_end = (i + 1) * N
        x_batch = x_test[batch_start:batch_end, :]
        x_len_batch = x_len_test[batch_start:batch_end]
        out = model(x_batch, x_len_batch)
        test_preds[batch_start: batch_end] = np.argmax(F.softmax(out.detach()), axis=1)
    if (batch_end < len(x_test)):
        x_batch = x_test[batch_end:, :]
        x_len_batch = x_len_test[batch_end:]
        out = model(x_batch, x_len_batch)
        test_preds[batch_end:] = np.argmax(F.softmax(out.detach()), axis=1)
    return test_preds

config = Config()
config.pretrained_embeddings = w2v.E
print('Model GRU')
seed_everything(15)
model = GRU_Model(config)
model.load_state_dict(torch.load('gru_model.pth')['net'])
model = model.cuda()
prediction = test_model(model, x, x_len).astype(int)
output = pd.DataFrame()
output['prediction'] = prediction
output.to_csv('../../data/output/gru_prediction.csv', encoding='utf-8', index=False)
