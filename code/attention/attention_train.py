#!/usr/bin/env python
# coding: utf-8
# -*- coding: utf-8 -*-

import warnings

from code.attention.attention_model import Attention_Model
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
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
import os
from sklearn.metrics import classification_report

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
    mask = np.zeros((n_sentences, c_max_len))
    for i in range(n_sentences):
        if (c_seqlen[i] == 0):
            mask[i,0] = 1
            count = count + 1
        elif c_seqlen[i] <= c_max_len:
            c_np[i, :c_seqlen[i]] = c_list[i]
            mask[i, :c_seqlen[i]] = 1
        else:
            c_np[i, :] = c_list[i][:c_max_len]
            mask[i,:] = 1

    return c_np, mask

pre_start = time.time()
print('Start Preprocessing')
output1='../../data/models/vec.model'
output2='../../data/models/word2vec_format'
w2v = word2vec('../../data/corpus_cut.txt')
w2v.embed(output1, 300, min_count=5)
w2i = lambda w: w2v.w2i[w] if w in w2v.w2i else w2v.w2i['<unk>']
all_data = pd.read_csv('../../data/train/train_data.csv')
all_data = all_data.fillna(' ')
shuffle_indices = np.random.permutation(np.arange(len(all_data)))
x = all_data['text'].values
y = all_data['domain'].values
y = torch.from_numpy(y).long()
x = [[w2i(w) for w in s] for s in preprocess_text(seed_word_path= '../../data/seed_words.txt',texts = x,output_path= 'corpus_cut.txt')]
x, mask = distinctConvert_np(x)
x = torch.from_numpy(x).long()
mask = torch.from_numpy(mask).long()
y = torch.from_numpy(np.array(y)).long()
print('Preprocess x,y done.')
gc.collect()
pre_end = time.time()
print('Preprocess done. Takes {:.2f}s'.format(pre_end - pre_start))

def custom_loss(predictions, targets):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    cross_entrophy_loss = nn.CrossEntropyLoss()(predictions, targets)
    return cross_entrophy_loss

def evaluate(prediction, target):
    report = classification_report(target, prediction, labels=labels, target_names=tags.split(), digits=4, output_dict = True)
    print(report)
    return report

def train_model(model, x, mask, y, loss_fn):
    x_train = x
    y_train = y
    train_dataset = data.TensorDataset(x_train, mask, y_train)
    for epoch in range(model.config.num_epochs):
        param_lrs = [{'params': param, 'lr': model.config.lr} for param in model.parameters()]
        optimizer = torch.optim.Adam(param_lrs, lr = model.config.lr * 0.8 ** (epoch // 50))
        start = time.time()
        print('Epoch {} starts.'.format(epoch + 1))
        train_loader = DataLoader(dataset=train_dataset, batch_size=model.config.batch_size, shuffle=True)
        model.train()
        avg_loss = 0
        # run batches
        count = 0
        optimizer.zero_grad()
        for i, x_y_data in enumerate(train_loader):
            x_train, mask, y_train = x_y_data
            out = model(x_train, mask)
            loss = loss_fn(out, y_train)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            count += 1
        print('average loss:', avg_loss / count)
    state_dict = {
        'net': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    model_version = "attention_model"  + ".pth"
    torch.save(state_dict, model_version)
    return

config = Config()
config.pretrained_embeddings = w2v.E
config.num_epochs = 100
print('Model Attention')
seed_everything(15)
model = Attention_Model(config)
model = model.cuda()
train_model(model, x, mask, y, custom_loss)
