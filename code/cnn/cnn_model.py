import torch
import torch.nn as nn
from codes.utils.utils import SpatialDropout
from torch.autograd import Variable
import numpy as np

class CNN_Model(nn.Module):
    def __init__(self, config):
        super(CNN_Model, self).__init__()
        self.config = config
        self.embedding_dropout = SpatialDropout(0.3)
        self.hidden_size1 = int(self.config.hidden_size * self.config.hidden_next)
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.config.pretrained_embeddings))
        self.conv = nn.ModuleList([nn.Conv2d(1,  self.config.kernel_num, (i, self.embedding.embedding_dim)) for i in self.config.kernel_size])
        self.maxpools = [nn.MaxPool2d((self.config.max_len + 1 - i, 1)) for i in self.config.kernel_size]
        self.linear1 = nn.Linear(len(self.config.kernel_size) * self.config.kernel_num, self.config.num_classes)

    def forward(self, train_x):
        train_x = train_x.cuda()
        embeds_x = self.embedding(train_x).float()
        embeds_x = self.embedding_dropout(embeds_x).unsqueeze(1)
        x = [self.maxpools[i](torch.tanh(cov(embeds_x))).squeeze(3).squeeze(2) for i, cov in enumerate(self.conv)]  # B X Kn
        x = torch.cat(x, dim=1)  # B X Kn * len(Kz)
        out = self.linear1(x)
        return out.cpu()