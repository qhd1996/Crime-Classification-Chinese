import torch
import torch.nn as nn
import torch.nn.functional as F

class BOW_Model(nn.Module):
    def __init__(self, config):
        super(BOW_Model, self).__init__()
        self.config = config
        self.hidden_size1 = int(self.config.hidden_size * self.config.hidden_next)
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.config.pretrained_embeddings))
        self.linear1 = nn.Linear(self.config.embed_size, self.config.hidden_size)
        self.linear2 = nn.Linear(self.config.hidden_size , self.hidden_size1)
        self.linear3 = nn.Linear(self.hidden_size1,self.config.num_classes)
        self.linear4 = nn.Linear(self.config.embed_size, self.config.num_classes)
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)

    def forward(self, train_x, train_x_len):
        train_x = train_x.cuda()
        train_x_len = train_x_len.cuda()
        embeds_x_tmp = self.embedding(train_x).float()
        embeds_x = torch.div(torch.sum(embeds_x_tmp, 1).view(-1, self.config.embed_size),train_x_len.view(-1, 1).float())
        # out1 = F.relu(self.linear1(embeds_x))
        # out2 = F.relu(self.linear2(out1))
        # out3 = self.linear3(out2)
        out3 = self.linear4(embeds_x)
        return out3.cpu()
