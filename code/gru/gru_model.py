import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from codes.utils.utils import SpatialDropout

class GRU_Model(nn.Module):
    def __init__(self, config):
        super(GRU_Model, self).__init__()
        self.config = config
        self.embedding_dropout = SpatialDropout(0.3)
        self.gru1 = nn.GRU(self.config.embed_size, self.config.hidden_size, batch_first = True, bidirectional = True )
        self.gru2 = nn.GRU(self.config.hidden_size * 2, self.config.hidden_size, batch_first = True, bidirectional=True)
        self.hidden_size1 = int(self.config.hidden_size * self.config.hidden_next)
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.config.pretrained_embeddings))
        self.linear1 = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.linear2 = nn.Linear(self.config.hidden_size, self.hidden_size1)
        self.linear3 = nn.Linear(self.hidden_size1, self.config.num_classes)
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)

    def forward(self, train_x, train_x_len):
        train_x = train_x.cuda()
        train_x_len = train_x_len.cuda()
        embeds_x = self.embedding(train_x).float()
        embeds_x = self.embedding_dropout(embeds_x)
        # pack  sequence
        pack_x = pack_padded_sequence(embeds_x, train_x_len, batch_first=True, enforce_sorted=False)
        output_h, h_h = self.gru1(pack_x)
        # output_h, h_h = self.gru2(output_h)
        h_l = h_h[0].view(-1, self.config.hidden_size)
        h_r = h_h[1].view(-1, self.config.hidden_size)
        h_h_f = torch.cat((h_l, h_r), dim=1)
        out1 = F.relu(self.linear1(h_h_f))
        out2 = F.relu(self.linear2(out1))
        out3 = self.linear3(out2)
        return out3.cpu()
