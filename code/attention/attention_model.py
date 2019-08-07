import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from codes.utils.utils import SpatialDropout

class Attention_Layer(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention_Layer, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask.float()
#         a[mask==0] = -np.inf
        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class Attention_Model(nn.Module):
    def __init__(self, config):
        super(Attention_Model, self).__init__()
        self.config = config
        self.embedding_dropout = SpatialDropout(0.3)
        self.gru1 = nn.GRU(self.config.embed_size, self.config.hidden_size, batch_first = True, bidirectional = True )
        self.gru2 = nn.GRU(self.config.hidden_size * 2, self.config.hidden_size, batch_first = True, bidirectional=True)
        self.hidden_size1 = int(self.config.hidden_size * self.config.hidden_next)
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.config.pretrained_embeddings))
        self.attention_layer = Attention_Layer(self.config.hidden_size * 2, self.config.max_len)
        self.linear1 = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.linear2 = nn.Linear(self.config.hidden_size, self.hidden_size1)
        self.linear3 = nn.Linear(self.hidden_size1, self.config.num_classes)
        self.linear4 = nn.Linear(self.config.hidden_size * 2, self.config.num_classes)
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        torch.nn.init.xavier_uniform_(self.linear4.weight)

    def forward(self, train_x, mask):
        train_x = train_x.cuda()
        mask = mask.cuda()
        embeds_x = self.embedding(train_x).float()
        embeds_x = self.embedding_dropout(embeds_x)
        output_h, _ = self.gru1(embeds_x)
        # output_h, _ = self.gru2(output_h)
        # attention
        atten = self.attention_layer(output_h, mask)
        # out1 = F.relu(self.linear1(atten))
        # out2 = F.relu(self.linear2(out1))
        # out3 = self.linear3(out2)
        out3 = self.linear4(atten)
        return out3.cpu()