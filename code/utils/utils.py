import torch.nn as nn

class Config:
    pretrained_embeddings = []  # not passed to config   - assigned in get_data
    hidden_size = 300
    hidden_next = 0.25
    num_classes = 12
    lr = 0.001
    num_epochs = 500
    batch_size = 32
    embed_size = 300
    max_len = 120
    kernel_num = 128
    fixed_length = 120
    kernel_size = [2, 5, 10]

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x