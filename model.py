import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, device):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.num_layers = num_layers
        self.directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim // self.directions, num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)
        self.device = device

    def forward(self, x, hidden=None):
        emb = self.embeddings(x)
        # (batch, seq, emb_dim) -> (seq, batch, emb_dim)
        emb = emb.transpose(0, 1).contiguous()
        S, B = emb.size(0), emb.size(1)
        # expected_hidden_size = (self.num_layers * num_directions, mini_batch, self.hidden_size)
        if hidden is None:
            hidden = (torch.zeros(self.num_layers * self.directions, B, self.hidden_dim // self.directions).to(self.device),
                      torch.zeros(self.num_layers * self.directions, B, self.hidden_dim // self.directions).to(self.device))

        lstm_out, hidden = self.lstm(emb, hidden)
        out = self.linear(lstm_out)
        out = out.transpose(0, 1).reshape(S * B, -1)
        return out, hidden
