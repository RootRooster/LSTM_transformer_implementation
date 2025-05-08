import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset


class LinearSum(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

    def forward(self, x, h1):
        return self.linear1(x) + self.linear2(h1)


class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.linear_sum1 = LinearSum(self.input_dim, self.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.linear_sum2 = LinearSum(self.input_dim, self.hidden_dim)
        self.linear_sum3 = LinearSum(self.input_dim, self.hidden_dim)
        self.linear_sum4 = LinearSum(self.input_dim, self.hidden_dim)
        self.tanh = nn.Tanh()

    def forward(self, x, C, h):
        # x - batch of encoded characters
        # C - Cell state of the previous iteration
        # h - Hidden state of the previous iteration

        # Returns: cell state C_out and the hidden state h_out

        # TODO: implement the forward pass of the LSTM cell

        # calculate the input cell
        i = self.linear_sum1(x, h)
        i = self.sigmoid(i)

        # calculate forget cell
        f = self.linear_sum2(x, h)
        f = self.sigmoid(f)

        # calculate output gates
        o = self.linear_sum3(x, h)
        o = self.sigmoid(o)
        g = self.linear_sum4(x, h)
        g = self.tanh(g)

        # update internal state
        C = f * C + i * g
        h = o * self.tanh(C)
        return C, h


class LSTMSimple(nn.Module):
    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(LSTMSimple, self).__init__()

        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.lstm_cell = LSTMCell(self.input_dim, self.hidden_dim, self.output_dim)
        self.proj = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        """
        x - One hot encoded batch - Shape: (batch, seq_len, onehot_char)

        return
        """
        batch_size = x.size(0)
        device = x.device
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        C = torch.zeros(batch_size, self.hidden_dim, device=device)

        outputs = []
        for t in range(min(x.size(1), self.seq_length)):
            x_t = x[:, t, :]
            C, h = self.lstm_cell(x_t, C, h)
            out = self.proj(h)
            outputs.append(out)
        outputs = torch.stack(outputs, dim=1)
        return outputs, (C, h)
