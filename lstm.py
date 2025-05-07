import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset


class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        ## TODO: Initialize the necessary components

    def forward(self, x, C, h):
        # x - batch of encoded characters
        # C - Cell state of the previous iteration
        # h - Hidden state of the previous iteration

        # Returns: cell state C_out and the hidden state h_out

        # TODO: implement the forward pass of the LSTM cell
        pass


class LSTMSimple(nn.Module):
    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(LSTMSimple, self).__init__()

        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        ## TODO: Initialize the LSTM Cell and other potential necessary components
        # You can use a nn.Linear layer to project the output of the LSTMCell to
        #
        #

    def forward(self, x):
        # x - One hot encoded batch - Shape: (batch, seq_len, onehot_char)

        # Returns the predicted next character for each character in the
        # sequence (outputs), also returns the cell state and hidden state of the
        # LSTMCell call on the last character. -- outputs, (c,t)

        # TODO: Implement the forward pass over the sequenece of characters
        pass


torch.zeros()
