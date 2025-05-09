from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from lstm import LSTMSimple

import unidecode
import string
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader


def greedy_sampling_lstm(lstm, x, num_chars, device):
    # x -- b x onehot_char
    outputs = torch.zeros((1, num_chars, x.shape[2]))
    t_outputs, (cell_state, hidden) = lstm(x.float())
    for c in range(num_chars):
        output_tmp = torch.softmax(lstm.proj(hidden), dim=1)
        top_ind = torch.argmax(output_tmp, dim=1)[0]
        tmp = torch.zeros_like(x[:, 0, :], device=device)
        tmp[:, top_ind] = 1
        outputs[:, c] = tmp

        cell_state, hidden = lstm.lstm_cell(tmp, cell_state, hidden)
    return outputs


def topk_sampling_lstm(lstm, x, num_chars, device):
    # x -- b x onehot_char
    outputs = torch.zeros((1, num_chars, x.shape[2]))
    t_outputs, (cell_state, hidden) = lstm(x.float())
    for c in range(num_chars):
        output_vals, output_ind = torch.topk(lstm.proj(hidden), 5, dim=1)
        output_tmp = torch.softmax(output_vals, dim=1)
        top_ind = torch.multinomial(output_tmp[0], 1)[0]
        tmp = torch.zeros_like(x[:, 0, :], device=device)
        tmp[:, output_ind[0, top_ind]] = 1
        outputs[:, c] = tmp

        cell_state, hidden = lstm.lstm_cell(tmp, cell_state, hidden)

    return outputs


class LSTMDataset(Dataset):
    def __init__(self, device, chunk_len=200, padded_chunks=False):
        # Character based dataset
        dataset_path = "./input.txt"
        # The tokens in the vocabulary (all_characters)
        # are just the printable characters of the string class
        self.all_characters = string.printable
        self.n_characters = len(self.all_characters)
        # Maps characters to indices
        self.char_dict = {x: i for i, x in enumerate(self.all_characters)}
        self.file, self.file_len = self.read_file(dataset_path)
        # Sequence length of the input
        self.chunk_len = chunk_len
        self.device = device

    def read_file(self, filename):
        file = unidecode.unidecode(open(filename).read())
        return file, len(file)

    def char_tensor(self, in_str):
        # in_str - input sequence - String
        # Return one-hot encoded characters of in_str
        tensor = torch.zeros(len(in_str), self.n_characters).long()
        char_ind = [self.char_dict[c] for c in in_str]
        tensor[torch.arange(tensor.shape[0]), char_ind] = 1
        return tensor

    def __getitem__(self, idx):
        inp, target = self.get_random_text()
        return {"input": inp, "target": target}

    def __len__(self):
        return 10000

    def get_random_text(self):
        # Pick a random string of length self.chunk_len from the dataset
        start_index = np.random.randint(0, self.file_len - self.chunk_len)
        end_index = start_index + self.chunk_len + 1
        chunk = self.file[start_index:end_index]
        # One-hot encode the chosen string
        inp = self.char_tensor(chunk[:-1])
        # The target string is the same as the
        # input string but shifted by 1 character
        target = self.char_tensor(chunk[1:])
        inp = Variable(inp).to(self.device)
        target = Variable(target).to(self.device)
        return inp, target


# make sure that we can use the accelerator
device_name = "mps"
if torch.accelerator.is_available():
    device = torch.device(device_name)
else:
    raise Exception("No accelerator is available")

batch_size = 256
chunk_len = 64
model_name = "LSTM"
train_dataset = LSTMDataset(device, chunk_len=chunk_len)
trainloader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=0, drop_last=True
)

# Sample parameters, use whatever you see fit.
input_dim = train_dataset.n_characters
hidden_dim = 256
output_dim = train_dataset.n_characters
learning_rate = 0.005
model = LSTMSimple(chunk_len, input_dim, hidden_dim, output_dim, batch_size)
model.train()
model.to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

epochs = 30

for epoch in range(epochs):
    with tqdm(
        total=len(trainloader.dataset),
        desc="Training - Epoch: " + str(epoch) + "/" + str(epochs),
        unit="chunks",
    ) as prog_bar:
        for i, data in enumerate(trainloader, 0):
            inputs = data["input"].float()
            labels = data["target"].float()
            # b x chunk_len x len(dataset.all_characters)
            target = torch.argmax(labels, dim=2)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(
                outputs.view(inputs.shape[0] * inputs.shape[1], -1),
                target.view(labels.shape[0] * labels.shape[1]),
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            prog_bar.set_postfix(
                **{"run:": model_name, "lr": learning_rate, "loss": loss.item()}
            )
            prog_bar.update(batch_size)
        # Intermediate output
        sample_text = "O Romeo, wherefore art thou"
        inp = train_dataset.char_tensor(sample_text)
        sample_input = Variable(inp).to(device).unsqueeze(0).float()
        out_test = topk_sampling_lstm(model, sample_input, 300, device)[0]
        out_char_index = torch.argmax(out_test, dim=1).detach().cpu().numpy()
        out_chars = sample_text + "".join(
            [train_dataset.all_characters[i] for i in out_char_index]
        )
        print("Top-K sampling -----------------")
        print(out_chars)

        out_test = greedy_sampling_lstm(model, sample_input, 300, device)[0]
        out_char_index = torch.argmax(out_test, dim=1).detach().cpu().numpy()
        out_chars = sample_text + "".join(
            [train_dataset.all_characters[i] for i in out_char_index]
        )
        print("Greedy sampling ----------------")
        print(out_chars)
