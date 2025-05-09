import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import unidecode
import string
import random
from torch.autograd import Variable
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.optim as optim
import math

# set the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        # Positional encoding adds the positional information to the
        # embedding. Without it the model would not be able to differentiate
        # between different characters orders such as between "dog" and "god".
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = 10000.0 ** (torch.arange(0, d_model, 2).float() / d_model)
        print(div_term.shape)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device)
        self.pe.requires_grad = False

    def forward(self, x):
        p = self.pe[:, : x.size(1)]
        return p


class AttentionMasking(nn.Module):
    def __init__(self, max_len):
        super(AttentionMasking, self).__init__()
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len),
        )

    def forward(self, x):
        length = x.shape[-1]
        out = x.masked_fill(self.mask[:, :, :length, :length] == 0, float("-inf"))
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, max_len):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        # Multiply with an upper triangular
        # matrix of dimensions (length x length) after the scale operation
        # in Figure 2 of the paper.
        self.mask_opt = AttentionMasking(max_len)
        self.max_len = max_len

    def forward(self, q, k, v):
        # length = number of input tokens
        # batch_size, num_heads, length, num_neuron = k.size()
        qkt = torch.matmul(q, k.transpose(-2, -1))
        qkt = qkt / math.sqrt(self.max_len)
        qkt = self.mask_opt(qkt)
        qkt = self.softmax(qkt)
        return torch.matmul(qkt, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, num_neuron, n_head, max_len):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.num_neuron = num_neuron
        self.scaled_dot_product = ScaledDotProductAttention(max_len)
        self.linearq = nn.Linear(dim_model, num_neuron * n_head)
        self.lineark = nn.Linear(dim_model, num_neuron * n_head)
        self.linearv = nn.Linear(dim_model, num_neuron * n_head)
        self.linearf = nn.Linear(num_neuron * n_head, dim_model)

    def split(self, tensor):
        batch_size, length, total_dim = tensor.size()
        # Reshape the tensor to enable the use in
        # the ScaledDotProductAttention module
        split_tensor = tensor.view(
            batch_size, length, self.n_head, self.num_neuron
        ).transpose(1, 2)
        return split_tensor

    def concat(self, tensor):
        batch_size, num_heads, length, num_neuron = tensor.size()
        # Reshape the tensor to its original size before the split operation.
        concat_tensor = (
            tensor.transpose(1, 2)
            .contiguous()
            .view(batch_size, length, self.n_head * self.num_neuron)
        )
        return concat_tensor

    def forward(self, q, k, v):
        q = self.linearq(q)
        k = self.lineark(k)
        v = self.linearv(v)
        q = self.split(q)
        k = self.split(k)
        v = self.split(v)
        qkv = self.scaled_dot_product(q, k, v)
        c = self.concat(qkv)
        return self.linearf(c)


class PositionFeedForwardNet(nn.Module):
    def __init__(self, dim_model):
        super(PositionFeedForwardNet, self).__init__()
        self.ff_net1 = nn.Linear(dim_model, dim_model * 4)
        self.ff_net2 = nn.Linear(dim_model * 4, dim_model)

    def forward(self, x):
        ff_out = self.ff_net1(x)
        ff_out = torch.nn.functional.relu(ff_out)
        ff_out = self.ff_net2(ff_out)
        return ff_out


class TransformerBlock(nn.Module):
    def __init__(self, dim_model, num_neuron, n_head, max_len):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(dim_model, num_neuron, n_head, max_len)
        self.l_norm = torch.nn.LayerNorm(dim_model)
        self.l_norm2 = torch.nn.LayerNorm(dim_model)
        self.ff_net = PositionFeedForwardNet(dim_model)
        # b, len_seq, n_head, num_neuron

    def forward(self, x):
        # A Transformer block as described in the
        # Attention is all you need paper. In Figure 1 the transformer
        # block is marked with a gray rectangle right of the text "Nx"
        _x = x
        mha1 = self.mha(x, x, x)
        lnorm = self.l_norm(_x + mha1)
        _x = lnorm
        ff_out = self.ff_net(lnorm)
        out = self.l_norm2(ff_out + _x)

        return out


class TransformerSimple(nn.Module):
    def __init__(self, seq_length, input_dim, output_dim, batch_size):
        super(TransformerSimple, self).__init__()
        num_neuron = 64
        n_head = 8
        dim_model = 256
        max_len = 512
        self.start_embedding = nn.Embedding(input_dim, dim_model)

        self.pos_embedding = PositionalEncoding(dim_model)

        # b x l x c*n_head
        self.t_block1 = TransformerBlock(dim_model, num_neuron, n_head, max_len)
        self.t_block2 = TransformerBlock(dim_model, num_neuron, n_head, max_len)
        self.t_block3 = TransformerBlock(dim_model, num_neuron, n_head, max_len)
        self.t_block4 = TransformerBlock(dim_model, num_neuron, n_head, max_len)
        self.t_block5 = TransformerBlock(dim_model, num_neuron, n_head, max_len)

        # self.out_layer_1 = nn.Linear(dim_model, dim_model)
        self.output_layer = nn.Linear(dim_model, output_dim)

    def forward(self, x):
        # x - Tensor - (b, seq_len)
        # Embeds the input tensor from tokens to features
        s_emb = self.start_embedding(x)
        # Adds positional embeddings
        p_emb = self.pos_embedding(s_emb)
        b_out = p_emb + s_emb
        # Transformer blocks - You can experiment with varying depth
        # For example GPT uses 12 blocks but this might be a bit memory intensive
        b_out = self.t_block1(b_out)
        b_out = self.t_block2(b_out)
        b_out = self.t_block3(b_out)
        b_out = self.t_block4(b_out)
        b_out = self.t_block5(b_out)

        # Output mapping to a classification of output tokens
        # For each token the network tries to predict the next token
        # based only on the previous tokens.
        # Output shape: (b x seq_len x vocabulary_size)
        out = self.output_layer(b_out)

        return out


class TextDataset(Dataset):
    def __init__(self, chunk_len=200, padded_chunks=False):
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
        self.encoded_file = [self.char_dict[x] for x in self.file]

    def read_file(self, filename):
        file = unidecode.unidecode(open(filename).read())
        return file, len(file)

    def encode_text(self, in_str):
        # in_str - input sequence - String
        # Returns - in_str mapped to tokens in char_dict
        tensor = torch.LongTensor([self.char_dict[x] for x in in_str])
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
        chunk = self.encoded_file[start_index:end_index]
        # input_tokens - random sequence of tokens from the dataset
        input_tokens = torch.LongTensor(chunk[:-1])
        # target - input token sequence shifted by 1
        # the idea is to predict next token for each token in the input sequence
        # therefore if the input is [1,2,3,4] the target is [2,3,4,5]
        target = torch.LongTensor(chunk[1:])
        input_tokens = input_tokens.to(device)
        target = target.to(device)
        return input_tokens, target


def topk_sampling_iter_transformer(model, x, num_chars, chunk_len, output_token):
    # x -- b x onehot_char
    # x = b x l
    outputs = torch.zeros((1, num_chars))
    inp = x

    for t in range(num_chars):
        # b x onehot_char
        output = model(inp.long())[0, -1:]
        # output = torch.softmax(output, dim=1)
        # b x 3
        output_vals, output_ind = torch.topk(output, 5, dim=1)
        # 3 -> int
        output_vals = torch.softmax(output_vals, dim=1)
        top_ind = torch.multinomial(output_vals[0], 1)[0]
        # int
        out_char_index = output_ind[0, top_ind]
        # int -> 1
        out_char_index = torch.ones(1).to(device)

        outputs[:, t] = out_char_index.item()
        if inp.shape[1] > chunk_len:
            inp = torch.cat((inp[:, 1:], out_char_index.unsqueeze(0)), dim=1)
        else:
            inp = torch.cat((inp, out_char_index.unsqueeze(0)), dim=1)

    return outputs


def greedy_sampling_iter_transformer(model, x, num_chars, chunk_len, output_token):
    # x -- shape (batch, tokens in x)
    outputs = torch.zeros((1, num_chars))
    inp = x

    for t in range(num_chars):
        # b x l x onehot_char
        output = model(inp.long())[0, -1:]
        output = torch.softmax(output, dim=1)
        out_char_index = torch.argmax(output, dim=1)
        outputs[:, t] = out_char_index.item()
        if inp.shape[1] > chunk_len:
            inp = torch.cat((inp[:, 1:], out_char_index.unsqueeze(0)), dim=1)
        else:
            inp = torch.cat((inp, out_char_index.unsqueeze(0)), dim=1)

    return outputs


# Sample parameters, use whatever you see fit.
batch_size = 256
chunk_len = 128
train_dataset = TextDataset(chunk_len=chunk_len)
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=0
)

input_dim = train_dataset.n_characters
output_dim = train_dataset.n_characters
learning_rate = 0.0006

model = TransformerSimple(chunk_len, input_dim, output_dim, batch_size)
model.train()
model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

epochs = 50

for epoch in range(epochs):
    with tqdm(
        total=len(trainloader.dataset),
        desc="Training - Epoch: " + str(epoch) + "/" + str(epochs),
        unit="chunks",
    ) as prog_bar:
        for i, data in enumerate(trainloader, 0):
            # inputs - shape (batch_size, chunk_len) - Tensor of vocabulary tokens
            inputs = data["input"].long()
            # labels - shape (batch_size, chunk_len) - Tensor of vocabulary tokens
            labels = data["target"].long()

            optimizer.zero_grad()
            outputs = model(inputs)
            target_t = labels
            loss = criterion(
                outputs.view(inputs.shape[0] * inputs.shape[1], -1),
                target_t.view(labels.shape[0] * labels.shape[1]),
            )
            loss.backward()
            optimizer.step()
            prog_bar.set_postfix(
                **{"run:": "Transformer", "lr": learning_rate, "loss": loss.item()}
            )
            prog_bar.update(batch_size)

        # Intermediate text output
        sample_texts = [
            "What authority surfeits on",
            "I say unto you, what he hath done famously, he did it to that end:",
            "That in submission will return to us: And then, as we have ta'en the sacrament,",
        ]
        output_token = torch.zeros(1, 1, device=device)
        output_token[0, 0] = train_dataset.n_characters - 1
        print("Top-K sampling")
        for sample_text in sample_texts:
            sample_encoding = train_dataset.encode_text(sample_text)
            sample_input = Variable(sample_encoding).to(device).unsqueeze(0).long()

            # out_test= greedy_sampling_iter_transformer(model, sample_input, 400, chunk_len, output_token)[0]
            out_test = topk_sampling_iter_transformer(
                model, sample_input, 400, chunk_len, output_token
            )[0]
            out_char_index = out_test.long().detach().cpu().numpy()
            out_chars = (
                sample_text
                + " "
                + "".join([train_dataset.all_characters[i] for i in out_char_index])
            )
            print("----------------------------------------")
            print(out_chars)
