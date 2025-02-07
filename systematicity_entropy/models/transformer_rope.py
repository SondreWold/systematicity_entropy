"""
The transformer model with RoPE.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(
        self,
        input_vocabulary_size: int,
        output_vocabulary_size: int,
        num_transformer_layers: int,
        hidden_size: int,
        dropout: float,
        pad_idx: int,
        rope_theta: int,
        input_bucket_size: int,
        output_bucket_size: int,
    ):
        super().__init__()
        self.input_vocabulary_size = input_vocabulary_size
        self.output_vocabulary_size = output_vocabulary_size
        self.num_transformer_layers = num_transformer_layers
        self.hidden_size = hidden_size
        self.num_heads = hidden_size // 64
        self.dropout = dropout
        self.pad_token = pad_idx

        self.positional_embedding = PositionalEncoding(hidden_size)

        self.input_embedding = nn.Embedding(
            self.input_vocabulary_size, self.hidden_size
        )
        self.output_embedding = nn.Embedding(
            self.output_vocabulary_size, self.hidden_size
        )
        self.encoder = Encoder(
            self.num_transformer_layers, self.hidden_size, self.num_heads, self.dropout, rope_theta
        )
        self.decoder = Decoder(
            self.num_transformer_layers, self.hidden_size, self.num_heads, self.dropout, rope_theta
        )
        self.projection = nn.Linear(self.hidden_size, self.output_vocabulary_size)
        self.init_weights()

    def forward(self, source, source_padding_mask, target, target_padding_mask):
        source = self.input_embedding(source)
        target = self.output_embedding(target)
        memory = self.encoder(source, source_padding_mask)
        output = self.decoder(target, target_padding_mask, memory, source_padding_mask)
        output = self.projection(output)
        return output

    def encode_source(self, source, source_padding_mask):
        source = self.input_embedding(source)
        return self.encoder(source, source_padding_mask)

    def decode_step(self, x, x_padding_mask, memory, memory_padding_mask):
        x = self.output_embedding(x)
        x = self.decoder(x, x_padding_mask, memory, memory_padding_mask)
        x = self.projection(x)
        return x

    def test_generate(
        self, source, source_mask, max_new_tokens, pad_idx, bos_idx, eos_idx, itoc
    ):
        source_encoding = self.encode_source(source, source_mask)
        target = torch.full(
            [source_encoding.size(0), 1], fill_value=bos_idx, device=source.device
        )
        stop = torch.zeros(target.size(0), dtype=torch.bool, device=target.device)
        for _ in range(max_new_tokens):
            prediction = self.decode_step(target, None, source_encoding, source_mask)[
                :, -1
            ]
            prediction = torch.where(stop, pad_idx, prediction.argmax(-1))
            stop |= prediction == eos_idx
            stop |= prediction == pad_idx
            target = torch.cat([target, prediction.unsqueeze(1)], dim=1)
            if stop.all():
                break
        sentences = []
        for batch in target:
            sent = []
            for token in batch[1:]:
                if token.item() == eos_idx:
                    break
                sent.append(token.item())
            sentences.append(sent)
        return sentences

    def init_weights(self):
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_embedding.weight.data.uniform_(-initrange, initrange)
        self.projection.bias.data.zero_()
        self.projection.weight.data.uniform_(-initrange, initrange)

    def __str__(self):
        return str(vars(self))


class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, dropout, rope_theta):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(hidden_size, num_heads, dropout, rope_theta) for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, x_padding_mask):
        for layer in self.layers:
            x = layer(x, x_padding_mask)
        x = self.output_norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, dropout, rope_theta):
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderLayer(hidden_size, num_heads, dropout, rope_theta) for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, x_padding_mask, memory, memory_padding_mask):
        attention_mask = self.get_attention_mask(x.size(1), x.size(1), x.device)
        for layer in self.layers:
            x = layer(x, x_padding_mask, memory, memory_padding_mask, attention_mask)
        x = self.output_norm(x)

        return x

    def get_attention_mask(self, query_length: int, key_length: int, device):
        return torch.triu(
            torch.full(
                (query_length, key_length), True, dtype=torch.bool, device=device
            ),
            diagonal=1,
        )


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, rope_theta):
        super().__init__()
        self.self_attention = RoPEMultiHeadAttention(hidden_size, num_heads, dropout, step_size=rope_theta)
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.feed_forward = FeedForward(hidden_size)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_padding_mask):
        x_ = self.norm_1(x)
        x = x + self.dropout(
            self.self_attention(x_, x_, x_, key_padding_mask=x_padding_mask)
        )
        x_ = self.norm_2(x)
        x = x + self.dropout(self.feed_forward(x_))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, rope_theta):
        super().__init__()
        self.self_attention = RoPEMultiHeadAttention(hidden_size, num_heads, dropout, step_size=rope_theta)
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.cross_attention = RoPEMultiHeadAttention(hidden_size, num_heads, dropout, step_size=rope_theta)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.feed_forward = FeedForward(hidden_size)
        self.norm_3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x, x_padding_mask, memory, memory_padding_mask, attention_mask=None
    ):
        x_ = self.norm_1(x)

        x = x + self.dropout(
            self.self_attention(
                x_,
                x_,
                x_,
                key_padding_mask=x_padding_mask,
                attention_mask=attention_mask,
            )
        )
        x_ = self.norm_2(x)
        x = x + self.dropout(
            self.cross_attention(
                x_, memory, memory, key_padding_mask=memory_padding_mask
            )
        )
        x_ = self.norm_3(x)
        x = x + self.dropout(self.feed_forward(x_))
        return x


# Follows "GLU Variants Improve Transformer": https://arxiv.org/abs/2002.05202
class FeedForward(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_1 = nn.Linear(hidden_size, hidden_size * 4 * 2)
        self.linear_2 = nn.Linear(hidden_size * 4, hidden_size)
        self.initialize()

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(
            self.linear_1.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
        )
        nn.init.trunc_normal_(
            self.linear_2.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
        )
        self.linear_1.bias.data.zero_()
        self.linear_2.bias.data.zero_()

    def forward(self, x):
        x = self.linear_1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert self.hidden_size % self.num_heads == 0

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(dropout)

        self.scale = 1.0 / math.sqrt(self.hidden_size // self.num_heads)
        self.initialize()

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(
            self.value.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
        )
        nn.init.trunc_normal_(
            self.output.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
        )
        nn.init.trunc_normal_(
            self.query.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
        )
        nn.init.trunc_normal_(self.key.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        self.query.bias.data.zero_()
        self.output.bias.data.zero_()

    def forward(
        self, queries, keys, values, key_padding_mask=None, attention_mask=None
    ):
        queries = self.query(queries)
        keys = self.key(keys)
        values = self.value(values)

        batch_size, key_len, hidden_size = keys.shape
        query_len = queries.size(1)

        queries = queries.view(
            batch_size, query_len, self.num_heads, hidden_size // self.num_heads
        )
        keys = keys.view(
            batch_size, key_len, self.num_heads, hidden_size // self.num_heads
        )
        values = values.view(
            batch_size, key_len, self.num_heads, hidden_size // self.num_heads
        )

        attention_weights = torch.einsum("bqhd,bkhd->bhqk", queries, keys)
        attention_weights = attention_weights * self.scale

        if key_padding_mask is not None:
            attention_weights = attention_weights.masked_fill(
                key_padding_mask.view(batch_size, 1, 1, key_len), value=float("-inf")
            )
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(
                attention_mask.view(1, 1, query_len, key_len), value=float("-inf")
            )

        attention_probs = torch.softmax(attention_weights, dim=-1)
        attention_probs = self.dropout(attention_probs)

        values = torch.einsum("bhqk,bkhd->bqhd", attention_probs, values)
        values = values.flatten(2, 3)
        values = self.output(values)
        return values


class RoPEMultiHeadAttention(nn.Module):
    def __init__(
        self, hidden_size, num_heads, dropout, max_seq_len=1_024, step_size=10_000
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert self.hidden_size % self.num_heads == 0

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(dropout)

        self.position_embedding = RotaryPositionalEmbeddings(
            self.hidden_size // self.num_heads, max_seq_len, step_size
        )

        self.scale = 1.0 / math.sqrt(self.hidden_size // self.num_heads)
        self.initialize()

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(
            self.value.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
        )
        nn.init.trunc_normal_(
            self.output.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
        )
        nn.init.trunc_normal_(
            self.query.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
        )
        nn.init.trunc_normal_(self.key.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        self.query.bias.data.zero_()
        self.output.bias.data.zero_()

    def forward(
        self, queries, keys, values, key_padding_mask=None, attention_mask=None
    ):
        queries = self.query(queries)
        keys = self.key(keys)
        values = self.value(values)

        batch_size, key_len, hidden_size = keys.shape
        query_len = queries.size(1)

        queries = queries.view(
            batch_size, query_len, self.num_heads, hidden_size // self.num_heads
        )
        queries = self.position_embedding(queries)
        keys = keys.view(
            batch_size, key_len, self.num_heads, hidden_size // self.num_heads
        )
        keys = self.position_embedding(keys)
        values = values.view(
            batch_size, key_len, self.num_heads, hidden_size // self.num_heads
        )

        attention_weights = torch.einsum("bqhd,bkhd->bhqk", queries, keys)
        attention_weights = attention_weights * self.scale

        if key_padding_mask is not None:
            attention_weights = attention_weights.masked_fill(
                key_padding_mask.view(batch_size, 1, 1, key_len), value=float("-inf")
            )
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(
                attention_mask.view(1, 1, query_len, key_len), value=float("-inf")
            )

        attention_probs = torch.softmax(attention_weights, dim=-1)
        attention_probs = self.dropout(attention_probs)

        values = torch.einsum("bhqk,bkhd->bqhd", attention_probs, values)
        values = values.flatten(2, 3)
        values = self.output(values)
        return values


class RotaryPositionalEmbeddings(nn.Module):

    def __init__(
        self, head_size: int = 64, max_seq_len: int = 1_024, step_size: int = 10_000
    ) -> None:
        super().__init__()

        self.cos_matrix: torch.Tensor
        self.sin_matrix: torch.Tensor
        thetas: torch.Tensor
        pos: torch.Tensor
        m_theta: torch.Tensor

        thetas = torch.tensor(
            [1 / (step_size ** ((2 * i) / head_size)) for i in range(head_size // 2)]
        )
        pos = torch.arange(max_seq_len)
        m_theta = torch.einsum("n, d -> nd", pos, thetas)
        m_theta = (
            torch.stack([m_theta, m_theta], dim=-1)
            .reshape(max_seq_len, head_size)
            .unsqueeze(0)
            .unsqueeze(2)
        )
        self.register_buffer("cos_matrix", m_theta.cos(), persistent=False)
        self.register_buffer("sin_matrix", m_theta.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len: int
        cos_matrix: torch.Tensor
        sin_matrix: torch.Tensor
        x_rotate_half: torch.Tensor
        out: torch.Tensor

        seq_len = x.shape[1]

        cos_matrix = self.cos_matrix[:, :seq_len]
        sin_matrix = self.sin_matrix[:, :seq_len]

        x_rotate_half = torch.cat([-x[..., x.size(-1)//2:], x[..., :x.size(-1)//2]], dim=-1)

        out = x * cos_matrix + x_rotate_half * sin_matrix

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=1024):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, hidden_size)  # shape: [T, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # shape: [T, 1]
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size)
        )  # shape: [D / 2]
        pe[:, 0::2] = torch.sin(position * div_term)  # shape: [T, D / 2]
        pe[:, 1::2] = torch.cos(position * div_term)  # shape: [T, D / 2]
        self.pe = nn.Parameter(pe.unsqueeze(0), requires_grad=False)  # shape: [1, T, D]

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


if __name__ == "__main__":
    model = Transformer(22, 32, 2, 256, 0.2, 2)
    print(model)
