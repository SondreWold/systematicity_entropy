"""
The transformer model.
"""

from __future__ import annotations

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

        self.input_embedding = nn.Embedding(
            self.input_vocabulary_size, self.hidden_size
        )
        self.output_embedding = nn.Embedding(
            self.output_vocabulary_size, self.hidden_size
        )
        self.source_positional_embedding = RelativePositionalEmbeddings(
            hidden_size, position_bucket_size=input_bucket_size
        )
        self.encoder = Encoder(
            self.num_transformer_layers, self.hidden_size, self.num_heads, self.dropout, input_bucket_size
        )
        self.target_positional_embedding = RelativePositionalEmbeddings(
            hidden_size, position_bucket_size=output_bucket_size
        )
        self.decoder = Decoder(
                self.num_transformer_layers, self.hidden_size, self.num_heads, self.dropout, input_bucket_size, output_bucket_size
        )
        self.projection = nn.Linear(self.hidden_size, self.output_vocabulary_size)
        self.init_weights()

    def forward(self, source, source_padding_mask, target, target_padding_mask):
        source = self.input_embedding(source)
        target = self.output_embedding(target)
        source_relative_embeddings = self.source_positional_embedding()
        memory = self.encoder(source, source_padding_mask, source_relative_embeddings)
        target_relative_embeddings = self.target_positional_embedding()
        output = self.decoder(
            target,
            target_padding_mask,
            target_relative_embeddings,
            memory,
            source_padding_mask,
            source_relative_embeddings,
        )
        output = self.projection(output)
        return output

    def encode_source(self, source, source_padding_mask):
        source = self.input_embedding(source)
        source_relative_embeddings = self.source_positional_embedding()
        return self.encoder(source, source_padding_mask, source_relative_embeddings)

    def decode_step(self, x, x_padding_mask, memory, memory_padding_mask):
        x = self.output_embedding(x)
        source_relative_embeddings = self.source_positional_embedding()
        target_relative_embeddings = self.target_positional_embedding()
        x = self.decoder(
            x,
            x_padding_mask,
            target_relative_embeddings,
            memory,
            memory_padding_mask,
            source_relative_embeddings,
        )
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
    def __init__(self, num_layers, hidden_size, num_heads, dropout, bucket_size):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(hidden_size, num_heads, dropout, bucket_size) for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, x_padding_mask, relative_embeddings):
        for layer in self.layers:
            x = layer(x, x_padding_mask, relative_embeddings)
        x = self.output_norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, dropout, input_bucket_size, output_bucket_size):
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderLayer(hidden_size, num_heads, dropout, input_bucket_size, output_bucket_size) for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x,
        x_padding_mask,
        x_relative_embeddings,
        memory,
        memory_padding_mask,
        memory_relative_embeddings,
    ):
        attention_mask = self.get_attention_mask(x.size(1), x.size(1), x.device)
        for layer in self.layers:
            x = layer(
                x,
                x_padding_mask,
                x_relative_embeddings,
                memory,
                memory_padding_mask,
                memory_relative_embeddings,
                attention_mask,
            )
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
    def __init__(self, hidden_size, num_heads, dropout, bucket_size):
        super().__init__()
        self.self_attention = RelativeMultiHeadAttention(
            hidden_size,
            num_heads,
            dropout,
            query_position_bucket_size=bucket_size,
            key_position_bucket_size=bucket_size,
        )
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.feed_forward = FeedForward(hidden_size)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_padding_mask, relative_embeddings):
        x_ = self.norm_1(x)
        x = x + self.dropout(
            self.self_attention(
                x_,
                x_,
                x_,
                relative_embeddings,
                relative_embeddings,
                key_padding_mask=x_padding_mask,
            )
        )
        x_ = self.norm_2(x)
        x = x + self.dropout(self.feed_forward(x_))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, input_bucket_size, output_bucket_size):
        super().__init__()
        self.self_attention = RelativeMultiHeadAttention(
            hidden_size,
            num_heads,
            dropout,
            query_position_bucket_size=output_bucket_size,
            key_position_bucket_size=output_bucket_size,
        )
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.cross_attention = RelativeMultiHeadAttention(
            hidden_size,
            num_heads,
            dropout,
            query_position_bucket_size=output_bucket_size,
            key_position_bucket_size=input_bucket_size,
        )
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.feed_forward = FeedForward(hidden_size)
        self.norm_3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        x_padding_mask,
        x_relative_embeddings,
        memory,
        memory_padding_mask,
        memory_relative_embeddings,
        attention_mask=None,
    ):
        x_ = self.norm_1(x)

        x = x + self.dropout(
            self.self_attention(
                x_,
                x_,
                x_,
                x_relative_embeddings,
                x_relative_embeddings,
                key_padding_mask=x_padding_mask,
                attention_mask=attention_mask,
            )
        )
        x_ = self.norm_2(x)
        x = x + self.dropout(
            self.cross_attention(
                x_,
                memory,
                memory,
                x_relative_embeddings,
                memory_relative_embeddings,
                key_padding_mask=memory_padding_mask,
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


class RelativeMultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        dropout,
        query_position_bucket_size=4,
        key_position_bucket_size=16,
        max_sequence_length=1_024,
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

        self.query_position_bucket_size = query_position_bucket_size
        self.key_position_bucket_size = key_position_bucket_size
        self.register_position_indices(max_sequence_length, "query")
        self.register_position_indices(max_sequence_length, "key")

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

    def register_position_indices(self, max_sequence_length: int, prefix: str) -> None:
        position_indices: torch.Tensor

        position_indices = torch.arange(
            max_sequence_length, dtype=torch.long
        ).unsqueeze(1) - torch.arange(max_sequence_length, dtype=torch.long).unsqueeze(
            0
        )
        position_indices = self.make_log_bucket_position(
            position_indices,
            getattr(self, f"{prefix}_position_bucket_size"),
            max_sequence_length,
        )
        position_indices = (
            getattr(self, f"{prefix}_position_bucket_size") - 1 + position_indices
        )
        self.register_buffer(
            f"{prefix}_position_indices", position_indices, persistent=False
        )

    def make_log_bucket_position(self, relative_pos, bucket_size, max_position):
        sign: torch.Tensor
        mid: int
        abs_pos: torch.Tensor
        log_pos: torch.Tensor
        bucket_pos: torch.Tensor

        sign = torch.sign(relative_pos)
        mid = bucket_size // 2
        abs_pos = torch.where(
            (relative_pos < mid) & (relative_pos > -mid),
            mid - 1,
            torch.abs(relative_pos).clamp(max=max_position - 1),
        )
        log_pos = (
            torch.ceil(
                torch.log(abs_pos / mid)
                / math.log((max_position - 1) / mid)
                * (mid - 1)
            ).int()
            + mid
        )
        bucket_pos = torch.where(abs_pos <= mid, relative_pos, log_pos * sign).long()

        return bucket_pos

    def create_relative_embeddings(
        self,
        query_relative_embeddings: torch.Tensor,
        key_relative_embeddings: torch.Tensor,
        query_length: int,
        key_length: int,
        head_size: int,
    ) -> torch.Tensor:
        relative_query_pos: torch.Tensor
        relative_key_pos: torch.Tensor

        if self.query_position_indices.size(0) < query_length:
            self.register_position_indices(query_length, "query")

        if self.key_position_indices.size(0) < key_length:
            self.register_position_indices(key_length, "key")

        relative_query_pos = self.query(self.dropout(query_relative_embeddings))
        relative_query_pos = F.embedding(
            self.query_position_indices[:query_length, :key_length], relative_query_pos
        )
        relative_query_pos = relative_query_pos.view(
            query_length, key_length, self.num_heads, head_size
        )

        relative_key_pos = self.key(self.dropout(key_relative_embeddings))
        relative_key_pos = F.embedding(
            self.key_position_indices[:query_length, :key_length], relative_key_pos
        )
        relative_key_pos = relative_key_pos.view(
            query_length, key_length, self.num_heads, head_size
        )

        return relative_query_pos, relative_key_pos

    def add_relative_embeddings(
        self,
        attention_scores: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        relative_query_pos: torch.Tensor,
        relative_key_pos: torch.Tensor,
    ) -> torch.Tensor:
        attention_scores.add_(
            torch.einsum("bqhd, qkhd -> bhqk", query, relative_key_pos * self.scale)
        )
        attention_scores.add_(
            torch.einsum("bkhd, qkhd -> bhqk", key * self.scale, relative_query_pos)
        )

        return attention_scores

    def forward(
        self,
        queries,
        keys,
        values,
        query_relative_embeddings,
        key_relative_embeddings,
        key_padding_mask=None,
        attention_mask=None,
    ):
        queries = self.query(queries)
        keys = self.key(keys)
        values = self.value(values)

        batch_size, key_len, hidden_size = keys.shape
        query_len = queries.size(1)

        relative_query_pos, relative_key_pos = self.create_relative_embeddings(
            query_relative_embeddings,
            key_relative_embeddings,
            query_len,
            key_len,
            hidden_size // self.num_heads,
        )

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
        attention_weights = self.add_relative_embeddings(
            attention_weights, queries, keys, relative_query_pos, relative_key_pos
        )

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


class RelativePositionalEmbeddings(nn.Module):

    def __init__(self, hidden_size: int, position_bucket_size: int = 8) -> None:
        super().__init__()

        self.relative_embedding: nn.Parameter
        self.relative_norm: nn.LayerNorm

        self.relative_embedding = nn.Parameter(
            torch.empty(2 * position_bucket_size - 1, hidden_size)
        )
        self.relative_norm = nn.LayerNorm(
            hidden_size, eps=1e-7, elementwise_affine=True
        )

        self.initialize(hidden_size, position_bucket_size)

    @torch.no_grad
    def initialize(self, hidden_size: int, bucket_size: int):
        std: float

        std = math.sqrt(2.0 / (hidden_size + (2 * bucket_size - 1)))
        nn.init.trunc_normal_(
            self.relative_embedding, mean=0.0, std=std, a=-2 * std, b=2 * std
        )

    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
        relative_embeddings: torch.Tensor

        relative_embeddings = self.relative_embedding
        relative_embeddings = self.relative_norm(relative_embeddings)

        return relative_embeddings


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
