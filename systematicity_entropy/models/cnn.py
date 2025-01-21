from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        dropout_p: int,
        input_vocab_size: int,
        output_vocab_size: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        pad_token: int,
        share_weights: bool = False,
    ) -> None:
        super().__init__()

        self.encoder: Encoder = Encoder(
            hidden_size,
            kernel_size,
            dropout_p,
            input_vocab_size,
            num_encoder_layers,
            num_decoder_layers,
        )
        self.decoder: Decoder = Decoder(
            hidden_size, kernel_size, dropout_p, output_vocab_size, num_decoder_layers
        )
        if share_weights:
            self.classifier: Classifier = Classifier(
                hidden_size, output_vocab_size, dropout_p, self.decoder.embedding.weight
            )
        else:
            self.classifier: Classifier = Classifier(
                hidden_size, output_vocab_size, dropout_p
            )
        self.pad_token = pad_token

    def decode_step(
        self,
        output_ids: torch.Tensor,
        encoder_outputs: torch.Tensor,
        input_embeddings: torch.Tensor,
        padding_mask: torch.Tensor
    ) -> torch.Tensor:
        decoder_outputs: torch.Tensor = self.decoder(
            output_ids, encoder_outputs, input_embeddings, padding_mask
        )

        logits: torch.Tensor = self.classifier(decoder_outputs)

        return logits

    def test_generate(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        max_new_tokens: int,
        pad_idx: int,
        bos_idx: int,
        eos_idx: int,
        itoc: dict
    ) -> torch.Tensor:
        encoder_outputs: torch.Tensor
        input_embeddings: torch.Tensor

        encoder_outputs, input_embeddings = self.encoder(input_ids)

        target = torch.full(
            [encoder_outputs.size(0), 1], fill_value=bos_idx, device=input_ids.device
        )
        stop = torch.zeros(target.size(0), dtype=torch.bool, device=target.device)
        for _ in range(max_new_tokens):
            prediction: torch.Tensor = self.decode_step(
                target, encoder_outputs, input_embeddings, padding_mask.unsqueeze(1)
            )[
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

    def forward(
        self,
        input_ids: torch.Tensor,
        output_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoder_outputs: torch.Tensor
        input_embeddings: torch.Tensor

        encoder_outputs, input_embeddings = self.encoder(input_ids)

        logits: torch.Tensor = self.decoder(
            output_ids, encoder_outputs, input_embeddings, padding_mask
        )

        predictions: torch.Tensor = self.classifier(logits)

        return predictions


class Encoder(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        dropout_p: float,
        vocab_size: int,
        num_layers: int,
        num_decoder_layers: int,
    ) -> None:
        super().__init__()

        self.embedding: nn.Embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_embedding: PositionalEncoding = PositionalEncoding(hidden_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

        self.in_proj: nn.Linear = nn.Linear(hidden_size, hidden_size)
        self.out_proj: nn.Linear = nn.Linear(hidden_size, hidden_size)

        self.layers: nn.ModuleList[EncoderLayer] = nn.ModuleList(
            [
                EncoderLayer(hidden_size, kernel_size, dropout_p, i)
                for i in range(num_layers)
            ]
        )

        self.num_decoder_layers: int = num_decoder_layers
        self.initialize(hidden_size, dropout_p)

    def initialize(self, hidden_size: int, dropout_p: float) -> None:
        nn.init.normal_(
            self.in_proj.weight, 0.0, math.sqrt((1.0 - dropout_p) / (hidden_size))
        )
        nn.init.normal_(self.out_proj.weight, 0.0, math.sqrt(1.0 / (hidden_size)))
        nn.init.normal_(self.embedding.weight, 0.0, 0.1)
        self.in_proj.bias.data.zero_()
        self.out_proj.bias.data.zero_()

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings: torch.Tensor = self.positional_embedding(self.embedding(input_ids))
        embeddings = self.dropout(embeddings)

        projected_embeddings: torch.Tensor = self.in_proj(embeddings)
        hidden_states: torch.Tensor = projected_embeddings

        x: torch.Tensor = embeddings

        for layer in self.layers:
            hidden_states = layer(hidden_states, x)
            x = hidden_states

        output: torch.Tensor = self.out_proj(x)

        output = GradMultiply.apply(output, 1.0 / (2.0 * self.num_decoder_layers))

        return output, projected_embeddings


class EncoderLayer(nn.Module):

    def __init__(
        self, hidden_size: int, kernel_size: int, dropout_p: float, layer: int
    ) -> None:
        super().__init__()

        self.dropout: nn.Dropout = nn.Dropout(dropout_p)
        if layer != 0:
            self.projection: nn.Linear = nn.Linear(hidden_size, hidden_size)
        else:
            self.projection: None = None
        self.convolution: nn.Conv1d = nn.Conv1d(
            hidden_size, 2 * hidden_size, kernel_size, padding="same"
        )
        self.activation: nn.GLU = nn.GLU(dim=-1)
        self.initialize(hidden_size, kernel_size, dropout_p)

    def initialize(self, hidden_size: int, kernel_size: int, dropout_p: float) -> None:
        nn.init.normal_(
            self.convolution.weight,
            0.0,
            math.sqrt(4 * (1.0 - dropout_p) / (kernel_size * hidden_size)),
        )
        self.convolution.bias.data.zero_()
        if self.projection is not None:
            nn.init.normal_(self.projection.weight, 0.0, math.sqrt(1 / hidden_size))
            self.projection.bias.data.zero_()

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = x.transpose(-1, -2)

        if self.projection is not None:
            residual = self.projection(residual)

        conv: torch.Tensor = self.convolution(x)
        conv = conv.transpose(-1, -2)

        activation: torch.Tensor = self.activation(conv)

        return (activation + residual) * math.sqrt(0.5)


class Decoder(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        dropout_p: float,
        vocab_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.embedding: nn.Embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_embedding: PositionalEncoding = PositionalEncoding(hidden_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

        self.in_proj: nn.Linear = nn.Linear(hidden_size, hidden_size)
        self.out_proj: nn.Linear = nn.Linear(hidden_size, hidden_size)

        self.layers: nn.ModuleList[DecoderLayer] = nn.ModuleList(
            [
                DecoderLayer(hidden_size, kernel_size, dropout_p, i)
                for i in range(num_layers)
            ]
        )
        self.initialize(hidden_size, dropout_p)

    def initialize(self, hidden_size: int, dropout_p: float) -> None:
        nn.init.normal_(
            self.in_proj.weight, 0.0, math.sqrt((1.0 - dropout_p) / (hidden_size))
        )
        nn.init.normal_(
            self.out_proj.weight, 0.0, math.sqrt((1.0 - dropout_p) / (hidden_size))
        )
        nn.init.normal_(self.embedding.weight, 0.0, 0.1)
        self.in_proj.bias.data.zero_()
        self.out_proj.bias.data.zero_()

    def forward(
        self,
        output_ids: torch.Tensor,
        encoder_outputs: torch.Tensor,
        input_embeddings: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        output_embeddings: torch.Tensor = self.positional_embedding(
            self.embedding(output_ids)
        )
        output_embeddings = self.dropout(output_embeddings)
        hidden_states: torch.Tensor = self.in_proj(output_embeddings)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                hidden_states,
                encoder_outputs,
                input_embeddings,
                output_embeddings,
                encoder_padding_mask,
            )

        output: torch.Tensor = self.out_proj(hidden_states)

        return output


class DecoderLayer(nn.Module):

    def __init__(
        self, hidden_size: int, kernel_size: int, dropout_p: float, layer: int
    ) -> None:
        super().__init__()

        self.dropout: nn.Dropout = nn.Dropout(dropout_p)
        if layer != 0:
            self.projection: nn.Linear = nn.Linear(hidden_size, hidden_size)
        else:
            self.projection = None
        self.convolution: nn.Conv1d = nn.Conv1d(
            hidden_size, 2 * hidden_size, kernel_size
        )
        self.activation: nn.GLU = nn.GLU(dim=-1)
        self.attention: Attention = Attention(hidden_size)
        self.kernel_size = kernel_size
        self.initialize(hidden_size, dropout_p)

    def initialize(self, hidden_size: int, dropout_p: float) -> None:
        nn.init.normal_(
            self.convolution.weight,
            0.0,
            math.sqrt(4 * (1.0 - dropout_p) / (self.kernel_size * hidden_size)),
        )
        self.convolution.bias.data.zero_()
        if self.projection is not None:
            nn.init.normal_(self.projection.weight, 0.0, math.sqrt(1 / hidden_size))
            self.projection.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        encoder_outputs: torch.Tensor,
        input_embeddings: torch.Tensor,
        output_embeddings: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.dropout(x)
        x = x.transpose(-1, -2)

        if self.projection is not None:
            residual = self.projection(residual)

        x = F.pad(x, (self.kernel_size - 1, 0))
        conv: torch.Tensor = self.convolution(x)
        conv = conv.transpose(-1, -2)

        activation: torch.Tensor = self.activation(conv)

        outputs: torch.Tensor = self.attention(
            activation,
            encoder_outputs,
            input_embeddings,
            output_embeddings,
            encoder_padding_mask,
        )

        return (outputs + residual) * math.sqrt(0.5)


class Classifier(nn.Module):

    def __init__(
        self: Classifier,
        hidden_size: int,
        vocab_size: int,
        dropout_p: float,
        embedding_weights: nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        self.emb_to_vocab = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_p)
        self.initialize(hidden_size, dropout_p, embedding_weights)

    def initialize(
        self: Classifier,
        hidden_size: int,
        dropout_p: float,
        embedding_weights: nn.Parameter | None,
    ) -> None:
        if embedding_weights is not None:
            self.emb_to_vocab.weight = embedding_weights
        else:
            nn.init.normal_(
                self.emb_to_vocab.weight, 0.0, math.sqrt((1 - dropout_p) / hidden_size)
            )
        self.emb_to_vocab.bias.data.zero_()

    def forward(self: Classifier, logits: torch.Tensor) -> torch.Tensor:
        predictions: torch.Tensor = self.emb_to_vocab(self.dropout(logits))
        return predictions


class Attention(nn.Module):

    def __init__(self, hidden_size: int) -> None:
        super().__init__()

        self.projection: nn.Linear = nn.Linear(hidden_size, hidden_size)
        self.initialize(hidden_size)

    def initialize(self, hidden_size: int) -> None:
        nn.init.normal_(self.projection.weight, 0.0, math.sqrt(1.0 / hidden_size))
        self.projection.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        encoder_outputs: torch.Tensor,
        input_embeddings: torch.Tensor,
        output_embeddings: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_sequence_length: int = input_embeddings.size(1)
        scale: float = float(input_sequence_length) * math.sqrt(
            1.0 / input_sequence_length
        )

        d: torch.Tensor = (self.projection(x) + output_embeddings) * math.sqrt(0.5)

        attention_weights: torch.Tensor = torch.bmm(
            d, encoder_outputs.transpose(-1, -2)
        )
        attention_weights = attention_weights.masked_fill(
            encoder_padding_mask, float("-inf")
        )

        attention_scores: torch.Tensor = F.softmax(attention_weights, dim=-1)

        output: torch.Tensor = (
            torch.bmm(
                attention_scores, input_embeddings
            )
            * scale
        )

        return (output + x) * math.sqrt(0.5)


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


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


if __name__ == "__main__":
    model = CNN(256, 5, 0.1, 22, 32, 2, 2)
    print(model)
