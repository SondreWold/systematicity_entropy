"""
MIT License

Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import abc

import six
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 64


class AbsSeq2Seq(six.with_metaclass(abc.ABCMeta, nn.Module)):
    """Abstract Seq2Seq base model

    Args:
        vocab_size (int): Laguage vocab size
        encoder_hidden_size (int): Encoder embedding dimensionality
        decoder_hidden_size (int): Decoder embedding dimensionality
        layer_type (str): Type of recurrent layer to be used
        use_attention (bool): Decoder uses attentive mechanisms
        drop_rate (0 <= float <= 1): Dropout rate to use in encoder / decoder
        bidirectional (bool): Encoder uses bidirectional mechanisms
        num_layers (int): number of hidden layers in recurrent modules
    """

    def __init__(
        self,
        vocab_size,
        encoder_hidden_size,
        decoder_hidden_size,
        layer_type,
        use_attention,
        drop_rate,
        bidirectional,
        num_layers,
        bos_idx,
        eos_idx,
    ):
        super(AbsSeq2Seq, self).__init__()

        self.max_length = MAX_LENGTH
        self.input_size = vocab_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        if layer_type not in ["GRU", "LSTM", "RNN", "GRNN", "GGRU", "GLSTM"]:
            raise NotImplementedError(
                "Supported cells: '(G)RNN', '(G)GRU', or '(G)LSTM'"
            )
        self.layer_type = layer_type
        self.output_size = self.input_size
        self.use_attention = use_attention
        self.drop_rate = drop_rate
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    @abc.abstractmethod
    def forward(
        self,
        input_tensor,
        syntax_tensor=None,
        target_tensor=None,
        use_teacher_forcing=False,
    ):
        """Forward pass for model. Conditions on sentence in input language and generates a sentence in output language

        Args:
            input_tensor (torch.tensor): Tensor representation (1-hot) of sentence in input language
            syntax_tensor (torch.tensor, optional): Tensor representation (1-hot) of sentence in input language syntax
            target_tensor (torch.tensor, optional): Tensor representation (1-hot) of target sentence in output language
            use_teacher_forcing (bool, optional): Indicates if true word is used as input to decoder. Default: False
        Returns:
            (torch.tensor) Tensor representation (softmax probabilities) of target sentence in output language
        """

    def init_forward_pass(self):
        """Helper function initialize required objects on forward pass"""
        encoder_hidden = self.encoder.init_hidden()
        decoder_outputs = torch.zeros(
            MAX_LENGTH, self.decoder.output_size, device=device
        )
        decoder_input = torch.tensor([[self.bos_idx]], device=device)
        return encoder_hidden, decoder_input, decoder_outputs

    def _arrange_hidden(self, hidden):
        """Reshape and rearrange the final encoder state for initialization of decoder state

        Args:
            hidden: (torch.tensor) Final output of the encoder
        Returns:
            (torch.tensor) Final output arranged for decoder
        """
        hidden = hidden.view(self.num_layers, self.num_directions, 1, -1)
        return hidden[-1].view(1, 1, -1)

    def _transfer_hidden(self, hidden):
        """Rearrange the final hidden state from encoder to be passed to decoder

        Args:
            hidden: (torch.tensor) final hidden state of encoder module
        Returns:
            (torch.tensor) initial hidden state to be passed to decoder
        """
        if "GRU" in self.layer_type or "RNN" in self.layer_type:
            return self._arrange_hidden(hidden)
        elif "LSTM" in self.layer_type:
            return tuple([self._arrange_hidden(h) for h in hidden])
        else:
            raise NotImplementedError

    def check_target_length(self, target_tensor):
        """Helper function to determine length of input sequence"""
        if target_tensor is not None:
            target_length = target_tensor.size(0)
            assert target_length <= MAX_LENGTH, print(
                "Max length exceeded. Max Length: %s, Target length: %s"
                % (MAX_LENGTH, target_length)
            )
            return target_length
        else:
            return None

    @property
    def num_params(self):
        """Count number of parameters in model"""
        count = 0
        for param in self.parameters():
            count += torch.tensor(param.shape).prod()
        return count


class BasicSeq2Seq(AbsSeq2Seq):
    """Standard implementation of sequence to sequence model for translation (potentially using attention)

    Args:
        vocab_size (int): Source language
        encoder_hidden_size (int): Encoder embedding dimensionality
        decoder_hidden_size (int): Decoder embedding dimensionality
        layer_type (str): Type of recurrent layer to be used
        use_attention (bool): Decoder uses attentive mechanisms
        drop_rate (0 <= float <= 1): Dropout rate to use in encoder / decoder
        bidirectional (bool): Encoder uses bidirectional mechanisms
        num_layers (int): number of hidden layers in recurrent modules
    """

    def __init__(
        self,
        vocab_size,
        encoder_hidden_size,
        decoder_hidden_size,
        layer_type,
        use_attention,
        drop_rate,
        bidirectional,
        num_layers,
        bos_token_idx,
        eos_token_idx,
    ):
        super(BasicSeq2Seq, self).__init__(
            vocab_size=vocab_size,
            encoder_hidden_size=encoder_hidden_size,
            decoder_hidden_size=decoder_hidden_size,
            layer_type=layer_type,
            use_attention=use_attention,
            drop_rate=drop_rate,
            bidirectional=bidirectional,
            num_layers=num_layers,
            bos_idx=bos_token_idx,
            eos_idx=eos_token_idx,
        )

        self.encoder = EncoderRNN(
            input_size=self.input_size,
            hidden_size=self.encoder_hidden_size,
            layer_type=self.layer_type,
            bidirectional=self.bidirectional,
            num_layers=self.num_layers,
        )
        if self.use_attention:
            self.decoder = AttnDecoderRNN(
                hidden_size=self.decoder_hidden_size * self.num_directions,
                output_size=self.output_size,
                layer_type=self.layer_type,
                dropout_p=self.drop_rate,
            )
        else:
            self.decoder = DecoderRNN(
                hidden_size=self.decoder_hidden_size,
                output_size=self.output_size,
                layer_type=self.layer_type,
            )

    def forward(
        self,
        input_tensor,
        syntax_tensor=None,
        target_tensor=None,
        use_teacher_forcing=False,
        target_length=None,
    ):
        """Forward pass for model. Conditions on sentence in input language and generates a sentence in output language

        Args:
            input_tensor (torch.tensor): Tensor representation (1-hot) of sentence in input language
            syntax_tensor (torch.tensor, optional): Tensor representation (1-hot) of sentence in input language syntax
            target_tensor (torch.tensor, optional): Tensor representation (1-hot) of target sentence in output language
            use_teacher_forcing (bool, optional): Indicates if true word is used as input to decoder. Default: False
        Returns:
            (torch.tensor) Tensor representation (softmax probabilities) of target sentence in output language
        """
        # Some bookkeeping and preparaion for forward pass
        encoder_hidden, decoder_input, decoder_outputs = self.init_forward_pass()

        encoder_output, encoder_hidden, _ = self.encoder(input_tensor, encoder_hidden)
        decoder_hidden = self._transfer_hidden(encoder_hidden)

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            target_length = self.check_target_length(target_tensor)
            for di in range(target_length):
                if self.use_attention:
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                        decoder_input, decoder_hidden, encoder_output
                    )
                else:
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden
                    )

                decoder_outputs[di] = decoder_output
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            length = target_length if target_length else self.max_length
            for di in range(length):
                if self.use_attention:
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                        decoder_input, decoder_hidden, encoder_output
                    )
                else:
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden
                    )
                topv, topi = decoder_output.topk(1)
                decoder_outputs[di] = decoder_output
                decoder_input = topi.squeeze().detach()  # detach from history as input

                if decoder_input.item() == self.eos_idx:
                    break

        return decoder_outputs


def get_cell_type(cell_type):
    assert cell_type in [
        "RNN",
        "GRU",
        "LSTM",
    ], "Please specify layer type as 'GRU' or 'LSTM'"
    if cell_type == "RNN":
        rnn_model = nn.RNN
    elif cell_type == "GRU":
        rnn_model = nn.GRU
    elif cell_type == "LSTM":
        rnn_model = nn.LSTM
    return rnn_model


class EncoderRNN(nn.Module):
    """Standard RNN encoder (using GRU hidden units in recurrent network)

    Args:
        input_size (int): Dimensionality of elements in input sequence number of words in input language)
        semantic_n_hidden (int): Dimensionality of semantic embeddings
        hidden_size (int): Dimensionality of elements in output sequence (number of units in layers)
        layer_type (str, optional): Specifies type of RNN layer to be used ('GRU' or 'LSTM'). Default: GRU
        bidirectional (bool, optional) Indicate whether to use a bi-directional encoder (slower). Default: False
        num_layers: (int, optional) Number of layers to place in encoder model. Default: 1
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        layer_type="GRU",
        semantic_n_hidden=120,
        bidirectional=False,
        num_layers=1,
    ):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.semantic_n_hidden = semantic_n_hidden
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.semantic_embedding = nn.Embedding(input_size, self.semantic_n_hidden)
        self.rnn_model = get_cell_type(layer_type)
        self.rnn = self.rnn_model(
            hidden_size,
            hidden_size,
            bidirectional=bidirectional,
            num_layers=self.num_layers,
        )

    def forward(self, input, hidden):
        """Forward pass through RNN-based encoder

        Args:
            input (torch.tensor): batch x length x input_size tensor of inputs to decoder
            hidden (torch.tensor): batch x 1 x hidden_size tensor of initial hidden states
        Returns:
            (tuple:: torch.tensor, torch.tensor, torch.tensor) decoder output, hidden states, semantic embeddings
        """
        embedded = self.embedding(input.squeeze())[:, None, :]
        semantics = self.semantic_embedding(input.squeeze())
        output = embedded
        output, hidden = self.rnn(output, hidden)
        return output, hidden, semantics

    def init_hidden(self):
        """Initialize the weights of the recurrent model"""
        if isinstance(self.rnn, nn.GRU) or isinstance(self.rnn, nn.RNN):
            return torch.zeros(
                self.num_directions * self.num_layers,
                1,
                self.hidden_size,
                device=device,
            )
        elif isinstance(self.rnn, nn.LSTM):
            return (
                torch.zeros(
                    self.num_directions * self.num_layers,
                    1,
                    self.hidden_size,
                    device=device,
                ),
                torch.zeros(
                    self.num_directions * self.num_layers,
                    1,
                    self.hidden_size,
                    device=device,
                ),
            )



class DecoderRNN(nn.Module):
    """Standard RNN decoder (using GRU hidden units in recurrent network)

    Args:
        hidden_size (int): Dimensionality of elements in input sequence
        output_size (int): Dimensionality of elements in output sequence (number of words in output language)
        layer_type (str, optional): Specifies type of RNN layer to be used ('GRU' or 'LSTM'). Default: GRU
    """

    def __init__(self, hidden_size, output_size, layer_type="GRU"):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(self.output_size, hidden_size)
        self.rnn_model = get_cell_type(layer_type)
        self.rnn = self.rnn_model(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        """Forward pass through RNN based decoder

        Args:
            input (torch.tensor): batch x length x input_size tensor of inputs to decoder
            hidden (torch.tensor): batch x 1 x hidden_size tensor of initial hidden states
        Returns:
            (tuple:: torch.tensor, torch.tensor, torch.tensor) decoder output, hidden states, semantic embeddings
        """
        embedded = self.embedding(input).view(1, 1, -1)
        output = F.relu(embedded)
        output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        """Initialize the weights of the recurrent model"""
        if isinstance(self.rnn, nn.GRU) or isinstance(self.rnn, nn.RNN):
            return torch.zeros(
                self.num_directions * self.num_layers,
                1,
                self.hidden_size,
                device=device,
            )
        elif isinstance(self.rnn, nn.LSTM):
            return (
                torch.zeros(
                    self.num_directions * self.num_layers,
                    1,
                    self.hidden_size,
                    device=device,
                ),
                torch.zeros(
                    self.num_directions * self.num_layers,
                    1,
                    self.hidden_size,
                    device=device,
                ),
            )


class AttnDecoderRNN(nn.Module):
    """RNN decoder (using GRU hidden units in recurrent network) with attention mechanism on the input sequence

    Args:
        hidden_size (int): Dimensionality of elements in input sequence
        output_size (int): Dimensionality of elements in output sequence (number of words in output language)
        layer_type (str, optional): Specifies type of RNN layer to be used ('GRU' or 'LSTM'). Default: GRU
        dropout_p (float, optional): Dropout rate for embeddings. Default: 0.1
        max_length (int, optional): Maximum allowable length of input sequence (required for attention).
                                    Default: MAX_LENGTH
    """

    def __init__(
        self,
        hidden_size,
        output_size,
        layer_type="GRU",
        dropout_p=0.1,
        max_length=MAX_LENGTH,
    ):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.rnn_model = get_cell_type(layer_type)
        self.rnn = self.rnn_model(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        """Forward pass through attention decoder

        Args:
            input (torch.tensor): batch x length x input_size tensor of inputs to decoder
            hidden (torch.tensor): batch x 1 x hidden_size tensor of initial hidden states
            encoder_outputs (torch.tensor): batch x length x hidden_size tensor of encoder hidden states
        Returns:
            (tuple:: torch.tensor, torch.tensor, torch.tensor) decoder output, hidden states, attention weights
        """
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_state = hidden[0] if isinstance(hidden, tuple) else hidden
        attn_weights = F.softmax(attn_state[0] @ encoder_outputs.squeeze().t(), dim=1)
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0), encoder_outputs.permute(1, 0, 2)
        )

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self):
        """Initialize the weights of the recurrent model"""
        if isinstance(self.rnn, nn.GRU):
            return torch.zeros(
                self.num_directions * self.num_layers,
                1,
                self.hidden_size,
                device=device,
            )
        elif isinstance(self.rnn, nn.LSTM):
            return (
                torch.zeros(
                    self.num_directions * self.num_layers,
                    1,
                    self.hidden_size,
                    device=device,
                ),
                torch.zeros(
                    self.num_directions * self.num_layers,
                    1,
                    self.hidden_size,
                    device=device,
                ),
            )
