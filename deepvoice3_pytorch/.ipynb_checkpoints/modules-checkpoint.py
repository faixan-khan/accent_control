# coding: utf-8

import torch
from torch import nn
import math
import numpy as np
from torch.nn import functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def position_encoding_init(n_position, d_pos_vec, position_rate=1.0,
                           sinusoidal=True):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [position_rate * pos / np.power(10000, 2 * (i // 2) / d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc = torch.from_numpy(position_enc).float()
    if sinusoidal:
        position_enc[1:, 0::2] = torch.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = torch.cos(position_enc[1:, 1::2])  # dim 2i+1

    return position_enc


def sinusoidal_encode(x, w):
    y = w * x
    y[1:, 0::2] = torch.sin(y[1:, 0::2].clone())
    y[1:, 1::2] = torch.cos(y[1:, 1::2].clone())
    return y


class SinusoidalEncoding(nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim,
                 *args, **kwargs):
        super(SinusoidalEncoding, self).__init__(num_embeddings, embedding_dim,
                                                 padding_idx=0,
                                                 *args, **kwargs)
        self.weight.data = position_encoding_init(num_embeddings, embedding_dim,
                                                  position_rate=1.0,
                                                  sinusoidal=False)

    def forward(self, x, w=1.0):
        isscaler = np.isscalar(w)
        assert self.padding_idx is not None

        if isscaler or w.size(0) == 1:
            weight = sinusoidal_encode(self.weight, w)
            return F.embedding(
                x, weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            # TODO: cannot simply apply for batch
            # better to implement efficient function
            pe = []
            for batch_idx, we in enumerate(w):
                weight = sinusoidal_encode(self.weight, we)
                pe.append(F.embedding(
                    x[batch_idx], weight, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse))
            pe = torch.stack(pe)
            return pe


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        ctx.mark_shared_storage((x, res))
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def Linear(in_features, out_features, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def Embedding(num_embeddings, embedding_dim, padding_idx, std=0.01):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, std)
    return m


def Conv1d(in_channels, out_channels, kernel_size, dropout=0, std_mul=4.0, **kwargs):
    from .conv import Conv1d
    m = Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((std_mul * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def ConvTranspose1d(in_channels, out_channels, kernel_size, dropout=0,
                    std_mul=1.0, **kwargs):
    m = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((std_mul * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


class Conv1dGLU(nn.Module):
    """(Dilated) Conv1d + Gated linear unit + (optionally) speaker embedding
    """

    def __init__(self, n_speakers, speaker_embed_dim,
                 in_channels, out_channels, kernel_size,
                 dropout, padding=None, dilation=1, causal=False, residual=False,
                 *args, **kwargs):
        super(Conv1dGLU, self).__init__()
        self.dropout = dropout
        self.residual = residual
        if padding is None:
            # no future time stamps available
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal

        self.conv = Conv1d(in_channels, 2 * out_channels, kernel_size,
                           dropout=dropout, padding=padding, dilation=dilation,
                           *args, **kwargs)
        if n_speakers > 1:
            self.speaker_proj = Linear(speaker_embed_dim, out_channels)
        else:
            self.speaker_proj = None

    def forward(self, x, speaker_embed=None):
        return self._forward(x, speaker_embed, False)

    def incremental_forward(self, x, speaker_embed=None):
        return self._forward(x, speaker_embed, True)

    def _forward(self, x, speaker_embed, is_incremental):
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            # remove future time steps
            x = x[:, :, :residual.size(-1)] if self.causal else x

        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
        if self.speaker_proj is not None:
            softsign = F.softsign(self.speaker_proj(speaker_embed))
            # Since conv layer assumes BCT, we need to transpose
            softsign = softsign if is_incremental else softsign.transpose(1, 2)
            a = a + softsign
        x = a * torch.sigmoid(b)
        return (x + residual) * math.sqrt(0.5) if self.residual else x

    def clear_buffer(self):
        self.conv.clear_buffer()


class HighwayConv1d(nn.Module):
    """Weight normzlized Conv1d + Highway network (support incremental forward)
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, padding=None,
                 dilation=1, causal=False, dropout=0, std_mul=None, glu=False):
        super(HighwayConv1d, self).__init__()
        if std_mul is None:
            std_mul = 4.0 if glu else 1.0
        if padding is None:
            # no future time stamps available
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal
        self.dropout = dropout
        self.glu = glu

        self.conv = Conv1d(in_channels, 2 * out_channels,
                           kernel_size=kernel_size, padding=padding,
                           dilation=dilation, dropout=dropout,
                           std_mul=std_mul)

    def forward(self, x):
        return self._forward(x, False)

    def incremental_forward(self, x):
        return self._forward(x, True)

    def _forward(self, x, is_incremental):
        """Forward

        Args:
            x: (B, in_channels, T)
        returns:
            (B, out_channels, T)
        """

        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            # remove future time steps
            x = x[:, :, :residual.size(-1)] if self.causal else x

        if self.glu:
            x = F.glu(x, dim=splitdim)
            return (x + residual) * math.sqrt(0.5)
        else:
            a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
            T = torch.sigmoid(b)
            return (T * a + (1 - T) * residual)

    def clear_buffer(self):
        self.conv.clear_buffer()


# def get_mask_from_lengths(memory, memory_lengths):
#     """Get mask tensor from list of length
#     Args:
#         memory: (batch, max_time, dim)
#         memory_lengths: array like
#     """
#     max_len = max(memory_lengths)
#     mask = torch.arange(max_len).expand(memory.size(0), max_len) < torch.tensor(memory_lengths).unsqueeze(-1)
#     mask = mask.to(memory.device)
#     return ~mask

def get_mask_from_lengths(memory, memory_lengths):
    """Get mask tensor from list of length
    Args:
        memory: (batch, max_time, dim)
        memory_lengths: array like
    """
    mask = memory.data.new(memory.size(0), memory.size(1)).byte().zero_()
    for idx, l in enumerate(memory_lengths):
        mask[idx][:l] = 1
    return ~mask

#  face_encoder_stuff

def conv_block(input,num_filters,kernel_size=3,strides=1,do_pad=True,act=True):
    if do_pad:
        h = input.shape[3]
        pad = (((h) - 1) - (h-kernel_size))/2
        pad = int(pad)
    else:
        pad=0
    in_chan = input.shape[1]
    conv = nn.Conv2d(in_channels=in_chan, out_channels=num_filters,kernel_size=kernel_size,stride=strides,padding=pad,dilation=1)
    conv = conv.to(device)
    x = conv(input)
    if x.shape[3] > 1:
        r_mean = x.shape[1]
        batch_norm = nn.BatchNorm2d(num_features=r_mean,momentum=0.8)
        batch_norm = batch_norm.to(device)
        x = batch_norm(x)
    if act:
        relu = nn.ReLU()
        relu =relu.to(device)
        out = relu(x)
    return out

def residual_block(inp, num_filters):
    x = conv_block(inp,num_filters)
    x = conv_block(x, num_filters)
    x = x + inp
    m = nn.ReLU()
    m = m.to(device)
    x = m(x)
    return x

class BatchConv2D_5D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        """ Batch convolve a set of inputs by groups in parallel. Similar to bmm.

        :param in_channels: (b_j, b_i, c_in, h, w) where b_j are the parallel convs to run
        :param out_channels: output channels from conv
        :param kernel_size: size of conv kernel
        :param stride: the stride of the filter
        :param padding: the padding around the input
        :param dilation: the filter dilation
        :param groups: number of parallel ops
        :param bias: whether of not to include a bias term in the conv (bool)
        :returns: tensor of (b_i, c_out, c_in, kh, kw) with batch convolve done
        :rtype: torch.Tensor

        """
        super(BatchConv2D_5D, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels*groups, out_channels*groups,
                              kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              groups=groups, bias=bias)

    def forward(self, x):
        """ (b_j, b_i, c_in, h, w) -> (b_j, b_i * c_in, h, w) --> (b_j, b_i, c_out, h, w)

        :param x: accepts an input of (b_j, b_i, c_in, h, w) where b_j are the parallel groups
        :returns: (b_j, b_i , c_out, kh, kh, kw)
        :rtype: torch.Tensor

        """
        assert len(x.shape) == 5, "batchconv2d expects a 5d [{}] tensor".format(x.shape)
        b_i, b_j, c, h, w = x.shape
        out = self.conv(x.permute([1, 0, 2, 3, 4]).contiguous().view(b_j, b_i * c, h, w))
        return out.view(b_j, b_i, self.out_channels,
                        out.shape[-2], out.shape[-1]).permute([1, 0, 2, 3, 4])
