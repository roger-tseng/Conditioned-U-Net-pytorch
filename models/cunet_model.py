import pytorch_lightning as pl
import torch
import torch.nn as nn

from models.FiLM_utils import FiLM_complex, FiLM_simple
from models.control_models import dense_control_block, dense_control_model
from models.unet_model import u_net_deconv_block, Conv2d_same, ConvTranspose2d_same


def get_func_by_name(activation_str):
    if activation_str == "leaky_relu":
        return nn.LeakyReLU
    elif activation_str == "relu":
        return nn.ReLU
    elif activation_str == "sigmoid":
        return nn.Sigmoid
    elif activation_str == "tanh":
        return nn.Tanh
    elif activation_str == "softmax":
        return nn.Softmax
    elif activation_str == "identity":
        return nn.Identity
    else:
        return None


class u_net_conv_block(pl.LightningModule):
    def __init__(self, conv_layer, bn_layer, FiLM_layer, activation):
        super(u_net_conv_block, self).__init__()
        self.bn_layer = bn_layer
        self.conv_layer = conv_layer
        self.FiLM_layer = FiLM_layer
        self.activation = activation
        self.in_channels = self.conv_layer.conv.in_channels
        self.out_channels = self.conv_layer.conv.out_channels

    def forward(self, x, gamma, beta):
        x = self.bn_layer(self.conv_layer(x))
        x = self.FiLM_layer(x, gamma, beta)
        return self.activation(x)


class CUNET(pl.LightningModule):

    @staticmethod
    def get_arg_keys():
        return ['n_layers',
                'input_channels',
                'filters_layer_1',
                'kernel_size',
                'stride',
                'film_type',
                'control_type',
                'encoder_activation',
                'decoder_activation',
                'last_activation',
                'control_input_dim',
                'control_n_layer']

    def __init__(self,
                 n_layers,
                 input_channels,
                 filters_layer_1,
                 kernel_size=(5, 5),
                 stride=(2, 2),
                 film_type='simple',
                 control_type='dense',
                 encoder_activation=nn.LeakyReLU,
                 decoder_activation=nn.ReLU,
                 last_activation=nn.Sigmoid,
                 control_input_dim=4,
                 control_n_layer=4
                 ):

        self.save_hyperparameters()
        encoder_activation = get_func_by_name(encoder_activation)
        decoder_activation = get_func_by_name(decoder_activation)
        last_activation = get_func_by_name(last_activation)

        super(CUNET, self).__init__()

        self.input_control_dims = control_input_dim
        self.n_layers = n_layers
        self.input_channels = input_channels
        self.filters_layer_1 = filters_layer_1

        # Encoder
        encoders = []
        for i in range(n_layers):
            output_channels = filters_layer_1 * (2 ** i)
            encoders.append(
                u_net_conv_block(
                    conv_layer=Conv2d_same(input_channels, output_channels, kernel_size, stride),
                    bn_layer=nn.BatchNorm2d(output_channels),
                    FiLM_layer=FiLM_simple if film_type == "simple" else FiLM_complex,
                    activation=encoder_activation()
                )
            )
            input_channels = output_channels
        self.encoders = nn.ModuleList(encoders)

        # Decoder
        decoders = []
        for i in range(n_layers):
            # parameters each decoder layer
            is_final_block = i == n_layers - 1  # the las layer is different
            # not dropout in the first block and the last two encoder blocks
            dropout = not (i == 0 or i == n_layers - 1 or i == n_layers - 2)
            # for getting the number of filters
            encoder_layer = self.encoders[n_layers - i - 1]
            skip = i > 0  # not skip in the first encoder block

            input_channels = encoder_layer.out_channels
            if skip:
                input_channels *= 2

            if is_final_block:
                output_channels = self.input_channels
                activation = last_activation
            else:
                output_channels = encoder_layer.in_channels
                activation = decoder_activation

            decoders.append(
                u_net_deconv_block(
                    deconv_layer=ConvTranspose2d_same(input_channels, output_channels, kernel_size, stride),
                    bn_layer=nn.BatchNorm2d(output_channels),
                    activation=activation(),
                    dropout=dropout
                )
            )

            self.decoders = nn.ModuleList(decoders)

        # Control Mechanism
        if film_type == "simple":
            control_output_dim = n_conditions = n_layers
            split = lambda tensor: [tensor[..., i] for i in range(n_layers)]
        else:
            output_channel_array = [encoder.conv_layer.conv.out_channels for encoder in self.encoders]
            control_output_dim = n_conditions = sum(output_channel_array)

            start_idx_per_layer = [sum(output_channel_array[:i]) for i in range(len(output_channel_array))]
            end_idx_per_layer = [sum(output_channel_array[:i + 1]) for i in range(len(output_channel_array))]

            split = lambda tensor: [tensor[..., start:end] for start, end in
                                    zip(start_idx_per_layer, end_idx_per_layer)]
        if control_type == "dense":
            self.condition_generator = dense_control_model(
                dense_control_block(control_input_dim, control_n_layer),
                control_output_dim,
                split
            )
        else:
            raise NotImplementedError

    def forward(self, input_spec, input_condition):

        gammas, betas = self.condition_generator(input_condition)

        x = input_spec
        if (torch.isnan(x).any()):
            print("input is nan!")
        # Encoding Phase
        encoder_outputs = []
        i = 0
        for encoder, gamma, beta in zip(self.encoders, gammas, betas):
            encoder_outputs.append(encoder(x, gamma, beta))  # TODO
            x = encoder_outputs[-1]
            if (torch.isnan(x).any()):
                print(f"encoder x is nan! At i={i}:", encoder)
                print("Previous x shapes:")
                for j in encoder_outputs:
                    print(j.shape)
                print("gamma shape:", gamma.shape)
                print("beta shape:", beta.shape)
            i += 1

        # Decoding Phase
        i = 0
        x = self.decoders[0](x)
        if (torch.isnan(x).any()):
            print("decoder[0] x is nan!")
        i += 1
        for decoder, x_encoded in zip(self.decoders[1:], reversed(encoder_outputs[:-1])):
            x = decoder(torch.cat([x, x_encoded], dim=-3))
            if (torch.isnan(x).any()):
                print(f"decoder x is nan! At i={i}:", decoder)
            i += 1
        if (torch.isnan(x).any()):
            raise NotImplementedError
        return x

import math
from typing import List, Tuple
import torch.nn.functional as f

def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = f.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x

class Conv2d_pad_lrelu(pl.LightningModule):
    def __init__(self, in_dim, out_dim, k, s, norm=True):
        super(Conv2d_pad_lrelu, self).__init__()
        self.k = k
        self.s = s
        self.norm = norm
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s)
        self.bnorm = nn.BatchNorm2d(out_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        y = pad_same(x, list(self.k), list(self.s))
        y = self.conv2d(y)
        #if self.norm:
        #    y = self.bnorm(y)
        y = self.act(y)
        return y

class Discriminator(pl.LightningModule):
    """
    Input shape: (batch_size, 2, num_frame, hop_length)
    Output shape: (batch_size, )
    """
    def __init__(self, channels, time, freq):
        self.save_hyperparameters()
        super(Discriminator, self).__init__()

        dim = 32

        #self.first = Conv2d_pad_lrelu(2, dim, (4,4), (2,2), norm=False)
        self.first = nn.Sequential(
            nn.Conv2d(2, dim, (4,4), (2,2)),
            nn.LeakyReLU(0.2)
        )
        self.mid_layers = nn.Sequential(
            Conv2d_pad_lrelu(32, 64, (4,4), (2,2)),
            Conv2d_pad_lrelu(64, 128, (4,4), (2,2)),
            Conv2d_pad_lrelu(128, 256, (4,4), (2,2)),
            Conv2d_pad_lrelu(256, 512, (4,4), (2,2)),
            Conv2d_pad_lrelu(512, 512, (2,4), (1,2)),
            Conv2d_pad_lrelu(512, 512, (2,4), (1,2))
        )
        self.last = nn.Conv2d(512, 1, 4, 2)

    def forward(self, x):
        y = self.first(x)
        y = self.mid_layers(y)
        y = self.last(y)
        y = y.view(-1).unsqueeze(dim=1)
        return y