import math

import numpy as np
import torch
from torch import nn

from model.shift_predictor.stylegan2.ops import upfirdn2d


@torch.no_grad()
def variance_scaling_init_(tensor, scale=1.0, mode="fan_avg", distribution="uniform"):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        scale /= fan_in
    elif mode == "fan_out":
        scale /= fan_out
    else:
        scale /= (fan_in + fan_out) / 2

    if distribution == "normal":
        std = math.sqrt(scale)
        return tensor.normal_(0, std)
    else:
        bound = math.sqrt(3 * scale)
        return tensor.uniform_(-bound, bound)


def conv2d(
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        scale=1.0,
        mode="fan_avg",
):
    conv = nn.Conv2d(
        in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias
    )

    variance_scaling_init_(conv.weight, scale, mode=mode)

    if bias:
        nn.init.zeros_(conv.bias)

    return conv


def linear(in_channel, out_channel, scale=1.0, mode="fan_avg"):
    linear = nn.Linear(in_channel, out_channel)

    variance_scaling_init_(linear.weight, scale, mode=mode)
    nn.init.zeros_(linear.bias)

    return linear


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


class Upsample(nn.Sequential):
    def __init__(self, channel):
        layers = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv2d(channel, channel, 3, padding=1),
        ]

        super().__init__(*layers)


class Downsample(nn.Sequential):
    def __init__(self, channel):
        layers = [conv2d(channel, channel, 3, stride=2, padding=1)]

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(
            self, in_channel, out_channel, time_dim, dropout, up=False, down=False, use_time_embedding=True,
    ):
        super().__init__()

        time_out_dim = out_channel
        time_scale = 1
        norm_affine = True

        self.norm1 = nn.GroupNorm(32, in_channel)
        self.activation1 = Swish()
        self.conv1 = conv2d(in_channel, out_channel, 3, padding=1)
        # self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        if use_time_embedding:
            self.time = nn.Sequential(
                Swish(),
                linear(time_dim, time_out_dim, scale=time_scale)
            )

        self.up = up
        self.down = down

        self.norm2 = nn.GroupNorm(32, out_channel, affine=norm_affine)
        self.activation2 = Swish()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv2d(out_channel, out_channel, 3, padding=1, scale=1e-10)

        self.skip = conv2d(in_channel, out_channel, 1) if in_channel != out_channel else nn.Identity()

        if self.up or self.down:
            self.filter_ = upfirdn2d.setup_filter([1, 3, 3, 1])

    def forward(self, input, time=None):
        out = self.activation1(self.norm1(input))

        if self.up:
            out = upfirdn2d.upsample2d(out, self.filter_.type_as(out))
            input = upfirdn2d.upsample2d(input, self.filter_.type_as(input))
        elif self.down:
            out = upfirdn2d.downsample2d(out, self.filter_.type_as(out))
            input = upfirdn2d.downsample2d(input, self.filter_.type_as(input))

        out = self.conv1(out)
        if time is not None:
            out = out + self.time(time).view(input.shape[0], -1, 1, 1)
        out = self.norm2(out)
        out = self.conv2(self.dropout(self.activation2(out)))
        input = self.skip(input)

        return (out + input) / np.sqrt(2.)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(32, in_channel)
        self.qkv = conv2d(in_channel, in_channel * 3, 1)
        self.out = conv2d(in_channel, in_channel, 1, scale=1e-10)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return (out + input) / np.sqrt(2.)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000) / dim)
        )

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.matmul(input.view(-1).unsqueeze(-1).float(), self.inv_freq.unsqueeze(0))
        # sinusoid_in = torch.outer(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)

        return pos_emb


class ResBlockWithAttention(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            time_dim,
            dropout,
            use_attention,
            attention_head,
            use_time_embedding=True
    ):
        super().__init__()

        self.residual_blocks = ResBlock(
            in_channel, out_channel, time_dim, dropout, use_time_embedding=use_time_embedding
        )
        self.attention = SelfAttention(out_channel, n_head=attention_head) if use_attention else nn.Identity()

    def forward(self, input, time):
        out = self.residual_blocks(input, time)
        out = self.attention(out)

        return out


class Combine(nn.Module):
    """Combine information from skip connections."""

    def __init__(self, in_channel):
        super().__init__()
        self.conv = conv2d(in_channel, in_channel, kernel_size=1, padding=0, stride=1)

    def __call__(self, x, y):
        h = self.conv(x)
        return h + y


class BetterUnet(nn.Module):
    def __init__(
            self,
            image_channel,
            base_channel,
            channel_multiplier,
            num_residual_blocks_of_a_block,
            attn_strides,
            attn_heads,
            dropout,
            use_time_embedding=True,
            res_up_down=True,
    ):
        super().__init__()

        time_dim = base_channel * 4
        self.progressive = None
        self.progressive_input = None
        progressive_combine = 'sum'

        num_blocks = len(channel_multiplier)
        self.num_blocks = num_blocks

        if use_time_embedding:
            self.time_embedding = nn.Sequential(
                TimeEmbedding(base_channel),
                linear(base_channel, time_dim),
                Swish(),
                linear(time_dim, time_dim),
            )

        down_layers = [conv2d(image_channel, base_channel, 3, padding=1)]
        pyramid_downsample = []
        combiner = []

        feature_channels = [base_channel]
        in_channel = base_channel
        for i in range(num_blocks):
            for _ in range(num_residual_blocks_of_a_block):
                out_channel = base_channel * channel_multiplier[i]

                down_layers.append(
                    ResBlockWithAttention(
                        in_channel,
                        out_channel,
                        time_dim,
                        dropout,
                        use_attention=2 ** i in attn_strides,
                        attention_head=attn_heads,
                        use_time_embedding=use_time_embedding
                    )
                )

                feature_channels.append(out_channel)
                in_channel = out_channel

            if i != num_blocks - 1:
                if res_up_down:
                    down_layers.append(ResBlock(in_channel, in_channel, time_dim, dropout, down=True,
                                                use_time_embedding=use_time_embedding))
                else:
                    down_layers.append(Downsample(in_channel))
                feature_channels.append(in_channel)

                # if self.progressive_input == 'input_skip':
                #     pyramid_downsample.append(Downsample(channel=in_channel, with_conv=False))
                #     combiner.append(Combine(in_channel=in_channel))
                #
                # elif self.progressive_input == 'residual':
                #     pyramid_downsample.append(Downsample(channel=in_channel, with_conv=True))

        self.down = nn.ModuleList(down_layers)
        self.pyramid_downsample = nn.ModuleList(pyramid_downsample)
        self.combiner = nn.ModuleList(combiner)

        self.mid = nn.ModuleList(
            [
                ResBlockWithAttention(
                    in_channel,
                    in_channel,
                    time_dim,
                    dropout=dropout,
                    use_attention=True,
                    attention_head=attn_heads,
                    use_time_embedding=use_time_embedding
                ),
                ResBlockWithAttention(
                    in_channel,
                    in_channel,
                    time_dim,
                    dropout=dropout,
                    use_attention=False,
                    attention_head=1,
                    use_time_embedding=use_time_embedding
                ),
            ]
        )

        up_layers = []
        # pyramid_upsample = []

        for i in reversed(range(num_blocks)):
            for _ in range(num_residual_blocks_of_a_block + 1):
                out_channel = base_channel * channel_multiplier[i]

                up_layers.append(
                    ResBlockWithAttention(
                        in_channel + feature_channels.pop(),
                        out_channel,
                        time_dim,
                        dropout=dropout,
                        use_attention=2 ** i in attn_strides,
                        attention_head=attn_heads,
                        use_time_embedding=use_time_embedding
                    )
                )

                in_channel = out_channel

            if i != 0:
                if res_up_down:
                    up_layers.append(ResBlock(in_channel, in_channel, time_dim, dropout, up=True,
                                              use_time_embedding=use_time_embedding))
                else:
                    up_layers.append(Upsample(in_channel))

            # if self.progressive != 'none':
            #     if i == num_blocks - 1:
            #         if self.progressive == 'output_skip':
            #             pyramid_upsample.append(
            #                 nn.Sequential(
            #                     nn.GroupNorm(32, in_channel),
            #                     Swish(),
            #                     conv2d(in_channel, image_channel, 3, padding=1, scale=1e-10),
            #                 )
            #             )
            #         elif self.progressive == 'residual':
            #             pyramid_upsample.append(
            #                 nn.Sequential(
            #                     nn.GroupNorm(32, in_channel),
            #                     Swish(),
            #                     conv2d(in_channel, image_channel, 3, padding=1),
            #                 )
            #             )
            #     else:
            #         if self.progressive == 'output_skip':
            #             pyramid_upsample.append(
            #                 nn.Sequential(
            #                     nn.GroupNorm(32, in_channel),
            #                     Swish(),
            #                     conv2d(in_channel, image_channel, 3, padding=1),
            #                 )
            #                 # Upsample(channel=in_channel, with_conv=False)
            #             )
            #         elif self.progressive == 'residual':
            #             pyramid_upsample.append(
            #                 Upsample(channel=in_channel, with_conv=True)
            #             )

        self.up = nn.ModuleList(up_layers)
        # self.pyramid_upsample = nn.ModuleList(pyramid_upsample)

        self.out = nn.Sequential(
            nn.GroupNorm(32, in_channel),
            Swish(),
            conv2d(in_channel, image_channel, 3, padding=1, scale=1e-10),
        )

    # input: batch_size x image_channel x height x width
    # time: batch_size
    def forward(self, input, time=None):
        if time is not None:
            time_embedding = self.time_embedding(time)
        else:
            time_embedding = None

        features = []

        # input_pyramid = input
        out = input
        for i, layer in enumerate(self.down):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, Downsample):
                out = layer(out)
            else:
                out = layer(out, time_embedding)

            # if i != self.num_blocks - 1:
            #     if self.progressive_input == 'input_skip':
            #         input_pyramid = self.pyramid_downsample[i](input_pyramid)
            #         out = self.combiner[i](input_pyramid, out)
            #     elif self.progressive_input == 'residual':
            #         input_pyramid = self.pyramid_downsample[i](input_pyramid)
            #         input_pyramid = (input_pyramid + out) / np.sqrt(2)
            #         out = input_pyramid

            features.append(out)

        for layer in self.mid:
            out = layer(out, time_embedding)

        # pyramid = None
        for i, layer in enumerate(self.up):
            if isinstance(layer, ResBlockWithAttention):
                out = layer(torch.cat((out, features.pop()), 1), time_embedding)
            elif isinstance(layer, Upsample):
                out = layer(out)
            else:
                out = layer(out, time_embedding)

            # if self.progressive != 'none':
            #     if i == 0:
            #         pyramid = self.pyramid_upsample[i](out)
            #     else:
            #         if self.progressive == 'output_skip':
            #             pyramid = upfirdn2d.upsample2d(pyramid, [1, 3, 3, 1])
            #             pyramid = pyramid + self.pyramid_upsample[i](out)
            #         elif self.progressive == 'residual':
            #             pyramid = self.pyramid_upsample[i](pyramid)
            #             pyramid = (pyramid + out) / np.sqrt(2.)
            #             out = pyramid

        out = self.out(out)

        return out