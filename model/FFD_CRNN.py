# Some codes are adopted from
# https://github.com/DCASE-REPO/DESED_task
import torch
import torch.nn as nn
import torch.nn.functional as F
from ddf import ddf
from ddf import FilterNorm


class GLU(nn.Module):
    def __init__(self, in_dim):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    # x size = [batch, chan, freq, frame]
    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))  # x size = [batch, freq, frame, chan]
        # x size = [batch, chan, freq, frame]
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, in_dim):
        super(ContextGating, self).__init__()
        # self.sigmoid = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    # x size = [batch, chan, frame, freq]
    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))  # x size = [batch, frame, freq, chan]
        # x size = [batch, chan, frame, freq]
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(lin)
        res = x * sig
        return res


class attention2d(nn.Module):
    def __init__(self, in_channels, head, kernel_size, stride, padding, temperature, pool_dim, se_ratio=0.2):
        super(attention2d, self).__init__()
        self.pool_dim = pool_dim
        self.temperature = temperature

        hidden_planes = int(in_channels * se_ratio)
        out_channels = head * kernel_size ** 2
        if hidden_planes < 4:
            hidden_planes = 4

        if not pool_dim == 'both':
            self.conv1d1 = nn.Conv1d(in_channels, hidden_planes, kernel_size, stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm1d(hidden_planes)
            self.act = ContextGating(hidden_planes)
            self.conv1d2 = nn.Conv1d(hidden_planes, out_channels, 1, bias=True)
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            self.fc1 = nn.Linear(in_channels, hidden_planes)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x size : [bs, chan, frames, freqs]
        if self.pool_dim == 'freq':
            x = torch.mean(x, dim=3)  # x size : [bs, chan, frames]
        elif self.pool_dim == 'time':
            x = torch.mean(x, dim=2)  # x size : [bs, chan, freqs]
        elif self.pool_dim == 'both':
            # x = torch.mean(torch.mean(x, dim=2), dim=1)  #x size : [bs, chan]
            x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        elif self.pool_dim == 'chan':
            x = torch.mean(x, dim=1)  # x size : [bs, freqs, frames]

        if not self.pool_dim == 'both':
            x = self.conv1d1(x)  # x size : [bs, hid_chan, freqs]
            x = self.bn(x).unsqueeze(2)
            x = self.act(x).squeeze(2)
            x = self.conv1d2(x)
        else:
            x = self.fc1(x)  # x size : [bs, hid_chan]
            x = self.relu(x)

        out = F.softmax(x / self.temperature, dim=1)
        if self.pool_dim == 'freq':
            out = out.unsqueeze(3)
        elif self.pool_dim == 'time':
            out = out.unsqueeze(2)

        return out


class build_spatial_branch(nn.Module):
    def __init__(self, spatial_kernel_type, in_channels, kernel_size, head=1, build_kernel_size_T=3, build_kernel_size_F=3,
                 static_build_kernel_size=3, nonlinearity='relu', stride=1, temperature=31, pool_dim='freq', se_ratio=0.2, atten_dim_k=9):
        super(build_spatial_branch, self).__init__()
        self.head = head
        self.kernel_size = kernel_size
        self.build_kernel_size_F = build_kernel_size_F
        self.build_kernel_size_T = build_kernel_size_T
        self.spatial_kernel_type = spatial_kernel_type

        if spatial_kernel_type == 'build_T':
            self.kernel_build = nn.Conv2d(in_channels, head * kernel_size ** 2, kernel_size=(static_build_kernel_size, build_kernel_size_F),
                                          padding=((static_build_kernel_size - 1) // 2, 0), stride=stride)
        elif spatial_kernel_type == 'build_F':
            self.kernel_build = nn.Conv2d(in_channels, head * kernel_size ** 2, kernel_size=(build_kernel_size_T, static_build_kernel_size),
                                            padding=(0, (static_build_kernel_size - 1) // 2), stride=stride)
        elif spatial_kernel_type == 'build_square':
            self.kernel_build = nn.Conv2d(in_channels, head * kernel_size ** 2, kernel_size=(static_build_kernel_size, static_build_kernel_size),
                                            padding=((static_build_kernel_size - 1) // 2, (static_build_kernel_size - 1) // 2), stride=stride)
        self.atten = attention2d(in_channels=in_channels, head=head, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1) // 2,
                                 temperature=temperature, pool_dim=pool_dim, se_ratio=se_ratio)
        self.FiltNorm = FilterNorm(head, kernel_size, 'spatial', nonlinearity)

    def forward(self, x):
        unique_kernel = self.kernel_build(x)
        atten2d = self.atten(x)
        # TODO add or mul
        kernel = atten2d * unique_kernel
        kernel = self.FiltNorm(kernel)

        if self.spatial_kernel_type == 'build_F':
            kernel = kernel.repeat_interleave(self.build_kernel_size_T, dim=2)
        elif self.spatial_kernel_type == 'build_T':
            kernel = kernel.repeat_interleave(self.build_kernel_size_F, dim=3)
        elif self.spatial_kernel_type == 'build_square':
            pass
        return kernel


def build_channel_branch(in_channels, kernel_size, build_kernel_size_F,
                         nonlinearity='linear', se_ratio=0.2):
    assert se_ratio > 0
    mid_channels = int(in_channels * se_ratio)
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Conv2d(in_channels, mid_channels, 1),
        ContextGating(mid_channels),
        nn.Conv2d(mid_channels, in_channels * kernel_size ** 2, 1),
        FilterNorm(in_channels, kernel_size, 'channel', nonlinearity, running_std=True))


class DDFPack(nn.Module):
    def __init__(self, spatial_kernel_type, in_channels, kernel_size=3, stride=1, padding=1, dilation=1, head=1, build_kernel_size_T=3, build_kernel_size_F=3,
                 static_build_kernel_size=3, se_ratio=[0.5, 0.5], nonlinearity='linear', kernel_combine='mul', temperature=31, pool_dim='freq'):
        super(DDFPack, self).__init__()
        assert kernel_size > 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.head = head
        self.kernel_combine = kernel_combine
        self.build_kernel_size_T = build_kernel_size_T
        self.build_kernel_size_F = build_kernel_size_F
        self.static_build_kernel_size = static_build_kernel_size

        self.spatial_branch = build_spatial_branch(spatial_kernel_type, in_channels, kernel_size, head, build_kernel_size_T, build_kernel_size_F,
                                                   static_build_kernel_size, nonlinearity, stride, temperature, pool_dim, se_ratio[1])

        self.channel_branch = build_channel_branch(in_channels, kernel_size, build_kernel_size_F, nonlinearity, se_ratio[0])

    def forward(self, x):
        b, c, h, w = x.shape
        g = self.head
        k = self.kernel_size
        s = self.stride

        # TODO
        spatial_filter_F = self.spatial_branch(x).reshape(b * g, -1, h // s, w // s)
        channel_filter_F = self.channel_branch(x).reshape(b * g, c // g, k, k)
        x = x.reshape(b * g, c // g, h, w)
        out_s = ddf(x, channel_filter_F, spatial_filter_F, self.kernel_size, self.dilation, self.stride, self.kernel_combine)

        return (out_s).reshape(b, c, h // s, w // s)


class FFD_Conv(nn.Module):
    def __init__(self,
                 spatial_kernel_type,
                 build_kernel_size_F,
                 build_kernel_size_T,
                 static_build_kernel_size,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 temperature=31,
                 pool_dim='freq',
                 se_ratio=[0.5, 0.5]):
        super(FFD_Conv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99)
        self.act1 = nn.ReLU(inplace=True)
        self.ddf_conv = DDFPack(spatial_kernel_type, in_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                 build_kernel_size_F=build_kernel_size_F, build_kernel_size_T=build_kernel_size_T,
                                 static_build_kernel_size=static_build_kernel_size, temperature=temperature, pool_dim=pool_dim, se_ratio=se_ratio)

    def forward(self, x):
        # x size : [bs, in_chan, frames, freqs]
        output1 = self.act1(self.norm1(self.conv1(x)))
        output2 = self.ddf_conv(output1)
        return output2


class CNN(nn.Module):
    def __init__(self,
                 n_input_ch,
                 static_build_kernel_size=3,
                 input_fdim=128,
                 input_tdim=626,
                 activation="Relu",
                 conv_dropout=0,
                 kernel=[3, 3, 3],
                 pad=[1, 1, 1],
                 stride=[1, 1, 1],
                 n_filt=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)],
                 normalization="batch",
                 DY_layers=[0, 1, 1, 1, 1, 1, 1],
                 f_dim=[128, 64, 32],
                 temperature=31,
                 pool_dim='freq',
                 spatial_kernel_type='build_F',
                 se_ratio=[0.5, 0.5]):
        super(CNN, self).__init__()
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.n_filt = n_filt
        self.n_filt_last = n_filt[-1]
        self.DY_layers = DY_layers
        cnn = nn.Sequential()

        def ratio_compute_F(i, pooling):
            if i == 0:
                return pooling[i][1]
            else:
                return pooling[i][1] * ratio_compute_F(i - 1, pooling)

        def ratio_compute_T(i, pooling):
            if i == 0:
                return pooling[i][0]
            else:
                return pooling[i][0] * ratio_compute_T(i - 1, pooling)

        def conv(i, normalization="batch",
                 dropout=None, activ='relu'):
            in_dim = n_input_ch if i == 0 else n_filt[i - 1]
            out_dim = n_filt[i]

            if DY_layers[i] == 1:
                build_kernel_size_F = input_fdim // ratio_compute_F(i - 1, pooling)
                build_kernel_size_T = input_tdim // ratio_compute_T(i - 1, pooling)
                cnn.add_module("FFD_Conv{0}".format(i), FFD_Conv(spatial_kernel_type, build_kernel_size_F, build_kernel_size_T, static_build_kernel_size,
                                                                   in_dim, out_dim, kernel[i], stride[i], pad[i], temperature=temperature, pool_dim=pool_dim, se_ratio=se_ratio))
            else:
                cnn.add_module(
                    "conv{0}".format(i), nn.Conv2d(in_dim, out_dim, kernel[i], stride[i], pad[i]))

            if normalization == "batch":
                cnn.add_module(
                    "batchnorm{0}".format(i), nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.99))
            elif normalization == "layer":
                cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, out_dim))

            if activ.lower() == "leakyrelu":
                cnn.add_module("Relu{0}".format(i), nn.LeakyReLu(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("Relu{0}".format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(out_dim))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(out_dim))

            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        for i in range(len(n_filt)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
            cnn.add_module("pooling{0}".format(i), nn.AvgPool2d(pooling[i]))
        self.cnn = cnn

    # x size : [bs, chan, frames, freqs]
    def forward(self, x):
        x = self.cnn(x)
        return x


class BiGRU(nn.Module):
    def __init__(self, n_in, n_hidden,
                 dropout=0, num_layers=1):
        super(BiGRU, self).__init__()
        self.rnn = nn.GRU(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True, num_layers=num_layers)

    def forward(self, x):
        # self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        return x


class FFD_CRNN(nn.Module):
    def __init__(self,
                 n_input_ch,
                 static_build_kernel_size=3,
                 input_fdim=128,
                 input_tdim=626,
                 n_class=10,
                 activation="glu",
                 conv_dropout=[0.5, 0.5],
                 n_RNN_cell=128,
                 n_RNN_layer=2,
                 rec_dropout=0,
                 attention=True,
                 **convkwargs):
        super(FFD_CRNN, self).__init__()
        self.n_input_ch = n_input_ch
        self.attention = attention
        self.n_class = n_class

        self.cnn = CNN(n_input_ch=n_input_ch, static_build_kernel_size=static_build_kernel_size,
                       input_fdim=input_fdim, input_tdim=input_tdim, activation=activation, conv_dropout=conv_dropout[0], **convkwargs)
        self.rnn = BiGRU(n_in=self.cnn.n_filt[-1], n_hidden=n_RNN_cell, dropout=rec_dropout, num_layers=n_RNN_layer)

        self.dropout = nn.Dropout(conv_dropout[1])
        self.sigmoid = nn.Sigmoid()
        self.dense = nn.Linear(
            n_RNN_cell * 2, n_class)

        if self.attention:
            self.dense_softmax = nn.Linear(
                n_RNN_cell * 2, n_class)
            if self.attention == "time":
                self.softmax = nn.Softmax(
                    dim=1)  # softmax on time dimension
            elif self.attention == "class":
                self.softmax = nn.Softmax(
                    dim=-1)  # softmax on class dimension

    # input size : [bs, freqs, frames]
    def forward(self, x):
        # cnn
        if self.n_input_ch > 1:
            x = x.transpose(2, 3)
        else:
            x = x.transpose(1, 2).unsqueeze(1)  # x size : [bs, chan, frames, freqs]
        x = self.cnn(x)
        bs, ch, frame, freq = x.size()
        if freq != 1:
            print("warning! frequency axis is large: " + str(freq))
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous.view(bs, frame, ch * freq)
        else:
            x = x.squeeze(-1)
            # x size : [bs, frames, chan]
            x = x.permute(0, 2, 1)

        # rnn
        # x size : [bs, frames, 2 * chan]
        x = self.rnn(x)
        x = self.dropout(x)

        # classifier
        # strong size : [bs, frames, n_class]
        strong = self.dense(x)
        strong = self.sigmoid(strong)
        if self.attention:
            # sof size : [bs, frames, n_class]
            sof = self.dense_softmax(x)
            # sof size : [bs, frames, n_class]
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, n_class]
        else:
            weak = strong.mean(1)

        return strong.transpose(1, 2), weak
