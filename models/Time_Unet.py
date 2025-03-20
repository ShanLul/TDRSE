import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN


# class EfficientAdditiveAttnetion(nn.Module):
#     """
#     Efficient Additive Attention module for SwiftFormer.
#     Input: tensor in shape [B, N, D]
#     Output: tensor in shape [B, N, D]
#     """
#
#     def __init__(self, in_dims=512, token_dim=256, num_heads=2):
#         super().__init__()
#
#         self.to_query = nn.Linear(in_dims, token_dim * num_heads)
#         self.to_key = nn.Linear(in_dims, token_dim * num_heads)
#
#         self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
#         self.scale_factor = token_dim ** -0.5
#         self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
#         self.final = nn.Linear(token_dim * num_heads, token_dim)
#
#     def forward(self, x):
#         query = self.to_query(x)
#         key = self.to_key(x)
#
#         query = torch.nn.functional.normalize(query, dim=-1) #BxNxD
#         key = torch.nn.functional.normalize(key, dim=-1) #BxNxD
#
#         query_weight = query @ self.w_g # BxNx1 (BxNxD @ Dx1)
#         A = query_weight * self.scale_factor # BxNx1
#
#         A = torch.nn.functional.normalize(A, dim=1) # BxNx1
#
#         G = torch.sum(A * query, dim=1) # BxD
#
#         G = einops.repeat(
#             G, "b d -> b repeat d", repeat=key.shape[1]
#         ) # BxNxD
#
#         out = self.Proj(G * key) + query #BxNxD
#
#         out = self.final(out) # BxNxD
#
#         return out
# 只是简单的计算频域能量，并没有将频域信息实际应用
class RFFTM(nn.Module):
    def __init__(self):
        super(RFFTM, self).__init__()
        # 初始化任何必要的参数或层
        self.energy_threshold = nn.Parameter(torch.tensor(0.0))  # 举例
        self.jump_net = nn.Identity()
        # self.jump_net = nn.Linear(in_features, out_features)

    def forward(self, x):
        # [B, T, C]
        xf = torch.fft.rfft(x, dim=1)

        # 计算每个频率分量的能量
        energy_list = (abs(xf) ** 2).mean(0).mean(-1)

        # 归一化能量
        total_energy = energy_list.sum()
        normalized_energy_list = energy_list / total_energy

        # 累积能量
        cumulative_energy = torch.cumsum(normalized_energy_list, dim=0)

        # 调整阈值范围并确定K值
        energy_threshold_adjusted = 0.95 + 0.1 * torch.sigmoid(self.energy_threshold)
        k = min(torch.searchsorted(cumulative_energy, energy_threshold_adjusted) + 1, xf.shape[1])

        # 其他处理步骤...
        # 获取频率分量的能量从大到小的索引
        sorted_indices = torch.argsort(-energy_list)

        # 保留前k个频率分量，将其余频率分量置零
        xf_filtered = torch.zeros_like(xf)
        top_k_indices = sorted_indices[:k]
        xf_filtered[:, top_k_indices, :] = xf[:, top_k_indices, :]

        # 逆傅里叶变换，从频域转换为时域
        x_reconstructed = torch.fft.irfft(xf_filtered, n=x.shape[1], dim=1)

        x_reconstructed = self.jump_net(x) * x_reconstructed

        return x_reconstructed


class MLPBlock(nn.Module):

    def __init__(
            self,
            dim,
            in_features: int,
            hid_features: int,
            out_features: int,
            activ="gelu",
            drop: float = 0.0,
            jump_conn='trunc',
    ):
        super().__init__()
        self.dim = dim
        self.out_features = out_features
        self.branch1 = nn.Sequential(
            nn.Linear(in_features, hid_features),
            nn.GELU(),
        )
        self.branch2 = nn.Linear(in_features, hid_features)
        self.combine = nn.Sequential(
            nn.Linear(hid_features, out_features),
            nn.GELU(),
            nn.LayerNorm(out_features),
            nn.Dropout(p=0.2)
        )

        if jump_conn == "trunc":
            self.jump_net = nn.Identity()
        elif jump_conn == 'proj':
            self.jump_net = nn.Sequential(
                nn.Linear(in_features, out_features),
            )
        else:
            raise ValueError(f"jump_conn:{jump_conn}")

    def forward(self, x):
        x = torch.transpose(x, self.dim, -1)
        branch1 = self.branch1(x)

        branch2 = self.branch2(x)

        combined = branch1 * branch2

        x1 = self.combine(combined)

        x = self.jump_net(x)[..., :self.out_features] + x1

        x = torch.transpose(x, self.dim, -1)
        return x


class ImprovedGatingMechanism(nn.Module):
    def __init__(self, L, hidden_size, C):
        super(ImprovedGatingMechanism, self).__init__()
        # 使用MLPBlock替换原有的MLP
        self.mlp = MLPBlock(
            dim=2,  # 时间步维度
            in_features=2 * C,
            hid_features=hidden_size,
            out_features=C,  # 输出特征数应与输入的通道数相同
            activ="gelu",
            drop=0.1,
            jump_conn='proj'  # 投影跳连
        )
        self.linear = nn.Linear(C, 2)  # 输出层，用于门控
        self.linear1 = nn.Linear(2 * C, C)  # 输出层，用于门控
        self.net = nn.Sequential(
            nn.Linear(2 * C, C),
            nn.SiLU(),
            nn.Dropout(p=0.1))

    def forward(self, x, y):
        h = torch.cat([x, y], dim=-1)  # [B, L, 2 * C]
        a = self.mlp(h)  # 通过MLPBlock处理
        B, L, _ = a.size()
        # 重塑形状
        a = a.reshape(B * L, -1)
        g = torch.softmax(self.linear(a), dim=-1).view(B, L, -1)  # [B, L, 2]
        g1, g2 = g[..., 0], g[..., 1]
        z = g1.unsqueeze(-1) * x + g2.unsqueeze(-1) * y + x + y  # [B, L, C]

        return z


class PatchMixerLayer(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_sizes=(7, 5, 3)):
        super().__init__()

        # Initialize the ResNet-style branches with specified kernel sizes
        self.resnet_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, 2 * input_dim, ks, groups=input_dim, padding='same'),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(2 * input_dim),
                nn.Conv1d(2 * input_dim, input_dim, 1, padding='same'),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(input_dim)
            ) for ks in kernel_sizes
        ])
        # for ks in kernel_sizes
        # Final 1x1 convolution layer
        self.Conv_1x1 = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(output_dim)
        )

        # Weight generator directly outputs 3-channel weights for each branch
        self.weight_generator = nn.Sequential(
            nn.Conv1d(input_dim, 3, kernel_size=1),
            nn.Softmax(dim=1)  # Softmax across channels ensures sum equals 1
        )

    def forward(self, x):
        # Apply each ResNet-style branch
        branch_outputs = [branch(x) for branch in self.resnet_branches]

        # Generate and apply weights
        weights = self.weight_generator(x).transpose(1, 2)
        weighted_outputs = torch.stack(branch_outputs, dim=-1) * weights.unsqueeze(1)
        suma = torch.sum(weighted_outputs, dim=-1)

        # Pass through final convolution
        x = self.Conv_1x1(x) + self.Conv_1x1(suma)
        return x


def efficient_exponential_weighting(x, decay_rate):
    """
    更高效地对时间序列x应用指数权重衰减。
    """
    # 计算衰减因子序列，形状为 [1, Input_length, 1] 以便广播与输入张量相乘
    decay_factors = decay_rate ** torch.arange(x.size(1), device=x.device).unsqueeze(0).unsqueeze(-1)

    # 直接应用衰减因子进行加权
    weighted_x = x * decay_factors

    return weighted_x


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.head1 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(nf, target_window * 2),
            nn.GELU(),
            # nn.LayerNorm(target_window*2),
            nn.Dropout(head_dropout),
            nn.Linear(target_window * 2, target_window),
            nn.LayerNorm(target_window),
            nn.Dropout(head_dropout)
        )

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]

        x = self.head1(x)

        return x


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class block_model(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, input_channels, input_len, out_len, individual):
        super(block_model, self).__init__()
        self.channels = input_channels
        self.input_len = input_len
        self.out_len = out_len
        self.individual = individual

        if self.individual:
            self.Linear_channel = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_channel.append(nn.Linear(self.input_len, self.out_len))
        else:
            self.Linear_channel = nn.Linear(self.input_len, self.out_len)
        self.ln = nn.LayerNorm(out_len)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.individual:
            output = torch.zeros([x.size(0), x.size(1), self.out_len], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, i, :] = self.Linear_channel[i](x[:, i, :])
        else:
            output = self.Linear_channel(x)
        # output = self.ln(output)
        # output = self.relu(output)
        return output  # [Batch, Channel, Output length]


# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.input_channels = configs.enc_in
#         self.input_len = configs.seq_len
#         self.out_len = configs.pred_len
#         self.individual = configs.individual
#         # 下采样设定
#         self.stage_num = configs.stage_num
#         self.stage_pool_kernel = configs.stage_pool_kernel
#         self.stage_pool_stride = configs.stage_pool_stride
#         self.stage_pool_padding = configs.stage_pool_padding
#
#         self.revin_layer = RevIN(self.input_channels, affine=True, subtract_last=False)
#
#
#
#         len_in = self.input_len
#         len_out = self.out_len
#         down_in = [len_in]
#         down_out = [len_out]
#         i = 0
#         while i < self.stage_num - 1:
#             linear_in = int((len_in + 2 * self.stage_pool_padding - self.stage_pool_kernel)/self.stage_pool_stride + 1 )
#             linear_out = int((len_out + 2 * self.stage_pool_padding - self.stage_pool_kernel)/self.stage_pool_stride + 1 )
#             down_in.append(linear_in)
#             down_out.append(linear_out)
#             len_in = linear_in
#             len_out = linear_out
#             i = i + 1
#
#         # 最大池化层
#         self.Maxpools = nn.ModuleList()
#         # 左边特征提取层
#         self.down_blocks = nn.ModuleList()
#         for in_len,out_len in zip(down_in,down_out):
#             self.down_blocks.append(block_model(self.input_channels, in_len, out_len, self.individual))
#             self.Maxpools.append(nn.AvgPool1d(kernel_size=self.stage_pool_kernel, stride=self.stage_pool_stride, padding=self.stage_pool_padding))
#
#         # 右边特征融合层
#         self.up_blocks = nn.ModuleList()
#         len_down_out = len(down_out)
#         for i in range(len_down_out -1):
#             print(len_down_out, len_down_out - i -1, len_down_out - i - 2)
#             in_len = down_out[len_down_out - i - 1] + down_out[len_down_out - i - 2]
#             out_len = down_out[len_down_out - i - 2]
#             self.up_blocks.append(block_model(self.input_channels, in_len, out_len, self.individual))
#
#         #self.linear_out = nn.Linear(self.out_len * 2, self.out_len)
#
#     def forward(self, x):
#         x = self.revin_layer(x, 'norm')
#         x1 = x.permute(0,2,1)
#         e_out = []
#         i = 0
#         for down_block in self.down_blocks:
#             e_out.append(down_block(x1))
#             x1 = self.Maxpools[i](x1)
#             i = i+1
#
#         e_last = e_out[self.stage_num - 1]
#         for i in range(self.stage_num - 1):
#             e_last = torch.cat((e_out[self.stage_num - i -2], e_last), dim=2)
#             e_last = self.up_blocks[i](e_last)
#         e_last = e_last.permute(0,2,1)
#         e_last = self.revin_layer(e_last, 'denorm')
#         return e_last
class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.layer = configs.e_layers
        self.output_attention = False
        # configs.d_model = configs.seq_len
        self.d_model = configs.d_model
        self.patch_len = patch_len
        self.stride = stride
        self.kernel_sizes = (1, 3, 5)
        self.kernel_sizes1 = (5, 5, 5)
        # unet
        self.input_channels = configs.enc_in
        # self.input_len = configs.seq_len
        # self.out_len = configs.pred_len
        self.individual = configs.individual
        # 下采样设定
        self.stage_num = configs.stage_num
        self.stage_pool_kernel = configs.stage_pool_kernel
        self.stage_pool_stride = configs.stage_pool_stride
        self.stage_pool_padding = configs.stage_pool_padding

        padding = stride

        # patching and embedding
        # self.patch_embedding = PatchEmbedding(
        #     configs.d_model, patch_len, stride, padding, configs.dropout)

        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

        self.patch_num = int((self.seq_len - self.patch_len) / self.stride + 1) + 1
        self.a = self.patch_num
        self.head_dropout = configs.dropout
        self.PatchMixer_blocks = PatchMixerLayer(input_dim=self.patch_num, output_dim=self.a,
                                                 kernel_sizes=self.kernel_sizes)
        self.PatchMixer_blocks1 = PatchMixerLayer(input_dim=self.d_model, output_dim=self.d_model,
                                                  kernel_sizes=self.kernel_sizes)

        self.Linear = nn.Sequential()
        self.Linear.add_module('Linear', nn.Linear(configs.seq_len, self.pred_len))
        self.w_dec = torch.nn.Parameter(torch.FloatTensor([configs.w_lin] * configs.enc_in), requires_grad=True)
        self.revin_layer = RevIN(configs.enc_in)
        self.RFFTM = RFFTM()
        self.ImprovedGatingMechanism = ImprovedGatingMechanism(self.pred_len, 128, configs.enc_in)
        self.mlp = MLPBlock(
            dim=1,  # 时间步维度
            in_features=configs.seq_len,
            hid_features=128,
            out_features=configs.pred_len,  # 输出特征数应与输入的通道数相同
            activ="gelu",
            drop=0.1,
            jump_conn='proj'  # 投影跳连
        )

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        # self.enc_embedding = DataEmbedding(
        #     configs.enc_in,
        #     configs.d_model,
        #     configs.embed,
        #     configs.freq,
        #     configs.dropout,
        # )
        # unet
        len_in = self.seq_len
        len_out = self.pred_len
        down_in = [len_in]
        down_out = [len_out]
        i = 0
        while i < self.stage_num - 1:
            linear_in = int(
                (len_in + 2 * self.stage_pool_padding - self.stage_pool_kernel) / self.stage_pool_stride + 1)
            linear_out = int(
                (len_out + 2 * self.stage_pool_padding - self.stage_pool_kernel) / self.stage_pool_stride + 1)
            down_in.append(linear_in)
            down_out.append(linear_out)
            len_in = linear_in
            len_out = linear_out
            i = i + 1

        # 最大池化层
        self.Maxpools = nn.ModuleList()
        self.Avgpools = nn.ModuleList()
        # 左边特征提取层
        self.down_blocks = nn.ModuleList()
        for in_len, out_len in zip(down_in, down_out):
            self.down_blocks.append(block_model(self.input_channels, in_len, out_len, self.individual))
            self.Maxpools.append(nn.AvgPool1d(kernel_size=self.stage_pool_kernel, stride=self.stage_pool_stride,
                                              padding=self.stage_pool_padding))
            self.Avgpools.append(nn.AvgPool1d(kernel_size=self.stage_pool_kernel, stride=self.stage_pool_stride,
                                              padding=self.stage_pool_padding))

        # 右边特征融合层
        self.up_blocks = nn.ModuleList()
        len_down_out = len(down_out)
        for i in range(len_down_out - 1):
            print(len_down_out, len_down_out - i - 1, len_down_out - i - 2)
            in_len = down_out[len_down_out - i - 1] + down_out[len_down_out - i - 2]
            out_len = down_out[len_down_out - i - 2]
            self.up_blocks.append(block_model(self.input_channels, in_len, out_len, self.individual))

    def forward(self, x_enc):
        # [B,L,C]
        x_enc = self.revin_layer(x_enc, 'norm')

        # [B,C,L]
        enc_out = x_enc.permute(0, 2, 1)

        # enc_out, n_vars = self.patch_embedding(enc_out)
        #
        # #---------patchcnn------------------------------------------------------------
        #
        #
        # # patch
        # # enc_out = enc_out.permute(0, 2, 1)
        # enc_out = self.PatchMixer_blocks(enc_out)
        # enc_out = enc_out.permute(0, 2, 1)
        # # time
        # enc_out = self.PatchMixer_blocks1(enc_out)
        #
        #
        #
        # enc_out = torch.reshape(
        #     enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # # z: [bs x nvars x d_model x patch_num]
        # enc_out = enc_out.permute(0, 1, 3, 2)
        #
        # # Decoder
        # dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        # dec_out = dec_out.permute(0, 2, 1)
        #
        #
        # dec_out = dec_out[:, -self.pred_len:, :]
        e_out = []
        i = 0
        for down_block in self.down_blocks:
            e_out.append(down_block(enc_out))
            enc_out = efficient_exponential_weighting(enc_out, 0.9)
            enc_out_maxpool = self.Maxpools[i](enc_out)
            enc_out_avgpool = self.Avgpools[i](enc_out)
            enc_out = enc_out_avgpool + enc_out_maxpool
            i = i + 1

        e_last = e_out[self.stage_num - 1]
        for i in range(self.stage_num - 1):
            e_last = torch.cat((e_out[self.stage_num - i - 2], e_last), dim=2)
            e_last = self.up_blocks[i](e_last)
        dec_out = e_last.permute(0, 2, 1)
        # dec_out = self.revin_layer(e_last, 'denorm')

        linear_enc = self.RFFTM(x_enc)
        #

        linear_out = self.mlp(linear_enc)
        #
        dec_out = dec_out * linear_out

        dec_out = self.revin_layer(dec_out, 'denorm')
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :]
        else:
            return dec_out