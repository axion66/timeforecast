import torch
import torch.nn as nn
import torch.nn.functional as F
from .revin import RevIN


class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, p=0.05) :
        super().__init__()
        self.l = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(out_dim, in_dim)
        )

    def forward(self, x):
        # [B, L, D] or [B, D, L]
        return self.l(x)


class TemporalMix(nn.Module):
    def __init__(self, input_dim, mlp_dim, sampling=[1, 2, 4, 8]) :
        super().__init__()

        assert sampling in [1, 2, 3, 4, 6, 8, 12]
        self.sampling = sampling
        self.temporal_fac = nn.ModuleList([
            MLPBlock(input_dim // sampling, mlp_dim) for _ in range(sampling)
        ])

    def merge(self, shape, x_list):
        x_new = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            x_new[:, :, idx::self.sampling] = x_pad

        return x_new

    def forward(self, x):
        x_samp = []
        for idx, samp in enumerate(self.temporal_fac):
            x_samp.append(samp(x[:, :, idx::self.sampling]))

        x = self.merge(x.shape, x_samp)

        return x


class ChannelMix(nn.Module):
    def __init__(self, input_dim, factorized_dim) :
        super().__init__()

        assert input_dim > factorized_dim
        self.channel_mixing = MLPBlock(input_dim, factorized_dim)

    def forward(self, x):

        return self.channel_mixing(x)




class MixerBlock(nn.Module):
    def __init__(self, 
                 tokens_dim, 
                 channels_dim, 
                 tokens_hidden_dim, 
                 channels_hidden_dim, 
                 sampling=[1, 2, 4, 8, 16],
                 norm_flag = True,
                 fac_T = True,
                 fac_C = True,
                ):
        super().__init__()
        self.tokens_mixing = TemporalMix(tokens_dim, tokens_hidden_dim, sampling) if fac_T else MLPBlock(tokens_dim, tokens_hidden_dim)
        self.channels_mixing = ChannelMix(channels_dim, channels_hidden_dim) if fac_C else None
        self.norm = nn.LayerNorm(channels_dim) if norm_flag else None

    def forward(self,x):
        # token-mixing [B, D, #tokens]
        y = self.norm(x) if self.norm else x
        y = self.tokens_mixing(y.transpose(1, 2)).transpose(1, 2)

        if self.channels_mixing:
            y += x
            res = y
            y = self.norm(y) if self.norm else y
            y = res + self.channels_mixing(y)

        return y



class ChannelProjection(nn.Module):
    def __init__(self, seq_len, pred_len):
        super().__init__()

        self.linears = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, L, D]
        x = self.linears(x.transpose(1, 2)).transpose(1, 2)

        return x
    
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.mlp_blocks = nn.ModuleList([
            MixerBlock(
                configs.seq_len,
                configs.enc_in,
                configs.d_model,
                configs.d_ff,
                configs.fac_T,
                configs.fac_C,
                configs.sampling,
                configs.norm) 
                for _ in range(configs.e_layers) # 2
        ])
        self.norm = nn.LayerNorm(
            configs.enc_in
        ) if configs.norm else None
        self.projection = ChannelProjection(
            configs.seq_len,
            configs.pred_len,
            configs.enc_in,
            configs.individual
        )
        self.rev = RevIN(configs.enc_in) if configs.rev else None

    def forward(self, x):
        x = self.rev(x, 'norm') if self.rev else x

        for block in self.mlp_blocks:
            x = block(x)

        x = self.norm(x) if self.norm else x
        x = self.projection(x)
        x = self.rev(x, 'denorm') if self.rev else x
        return x
    