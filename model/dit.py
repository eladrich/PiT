# From the great https://github.com/cloneofsimo/minRF/blob/main/dit.py
# Code heavily based on https://github.com/Alpha-VLLM/LLaMA2-Accessory
# this is modeling code for DiT-LLaMA model

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half) / half).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype=next(self.parameters()).dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = int(dropout_prob > 0)
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]) < self.dropout_prob
            drop_ids = drop_ids.cuda()
            drop_ids = drop_ids.to(labels.device)
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.n_rep = 1
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
        self.k_norm = nn.LayerNorm(self.n_heads * self.head_dim)

    @staticmethod
    def reshape_for_broadcast(freqs_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        # assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        _freqs_cis = freqs_cis[: x.shape[1]]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return _freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(xq, xk, freqs_cis):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis_xq = Attention.reshape_for_broadcast(freqs_cis, xq_)
        freqs_cis_xk = Attention.reshape_for_broadcast(freqs_cis, xk_)

        xq_out = torch.view_as_real(xq_ * freqs_cis_xq).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis_xk).flatten(3)
        return xq_out, xk_out

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        dtype = xq.dtype

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq, xk = xq.to(dtype), xk.to(dtype)

        output = F.scaled_dot_product_attention(
            xq.permute(0, 2, 1, 3),
            xk.permute(0, 2, 1, 3),
            xv.permute(0, 2, 1, 3),
            dropout_p=0.0,
            is_causal=False,
        ).permute(0, 2, 1, 3)
        output = output.flatten(-2)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id,
        dim,
        n_heads,
        multiple_of,
        ffn_dim_multiplier,
        norm_eps,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 1024), 6 * dim, bias=True),
        )

    def forward(self, x, freqs_cis, adaln_input=None):
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(
                6, dim=1
            )

            x = x + gate_msa.unsqueeze(1) * self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa), freqs_cis
            )
            x = x + gate_mlp.unsqueeze(1) * self.feed_forward(modulate(self.ffn_norm(x), shift_mlp, scale_mlp))
        else:
            x = x + self.attention(self.attention_norm(x), freqs_cis)
            x = x + self.feed_forward(self.ffn_norm(x))

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, 1024), 2 * hidden_size, bias=True),
        )
        # # init zero
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT_Llama(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        embedding_dim=3,
        hidden_dim=512,
        n_layers=5,
        n_heads=16,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
    ):
        super().__init__()

        self.in_channels = embedding_dim
        self.out_channels = embedding_dim

        self.x_embedder = nn.Linear(embedding_dim, hidden_dim, bias=True)
        nn.init.constant_(self.x_embedder.bias, 0)

        self.t_embedder = TimestepEmbedder(min(hidden_dim, 1024))
        # self.y_embedder = LabelEmbedder(num_classes, min(dim, 1024), class_dropout_prob)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    layer_id,
                    hidden_dim,
                    n_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                )
                for layer_id in range(n_layers)
            ]
        )
        self.final_layer = FinalLayer(hidden_dim, self.out_channels)

        self.freqs_cis = DiT_Llama.precompute_freqs_cis(hidden_dim // n_heads, 4096)

    def forward(self, x, t, cond):
        self.freqs_cis = self.freqs_cis.to(x.device)

        x = torch.cat([x, cond], dim=1)

        x = self.x_embedder(x)

        t = self.t_embedder(t)  # (N, D)
        adaln_input = t.to(x.dtype)

        for layer in self.layers:
            x = layer(x, self.freqs_cis[: x.size(1)], adaln_input=adaln_input)

        x = self.final_layer(x, adaln_input)
        # Drop the cond part
        x = x[:, : -cond.size(1)]
        return x

    def forward_with_cfg(self, x, t, cond, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, cond)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    @staticmethod
    def precompute_freqs_cis(dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


def DiT_base(**kwargs):
    return DiT_Llama(in_dim=2048, hidden_dim=2048, n_layers=8, n_heads=32, **kwargs)


if __name__ == "__main__":
    model = DiT_Llama_600M_patch2()
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 100, (2,))
    y = torch.randint(0, 10, (2,))

    with torch.no_grad():
        out = model(x, t, y)
        print(out.shape)
        out = model.forward_with_cfg(x, t, y, 0.5)
        print(out.shape)
