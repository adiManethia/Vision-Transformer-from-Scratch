# model.py
import torch
import torch.nn as nn
from config import Config as cfg

class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(
            cfg.NUM_CHANNELS, cfg.EMBEDDING_DIMS,
            kernel_size=cfg.PATCH_SIZE, stride=cfg.PATCH_SIZE
        )

    def forward(self, x):
        x = self.proj(x)           # (B, D, 4, 4)
        x = x.flatten(2)           # (B, D, 16)
        x = x.transpose(1, 2)      # (B, 16, D)
        return x


class TransformerEncode(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.EMBEDDING_DIMS)
        self.ln2 = nn.LayerNorm(cfg.EMBEDDING_DIMS)
        self.mha = nn.MultiheadAttention(
            cfg.EMBEDDING_DIMS, cfg.ATTENTION_HEADS, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(cfg.EMBEDDING_DIMS, cfg.MLP_HIDDEN),
            nn.GELU(),
            nn.Linear(cfg.MLP_HIDDEN, cfg.EMBEDDING_DIMS),
        )

    def forward(self, x):
        res = x
        x = self.ln1(x)
        attn_out, _ = self.mha(x, x, x)
        x = attn_out + res

        res = x
        x = self.ln2(x)
        x = self.mlp(x) + res
        return x

    def forward_with_attn(self, x):
        res = x
        x = self.ln1(x)
        attn_out, attn_weight = self.mha(x, x, x, average_attn_weights=False)
        x = attn_out + res
        res = x
        x = self.ln2(x)
        x = self.mlp(x) + res
        attn_weight = attn_weight.transpose(1, 2)  # (B, heads, S, S)
        return x, attn_weight


class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.EMBEDDING_DIMS)
        self.linear = nn.Linear(cfg.EMBEDDING_DIMS, cfg.NUM_CLASSES)

    def forward(self, x):
        return self.linear(self.ln(x))


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_emb = PatchEmbedding()
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.EMBEDDING_DIMS))
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.NUM_PATCHES + 1, cfg.EMBEDDING_DIMS))
        self.blocks = nn.Sequential(*[TransformerEncode() for _ in range(cfg.TRANSFORMER_BLOCKS)])
        self.head = MLPHead()

    def forward(self, x):
        x = self.patch_emb(x)
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_emb
        x = self.blocks(x)
        return self.head(x[:, 0])

    def forward_with_attention(self, x):
        x = self.patch_emb(x)
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_emb

        attn_weights = []
        for blk in self.blocks:
            x, attn_weight = blk.forward_with_attn(x)
            attn_weights.append(attn_weight)
        return self.head(x[:, 0]), attn_weights