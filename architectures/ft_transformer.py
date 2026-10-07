import torch
import torch.nn as nn


# =====================================================
# Feature Tokenizer
# =====================================================

class FeatureTokenizer(nn.Module):

    def __init__(self, num_features, embedding_dim):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(num_features, embedding_dim)
        )

        self.bias = nn.Parameter(
            torch.randn(num_features, embedding_dim)
        )

    def forward(self, x):

        # x : (batch_size, num_features)

        x = x.unsqueeze(-1)

        return x * self.weight + self.bias


# =====================================================
# CLS Token
# =====================================================

class CLSToken(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.cls = nn.Parameter(
            torch.randn(1, 1, embedding_dim)
        )

    def forward(self, x):

        batch_size = x.size(0)

        cls = self.cls.expand(batch_size, -1, -1)

        return torch.cat([cls, x], dim=1)


# =====================================================
# Feed Forward Network
# =====================================================

class FeedForward(nn.Module):

    def __init__(self, embedding_dim, dropout):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embedding_dim,
                embedding_dim * 4
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                embedding_dim * 4,
                embedding_dim
            )

        )

    def forward(self, x):

        return self.network(x)


# =====================================================
# Transformer Block
# =====================================================

class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim, heads, dropout):

        super().__init__()

        self.attention = nn.MultiheadAttention(

            embed_dim=embedding_dim,

            num_heads=heads,

            dropout=dropout,

            batch_first=True

        )

        self.norm1 = nn.LayerNorm(embedding_dim)

        self.norm2 = nn.LayerNorm(embedding_dim)

        self.ff = FeedForward(
            embedding_dim,
            dropout
        )

    def forward(self, x):

        attention_output, _ = self.attention(
            x,
            x,
            x
        )

        x = self.norm1(
            x + attention_output
        )

        ff = self.ff(x)

        x = self.norm2(
            x + ff
        )

        return x


# =====================================================
# FT Transformer
# =====================================================

class FTTransformer(nn.Module):

    def __init__(

        self,

        num_features,

        embedding_dim=64,

        depth=4,

        heads=8,

        dropout=0.1

    ):

        super().__init__()

        self.tokenizer = FeatureTokenizer(
            num_features,
            embedding_dim
        )

        self.cls = CLSToken(
            embedding_dim
        )

        self.transformer = nn.Sequential(

            *[
                TransformerBlock(
                    embedding_dim,
                    heads,
                    dropout
                )
                for _ in range(depth)
            ]

        )

        self.classifier = nn.Sequential(

            nn.LayerNorm(
                embedding_dim
            ),

            nn.Linear(
                embedding_dim,
                128
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                128,
                1
            )

        )

    def forward(self, x):

        x = self.tokenizer(x)

        x = self.cls(x)

        x = self.transformer(x)

        cls_token = x[:, 0]

        logits = self.classifier(cls_token)

        return logits