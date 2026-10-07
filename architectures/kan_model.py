import torch
import torch.nn as nn
from kan import KAN


class KANClassifier(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.kan = KAN(
            width=[input_dim, 64, 32, 1],
            grid=5,
            k=3,
            seed=42
        )

    def forward(self, x):

        return self.kan(x)