import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import torch

from architectures.ft_transformer import FTTransformer


model = FTTransformer(
    num_features=25
)

print(model)

x = torch.randn(8, 25)

output = model(x)

print("\nInput Shape :", x.shape)
print("Output Shape:", output.shape)