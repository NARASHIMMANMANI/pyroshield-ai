import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import torch

from architectures.kan_model import KANClassifier

model = KANClassifier(input_dim=25)

print(model)

x = torch.randn(8, 25)

y = model(x)

print()
print("Input Shape :", x.shape)
print("Output Shape:", y.shape)