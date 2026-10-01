import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_size=3072,
        hidden1_size=256,
        hidden2_size=128,
        hidden3_size=64,
        output_size=10
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden1_size),
            nn.ReLU(),

            nn.Linear(hidden1_size, hidden2_size),
            nn.ReLU(),

            nn.Linear(hidden2_size, hidden3_size),
            nn.ReLU(),

            nn.Linear(hidden3_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def count_parameters(self):
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )