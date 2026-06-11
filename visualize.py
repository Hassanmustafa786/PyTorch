import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Define a simple model
class SinModel(nn.Module):
    def forward(self, x):
        return torch.sin(x ** 2)

# Create model + dummy input
model = SinModel()
x = torch.randn(1, 3)  # example input tensor

# TensorBoard writer
writer = SummaryWriter()

# Correct graph logging
writer.add_graph(model, x)

writer.close()


# run the following command to view the graph:
# tensorboard --logdir=runs