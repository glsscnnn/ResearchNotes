import torch
import torch.nn as nn
import torch.nn.functional as F

# do it the imperitive way no nn.Seq
class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()

        self.input_layer = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.hl1 = nn.Linear(64, 64)
        self.hl2 = nn.Linear(64, 32)
        self.hl3 = nn.Linear(32, 64)
        self.output_layer = nn.Linear(64, 10)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hl1(x)
        x = self.hl2(x)
        x = self.hl3(x)
        return self.output_layer(x)

with torch.no_grad():
    x = torch.randn(128)
    model = FFNN()
    print(model.forward(x))
