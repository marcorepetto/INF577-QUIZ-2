import torch
import torch.nn as nn

class MLP_classifier(nn.Module):
    def __init__(self, cfg):
        super(MLP_classifier, self).__init__()
        self.cfg = cfg
        # Assuming input size 128 from CNN backbone (Block 3 size)
        self.fc1 = nn.Linear(128, self.cfg.linear[0].layer_1)
        self.dropout = nn.Dropout(self.cfg.dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.cfg.linear[0].layer_1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x