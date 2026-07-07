"""AlphaZero-style policy+value ResNet for 15x15 Gomoku."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from game import BOARD, A, IN_PLANES


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, channels=64, blocks=6):
        super().__init__()
        self.channels, self.blocks_n = channels, blocks
        self.stem = nn.Sequential(
            nn.Conv2d(IN_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        # policy head
        self.p_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * A, A)
        # value head
        self.v_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(A, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        h = self.blocks(self.stem(x))
        p = F.relu(self.p_bn(self.p_conv(h))).flatten(1)
        logits = self.p_fc(p)
        v = F.relu(self.v_bn(self.v_conv(h))).flatten(1)
        v = torch.tanh(self.v_fc2(F.relu(self.v_fc1(v)))).squeeze(-1)
        return logits, v


def build_net(channels=64, blocks=6, device="cuda"):
    return PolicyValueNet(channels, blocks).to(device)
