"""Replay buffer (CPU numpy ring) with dihedral augmentation at sample time."""
import numpy as np
import torch

from game import A, BOARD, IN_PLANES, dihedral


class ReplayBuffer:
    def __init__(self, capacity=500_000, planes=IN_PLANES, board=BOARD):
        self.cap = capacity
        self.board = board
        self.actions = board * board
        self.feats = np.zeros((capacity, planes, board, board), dtype=np.uint8)
        self.pis = np.zeros((capacity, self.actions), dtype=np.float16)
        self.zs = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add_game(self, feats, pis, zs):
        n = len(zs)
        idx = (self.ptr + np.arange(n)) % self.cap
        self.feats[idx] = feats
        self.pis[idx] = pis
        self.zs[idx] = zs
        self.ptr = (self.ptr + n) % self.cap
        self.size = min(self.size + n, self.cap)

    def sample(self, batch_size, device="cuda"):
        idx = np.random.randint(0, self.size, batch_size)
        f = torch.from_numpy(self.feats[idx]).to(device).float()
        p = torch.from_numpy(self.pis[idx].astype(np.float32)).to(device)
        z = torch.from_numpy(self.zs[idx]).to(device)
        # random dihedral transform (one per batch)
        k = np.random.randint(8)
        f = dihedral(f, k)
        p = dihedral(p.view(-1, self.board, self.board), k).reshape(-1, self.actions)
        return f, p, z
