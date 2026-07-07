"""Batched, fully-tensorized 15x15 Gomoku rules on GPU.

Board representation: (G, 2, 225) float32 — plane 0 = black stones, plane 1 = white.
Player: (G,) long — 0 = black (moves first), 1 = white.
Actions: 0..224, action = row * 15 + col.
"""
import torch
import torch.nn.functional as F

BOARD = 15
A = BOARD * BOARD  # 225
IN_PLANES = 4      # [my stones, opp stones, last move, am-I-black]


def win_kernels(device):
    """Conv kernels detecting 5-in-a-row: horizontal, vertical, both diagonals."""
    w = torch.zeros(4, 1, 5, 5, device=device)
    w[0, 0, 2, :] = 1.0                      # horizontal
    w[1, 0, :, 2] = 1.0                      # vertical
    w[2, 0] = torch.eye(5, device=device)    # main diagonal
    w[3, 0] = torch.eye(5, device=device).flip(0)  # anti-diagonal
    return w


def check_win(planes, kernels):
    """planes: (G, A) float32 of ONE player's stones. Returns (G,) bool: any 5-in-a-row."""
    x = F.conv2d(planes.view(-1, 1, BOARD, BOARD), kernels, padding=2)
    return x.flatten(1).amax(1) > 4.5


def legal_mask(boards):
    """boards: (G, 2, A) -> (G, A) bool, True where cell empty."""
    return boards.sum(1) == 0


def make_features(boards, player, last):
    """Features from the perspective of the player to move.

    boards: (G, 2, A), player: (G,) long, last: (G,) long (-1 if none).
    Returns (G, 4, 15, 15) float32.
    """
    G = boards.shape[0]
    ar = torch.arange(G, device=boards.device)
    me = boards[ar, player]
    opp = boards[ar, 1 - player]
    lastp = torch.zeros(G, A, device=boards.device)
    has_last = last >= 0
    lastp[ar[has_last], last[has_last]] = 1.0
    color = (player == 0).float().unsqueeze(1).expand(G, A)
    return torch.stack([me, opp, lastp, color], 1).view(G, IN_PLANES, BOARD, BOARD)


def dihedral(x, k):
    """Apply one of the 8 board symmetries to (..., 15, 15). k in 0..7."""
    if k >= 4:
        x = x.flip(-1)
    return torch.rot90(x, k % 4, dims=(-2, -1))


def render(board_g, last=-1):
    """ASCII render of one game's (2, A) board tensor."""
    b = board_g.view(2, BOARD, BOARD).cpu()
    cols = "ABCDEFGHIJKLMNO"
    lines = ["   " + " ".join(cols)]
    for r in range(BOARD):
        row = []
        for c in range(BOARD):
            i = r * BOARD + c
            if b[0, r, c] > 0:
                ch = "X" if i != last else "#"
            elif b[1, r, c] > 0:
                ch = "O" if i != last else "@"
            else:
                ch = "."
            row.append(ch)
        lines.append(f"{r:2d} " + " ".join(row))
    return "\n".join(lines)
