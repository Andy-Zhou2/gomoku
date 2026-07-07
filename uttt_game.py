"""Batched, fully-tensorized Ultimate Tic-Tac-Toe on GPU.

Rules: 3x3 grid of 3x3 sub-boards. Your move's position WITHIN its sub-board
determines which sub-board the opponent must play in next. If that target
board is already decided (won or full), the opponent may play anywhere.
Win 3 sub-boards in a row to win the game; no legal moves left = draw.
Cells inside a decided board are dead.

Action layout is image row-major on the 9x9 grid: a = r*9 + c.
sub-board b = (r//3)*3 + c//3, position within it p = (r%3)*3 + c%3.

State (per batch of G games):
  cells:  (G, 2, 81) float32  occupancy per player (0 = X/first, 1 = O)
  bstat:  (G, 9)     long     0 open, 1 won by X, 2 won by O, 3 full-drawn
  forced: (G,)       long     sub-board the mover MUST play in, -1 = any
  player: (G,)       long     side to move
  last:   (G,)       long     last move, -1 none
"""
import torch

B = 9
A = 81
IN_PLANES = 8

_MAPS = {}


def maps(device):
    """(BOARD_OF_A (81,), POS_OF_A (81,), CELLS_OF_BOARD (9,9), LINES (8,3))"""
    if device not in _MAPS:
        a = torch.arange(A, device=device)
        r, c = a // 9, a % 9
        board = (r // 3) * 3 + (c // 3)
        pos = (r % 3) * 3 + (c % 3)
        cells = torch.empty(9, 9, dtype=torch.long, device=device)
        cells[board, pos] = a
        lines = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8],
                              [0, 3, 6], [1, 4, 7], [2, 5, 8],
                              [0, 4, 8], [2, 4, 6]], device=device)
        _MAPS[device] = (board, pos, cells, lines)
    return _MAPS[device]


class UtttState:
    def __init__(self, G, device):
        self.G, self.dev = G, device
        self.cells = torch.zeros(G, 2, A, device=device)
        self.bstat = torch.zeros(G, 9, dtype=torch.long, device=device)
        self.forced = torch.full((G,), -1, dtype=torch.long, device=device)
        self.player = torch.zeros(G, dtype=torch.long, device=device)
        self.last = torch.full((G,), -1, dtype=torch.long, device=device)

    def clone(self):
        s = UtttState.__new__(UtttState)
        s.G, s.dev = self.G, self.dev
        for k in ("cells", "bstat", "forced", "player", "last"):
            setattr(s, k, getattr(self, k).clone())
        return s

    def reset_(self, g):
        self.cells[g] = 0
        self.bstat[g] = 0
        self.forced[g] = -1
        self.player[g] = 0
        self.last[g] = -1


def legal_mask(st):
    """(G, 81) bool."""
    BOARD_OF_A, _, _, _ = maps(st.dev)
    empty = st.cells.sum(1) == 0
    open_b = st.bstat == 0                                   # (G,9)
    forced_1h = torch.zeros_like(open_b)
    has_f = st.forced >= 0
    forced_1h[has_f] = torch.nn.functional.one_hot(st.forced[has_f], 9).bool()
    allowed = torch.where(has_f.unsqueeze(1), forced_1h & open_b, open_b)
    return empty & allowed.gather(1, BOARD_OF_A.expand(st.G, A))


def apply_move(st, move, mask):
    """Play `move` for games where mask is True. Mutates st (including player
    flip and forced-board update). Returns (win, draw) bool tensors — from the
    MOVER's perspective."""
    BOARD_OF_A, POS_OF_A, CELLS_OF_BOARD, LINES = maps(st.dev)
    ar = torch.arange(st.G, device=st.dev)
    g = mask
    p = st.player.clone()                                    # mover
    st.cells[ar[g], p[g], move[g]] = 1.0

    b = BOARD_OF_A[move.clamp(min=0)]                        # (G,)
    idx = CELLS_OF_BOARD[b]                                  # (G,9)
    mine = st.cells[ar, p].gather(1, idx)                    # (G,9)
    sub_win = (mine[:, LINES] > 0).all(-1).any(-1) & g       # (G,8,3) -> (G,)
    filled = (st.cells[:, 0].gather(1, idx) + st.cells[:, 1].gather(1, idx)).sum(-1) == 9
    st.bstat[ar[sub_win], b[sub_win]] = 1 + p[sub_win]
    became_full = g & ~sub_win & filled
    st.bstat[ar[became_full], b[became_full]] = 3

    mymac = st.bstat == (1 + p).unsqueeze(1)                 # (G,9) bool
    win = mymac[:, LINES].all(-1).any(-1) & g
    draw = g & ~win & (st.bstat != 0).all(-1)

    tgt = POS_OF_A[move.clamp(min=0)]
    tgt_open = st.bstat.gather(1, tgt.unsqueeze(1)).squeeze(1) == 0
    st.forced = torch.where(g, torch.where(tgt_open, tgt, torch.full_like(tgt, -1)),
                            st.forced)
    st.last = torch.where(g, move, st.last)
    st.player = torch.where(g, 1 - p, st.player)
    return win, draw


def make_features(st):
    """(G, 8, 9, 9) from the mover's perspective: [my cells, opp cells,
    my won boards, opp won boards, drawn boards, legal mask, last move, color]."""
    BOARD_OF_A, _, _, _ = maps(st.dev)
    G, ar = st.G, torch.arange(st.G, device=st.dev)
    p = st.player
    me = st.cells[ar, p]
    opp = st.cells[ar, 1 - p]
    exp = BOARD_OF_A.expand(G, A)
    myb = (st.bstat == (1 + p).unsqueeze(1)).float().gather(1, exp)
    oppb = (st.bstat == (2 - p).unsqueeze(1)).float().gather(1, exp)
    drawn = (st.bstat == 3).float().gather(1, exp)
    legal = legal_mask(st).float()
    lastp = torch.zeros(G, A, device=st.dev)
    has = st.last >= 0
    lastp[ar[has], st.last[has]] = 1.0
    color = (p == 0).float().unsqueeze(1).expand(G, A)
    return torch.stack([me, opp, myb, oppb, drawn, legal, lastp, color], 1) \
        .view(G, IN_PLANES, B, B)


def render(st, g=0):
    """ASCII render of one game."""
    cells = st.cells[g].cpu()
    out = []
    for r in range(9):
        row = []
        for c in range(9):
            a = r * 9 + c
            ch = "X" if cells[0, a] > 0 else "O" if cells[1, a] > 0 else "."
            row.append(ch + (" |" if c % 3 == 2 and c < 8 else ""))
        out.append(" ".join(row))
        if r % 3 == 2 and r < 8:
            out.append("-" * 25)
    st_txt = " ".join("XO=?"[int(s) - 1] if s > 0 else "." for s in st.bstat[g].cpu())
    return "\n".join(out) + f"\nboards: {st_txt}  forced: {int(st.forced[g])} " \
        f"to move: {'X' if int(st.player[g]) == 0 else 'O'}"
