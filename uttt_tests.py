"""UTTT correctness: fuzz the tensor rules against a slow python reference,
plus MCTS tactical sanity.

    python uttt_tests.py
"""
import random

import torch

from net import build_net
from uttt_game import A, B, IN_PLANES, UtttState, legal_mask, apply_move, maps
from uttt_mcts import UtttMCTS

DEV = "cuda"
LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8],
         [0, 4, 8], [2, 4, 6]]


def board_of(a):
    r, c = a // 9, a % 9
    return (r // 3) * 3 + (c // 3)


def pos_of(a):
    r, c = a // 9, a % 9
    return (r % 3) * 3 + (c % 3)


CELLS_OF = [[None] * 9 for _ in range(9)]
for a in range(81):
    CELLS_OF[board_of(a)][pos_of(a)] = a


class RefUttt:
    """Slow reference implementation."""

    def __init__(self):
        self.cells = [[0] * 81, [0] * 81]
        self.bstat = [0] * 9
        self.forced = -1
        self.player = 0

    def legal(self):
        out = []
        for a in range(81):
            if self.cells[0][a] or self.cells[1][a]:
                continue
            b = board_of(a)
            if self.bstat[b] != 0:
                continue
            if self.forced >= 0 and b != self.forced:
                continue
            out.append(a)
        return out

    def play(self, a):
        p = self.player
        self.cells[p][a] = 1
        b, pos = board_of(a), pos_of(a)
        sub = [self.cells[p][CELLS_OF[b][i]] for i in range(9)]
        if any(all(sub[i] for i in ln) for ln in LINES):
            self.bstat[b] = 1 + p
        elif all(self.cells[0][CELLS_OF[b][i]] or self.cells[1][CELLS_OF[b][i]]
                 for i in range(9)):
            self.bstat[b] = 3
        mac = [1 if s == 1 + p else 0 for s in self.bstat]
        win = any(all(mac[i] for i in ln) for ln in LINES)
        draw = (not win) and all(s != 0 for s in self.bstat)
        self.forced = pos if self.bstat[pos] == 0 else -1
        self.player = 1 - p
        return win, draw


def test_fuzz(games=64, plies=4000, seed=0):
    random.seed(seed)
    st = UtttState(games, DEV)
    refs = [RefUttt() for _ in range(games)]
    ar = torch.arange(games, device=DEV)
    ones = torch.ones(games, dtype=torch.bool, device=DEV)
    total = 0
    finished = 0
    while total < plies:
        lm = legal_mask(st).cpu()
        moves = []
        for g in range(games):
            ref_legal = refs[g].legal()
            got = sorted(lm[g].nonzero().flatten().tolist())
            assert got == ref_legal, \
                f"legal mismatch g={g}: tensor={got} ref={ref_legal}"
            moves.append(random.choice(ref_legal))
        mv = torch.tensor(moves, device=DEV)
        win, draw = apply_move(st, mv, ones)
        win_np, draw_np = win.cpu().tolist(), draw.cpu().tolist()
        bs = st.bstat.cpu().tolist()
        fc = st.forced.cpu().tolist()
        pl = st.player.cpu().tolist()
        for g in range(games):
            rw, rd = refs[g].play(moves[g])
            assert (win_np[g], draw_np[g]) == (rw, rd), \
                f"outcome mismatch g={g} move={moves[g]}: tensor={(win_np[g], draw_np[g])} ref={(rw, rd)}"
            assert bs[g] == refs[g].bstat, f"bstat mismatch g={g}"
            assert fc[g] == refs[g].forced, f"forced mismatch g={g}: {fc[g]} vs {refs[g].forced}"
            assert pl[g] == refs[g].player
            if rw or rd:
                st.reset_(g)
                refs[g] = RefUttt()
                finished += 1
        total += games
    print(f"PASS fuzz: {total} plies, {finished} full games, legal/outcome/bstat/forced all match")


def test_mcts_win_in_1():
    torch.manual_seed(0)
    net = build_net(32, 3, DEV, board_size=B, in_planes=IN_PLANES).eval()
    st = UtttState(1, DEV)
    # X has won boards 0 and 1; board 2 has X at pos 0,1 -> pos 2 completes macro row
    st.bstat[0, 0] = 1
    st.bstat[0, 1] = 1
    st.cells[0, 0, CELLS_OF[2][0]] = 1
    st.cells[0, 0, CELLS_OF[2][1]] = 1
    # O stones somewhere harmless (board 4)
    st.cells[0, 1, CELLS_OF[4][0]] = 1
    st.cells[0, 1, CELLS_OF[4][1]] = 1
    st.forced[0] = -1
    st.player[0] = 0
    win_cell = CELLS_OF[2][2]

    m = UtttMCTS(net, 1, 300, device=DEV)
    pi = m.run(st, add_noise=False)
    best = int(pi[0].argmax())
    q = (m.Wsa[0, 0, best] / m.Nsa[0, 0, best]).item()
    assert best == win_cell, f"expected {win_cell}, got {best}"
    assert q > 0.9, f"win move Q={q}"
    print(f"PASS MCTS finds macro win-in-1 (cell {win_cell}, Q={q:.2f})")


def test_batch_pi_valid():
    torch.manual_seed(0)
    net = build_net(32, 3, DEV, board_size=B, in_planes=IN_PLANES).eval()
    G = 64
    st = UtttState(G, DEV)
    ones = torch.ones(G, dtype=torch.bool, device=DEV)
    # random 4-ply openings
    for _ in range(4):
        lm = legal_mask(st)
        mv = torch.multinomial(lm.float(), 1).squeeze(1)
        apply_move(st, mv, ones)
    m = UtttMCTS(net, G, 100, device=DEV)
    pi = m.run(st)
    assert torch.allclose(pi.sum(-1), torch.ones(G, device=DEV), atol=1e-4)
    assert (pi[~legal_mask(st)] == 0).all(), "pi mass on illegal moves"
    assert (m.Nsa[:, 0].sum(-1) == 100).all()
    print("PASS batched MCTS pi validity (G=64, forced-board rule active)")


if __name__ == "__main__":
    test_fuzz()
    test_batch_pi_valid()
    test_mcts_win_in_1()
    print("\nALL PASS")
