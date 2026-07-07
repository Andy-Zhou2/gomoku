"""Correctness checks: win detection vs a slow reference, feature planes,
and MCTS tactical sanity (finds win-in-1 with an untrained net).

    python tests_sanity.py
"""
import numpy as np
import torch

from game import A, BOARD, win_kernels, check_win, make_features
from mcts import BatchedMCTS
from net import build_net

DEV = "cuda"


def slow_win_check(plane):
    """Reference python 5-in-a-row check. plane: (15,15) numpy 0/1."""
    for r in range(BOARD):
        for c in range(BOARD):
            if plane[r, c] == 0:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                rr, cc, k = r, c, 0
                while 0 <= rr < BOARD and 0 <= cc < BOARD and plane[rr, cc] == 1:
                    k += 1
                    if k >= 5:
                        return True
                    rr += dr
                    cc += dc
    return False


def test_win_detection():
    w = win_kernels(DEV)
    # directed cases: rows/cols/diags incl. edges and corners
    cases = []
    for (r, c, dr, dc) in [(0, 0, 0, 1), (14, 10, 0, 1), (0, 0, 1, 0), (10, 14, 1, 0),
                           (0, 0, 1, 1), (10, 10, 1, 1), (0, 4, 1, -1), (10, 4, 1, -1),
                           (7, 5, 0, 1), (3, 7, 1, 1)]:
        p = np.zeros((BOARD, BOARD), dtype=np.float32)
        for k in range(5):
            p[r + dr * k, c + dc * k] = 1
        cases.append((p, True))
    # negatives: 4-in-a-row, broken 5, L-shapes
    for (r, c, dr, dc) in [(0, 0, 0, 1), (5, 5, 1, 1)]:
        p = np.zeros((BOARD, BOARD), dtype=np.float32)
        for k in range(4):
            p[r + dr * k, c + dc * k] = 1
        cases.append((p, False))
    p = np.zeros((BOARD, BOARD), dtype=np.float32)
    p[7, 2:6] = 1; p[7, 7] = 1  # broken five
    cases.append((p, False))

    planes = torch.tensor(np.stack([c[0] for c in cases]).reshape(-1, A), device=DEV)
    got = check_win(planes, w).cpu().numpy()
    want = np.array([c[1] for c in cases])
    assert (got == want).all(), f"directed win cases failed: got {got}, want {want}"

    # fuzz vs slow reference
    rng = np.random.default_rng(0)
    boards = (rng.random((500, BOARD, BOARD)) < 0.25).astype(np.float32)
    got = check_win(torch.tensor(boards.reshape(500, A), device=DEV), w).cpu().numpy()
    want = np.array([slow_win_check(b) for b in boards])
    assert (got == want).all(), f"fuzz mismatch on {np.where(got != want)[0]}"
    print("PASS win detection (10 directed + 3 negative + 500 fuzz)")


def test_features():
    boards = torch.zeros(2, 2, A, device=DEV)
    boards[0, 0, 5] = 1   # game0: black stone at 5
    boards[0, 1, 6] = 1   # game0: white stone at 6
    player = torch.tensor([1, 0], device=DEV)  # game0: white to move
    last = torch.tensor([5, -1], device=DEV)
    f = make_features(boards, player, last)
    assert f.shape == (2, 4, BOARD, BOARD)
    f0 = f[0].flatten(1)
    assert f0[0, 6] == 1 and f0[0, 5] == 0     # "my" = white
    assert f0[1, 5] == 1 and f0[1, 6] == 0     # "opp" = black
    assert f0[2, 5] == 1 and f0[2].sum() == 1  # last move
    assert f0[3].sum() == 0                    # white to move -> color plane 0
    assert f[1, 3].sum() == A                  # black to move -> all ones
    assert f[1, 2].sum() == 0                  # no last move
    print("PASS feature planes")


def _tactical_position(win_for):
    """Board where `win_for` (0=black,1=white) has 4-in-a-row at row 7, cols 3..6,
    open at col 7 (=action 112). Opponent stones scattered. Returns board, mover."""
    boards = torch.zeros(1, 2, A, device=DEV)
    me, opp = win_for, 1 - win_for
    for c in range(3, 7):
        boards[0, me, 7 * BOARD + c] = 1
    # opponent stones (non-threatening, keeps stone counts legal)
    opp_cells = [1 * BOARD + 1, 2 * BOARD + 12, 12 * BOARD + 2, 13 * BOARD + 11]
    for cell in opp_cells[:4 if win_for == 1 else 3]:
        boards[0, opp, cell] = 1
    return boards


def test_mcts_win_in_1():
    torch.manual_seed(0)
    net = build_net(32, 3, DEV).eval()  # untrained
    for color in (0, 1):
        boards = _tactical_position(color)
        mcts = BatchedMCTS(net, num_games=1, sims=800, device=DEV)
        player = torch.tensor([color], device=DEV)
        last = torch.tensor([-1], device=DEV)
        pi = mcts.run(boards, player, last, add_noise=False)
        best = int(pi.argmax())
        target = 7 * BOARD + 7  # completes five: cols 3..7 — or col 2 (action 107) also works
        alt = 7 * BOARD + 2
        assert best in (target, alt), \
            f"color {color}: expected win move {target} or {alt}, got {best} (pi={pi[0, best]:.2f})"
        # winning move should have Q near +1 at root
        q = (mcts.Wsa[0, 0, best] / mcts.Nsa[0, 0, best]).item()
        assert q > 0.9, f"root Q of winning move = {q}"
    print("PASS MCTS finds win-in-1 for both colors (untrained net, 800 sims)")


def test_mcts_batch_consistency():
    """Batched G=64 run: pi rows are valid distributions over legal moves."""
    torch.manual_seed(0)
    net = build_net(32, 3, DEV).eval()
    G = 64
    boards = torch.zeros(G, 2, A, device=DEV)
    # random legal openings of 6 plies
    for g in range(G):
        cells = torch.randperm(A)[:6]
        for i, cell in enumerate(cells):
            boards[g, i % 2, cell] = 1
    player = torch.zeros(G, dtype=torch.long, device=DEV)
    last = torch.full((G,), -1, dtype=torch.long, device=DEV)
    mcts = BatchedMCTS(net, num_games=G, sims=100, device=DEV)
    pi = mcts.run(boards, player, last)
    assert torch.allclose(pi.sum(-1), torch.ones(G, device=DEV), atol=1e-4)
    occupied = boards.sum(1) > 0
    assert (pi[occupied] == 0).all(), "visits assigned to illegal moves"
    assert (mcts.Nsa[:, 0].sum(-1) == 100).all(), "root visit count != sims"
    print("PASS batched MCTS consistency (G=64)")


if __name__ == "__main__":
    test_win_detection()
    test_features()
    test_mcts_batch_consistency()
    test_mcts_win_in_1()
    print("\nALL PASS")
