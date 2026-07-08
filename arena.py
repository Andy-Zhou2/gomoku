"""Pit two nets against each other with batched MCTS, alternating colors.

    python arena.py runs/az/latest.pt random --games 128 --sims 200
    python arena.py ckptA.pt ckptB.pt

Second argument may be "random" (an untrained net) or a checkpoint path.
"""
import argparse

import torch

from game import A, win_kernels, check_win
from mcts import BatchedMCTS
from net import build_net

DEV = "cuda"


def load_net(path, channels=64, blocks=6):
    if path == "random":
        torch.manual_seed(123)
        return build_net(channels, blocks, DEV).eval()
    ck = torch.load(path, map_location=DEV, weights_only=True)
    net = build_net(ck.get("channels", channels), ck.get("blocks", blocks), DEV)
    net.load_state_dict(ck["net"])
    return net.eval()


@torch.inference_mode()
def arena(net_a, net_b, games=128, sims=200, temp_moves=4, seed=0, open_stones=0):
    """Returns (wins_a, draws, wins_b). Slot g: A plays black iff g even.

    open_stones > 0: seed each PAIR of slots (2k, 2k+1) with the same random
    opening (alternating colors, even count keeps black to move) and set
    temp_moves=0 — position-diverse, color-paired, fully greedy games. This
    discriminates strength better than temperature sampling, which in gomoku
    turns into a tactical lottery that punishes nuanced nets.
    """
    torch.manual_seed(seed)
    G = games
    boards = torch.zeros(G, 2, A, device=DEV)
    player = torch.zeros(G, dtype=torch.long, device=DEV)
    last = torch.full((G,), -1, dtype=torch.long, device=DEV)
    if open_stones > 0:
        temp_moves = 0
        k = open_stones - (open_stones % 2)  # even count: black still to move
        for g0 in range(0, G - 1, 2):
            cells = torch.randperm(A)[:k]
            for i in range(k):
                boards[g0, i % 2, cells[i]] = 1.0
                boards[g0 + 1, i % 2, cells[i]] = 1.0
            last[g0] = last[g0 + 1] = int(cells[-1])
    done = torch.zeros(G, dtype=torch.bool, device=DEV)
    winner = torch.full((G,), -1, dtype=torch.long, device=DEV)  # 0 black, 1 white, -1 draw
    ar = torch.arange(G, device=DEV)
    a_is_black = (ar % 2 == 0)
    win_w = win_kernels(DEV)
    m_a = BatchedMCTS(net_a, G, sims, device=DEV)
    m_b = BatchedMCTS(net_b, G, sims, device=DEV)

    for ply in range(A):
        if bool(done.all()):
            break
        # all live games are at the same ply -> same side to move everywhere
        p = int(player[(~done).nonzero()[0]]) if not bool(done.all()) else 0
        pi_a = m_a.run(boards, player, last, add_noise=False)
        pi_b = m_b.run(boards, player, last, add_noise=False)
        a_to_move = a_is_black == (torch.full_like(player, p) == 0)
        pi = torch.where(a_to_move.unsqueeze(1), pi_a, pi_b)
        if ply < temp_moves:  # opening diversity
            move = torch.multinomial(pi.clamp(min=1e-8), 1).squeeze(1)
        else:
            move = pi.argmax(-1)
        live = ~done
        boards[ar[live], player[live], move[live]] = 1.0
        win = check_win(boards[ar, player], win_w) & live
        last = torch.where(live, move, last)
        winner = torch.where(win, player, winner)
        full = (boards.sum((1, 2)) == A) & live & ~win
        done = done | win | full
        player = torch.where(live, 1 - player, player)

    w = winner.cpu()
    black_wins_a = ((w == 0) & a_is_black.cpu()).sum().item()
    white_wins_a = ((w == 1) & ~a_is_black.cpu()).sum().item()
    wins_a = black_wins_a + white_wins_a
    draws = (w == -1).sum().item()
    return wins_a, draws, G - wins_a - draws


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("net_a")
    ap.add_argument("net_b")
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    args = ap.parse_args()
    na = load_net(args.net_a, args.channels, args.blocks)
    nb = load_net(args.net_b, args.channels, args.blocks)
    wa, d, wb = arena(na, nb, args.games, args.sims)
    print(f"A ({args.net_a}) vs B ({args.net_b}): {wa} - {d} - {wb}  "
          f"(A winrate {100 * (wa + 0.5 * d) / args.games:.1f}%)")
