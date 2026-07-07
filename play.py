"""Play against a trained checkpoint in the terminal.

    python play.py runs/az/latest.pt --sims 800 --human black
Moves are entered as e.g. "H8" (column letter A-O + row number 0-14).
"""
import argparse

import torch

from arena import load_net
from game import A, BOARD, win_kernels, check_win, render
from mcts import BatchedMCTS

DEV = "cuda"
COLS = "ABCDEFGHIJKLMNO"


def parse_move(s):
    s = s.strip().upper()
    col = COLS.index(s[0])
    row = int(s[1:])
    assert 0 <= row < BOARD
    return row * BOARD + col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--human", choices=["black", "white"], default="black")
    args = ap.parse_args()

    net = load_net(args.ckpt)
    mcts = BatchedMCTS(net, 1, args.sims, device=DEV)
    win_w = win_kernels(DEV)
    boards = torch.zeros(1, 2, A, device=DEV)
    player = torch.zeros(1, dtype=torch.long, device=DEV)
    last = torch.tensor([-1], device=DEV)
    human = 0 if args.human == "black" else 1

    for ply in range(A):
        print("\n" + render(boards[0], int(last[0])))
        p = int(player[0])
        if p == human:
            while True:
                try:
                    mv = parse_move(input(f"your move ({'X' if p == 0 else 'O'}): "))
                    if boards[0, :, mv].sum() == 0:
                        break
                except (ValueError, IndexError, AssertionError):
                    pass
                print("invalid, try again (e.g. H8)")
        else:
            pi = mcts.run(boards, player, last, add_noise=False)
            mv = int(pi.argmax())
            v = (mcts.Wsa[0, 0, mv] / mcts.Nsa[0, 0, mv]).item()
            print(f"engine plays {COLS[mv % BOARD]}{mv // BOARD}  (Q={v:+.2f})")
        boards[0, p, mv] = 1.0
        last[0] = mv
        if bool(check_win(boards[:, p], win_w)[0]):
            print("\n" + render(boards[0], mv))
            print(("You win!" if p == human else "Engine wins!"))
            return
        player[0] = 1 - p
    print("Draw.")


if __name__ == "__main__":
    main()
