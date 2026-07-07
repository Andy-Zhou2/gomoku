"""Continuous batched self-play: G game slots run in lockstep; finished games
emit training records and the slot resets immediately, so every MCTS step is
always a full batch.
"""
import numpy as np
import torch

from game import A, win_kernels, check_win, make_features
from mcts import BatchedMCTS


class SelfPlayManager:
    def __init__(self, net, num_games=512, sims=200, c_puct=1.5,
                 dirichlet_alpha=0.15, dirichlet_eps=0.25,
                 temp_moves=8, device="cuda", log_moves=False, rand_open=0,
                 forced_playouts=False, q_filter=False):
        self.G = num_games
        self.dev = device
        self.temp_moves = temp_moves
        self.mcts = BatchedMCTS(net, num_games, sims, c_puct,
                                dirichlet_alpha, dirichlet_eps, device,
                                forced_playouts=forced_playouts, q_filter=q_filter)
        self.boards = torch.zeros(num_games, 2, A, device=device)
        self.player = torch.zeros(num_games, dtype=torch.long, device=device)
        self.move_num = torch.zeros(num_games, dtype=torch.long, device=device)
        self.last = torch.full((num_games,), -1, dtype=torch.long, device=device)
        self.ar = torch.arange(num_games, device=device)
        self.win_w = win_kernels(device)

        # randomized openings: seed each game with 0..rand_open random stones so
        # positions differ and either side can start with the stronger threats
        # (rand_open <= 8 guarantees no pre-made five)
        self.rand_open = min(rand_open, 8)
        self.slot_open = [[] for _ in range(num_games)]  # [[cell, color], ...]
        for g in range(num_games):
            self._random_opening(g)

        self.slot_records = [[] for _ in range(num_games)]  # (feat u8, pi f16, player)
        self.finished = []          # (feats, pis, zs) numpy arrays per game
        self.log_moves = log_moves  # also record per-move MCTS stats for viewing
        self.slot_logs = [[] for _ in range(num_games)]
        self.finished_logs = []     # dicts: winner, length, moves[{a,p,q,v,top}]
        self.completed_games = 0
        self.total_moves = 0
        self.results = {"black": 0, "white": 0, "draw": 0}
        self.game_lengths = []

    def _random_opening(self, g):
        self.slot_open[g] = []
        if self.rand_open == 0:
            return
        k = int(torch.randint(0, self.rand_open + 1, (1,)).item())
        if k == 0:
            return
        cells = torch.randperm(A)[:k]
        for i in range(k):
            c = int(cells[i])
            self.boards[g, i % 2, c] = 1.0
            self.slot_open[g].append([c, i % 2])
        self.player[g] = k % 2
        self.move_num[g] = k
        self.last[g] = int(cells[-1])

    @torch.inference_mode()
    def step(self):
        """Play one move in every slot."""
        pi = self.mcts.run(self.boards, self.player, self.last, add_noise=True)
        feats = make_features(self.boards, self.player, self.last)

        # move selection: sample (tau=1) early, argmax after
        sampled = torch.multinomial(pi.clamp(min=1e-8), 1).squeeze(1)
        greedy = pi.argmax(-1)
        move = torch.where(self.move_num < self.temp_moves, sampled, greedy)

        # record training targets (async-ish copies, then numpy slicing)
        feats_np = feats.to(torch.uint8).cpu().numpy()
        pi_np = pi.to(torch.float16).cpu().numpy()
        player_np = self.player.cpu().numpy()
        for g in range(self.G):
            self.slot_records[g].append((feats_np[g], pi_np[g], int(player_np[g])))

        if self.log_moves:
            n_root = self.mcts.Nsa[:, 0]
            q_root = self.mcts.Wsa[:, 0] / n_root.clamp(min=1.0)
            q_chosen = q_root[self.ar, move]                     # mover's perspective
            v_root = (self.mcts.Wsa[:, 0].sum(-1) / n_root.sum(-1).clamp(min=1.0))
            topv, topi = pi.topk(5, dim=-1)
            mv_np = move.cpu().numpy()
            qc_np, vr_np = q_chosen.cpu().numpy(), v_root.cpu().numpy()
            tv_np, ti_np = topv.cpu().numpy(), topi.cpu().numpy()
            for g in range(self.G):
                self.slot_logs[g].append({
                    "a": int(mv_np[g]), "p": int(player_np[g]),
                    "q": round(float(qc_np[g]), 3), "v": round(float(vr_np[g]), 3),
                    "top": [[int(ti_np[g, k]), round(float(tv_np[g, k]), 3)]
                            for k in range(5)],
                })

        # apply moves
        self.boards[self.ar, self.player, move] = 1.0
        win = check_win(self.boards[self.ar, self.player], self.win_w)
        self.move_num += 1
        draw = ~win & (self.move_num == A)
        done = win | draw
        self.last = move.clone()
        mover = self.player.clone()
        self.player = 1 - self.player
        self.total_moves += self.G

        if bool(done.any()):
            done_idx = done.nonzero(as_tuple=True)[0].cpu().numpy()
            win_np = win.cpu().numpy()
            mover_np = mover.cpu().numpy()
            for g in done_idx:
                g = int(g)
                winner = int(mover_np[g]) if win_np[g] else -1
                recs = self.slot_records[g]
                fs = np.stack([r[0] for r in recs])
                ps = np.stack([r[1] for r in recs])
                zs = np.array([1.0 if r[2] == winner else
                               (0.0 if winner < 0 else -1.0) for r in recs],
                              dtype=np.float32)
                self.finished.append((fs, ps, zs))
                self.completed_games += 1
                self.game_lengths.append(len(recs))
                if winner == 0:
                    self.results["black"] += 1
                elif winner == 1:
                    self.results["white"] += 1
                else:
                    self.results["draw"] += 1
                if self.log_moves:
                    self.finished_logs.append({
                        "winner": winner, "length": len(recs),
                        "open": list(self.slot_open[g]),
                        "moves": self.slot_logs[g],
                    })
                    self.slot_logs[g] = []
                # reset slot
                self.slot_records[g] = []
                self.boards[g].zero_()
                self.player[g] = 0
                self.move_num[g] = 0
                self.last[g] = -1
                self._random_opening(g)

    def drain(self):
        out = self.finished
        self.finished = []
        return out

    def drain_logs(self):
        out = self.finished_logs
        self.finished_logs = []
        return out
