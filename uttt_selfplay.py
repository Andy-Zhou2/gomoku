"""Continuous batched self-play for Ultimate Tic-Tac-Toe (mirrors selfplay.py)."""
import numpy as np
import torch

from uttt_game import A, UtttState, apply_move, make_features
from uttt_mcts import UtttMCTS


class UtttSelfPlay:
    def __init__(self, net, num_games=1024, sims=160, c_puct=1.5,
                 temp_moves=12, device="cuda", log_moves=False,
                 forced_playouts=False, q_filter=False):
        self.G = num_games
        self.dev = device
        self.temp_moves = temp_moves
        self.mcts = UtttMCTS(net, num_games, sims, c_puct, device=device,
                             forced_playouts=forced_playouts, q_filter=q_filter)
        self.st = UtttState(num_games, device)
        self.move_num = torch.zeros(num_games, dtype=torch.long, device=device)
        self.ar = torch.arange(num_games, device=device)

        self.slot_records = [[] for _ in range(num_games)]
        self.finished = []
        self.log_moves = log_moves
        self.slot_logs = [[] for _ in range(num_games)]
        self.finished_logs = []
        self.completed_games = 0
        self.total_moves = 0
        self.results = {"x": 0, "o": 0, "draw": 0}
        self.game_lengths = []

    @torch.inference_mode()
    def step(self):
        pi = self.mcts.run(self.st, add_noise=True)
        feats = make_features(self.st)

        sampled = torch.multinomial(pi.clamp(min=1e-8), 1).squeeze(1)
        greedy = pi.argmax(-1)
        move = torch.where(self.move_num < self.temp_moves, sampled, greedy)

        feats_np = feats.to(torch.uint8).cpu().numpy()
        pi_np = pi.to(torch.float16).cpu().numpy()
        player_np = self.st.player.cpu().numpy()
        for g in range(self.G):
            self.slot_records[g].append((feats_np[g], pi_np[g], int(player_np[g])))

        if self.log_moves:
            n_root = self.mcts.Nsa[:, 0]
            q_root = self.mcts.Wsa[:, 0] / n_root.clamp(min=1.0)
            q_chosen = q_root[self.ar, move]
            v_root = self.mcts.Wsa[:, 0].sum(-1) / n_root.sum(-1).clamp(min=1.0)
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

        mover = self.st.player.clone()
        ones = torch.ones(self.G, dtype=torch.bool, device=self.dev)
        win, draw = apply_move(self.st, move, ones)
        self.move_num += 1
        done = win | draw
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
                self.results["x" if winner == 0 else "o" if winner == 1 else "draw"] += 1
                if self.log_moves:
                    self.finished_logs.append({"winner": winner, "length": len(recs),
                                               "moves": self.slot_logs[g]})
                    self.slot_logs[g] = []
                self.slot_records[g] = []
                self.st.reset_(g)
                self.move_num[g] = 0

    def drain(self):
        out = self.finished
        self.finished = []
        return out

    def drain_logs(self):
        out = self.finished_logs
        self.finished_logs = []
        return out
