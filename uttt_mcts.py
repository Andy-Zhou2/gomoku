"""Batched lockstep MCTS for Ultimate Tic-Tac-Toe. Same tree machinery as the
gomoku mcts.py (flat GPU tensors, one NN batch per simulation step); the game
state carried during descent is the richer UTTT state (cells, board status,
forced-board pointer)."""
import torch

from uttt_game import A, legal_mask, apply_move, make_features

NEG_INF = -1e30


class UtttMCTS:
    def __init__(self, net, num_games, sims, c_puct=1.5,
                 dirichlet_alpha=0.6, dirichlet_eps=0.25, device="cuda",
                 forced_playouts=False, q_filter=False, q_margin=0.05):
        self.net = net
        self.G = num_games
        self.sims = sims
        self.c_puct = c_puct
        self.alpha = dirichlet_alpha   # ~10/avg_legal (~9-ish branching) -> higher than gomoku
        self.eps = dirichlet_eps
        self.dev = device
        self.forced = forced_playouts
        self.q_filter = q_filter
        self.q_margin = q_margin
        self.root_P = None

        G, N = num_games, sims + 2
        self.Nmax = N
        self.P = torch.zeros(G, N, A, device=device)
        self.Nsa = torch.zeros(G, N, A, device=device)
        self.Wsa = torch.zeros(G, N, A, device=device)
        self.children = torch.full((G, N, A), -1, dtype=torch.long, device=device)
        self.is_term = torch.zeros(G, N, dtype=torch.bool, device=device)
        self.term_val = torch.zeros(G, N, device=device)
        self.node_count = torch.zeros(G, dtype=torch.long, device=device)
        self.path_nodes = torch.zeros(G, sims + 8, dtype=torch.long, device=device)
        self.path_acts = torch.zeros(G, sims + 8, dtype=torch.long, device=device)
        self.ar = torch.arange(G, device=device)
        self.nn_evals = 0

    @torch.inference_mode()
    def _eval(self, st):
        feats = make_features(st)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, v = self.net(feats)
        logits = logits.float()
        lm = legal_mask(st)
        pri = logits.masked_fill(~lm, -1e9).softmax(-1)
        self.nn_evals += st.G
        return pri, v.float()

    @torch.inference_mode()
    def run(self, st, add_noise=True):
        self.start(st, add_noise)
        self.advance(self.sims)
        return self.policy_target()

    @torch.inference_mode()
    def start(self, st, add_noise=True):
        self.P.zero_(); self.Nsa.zero_(); self.Wsa.zero_()
        self.children.fill_(-1)
        self.is_term.zero_(); self.term_val.zero_()
        self.node_count.fill_(1)

        pri, _ = self._eval(st)
        if add_noise:
            lm = legal_mask(st).float()
            conc = torch.full_like(pri, self.alpha)
            noise = torch.distributions.Gamma(conc, torch.ones_like(conc)).sample() * lm
            noise = noise / noise.sum(-1, keepdim=True).clamp(min=1e-8)
            pri = (1 - self.eps) * pri + self.eps * noise
        self.P[:, 0] = pri
        self.root_P = pri
        self._root = st.clone()
        self.sims_done = 0

    @torch.inference_mode()
    def advance(self, k):
        k = max(0, min(int(k), self.Nmax - 2 - self.sims_done))
        for _ in range(k):
            self._simulate()
        self.sims_done += k
        return k

    @torch.inference_mode()
    def policy_target(self):
        visits = self.Nsa[:, 0]
        pi = visits / visits.sum(-1, keepdim=True).clamp(min=1e-8)
        if self.q_filter:
            q = self.Wsa[:, 0] / visits.clamp(min=1.0)
            vis_ok = visits >= 2.0
            q_best = torch.where(vis_ok, q, torch.full_like(q, -1e9)).amax(-1, keepdim=True)
            keep = vis_ok & (q >= q_best - self.q_margin)
            filt = visits * keep
            s = filt.sum(-1, keepdim=True)
            pi = torch.where(s > 0, filt / s.clamp(min=1e-8), pi)
        return pi

    def _simulate(self):
        G, ar = self.G, self.ar
        cur = torch.zeros(G, dtype=torch.long, device=self.dev)
        s = self._root.clone()
        active = torch.ones(G, dtype=torch.bool, device=self.dev)
        needs_nn = torch.zeros(G, dtype=torch.bool, device=self.dev)
        leaf_val = torch.zeros(G, device=self.dev)
        plen = torch.zeros(G, dtype=torch.long, device=self.dev)

        depth = 0
        while bool(active.any()):
            p = self.P[ar, cur]
            n = self.Nsa[ar, cur]
            w = self.Wsa[ar, cur]
            q = w / n.clamp(min=1.0)
            u = self.c_puct * p * (n.sum(-1, keepdim=True) + 1.0).sqrt() / (1.0 + n)
            lm = legal_mask(s)
            score = (q + u).masked_fill(~lm, NEG_INF)
            if self.forced and depth == 0:
                forced_n = (2.0 * self.root_P * n.sum(-1, keepdim=True)).sqrt()
                need = lm & (n < forced_n)
                score = torch.where(need, 1e9 * (1.0 + self.root_P), score)
            score += torch.rand_like(score) * 1e-4
            a = score.argmax(-1)

            self.path_nodes[:, depth] = cur
            self.path_acts[:, depth] = a
            plen += active.long()

            ga = active
            win, draw = apply_move(s, a, ga)   # flips s.player, updates forced

            child = self.children[ar, cur, a]
            nid = self.node_count.clone()
            is_new = ga & (child < 0)
            self.children[ar[is_new], cur[is_new], a[is_new]] = nid[is_new]
            self.node_count += is_new.long()

            term_now = is_new & (win | draw)
            self.is_term[ar[term_now], nid[term_now]] = True
            self.term_val[ar[term_now], nid[term_now]] = torch.where(
                win[term_now], torch.full_like(leaf_val[term_now], -1.0),
                torch.zeros_like(leaf_val[term_now]))

            child_c = child.clamp(min=0)
            old_term = ga & ~is_new & self.is_term[ar, child_c]

            leaf_val = torch.where(term_now,
                                   torch.where(win, torch.full_like(leaf_val, -1.0),
                                               torch.zeros_like(leaf_val)),
                                   leaf_val)
            leaf_val = torch.where(old_term, self.term_val[ar, child_c], leaf_val)
            needs_nn = needs_nn | (is_new & ~term_now)

            cur = torch.where(is_new, nid, torch.where(ga, child_c, cur))
            active = ga & ~is_new & ~old_term
            depth += 1

        pri, v = self._eval(s)
        gi = needs_nn
        self.P[ar[gi], cur[gi]] = pri[gi]
        leaf_val = torch.where(gi, v, leaf_val)

        for d in range(depth):
            m = plen > d
            sign = torch.where((plen - d) % 2 == 0,
                               torch.ones_like(leaf_val), -torch.ones_like(leaf_val))
            idx = (ar[m], self.path_nodes[m, d], self.path_acts[m, d])
            self.Nsa[idx] += 1.0
            self.Wsa[idx] += (sign * leaf_val)[m]
