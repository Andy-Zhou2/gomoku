"""Fully-tensorized batched MCTS on GPU.

G games run in lockstep. Each game's search tree lives in flat GPU tensors:
  P, Nsa, Wsa : (G, Nmax, A)  edge priors / visit counts / total values
  children    : (G, Nmax, A)  child node index, -1 if unexpanded
  is_term/term_val : (G, Nmax) terminal flag and value

Values (Wsa, term_val, leaf values) are always from the perspective of the
player to move at that node. One simulation per game per step; each step
batches all leaf evaluations into a single NN forward of size G — this is
what keeps the GPU saturated.
"""
import torch

from game import A, BOARD, win_kernels, check_win, legal_mask, make_features

NEG_INF = -1e30


class BatchedMCTS:
    def __init__(self, net, num_games, sims, c_puct=1.5,
                 dirichlet_alpha=0.15, dirichlet_eps=0.25, device="cuda",
                 forced_playouts=False, q_filter=False, q_margin=0.05):
        self.net = net
        self.G = num_games
        self.sims = sims
        self.c_puct = c_puct
        self.alpha = dirichlet_alpha
        self.eps = dirichlet_eps
        self.dev = device
        # anti-capitulation levers (self-play only; see README):
        # forced_playouts: root children get >= sqrt(2*P*N) visits so low-prior
        #   moves (blocks) are honestly evaluated instead of dying in the prior trap
        # q_filter: policy target keeps only visited moves with Q near the best
        #   visited Q, so "least losing" moves (blocks) dominate the target even
        #   when raw visit counts are flat
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
        # +8 slack: descent checks active.any() only every 4 levels (sync reduction)
        self.path_nodes = torch.zeros(G, sims + 8, dtype=torch.long, device=device)
        self.path_acts = torch.zeros(G, sims + 8, dtype=torch.long, device=device)
        self.ar = torch.arange(G, device=device)
        self.win_w = win_kernels(device)
        self.nn_evals = 0  # counts states sent through the net (throughput stat)

    @torch.inference_mode()
    def _eval(self, boards, player, last):
        """Batched net eval -> (priors (G,A) masked+normalized, value (G,))."""
        feats = make_features(boards, player, last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, v = self.net(feats)
        logits = logits.float()
        lm = legal_mask(boards)
        # -1e9 (finite) so an all-illegal row yields uniform, not NaN
        pri = logits.masked_fill(~lm, -1e9).softmax(-1)
        self.nn_evals += boards.shape[0]
        return pri, v.float()

    @torch.inference_mode()
    def run(self, boards, player, last, add_noise=True):
        """Run `sims` simulations from the given root states.

        boards: (G,2,A) float32, player: (G,) long, last: (G,) long.
        Returns visit distribution pi: (G, A).
        """
        self.start(boards, player, last, add_noise)
        self.advance(self.sims)
        return self.policy_target()

    @torch.inference_mode()
    def start(self, boards, player, last, add_noise=True):
        """Reset trees and expand the root. Follow with advance(k) calls."""
        self.P.zero_(); self.Nsa.zero_(); self.Wsa.zero_()
        self.children.fill_(-1)
        self.is_term.zero_(); self.term_val.zero_()
        self.node_count.fill_(1)  # node 0 = root

        pri, _ = self._eval(boards, player, last)
        if add_noise:
            lm = legal_mask(boards).float()
            conc = torch.full_like(pri, self.alpha)
            noise = torch.distributions.Gamma(conc, torch.ones_like(conc)).sample() * lm
            noise = noise / noise.sum(-1, keepdim=True).clamp(min=1e-8)
            pri = (1 - self.eps) * pri + self.eps * noise
        self.P[:, 0] = pri
        self.root_P = pri
        self._root = (boards.clone(), player.clone(), last.clone(),
                      boards.sum((1, 2)).long())
        self.sims_done = 0

    @torch.inference_mode()
    def advance(self, k):
        """Run up to k more simulations (bounded by node capacity)."""
        k = max(0, min(int(k), self.Nmax - 2 - self.sims_done))
        rb, rp, rl, s0 = self._root
        for _ in range(k):
            self._simulate(rb, rp, rl, s0)
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

    def _simulate(self, rb, rp, rl, stones0):
        G, ar = self.G, self.ar
        cur = torch.zeros(G, dtype=torch.long, device=self.dev)
        sb = rb.clone()
        sp = rp.clone()
        sl = rl.clone()
        stones = stones0.clone()
        active = torch.ones(G, dtype=torch.bool, device=self.dev)
        needs_nn = torch.zeros(G, dtype=torch.bool, device=self.dev)
        leaf_val = torch.zeros(G, device=self.dev)
        plen = torch.zeros(G, dtype=torch.long, device=self.dev)

        depth = 0
        while bool(active.any()):
            # --- selection: PUCT over edges of `cur` ---
            p = self.P[ar, cur]
            n = self.Nsa[ar, cur]
            w = self.Wsa[ar, cur]
            q = w / n.clamp(min=1.0)
            u = self.c_puct * p * (n.sum(-1, keepdim=True) + 1.0).sqrt() / (1.0 + n)
            lm = sb.sum(1) == 0
            score = (q + u).masked_fill(~lm, NEG_INF)
            if self.forced and depth == 0:
                # root forced playouts (KataGo-style): any legal child still
                # below its forced quota outranks everything
                forced_n = (2.0 * self.root_P * n.sum(-1, keepdim=True)).sqrt()
                need = lm & (n < forced_n)
                score = torch.where(need, 1e9 * (1.0 + self.root_P), score)
            # tiny random tie-break: near-equal candidates (fresh nodes) would
            # otherwise argmax to a fixed cell and self-reinforce
            score += torch.rand_like(score) * 1e-4
            a = score.argmax(-1)

            self.path_nodes[:, depth] = cur
            self.path_acts[:, depth] = a
            plen += active.long()

            # --- apply move for active games (sp not yet switched = mover) ---
            ga = active
            sb[ar[ga], sp[ga], a[ga]] = 1.0
            mover_planes = sb[ar, sp]
            win = check_win(mover_planes, self.win_w) & ga
            stones += ga.long()
            draw = ga & ~win & (stones == A)
            sl = torch.where(ga, a, sl)

            # --- expansion bookkeeping ---
            child = self.children[ar, cur, a]
            nid = self.node_count.clone()
            is_new = ga & (child < 0)
            self.children[ar[is_new], cur[is_new], a[is_new]] = nid[is_new]
            self.node_count += is_new.long()

            term_now = is_new & (win | draw)
            self.is_term[ar[term_now], nid[term_now]] = True
            # player to move at the new node just lost (win) or drew
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
            sp = torch.where(ga, 1 - sp, sp)
            active = ga & ~is_new & ~old_term
            depth += 1

        # --- batched leaf evaluation (full G batch, constant shape) ---
        # unconditional: needs_nn is true for ~all games in ~all sims, and
        # skipping the .any() check saves one host sync per simulation
        pri, v = self._eval(sb, sp, sl)
        gi = needs_nn
        self.P[ar[gi], cur[gi]] = pri[gi]
        leaf_val = torch.where(gi, v, leaf_val)

        # --- backup: edge at depth d gets leaf_val * (-1)^(plen - d) ---
        # `depth` (python int) bounds max path length -> no .max() sync
        for d in range(depth):
            m = plen > d
            sign = torch.where((plen - d) % 2 == 0,
                               torch.ones_like(leaf_val), -torch.ones_like(leaf_val))
            idx = (ar[m], self.path_nodes[m, d], self.path_acts[m, d])
            self.Nsa[idx] += 1.0
            self.Wsa[idx] += (sign * leaf_val)[m]
