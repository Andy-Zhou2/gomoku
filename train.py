"""AlphaZero training loop: alternate batched GPU self-play and SGD on the
replay buffer. Checkpoints to --out.

Example:
    python train.py --iters 50 --num-games 512 --sims 200
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from data import ReplayBuffer
from net import build_net
from selfplay import SelfPlayManager


def train_step(net, opt, buffer, batch_size, device):
    f, p_tgt, z = buffer.sample(batch_size, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, v = net(f)
    logits, v = logits.float(), v.float()
    loss_p = -(p_tgt * F.log_softmax(logits, -1)).sum(-1).mean()
    loss_v = F.mse_loss(v, z)
    loss = loss_p + loss_v
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss_p.item(), loss_v.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--num-games", type=int, default=512, help="parallel self-play games")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--games-per-iter", type=int, default=250)
    ap.add_argument("--train-steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--buffer", type=int, default=500_000)
    ap.add_argument("--min-buffer", type=int, default=10_000)
    ap.add_argument("--temp-moves", type=int, default=8,
                    help="plies played at tau=1 (225 = whole game)")
    ap.add_argument("--temp-moves-final", type=int, default=None,
                    help="anneal temp-moves to this value at --temp-switch-iter")
    ap.add_argument("--temp-switch-iter", type=int, default=0)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--out", default="runs/az")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the net (~15-20%% faster self-play)")
    ap.add_argument("--log-games", type=int, default=8,
                    help="save up to N sample games per iter to games.jsonl (0=off)")
    ap.add_argument("--random-openings", type=int, default=0,
                    help="seed each game with 0..N random stones (diversity/defense)")
    ap.add_argument("--forced-playouts", action="store_true",
                    help="root forced playouts: honestly evaluate low-prior moves")
    ap.add_argument("--q-filter", action="store_true",
                    help="policy targets keep only moves with Q near best visited Q")
    args = ap.parse_args()

    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.out, exist_ok=True)

    net = build_net(args.channels, args.blocks, dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    start_iter = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=True)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        start_iter = ck.get("iter", 0)
        print(f"resumed from {args.resume} at iter {start_iter}")

    model = torch.compile(net) if args.compile else net  # shares params with net
    mgr = SelfPlayManager(model, args.num_games, args.sims, args.c_puct,
                          temp_moves=args.temp_moves, device=dev,
                          log_moves=args.log_games > 0,
                          rand_open=args.random_openings,
                          forced_playouts=args.forced_playouts,
                          q_filter=args.q_filter)
    buf = ReplayBuffer(args.buffer)
    log_path = os.path.join(args.out, "log.jsonl")

    for it in range(start_iter + 1, args.iters + 1):
        if args.temp_moves_final is not None and it > args.temp_switch_iter:
            mgr.temp_moves = args.temp_moves_final
        # ---- self-play ----
        net.eval()
        t0 = time.time()
        g0, m0 = mgr.completed_games, mgr.total_moves
        hb = time.time()
        while mgr.completed_games - g0 < args.games_per_iter:
            mgr.step()
            if time.time() - hb > 30:  # heartbeat so the log never goes silent
                print(f"  [selfplay] {mgr.completed_games - g0}/{args.games_per_iter} games, "
                      f"{mgr.total_moves - m0} moves, {(mgr.total_moves - m0) / (time.time() - t0):.0f} mv/s",
                      flush=True)
                hb = time.time()
        for fs, ps, zs in mgr.drain():
            buf.add_game(fs, ps, zs)
        if args.log_games > 0:
            logs = mgr.drain_logs()
            with open(os.path.join(args.out, "games.jsonl"), "a") as fh:
                for gm in logs[:args.log_games]:
                    gm["iter"] = it
                    fh.write(json.dumps(gm) + "\n")
        sp_t = time.time() - t0
        sp_moves = mgr.total_moves - m0
        sp_games = mgr.completed_games - g0

        # ---- training ----
        lp = lv = float("nan")
        tr_t = 0.0
        if buf.size >= args.min_buffer:
            net.train()
            t0 = time.time()
            lps, lvs = [], []
            for _ in range(args.train_steps):
                a, b = train_step(model, opt, buf, args.batch_size, dev)
                lps.append(a); lvs.append(b)
            torch.cuda.synchronize()
            tr_t = time.time() - t0
            lp, lv = sum(lps) / len(lps), sum(lvs) / len(lvs)

        rec = {
            "iter": it, "buffer": buf.size,
            "selfplay_s": round(sp_t, 1), "games": sp_games,
            "moves_per_s": round(sp_moves / sp_t, 1),
            "avg_game_len": round(sum(mgr.game_lengths[-200:]) / max(1, len(mgr.game_lengths[-200:])), 1),
            "train_s": round(tr_t, 1),
            "loss_p": round(lp, 4), "loss_v": round(lv, 4),
            "results": dict(mgr.results),
        }
        print(json.dumps(rec), flush=True)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

        ck = {"net": net.state_dict(), "opt": opt.state_dict(), "iter": it,
              "channels": args.channels, "blocks": args.blocks}
        torch.save(ck, os.path.join(args.out, "latest.pt"))
        if it % 10 == 0:
            torch.save(ck, os.path.join(args.out, f"iter{it:04d}.pt"))


if __name__ == "__main__":
    main()
