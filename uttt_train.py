"""AlphaZero training for Ultimate Tic-Tac-Toe (9x9, forced-board rule:
if sent to a decided board, play anywhere).

    python uttt_train.py --iters 200 --out runs/uttt
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from data import ReplayBuffer
from net import build_net
from uttt_game import B, IN_PLANES
from uttt_selfplay import UtttSelfPlay


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
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--num-games", type=int, default=1024)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--games-per-iter", type=int, default=500)
    ap.add_argument("--train-steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--buffer", type=int, default=400_000)
    ap.add_argument("--min-buffer", type=int, default=10_000)
    ap.add_argument("--temp-moves", type=int, default=12)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--forced-playouts", action="store_true")
    ap.add_argument("--q-filter", action="store_true")
    ap.add_argument("--log-games", type=int, default=8)
    ap.add_argument("--out", default="runs/uttt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.out, exist_ok=True)

    net = build_net(args.channels, args.blocks, dev, board_size=B, in_planes=IN_PLANES)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    start_iter = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=True)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        start_iter = ck.get("iter", 0)
        print(f"resumed from {args.resume} at iter {start_iter}", flush=True)

    model = torch.compile(net) if args.compile else net
    mgr = UtttSelfPlay(model, args.num_games, args.sims, args.c_puct,
                       temp_moves=args.temp_moves, device=dev,
                       log_moves=args.log_games > 0,
                       forced_playouts=args.forced_playouts, q_filter=args.q_filter)
    buf = ReplayBuffer(args.buffer, planes=IN_PLANES, board=B)
    log_path = os.path.join(args.out, "log.jsonl")

    for it in range(start_iter + 1, args.iters + 1):
        net.eval()
        t0 = time.time()
        g0, m0 = mgr.completed_games, mgr.total_moves
        hb = time.time()
        while mgr.completed_games - g0 < args.games_per_iter:
            mgr.step()
            if time.time() - hb > 30:
                print(f"  [selfplay] {mgr.completed_games - g0}/{args.games_per_iter} games, "
                      f"{(mgr.total_moves - m0) / (time.time() - t0):.0f} mv/s", flush=True)
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

        lp = lv = float("nan")
        tr_t = 0.0
        if buf.size >= args.min_buffer:
            net.train()
            t0 = time.time()
            lps, lvs = [], []
            for _ in range(args.train_steps):
                a, b_ = train_step(model, opt, buf, args.batch_size, dev)
                lps.append(a); lvs.append(b_)
            torch.cuda.synchronize()
            tr_t = time.time() - t0
            lp, lv = sum(lps) / len(lps), sum(lvs) / len(lvs)

        recent = mgr.game_lengths[-300:]
        rec = {
            "iter": it, "buffer": buf.size,
            "selfplay_s": round(sp_t, 1),
            "games": mgr.completed_games - g0,
            "moves_per_s": round(sp_moves / sp_t, 1),
            "avg_game_len": round(sum(recent) / max(1, len(recent)), 1),
            "train_s": round(tr_t, 1),
            "loss_p": round(lp, 4), "loss_v": round(lv, 4),
            "results": dict(mgr.results),
        }
        print(json.dumps(rec), flush=True)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

        ck = {"net": net.state_dict(), "opt": opt.state_dict(), "iter": it,
              "channels": args.channels, "blocks": args.blocks,
              "board_size": B, "in_planes": IN_PLANES}
        torch.save(ck, os.path.join(args.out, "latest.pt"))
        if it % 20 == 0:
            torch.save(ck, os.path.join(args.out, f"iter{it:04d}.pt"))


if __name__ == "__main__":
    main()
