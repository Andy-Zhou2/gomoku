"""Throughput benchmarks: raw NN inference, self-play (MCTS+NN), training steps.

    python benchmark.py                 # full suite, default net 64ch x 6 blocks
    python benchmark.py --quick         # smaller sweep
"""
import argparse
import time

import numpy as np
import torch

from data import ReplayBuffer
from game import A, BOARD, IN_PLANES
from net import build_net
from selfplay import SelfPlayManager
from train import train_step


def bench_inference(net, batches, device):
    print("\n== Raw NN inference (bf16 autocast) ==")
    net.eval()
    for bs in batches:
        x = torch.randn(bs, IN_PLANES, BOARD, BOARD, device=device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(5):
                net(x)
            torch.cuda.synchronize()
            t0 = time.time()
            iters = 50
            for _ in range(iters):
                net(x)
            torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"  batch {bs:5d}: {bs * iters / dt:>12,.0f} states/s   ({dt / iters * 1e3:.2f} ms/fwd)")


def bench_selfplay(net, configs, device, moves=40):
    print("\n== Self-play throughput (MCTS + NN, fresh net) ==")
    for G, sims in configs:
        mgr = SelfPlayManager(net, num_games=G, sims=sims, device=device)
        for _ in range(3):  # warmup
            mgr.step()
        torch.cuda.synchronize()
        e0 = mgr.mcts.nn_evals
        g0, m0 = mgr.completed_games, mgr.total_moves
        t0 = time.time()
        for _ in range(moves):
            mgr.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        mv = mgr.total_moves - m0
        ev = mgr.mcts.nn_evals - e0
        gl = np.mean(mgr.game_lengths) if mgr.game_lengths else 50.0
        games_hr = mv / dt / gl * 3600
        print(f"  G={G:4d} sims={sims:3d}: {mv / dt:8.1f} moves/s | "
              f"{ev / dt:>10,.0f} NN evals/s | ~{games_hr:>8,.0f} games/hr "
              f"(avg len {gl:.0f}, {mgr.completed_games} done)")
        del mgr
        torch.cuda.empty_cache()


def bench_training(net, batch_sizes, device, steps=50):
    print("\n== Training step throughput ==")
    buf = ReplayBuffer(100_000)
    # fill with random data
    n = 60_000
    buf.feats[:n] = np.random.randint(0, 2, (n, IN_PLANES, BOARD, BOARD), dtype=np.uint8)
    pis = np.random.rand(n, A).astype(np.float16)
    buf.pis[:n] = pis / pis.sum(-1, keepdims=True)
    buf.zs[:n] = np.random.choice([-1.0, 0.0, 1.0], n)
    buf.size = n
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    net.train()
    for bs in batch_sizes:
        for _ in range(5):
            train_step(net, opt, buf, bs, device)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(steps):
            train_step(net, opt, buf, bs, device)
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"  batch {bs:5d}: {steps / dt:6.1f} steps/s | {bs * steps / dt:>10,.0f} samples/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"net: {args.channels}ch x {args.blocks} blocks "
          f"({sum(p.numel() for p in build_net(args.channels, args.blocks, 'cpu').parameters()):,} params)")

    net = build_net(args.channels, args.blocks, dev)

    if args.quick:
        bench_inference(net, [512, 2048], dev)
        bench_selfplay(net, [(512, 200)], dev, moves=20)
        bench_training(net, [1024], dev)
    else:
        bench_inference(net, [256, 512, 1024, 2048, 4096], dev)
        bench_selfplay(net, [(256, 200), (512, 200), (1024, 200), (512, 400)], dev)
        bench_training(net, [512, 1024, 2048], dev)


if __name__ == "__main__":
    main()
