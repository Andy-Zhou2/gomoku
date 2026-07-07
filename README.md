# AlphaZero-style Gomoku (15×15, freestyle)

A fully GPU-resident AlphaZero implementation for 15×15 Gomoku (black first,
five-in-a-row wins, no forbidden moves). Board rules, MCTS trees, and the
network all live on the GPU — self-play never round-trips per node to Python.

## Why it's fast

The classic bottleneck in AlphaZero self-play is per-node Python MCTS with
batch-1 NN evals. Here instead:

- **G games run in lockstep** (default 512–1024). Each game's search tree is a
  slab of flat GPU tensors `(G, nodes, 225)` for priors / visit counts / values
  / child pointers.
- **One simulation step across all games = one NN forward of batch G.**
  Selection (PUCT argmax), move application, 5-in-a-row detection (a 4-kernel
  conv2d), expansion, and backup are all batched tensor ops.
- bf16 autocast everywhere; optional `torch.compile` (+~19%); binary uint8
  replay buffer on CPU with dihedral (×8 symmetry) augmentation at sample time.

## Benchmarks (RTX 4090, 64ch × 6-block ResNet, 562k params)

Raw NN inference (bf16):

| batch | states/s |
|------:|---------:|
| 512   | 235k |
| 1024  | **251k** |

Self-play throughput (MCTS + NN, eager):

| parallel games G | sims/move | moves/s | NN evals/s | est. games/hr |
|---:|---:|---:|---:|---:|
| 256  | 200 | 397 | 80k  | ~40k |
| 512  | 200 | 606 | 122k | ~60k |
| 1024 | 200 | **836** | **168k** | **~84k** |
| 512  | 400 | 325 | 130k | ~33k |

`torch.compile` adds ~19% on top (586 → 698 moves/s at G=512).

Training: **~52k samples/s** at batch 512–1024 (≈100 steps/s @ 512).

At the default training config (G=1024, 200 sims) the GPU generates ~15k
positions and trains on ~60k augmented samples in roughly 20 s per iteration.

## Files

| file | what |
|---|---|
| `game.py` | batched tensor rules: win-detection conv kernels, legality, features, symmetries |
| `net.py` | policy+value ResNet (configurable channels/blocks) |
| `mcts.py` | batched lockstep MCTS, trees as flat GPU tensors |
| `selfplay.py` | continuous self-play manager (finished slots reset instantly → always full batches) |
| `data.py` | uint8 replay ring buffer + dihedral augmentation |
| `train.py` | training loop, checkpointing, JSONL logging |
| `benchmark.py` | inference / self-play / training throughput suite |
| `arena.py` | pit two checkpoints (or vs `random`), color-alternating batched matches |
| `play.py` | play vs a checkpoint in the terminal |
| `play_web.py` + `play_gui.html` | play vs a checkpoint in the browser (local server, stdlib only) |
| `tests_sanity.py` | rules fuzz test vs slow reference, feature checks, MCTS finds win-in-1 |

## Usage

```bash
python tests_sanity.py                  # correctness checks
python benchmark.py                     # throughput suite
python train.py --iters 240 --num-games 1024 --sims 400 \
    --temp-moves 225 --temp-moves-final 16 --temp-switch-iter 140 \
    --random-openings 6 --forced-playouts --q-filter \
    --compile --log-games 8 --out runs/az3    # recommended recipe (see below)
python arena.py runs/az3/latest.pt random --games 128 --sims 200
python play.py runs/az3/latest.pt --sims 800 --human black
python play_web.py runs/az3/latest.pt          # browser GUI at http://127.0.0.1:8000
python make_viewer.py runs/az3/games.jsonl -o viewer_built.html --last 120
```

Open `viewer.html` in a browser and load any run's `games.jsonl` to replay
rollouts with policy overlays, eval charts, and rule-based commentary.

## Training dynamics: the capitulation collapse and its fixes

Vanilla AlphaZero settings collapse on gomoku at this scale. The failure
chain, observed twice: the value net learns "the attacker wins," every
defensive move backs up Q ≈ −1, the defender's π flattens to uniform,
defense vanishes from the data, and the belief becomes self-fulfilling —
9-ply games where five stones go down uncontested. Three levers fixed it,
each verified with probe positions (a must-block four / open three):

1. `--random-openings 6` — seed 0..6 random stones pre-game. Breaks
   position monotony and hands either color the initiative, so the value
   net must read the board, not the color plane.
2. `--forced-playouts` — root children get ≥ √(2·P·N) visits (KataGo),
   so low-prior blocks are honestly evaluated instead of starving.
3. `--q-filter` — π targets keep only visited moves with Q within 0.05 of
   the best visited Q. In "everything loses" positions the target
   concentrates on the least-losing moves — i.e., the blocks. This is the
   deliberate deviation from vanilla AZ that converts honest evaluation
   into a learning signal (π(block) on a must-block four: 0.27 → 1.00).

Full-game τ=1 (`--temp-moves 225`) annealed to 16 plies at iter 140
provides the counterexample games; avg game length recovering from ~9 to
~26 plies is the health metric to watch. Final 240-iter net: 98.4% vs its
pre-patch self, 100% vs random, blocks fours and threes with vanilla search.

## Details

- **Features** (4 planes, mover-relative): my stones, opponent stones, last
  move, am-I-black.
- **MCTS**: PUCT with c=1.5, Dirichlet(0.15) root noise (ε=0.25), fresh tree
  per move, τ=1 sampling for the first 8 plies then greedy. Values are stored
  from the perspective of the player to move at each node; terminal nodes
  short-circuit the net.
- **Training**: AdamW (lr 1e-3, wd 1e-4), loss = policy cross-entropy vs MCTS
  visit distribution + value MSE vs game outcome; one random dihedral
  transform per batch.
- **Scaling up strength**: raise `--sims` (400–800), `--channels 128
  --blocks 10`, and train longer; throughput numbers above tell you the cost.

Known simplifications (deliberate, v1): no tree reuse between moves, no
separate eval-gating of checkpoints (continuous training like AZ), lockstep
sims mean tree depth syncs per step.

## Ultimate Tic-Tac-Toe (`uttt_*.py`)

The same machinery applied to Ultimate Tic-Tac-Toe (9×9, forced-board rule:
your move's cell position sends the opponent to that sub-board; if that board
is decided, they play anywhere). Fully tensorized rules (`uttt_game.py`),
batched MCTS (`uttt_mcts.py`, same tree code with the richer scratch state),
self-play (`uttt_selfplay.py`), and training:

```bash
python uttt_tests.py     # rules fuzzed vs slow reference + MCTS tactics
python uttt_train.py --iters 200 --compile --q-filter --forced-playouts --out runs/uttt
```

Features (8 planes): my/opp cells, my/opp won boards, drawn boards, the legal
mask (encodes the forced-board rule for the net), last move, color. Dihedral
augmentation is valid — the 8 symmetries of the macro grid map sub-boards
consistently. `net.py`/`data.py` take board_size/planes params shared by both
games.
