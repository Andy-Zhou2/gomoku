"""Play against a checkpoint in the browser.

    python play_web.py runs/az3/latest.pt [--port 8000] [--host 127.0.0.1]

Then open http://127.0.0.1:8000 — click intersections to play.
Stateless API: the client sends the full move list each turn; the server
replays it, checks the game state, and answers with the engine's move.
"""
import argparse
import json
import os
import re
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from arena import load_net
from game import A, check_win, win_kernels
from mcts import BatchedMCTS

DEV = "cuda"
WIN_W = None
DEFAULT_CKPT = None      # checkpoint main() was launched with
CKPT_DIR = None          # its directory — scanned for iterNNNN.pt siblings
_ITER_RE = re.compile(r"^iter(\d+)\.pt$")

NET_CACHE = OrderedDict()  # ckpt path -> loaded net, LRU-capped
NET_CACHE_MAX = 24
MCTS_CACHE = {}            # (ckpt, sims) -> one-shot BatchedMCTS
ANALYSIS_CAP = 20000       # max sims per analysed position (tree node capacity)
BASE_C_PUCT = 1.5
EXPLORE_C_PUCT = 5.0       # "force wider exploration" — see handle_analyze
ANA = {"key": None, "mcts": None, "explore": False, "probe": frozenset(), "model": None}


def get_net(ckpt):
    if ckpt in NET_CACHE:
        NET_CACHE.move_to_end(ckpt)
        return NET_CACHE[ckpt]
    net = load_net(ckpt)
    NET_CACHE[ckpt] = net
    if len(NET_CACHE) > NET_CACHE_MAX:
        NET_CACHE.popitem(last=False)
    return net


def list_checkpoints():
    """Every checkpoint the client is allowed to select, weakest (lowest
    iter) to strongest — discovered by scanning CKPT_DIR, not client input,
    so a request can't point load_net() at an arbitrary path."""
    out = []
    if CKPT_DIR and os.path.isdir(CKPT_DIR):
        for fn in os.listdir(CKPT_DIR):
            m = _ITER_RE.match(fn)
            if m:
                it = int(m.group(1))
                out.append({"id": os.path.join(CKPT_DIR, fn), "label": f"iter {it}", "iter": it})
        out.sort(key=lambda d: d["iter"])
        latest_path = os.path.join(CKPT_DIR, "latest.pt")
        if os.path.isfile(latest_path):
            out.append({"id": latest_path, "label": "latest (strongest)",
                        "iter": (out[-1]["iter"] + 1) if out else 0})
    if not any(d["id"] == DEFAULT_CKPT for d in out):
        out.append({"id": DEFAULT_CKPT, "label": os.path.basename(DEFAULT_CKPT), "iter": -1})
        out.sort(key=lambda d: d["iter"])
    return out


def resolve_model(model_id):
    """Map a client-supplied checkpoint id (the opponent-strength slider) to
    a safe, known-good path."""
    if not model_id:
        return DEFAULT_CKPT
    valid = {d["id"] for d in list_checkpoints()}
    return model_id if model_id in valid else DEFAULT_CKPT


def analysis_model():
    """The net that powers hints/candidates/win-rate — always the strongest
    checkpoint available, independent of the opponent-strength slider. You
    want an honest read of the position even while sparring against a
    deliberately weak opponent, not the weak net's own shaky self-assessment."""
    models = list_checkpoints()
    return models[-1]["id"] if models else DEFAULT_CKPT


def get_mcts(sims, ckpt):
    key = (ckpt, sims)
    if key not in MCTS_CACHE:
        MCTS_CACHE[key] = BatchedMCTS(get_net(ckpt), 1, sims, device=DEV)
    return MCTS_CACHE[key]


def setup_fp(setup):
    """Canonical, hashable fingerprint of a setup position for cache keys."""
    if not setup:
        return ((), (), 0)
    b = tuple(sorted(int(a) for a in setup.get("black", [])))
    w = tuple(sorted(int(a) for a in setup.get("white", [])))
    return (b, w, int(setup.get("first", 0)) & 1)


def game_state(moves, setup=None):
    """Replay moves from an optional custom `setup` position, else the empty
    board with black to move. setup: {"black":[...], "white":[...],
    "first":0|1} — stones placed in any order/mix of colors (not necessarily
    alternating), with `first` saying who moves next from that position.
    Returns (boards, player, winner, err)."""
    boards = torch.zeros(1, 2, A, device=DEV)
    p = 0
    if setup:
        black = [int(a) for a in setup.get("black", [])]
        white = [int(a) for a in setup.get("white", [])]
        seen = set()
        for a in black + white:
            if not (0 <= a < A) or a in seen:
                return None, None, None, f"illegal setup stone {a}"
            seen.add(a)
        for a in black:
            boards[0, 0, a] = 1.0
        for a in white:
            boards[0, 1, a] = 1.0
        p = int(setup.get("first", 0)) & 1
        if not moves:
            for pl in (0, 1):
                if bool(check_win(boards[:, pl], WIN_W)[0]):
                    return boards, None, pl, None
    for a in moves:
        if not (0 <= a < A) or boards[0, :, a].sum() > 0:
            return None, None, None, f"illegal move {a}"
        boards[0, p, a] = 1.0
        if bool(check_win(boards[:, p], WIN_W)[0]):
            winner = p if a == moves[-1] else None
            if winner is None:  # win occurred before the last move: bad input
                return None, None, None, "moves continue past a finished game"
            return boards, 1 - p, winner, None
        p ^= 1
    if int(boards.sum().item()) == A:
        return boards, p, -1, None  # draw
    return boards, p, None, None


@torch.inference_mode()
def handle_move(req):
    moves = [int(a) for a in req.get("moves", [])]
    sims = max(50, min(int(req.get("sims", 800)), 5000))
    model = resolve_model(req.get("model"))
    setup = req.get("setup")
    boards, player, winner, err = game_state(moves, setup)
    if err:
        return {"error": err}
    if winner is not None:
        return {"winner": winner, "moves": moves}

    key = (setup_fp(setup), tuple(moves))
    # If the analysis tree already covers this position at >= the requested
    # depth, play from it (instant — the client streams the thinking instead).
    # Only reuse a "clean" tree: one built with exploration noise or a probe
    # boost would bias the actual move choice, so those fall back to a fresh
    # no-noise search below instead.
    if ANA["key"] == key and ANA["mcts"] is not None \
            and not ANA["explore"] and not ANA["probe"] and ANA["model"] == model \
            and ANA["mcts"].sims_done >= sims:
        m = ANA["mcts"]
        visits = m.Nsa[0, 0]
        pi = (visits / visits.sum().clamp(min=1.0)).unsqueeze(0)
        q_all = m.Wsa[0, 0] / visits.clamp(min=1.0)
    else:
        mcts = get_mcts(sims, model)
        last = torch.tensor([moves[-1] if moves else -1], device=DEV)
        pl = torch.tensor([player], device=DEV)
        pi = mcts.run(boards, pl, last, add_noise=False)
        q_all = mcts.Wsa[0, 0] / mcts.Nsa[0, 0].clamp(min=1.0)
    a = int(pi[0].argmax())
    topv, topi = pi[0].topk(5)
    resp = {
        "move": a,
        "q": round(float(q_all[a]), 3),          # engine's view of its move
        "top": [[int(i), round(float(v), 3)] for v, i in zip(topv, topi) if v > 0.001],
        "sims": sims,
    }
    moves.append(a)
    _, _, winner, _ = game_state(moves, setup)
    if winner is not None:
        resp["winner"] = winner
    return resp


@torch.inference_mode()
def handle_analyze(req):
    """Continuous analysis of the current position: keeps one persistent tree
    per position and deepens it by `chunk` sims per call.

    explore: bool — widen the search with three levers together: root
      Dirichlet noise, a raised c_puct for the whole tree (not just at the
      start — noise alone fades after a few hundred sims of a 20000-sim
      persistent tree), and forced_playouts (KataGo-style: every legal root
      move gets an unconditional sqrt(2*P*N) visit quota, not just a nudge —
      see mcts.py). The quota is what actually guarantees breadth; noise/
      c_puct alone just make the dominant move somewhat less dominant.
    probe: [action,...] — points the user wants a genuine win-rate readout
      for even if the engine wouldn't naturally visit them much; their root
      prior is floored so they accumulate real sims (see BatchedMCTS.start).
    Analysis always uses the strongest checkpoint (analysis_model()) — the
    opponent-strength slider (handle_move's `model`) only affects which net
    actually plays the engine's moves, not what powers the hints/eval. A
    client-sent `model` here is ignored on purpose.
    setup: optional {"black":[...], "white":[...], "first":0|1} — a custom
      position (stones placed in any order/mix of colors) to analyze/play
      from instead of the empty board; see game_state.
    Any of these changes the tree the position needs, so they all
    participate in the same-key cache check as the move list itself.
    """
    moves = [int(a) for a in req.get("moves", [])]
    chunk = max(32, min(int(req.get("chunk", 250)), 2000))
    topn = max(1, min(int(req.get("topn", 10)), A))
    explore = bool(req.get("explore", False))
    probe = frozenset(int(a) for a in req.get("probe", []) if 0 <= int(a) < A)
    model = analysis_model()
    setup = req.get("setup")
    boards, player, winner, err = game_state(moves, setup)
    if err:
        return {"error": err}
    if winner is not None:
        return {"winner": winner}

    key = (setup_fp(setup), tuple(moves))
    if ANA["key"] != key or ANA["explore"] != explore or ANA["probe"] != probe \
            or ANA["model"] != model:
        if ANA["mcts"] is None:
            ANA["mcts"] = BatchedMCTS(get_net(model), 1, ANALYSIS_CAP, device=DEV)
        elif ANA["model"] != model:
            ANA["mcts"].net = get_net(model)
        last = torch.tensor([moves[-1] if moves else -1], device=DEV)
        boost = None
        if probe:
            boost = torch.zeros(1, A, dtype=torch.bool, device=DEV)
            boost[0, list(probe)] = True
        ANA["mcts"].start(boards, torch.tensor([player], device=DEV), last,
                          add_noise=explore, boost=boost)
        ANA["key"] = key
        ANA["explore"] = explore
        ANA["probe"] = probe
        ANA["model"] = model
    m = ANA["mcts"]
    m.c_puct = EXPLORE_C_PUCT if explore else BASE_C_PUCT
    # forced_playouts (KataGo-style, already used for the self-play
    # anti-collapse recipe — see mcts.py) gives every legal root move an
    # unconditional sqrt(2*P*N) visit quota. That's a much harder guarantee
    # than noise/c_puct alone: those just make dominant moves *less* dominant,
    # while forced quotas make the engine actually spend sims confirming a
    # long-shot move is bad rather than never trying it.
    m.forced = explore
    m.advance(chunk)

    visits = m.Nsa[0, 0]
    tot = float(visits.sum())
    pi = visits / max(tot, 1.0)
    q_all = m.Wsa[0, 0] / visits.clamp(min=1.0)
    topv, topi = pi.topk(topn)
    value = float(m.Wsa[0, 0].sum()) / max(tot, 1.0)  # perspective: player to move
    resp = {
        "sims": m.sims_done, "cap": ANALYSIS_CAP, "player": player,
        "value": round(value, 3),
        "top": [[int(i), round(float(v), 4), round(float(q_all[i]), 3), int(visits[i])]
                for v, i in zip(topv, topi) if v > 0.001],
    }
    if probe:
        resp["probe"] = [[a, round(float(pi[a]), 4), round(float(q_all[a]), 3), int(visits[a])]
                          for a in sorted(probe)]
    return resp


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = open(os.path.join(os.path.dirname(__file__), "play_gui.html"), "rb").read()
            self._send(200, page, "text/html; charset=utf-8")
        elif self.path == "/api/models":
            self._send(200, json.dumps({"models": list_checkpoints(), "default": DEFAULT_CKPT}))
        else:
            self._send(404, '{"error":"not found"}')

    def do_POST(self):
        routes = {"/api/move": handle_move, "/api/analyze": handle_analyze}
        fn = routes.get(self.path)
        if fn is None:
            self._send(404, '{"error":"not found"}')
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or "{}")
            self._send(200, json.dumps(fn(req)))
        except Exception as e:  # noqa: BLE001
            self._send(500, json.dumps({"error": str(e)}))


def main():
    global DEFAULT_CKPT, CKPT_DIR, WIN_W
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    DEFAULT_CKPT = os.path.abspath(args.ckpt)
    CKPT_DIR = os.path.dirname(DEFAULT_CKPT)
    WIN_W = win_kernels(DEV)
    get_mcts(800, DEFAULT_CKPT)  # warm up allocations
    print(f"engine: {DEFAULT_CKPT}")
    print(f"{len(list_checkpoints())} checkpoint(s) selectable from {CKPT_DIR}")
    print(f"open http://{args.host}:{args.port}")
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
