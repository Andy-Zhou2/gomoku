"""Play against a checkpoint in the browser.

    python play_web.py runs/az3/latest.pt [--port 8000] [--host 127.0.0.1]

Then open http://127.0.0.1:8000 — click intersections to play.
Stateless API: the client sends the full move list each turn; the server
replays it, checks the game state, and answers with the engine's move.
"""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from arena import load_net
from game import A, check_win, win_kernels
from mcts import BatchedMCTS

DEV = "cuda"
NET = None
WIN_W = None
MCTS_CACHE = {}
ANALYSIS_CAP = 20000  # max sims per analysed position (tree node capacity)
ANA = {"key": None, "mcts": None}


def get_mcts(sims):
    if sims not in MCTS_CACHE:
        MCTS_CACHE[sims] = BatchedMCTS(NET, 1, sims, device=DEV)
    return MCTS_CACHE[sims]


def game_state(moves):
    """Replay moves (black first). Returns (boards, player, winner, err)."""
    boards = torch.zeros(1, 2, A, device=DEV)
    p = 0
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
    if len(moves) == A:
        return boards, p, -1, None  # draw
    return boards, p, None, None


@torch.inference_mode()
def handle_move(req):
    moves = [int(a) for a in req.get("moves", [])]
    sims = max(50, min(int(req.get("sims", 800)), 5000))
    boards, player, winner, err = game_state(moves)
    if err:
        return {"error": err}
    if winner is not None:
        return {"winner": winner, "moves": moves}

    mcts = get_mcts(sims)
    last = torch.tensor([moves[-1] if moves else -1], device=DEV)
    pl = torch.tensor([player], device=DEV)
    pi = mcts.run(boards, pl, last, add_noise=False)
    a = int(pi[0].argmax())
    q_all = mcts.Wsa[0, 0] / mcts.Nsa[0, 0].clamp(min=1.0)
    topv, topi = pi[0].topk(5)
    resp = {
        "move": a,
        "q": round(float(q_all[a]), 3),          # engine's view of its move
        "top": [[int(i), round(float(v), 3)] for v, i in zip(topv, topi) if v > 0.001],
        "sims": sims,
    }
    moves.append(a)
    _, _, winner, _ = game_state(moves)
    if winner is not None:
        resp["winner"] = winner
    return resp


@torch.inference_mode()
def handle_analyze(req):
    """Continuous analysis of the current position: keeps one persistent tree
    per position and deepens it by `chunk` sims per call."""
    moves = [int(a) for a in req.get("moves", [])]
    chunk = max(50, min(int(req.get("chunk", 250)), 2000))
    boards, player, winner, err = game_state(moves)
    if err:
        return {"error": err}
    if winner is not None:
        return {"winner": winner}

    key = tuple(moves)
    if ANA["key"] != key:
        if ANA["mcts"] is None:
            ANA["mcts"] = BatchedMCTS(NET, 1, ANALYSIS_CAP, device=DEV)
        last = torch.tensor([moves[-1] if moves else -1], device=DEV)
        ANA["mcts"].start(boards, torch.tensor([player], device=DEV), last,
                          add_noise=False)
        ANA["key"] = key
    m = ANA["mcts"]
    m.advance(chunk)

    visits = m.Nsa[0, 0]
    tot = float(visits.sum())
    pi = visits / max(tot, 1.0)
    q_all = m.Wsa[0, 0] / visits.clamp(min=1.0)
    topv, topi = pi.topk(8)
    value = float(m.Wsa[0, 0].sum()) / max(tot, 1.0)  # perspective: player to move
    return {
        "sims": m.sims_done, "cap": ANALYSIS_CAP, "player": player,
        "value": round(value, 3),
        "top": [[int(i), round(float(v), 4), round(float(q_all[i]), 3)]
                for v, i in zip(topv, topi) if v > 0.001],
    }


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
    global NET, WIN_W
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    NET = load_net(args.ckpt)
    WIN_W = win_kernels(DEV)
    get_mcts(800)  # warm up allocations
    print(f"engine: {args.ckpt}")
    print(f"open http://{args.host}:{args.port}")
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
