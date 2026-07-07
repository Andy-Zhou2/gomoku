"""Embed a games.jsonl into viewer.html -> a self-contained shareable page.

    python make_viewer.py runs/az3/games.jsonl [-o viewer_built.html] [--last N]
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("games")
ap.add_argument("-o", "--out", default="viewer_built.html")
ap.add_argument("--last", type=int, default=0, help="embed only the last N games")
args = ap.parse_args()

html = open("viewer.html").read()
lines = [l for l in open(args.games).read().splitlines() if l.strip()]
if args.last:
    lines = lines[-args.last:]
data = "\n".join(lines)
assert "</script" not in data
open(args.out, "w").write(html.replace("__GAMES__", data))
print(f"wrote {args.out}: {len(lines)} games embedded")
