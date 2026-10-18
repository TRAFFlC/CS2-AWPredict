#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Series-MVP labels from csapi stats, and how predictable they look.

Q-006 settled that item 4 scores the SERIES MVP of a single match, not the
tournament MVP.  csapi exposes no MVP field, so the label must be derived from
`/matches/{id}/stats`.  This script:

1. Derives the label as "highest series rating on the winning side".
2. Measures the margin between the winner's #1 and #2 players, i.e. how often
   the call is close and therefore hard.
3. Checks how often the best rating in the series belongs to the LOSING side.
   If that happens often, then "take the best player overall" is a different
   (and worse) rule than "take the best player on the predicted winner", which
   is the conditional structure the joint objective already assumes.
4. Runs integrity checks on the payload (both teams present, 5 players each).

IMPORTANT: this derives labels, it does not validate them.  Confirming that the
derived label equals HLTV's official Player of the Match needs the HLTV side,
and so far exactly one such sample has been compared (it agreed).

Usage
-----
  python -X utf8 mvp_analysis.py report [--stats csapi_stats_cologne.jsonl]
  python -X utf8 mvp_analysis.py selftest
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _overall_block(payload):
    """The aggregate block: csapi returns a list whose first element is id=0."""
    if isinstance(payload, list):
        for blk in payload:
            if blk.get("id") == 0:
                return blk
        return payload[0] if payload else {}
    return payload or {}


def analyse(rec, match):
    """Return a per-match MVP analysis dict, or an error marker."""
    payload = rec.get("by_map") if rec.get("by_map") else rec.get("overall")
    blk = _overall_block(payload)
    if not blk:
        return {"matchid": rec.get("matchid"), "error": "empty stats payload"}

    winner = (match or {}).get("winner") or {}
    wid = winner.get("id")
    if wid is None:
        return {"matchid": rec.get("matchid"), "error": "match has no winner"}

    sides = {}
    for side in ("team1", "team2"):
        t = blk.get(side) or {}
        sides[t.get("id")] = {
            "name": t.get("name"),
            "players": [(p.get("name"), p.get("rating")) for p in (t.get("players") or [])],
        }
    if wid not in sides:
        return {"matchid": rec.get("matchid"), "error": "winner not in stats payload"}

    problems = []
    for tid, s in sides.items():
        if len(s["players"]) != 5:
            problems.append("team %s has %d players" % (tid, len(s["players"])))
    all_players = [(n, r, tid) for tid, s in sides.items() for n, r in s["players"]]
    if any(r is None for _, r, _ in all_players):
        problems.append("missing rating")

    win_players = sorted(
        [(n, r) for n, r in sides[wid]["players"] if r is not None],
        key=lambda x: -x[1])
    lose_players = sorted(
        [(n, r) for tid, s in sides.items() if tid != wid
         for n, r in s["players"] if r is not None],
        key=lambda x: -x[1])
    if not win_players:
        # Keep the integrity problems: an error return that drops them would
        # hide *why* the roster could not be scored.
        return {"matchid": rec.get("matchid"),
                "error": "winner has no rated players",
                "problems": problems}

    top_all = max(all_players, key=lambda x: (x[1] is not None, x[1] or 0))
    out = {
        "matchid": rec.get("matchid"),
        "event": (match or {}).get("event"),
        "date": (match or {}).get("date"),
        "winner": winner.get("name"),
        "bo": (match or {}).get("best_of"),
        "mvp": win_players[0][0],
        "mvp_rating": win_players[0][1],
        "runner_up": win_players[1][0] if len(win_players) > 1 else None,
        "runner_up_rating": win_players[1][1] if len(win_players) > 1 else None,
        "margin": (round(win_players[0][1] - win_players[1][1], 3)
                   if len(win_players) > 1 else None),
        "best_overall_is_loser": bool(
            top_all[2] != wid and lose_players and top_all[1] > win_players[0][1]),
        "best_overall_player": top_all[0],
        "problems": problems,
    }
    return out


def cmd_report(args):
    stats = load_jsonl(os.path.join(HERE, args.stats))
    matches = {m["id"]: m for m in load_jsonl(os.path.join(HERE, args.matches))}
    rows = [analyse(r, matches.get(r["matchid"])) for r in stats]

    errs = [r for r in rows if r.get("error")]
    oks = [r for r in rows if not r.get("error")]
    print("stats records=%d parsed=%d errors=%d" % (len(stats), len(oks), len(errs)))
    for e in errs[:5]:
        print("   ERR %s: %s" % (e["matchid"], e["error"]))
    if not oks:
        return 1

    prob = [p for r in oks for p in r["problems"]]
    print("integrity problems: %d" % len(prob))
    for p in sorted(set(prob))[:8]:
        print("   %s" % p)

    margins = [r["margin"] for r in oks if r["margin"] is not None]
    print("\n-- derived label: highest series rating on the winning side --")
    print("matches=%d  distinct MVP players=%d" % (len(oks), len({r["mvp"] for r in oks})))
    print("margin winner#1 - winner#2: min=%.2f p25=%.2f median=%.2f p75=%.2f max=%.2f"
          % (
              min(margins),
              sorted(margins)[len(margins) // 4],
              sorted(margins)[len(margins) // 2],
              sorted(margins)[3 * len(margins) // 4],
              max(margins),
          ))
    for thr in (0.05, 0.10, 0.20):
        n = sum(1 for m in margins if m <= thr)
        print("   margin <= %.2f : %d/%d (%.1f%%)  <- near-ties, hard to call"
              % (thr, n, len(margins), 100.0 * n / len(margins)))

    losers = [r for r in oks if r["best_overall_is_loser"]]
    print("\nbest series rating belongs to the LOSING side: %d/%d (%.1f%%)"
          % (len(losers), len(oks), 100.0 * len(losers) / len(oks)))
    print("  -> with those, 'best player overall' picks a wrong-side name, while"
          " 'best on the predicted winner' cannot.")

    by_bo = collections.Counter(r["bo"] for r in oks)
    print("\nby best_of: %s" % dict(sorted(by_bo.items(), key=lambda kv: str(kv[0]))))
    top = collections.Counter(r["mvp"] for r in oks).most_common(8)
    print("most frequent derived MVPs: %s" % top)

    out = os.path.join(HERE, args.out)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "margins": margins}, fh, ensure_ascii=False, indent=1)
    print("\nMVP_ANALYSIS_OK -> %s" % args.out)
    return 0


def _rec(mid, t1, t2, p1, p2, by_map=False):
    def players(names):
        return [{"name": n, "rating": r} for n, r in names]
    blk = {"id": 0, "name": "All",
           "team1": {"id": t1[0], "name": t1[1], "players": players(p1)},
           "team2": {"id": t2[0], "name": t2[1], "players": players(p2)}}
    return {"matchid": mid, "overall": [blk], "by_map": [blk] if by_map else [blk]}


def cmd_selftest(args):
    ok = True
    m = {"id": 1, "winner": {"id": 10, "name": "W"}, "event": "E", "date": "d",
         "best_of": 3}
    r = _rec(1, (10, "W"), (20, "L"),
             [("a", 1.32), ("b", 1.05), ("c", 1.04), ("d", 0.94), ("e", 0.90)],
             [("x", 1.19), ("y", 1.08), ("z", 1.06), ("u", 1.03), ("v", 0.83)])
    a = analyse(r, m)
    if a.get("mvp") != "a" or abs(a["mvp_rating"] - 1.32) > 1e-9:
        print("FAIL mvp pick", a); ok = False
    if abs(a["margin"] - 0.27) > 1e-6:
        print("FAIL margin", a["margin"]); ok = False
    if a["best_overall_is_loser"]:
        print("FAIL loser flag"); ok = False

    # the loser owning the best rating must be flagged
    r2 = _rec(2, (10, "W"), (20, "L"),
              [("a", 1.05), ("b", 1.01), ("c", 1.0), ("d", 0.9), ("e", 0.8)],
              [("x", 1.40), ("y", 1.1), ("z", 1.0), ("u", 0.9), ("v", 0.8)])
    a2 = analyse(r2, m)
    if not a2["best_overall_is_loser"] or a2["mvp"] != "a":
        print("FAIL loser-owns-best", a2); ok = False

    # winner must be looked up by ID, not by side order
    r3 = _rec(3, (20, "L"), (10, "W"),
              [("x", 1.40), ("y", 1.1), ("z", 1.0), ("u", 0.9), ("v", 0.8)],
              [("a", 1.05), ("b", 1.01), ("c", 1.0), ("d", 0.9), ("e", 0.8)])
    a3 = analyse(r3, m)
    if a3["mvp"] != "a":
        print("FAIL side-order lookup", a3); ok = False

    # by_map=True payload must yield the same aggregate answer
    r4 = _rec(4, (10, "W"), (20, "L"),
              [("a", 1.32), ("b", 1.05), ("c", 1.04), ("d", 0.94), ("e", 0.90)],
              [("x", 1.19), ("y", 1.08), ("z", 1.06), ("u", 1.03), ("v", 0.83)],
              by_map=True)
    if analyse(r4, m)["mvp"] != "a":
        print("FAIL by_map aggregate"); ok = False

    # missing winner / short roster must be reported, not silently accepted
    a5 = analyse(_rec(5, (10, "W"), (20, "L"),
                      [("a", 1.1)], [("x", 1.0)]), m)
    if not any("has 1 players" in p for p in a5["problems"]):
        print("FAIL roster integrity", a5); ok = False
    a6 = analyse(_rec(6, (10, "W"), (20, "L"),
                      [("a", None)], [("x", 1.0)]),
                 {"id": 6, "winner": {"id": 10, "name": "W"}})
    if not any("missing rating" in p for p in a6["problems"]):
        print("FAIL missing rating", a6); ok = False
    a7 = analyse(_rec(7, (10, "W"), (20, "L"),
                      [("a", 1.1)], [("x", 1.0)]), {"id": 7})
    if a7.get("error") != "match has no winner":
        print("FAIL no-winner", a7); ok = False
    a8 = analyse(_rec(8, (10, "W"), (20, "L"),
                      [("a", 1.1)], [("x", 1.0)]),
                 {"id": 8, "winner": {"id": 99, "name": "Z"}})
    if a8.get("error") != "winner not in stats payload":
        print("FAIL winner-missing", a8); ok = False

    print("MVP_ANALYSIS_SELFTEST_%s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("report")
    p.add_argument("--stats", default="csapi_stats_cologne.jsonl")
    p.add_argument("--matches", default="csapi_matches_all.jsonl")
    p.add_argument("--out", default="mvp_derived_cologne.json")
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("selftest")
    p.set_defaults(func=cmd_selftest)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
