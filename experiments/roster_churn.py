#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Roster churn: how long does a CS2 lineup actually stay intact?

Motivation
----------
The project owner's argument (2026-10-08) is that CS2 roster churn is fast
enough that long historical data says almost nothing about a team's *current*
strength — so chasing pre-2025 history for the match corpus may be pointless.
That is a testable claim, and the weekly ranking archive already contains each
top-250 team's five players, so it can be measured instead of assumed.

Method
------
For every team, walk the weekly snapshots and take maximal runs of snapshots
sharing an identical five-player lineup.  Only runs whose consecutive snapshots
are <= `MAX_GAP` days apart count, so a team dropping out of the top 250 does not
get credited with a long-lived lineup it never had.  A run's duration is the
span from its first to last snapshot.

Outputs feed three decisions:
* how big a training window is actually informative (lineup tenure),
* whether the 11-month csapi corpus spans enough lineup generations,
* whether churn has changed over the years (so a fixed window stays valid).

Usage
-----
  python -X utf8 roster_churn.py report [--matches csapi_matches_all.jsonl]
  python -X utf8 roster_churn.py selftest
"""
import argparse
import collections
import datetime as dt
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from control_arm import norm_team                               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）
MAX_GAP = 10                # days; larger means the team left the top 250
RECENT_DAYS = 28            # "recently changed lineup" horizon for the corpus


def load_lineups(root):
    """Return (snapshots, players_by_date).

    snapshots: [(date_str, {norm_team: rank})]
    players_by_date: {date_str: {norm_team: frozenset(player_names)}}
    """
    snapshots, players = [], {}
    for path in sorted(glob.glob(os.path.join(root, "*", "*.json"))):
        base = os.path.basename(path)
        if len(base) != 15 or not base.endswith(".json"):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:                                   # noqa: BLE001
            continue
        if "teams" not in d or "date" not in d:
            continue
        rank, pl = {}, {}
        for t in d["teams"]:
            key = norm_team(t.get("name") or "")
            rank[key] = t.get("rank")
            squad = tuple(sorted(p.get("name") or ""
                                 for p in (t.get("players") or [])))
            if len(squad) == 5:
                pl[key] = frozenset(squad)
        snapshots.append((d["date"], rank))
        players[d["date"]] = pl
    snapshots.sort(key=lambda x: x[0])
    return snapshots, players


def _d(s):
    return dt.date.fromisoformat(s)


def lineup_runs(players_by_date, dates, max_gap=MAX_GAP, ranks_by_date=None,
                max_rank=None):
    """Yield (team, frozenset(lineup), first_date, last_date, n_snapshots).

    With `max_rank`, a team only counts on weeks where it sits inside that
    rank, which makes cohorts comparable across the archive's very uneven
    coverage (30 teams per week until 2023, ~250 afterwards).
    """
    for team in sorted({t for pl in players_by_date.values() for t in pl}):
        cur, cur_dates = None, []
        for date in dates:
            squad = players_by_date[date].get(team)
            if squad is None:
                continue                                # team not ranked that week
            if max_rank is not None and ranks_by_date is not None:
                r = (ranks_by_date.get(date) or {}).get(team)
                if r is None or r > max_rank:
                    continue
            if squad == cur:
                if cur_dates and (_d(date) - _d(cur_dates[-1])).days > max_gap:
                    yield (team, cur, cur_dates[0], cur_dates[-1], len(cur_dates))
                    cur_dates = [date]
                else:
                    cur_dates.append(date)
            else:
                if cur is not None:
                    yield (team, cur, cur_dates[0], cur_dates[-1], len(cur_dates))
                cur, cur_dates = squad, [date]
        if cur is not None:
            yield (team, cur, cur_dates[0], cur_dates[-1], len(cur_dates))


def _per_team(players_by_date, dates):
    out = {}
    for d in dates:
        for t, squad in players_by_date[d].items():
            out.setdefault(t, []).append(squad)
    return out


def lineup_in_force(players_by_date, dates, date, team):
    """(lineup, snapshot_date) from the latest snapshot on or before `date`."""
    best = None
    for d in dates:
        if d > date:
            break
        squad = players_by_date[d].get(team)
        if squad is not None:
            best = (squad, d)
    return best


def span_days(first, last):
    return (_d(last) - _d(first)).days


def cmd_report(args):
    snapshots, players = load_lineups(os.path.join(HERE, "rankings"))
    dates = [d for d, _ in snapshots]
    print("snapshots=%d  %s .. %s" % (len(snapshots), dates[0], dates[-1]))

    ranks = dict(snapshots)
    # (a) top-30 cohort: the only scope the WHOLE archive covers evenly
    runs = list(lineup_runs(players, dates, ranks_by_date=ranks, max_rank=30))
    by_team = collections.defaultdict(list)
    for team, squad, first, last, n in runs:
        by_team[team].append((span_days(first, last), n, first, last, squad))
    print("scope=top30: runs=%d across %d teams" % (len(runs), len(by_team)))

    spans = sorted(s for r in by_team.values() for s, *_ in r)
    def pct(q):
        return spans[min(len(spans) - 1, int(q * len(spans)))]
    print("\n-- lineup tenure (days from first to last week it was intact) --")
    for q, label in ((0.10, "p10"), (0.25, "p25"), (0.50, "median"),
                     (0.75, "p75"), (0.90, "p90")):
        print("   %-7s %4d days (%.1f months)" % (label, pct(q), pct(q) / 30.4))

    freed = [r for r in by_team.values() if any(s >= 182 for s, *_ in r)]
    print("   runs living >= 6 months: %d/%d (%.1f%%)"
          % (len(freed), len(by_team), 100.0 * len(freed) / len(by_team)))

    # churn per calendar year: how many lineups did a typical ranked team use?
    print("\n-- lineup changes per year (teams ranked that year) --")
    years = sorted({d[:4] for d in dates})
    for y in years:
        yd = [d for d in dates if d.startswith(y)]
        if len(yd) < 8:
            continue
        teams = collections.defaultdict(list)
        for d in yd:
            for t, squad in players[d].items():
                teams[t].append(squad)
        changes = [sum(1 for a, b in zip(v, v[1:]) if a != b) for v in teams.values()
                   if len(v) >= 8]
        if changes:
            changes.sort()
            med = changes[len(changes) // 2]
            print("   %s  teams=%-3d  median changes/team=%d  mean=%.2f"
                  % (y, len(changes), med, sum(changes) / len(changes)))

    # ---- (b) deep-coverage era, all ranked teams -------------------------
    deep_since = args.since or "2025-01-01"
    deep_dates = [d for d in dates if d >= deep_since]
    deep_runs = list(lineup_runs(players, deep_dates, ranks_by_date=ranks))
    spans_deep = sorted(span_days(f_, l_) for _, _, f_, l_, _ in deep_runs)
    print("\n-- scope: deep era (%s..), all ranked teams = %d runs --"
          % (deep_since, len(deep_runs)))
    if spans_deep:
        def pd(q):
            return spans_deep[min(len(spans_deep) - 1, int(q * len(spans_deep)))]
        print("   median tenure %d days (%.1f months) | p75 %d | p90 %d"
              % (pd(.5), pd(.5) / 30.4, pd(.75), pd(.90)))
        print("   runs >= 6 months: %d (%.1f%%)"
              % (sum(1 for x in spans_deep if x >= 182),
                 100.0 * sum(1 for x in spans_deep if x >= 182) / len(spans_deep)))
        changed = [sum(1 for x, y in zip(v, v[1:]) if x != y)
                   for t, v in _per_team(players, deep_dates).items() if len(v) >= 8]
        if changed:
            changed.sort()
            print("   lineup changes per team in era: median=%d mean=%.2f (n=%d teams)"
                  % (changed[len(changed) // 2], sum(changed) / len(changed),
                     len(changed)))

    # ---- corpus coverage against the archive -----------------------------
    mp = os.path.join(HERE, args.matches)
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as fh:
            matches = [json.loads(l) for l in fh if l.strip()]
        keys, recent, comparable, nodata = set(), 0, 0, 0
        for m in matches:
            date = m.get("date") or ""
            then = (_d(date) - dt.timedelta(days=RECENT_DAYS)).isoformat()
            for side in ("team1", "team2"):
                t = norm_team(m[side]["name"])
                base = lineup_in_force(players, dates, date, t)
                prior = lineup_in_force(players, dates, then, t)
                if base:
                    keys.add((t, base[0]))
                if base and prior:
                    comparable += 1
                    if base[0] != prior[0]:
                        recent += 1
                else:
                    nodata += 1
        print("\n-- match corpus (n=%d) --" % len(matches))
        print("   distinct (team, lineup) pairs: %d" % len(keys))
        print("   team-slots with lineups at both ends: %d (%.1f%%) | not enough data: %d"
              % (comparable, 100.0 * comparable / max(1, comparable + nodata), nodata))
        print("   ...of those, lineup changed within %d days before the match: %d (%.1f%%)"
              % (RECENT_DAYS, recent, 100.0 * recent / max(1, comparable)))

    out = os.path.join(HERE, args.out)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"spans": spans,
                   "per_team_runs": {k: len(v) for k, v in by_team.items()}},
                  fh, ensure_ascii=False)
    print("\nROSTER_CHURN_OK -> %s" % args.out)
    return 0


def cmd_timelines(args):
    """Lineup segments inside the corpus window for the teams we actually predict.

    The aggregate tenure distribution is bimodal and dominated by volatile
    tier-3 teams that flicker in and out of the top 250.  The prediction targets
    are major participants, so their lineup turnover is the number that matters.
    """
    import collections
    snapshots, players = load_lineups(os.path.join(HERE, "rankings"))
    dates = [d for d, _ in snapshots]
    ranks = dict(snapshots)

    with open(os.path.join(HERE, args.matches), encoding="utf-8") as fh:
        matches = [json.loads(l) for l in fh if l.strip()]
    appear = collections.Counter()
    for m in matches:
        for side in ("team1", "team2"):
            appear[norm_team(m[side]["name"])] += 1
    names = {norm_team(m[side]["name"]): m[side]["name"] for m in matches
             for side in ("team1", "team2")}

    lo = min(m["date"] for m in matches if m.get("date"))
    hi = max(m["date"] for m in matches if m.get("date"))
    win = [d for d in dates if lo <= d <= hi]
    print("corpus window %s .. %s  (%d weekly snapshots)" % (lo, hi, len(win)))
    print("top %d teams by matches played:" % args.top)
    print()
    rows = []
    for team, n in appear.most_common(args.top):
        runs = [r for r in lineup_runs(players, win, ranks_by_date=ranks)
                if r[0] == team]
        segs = ["%s..%s" % (f_[5:], l_[5:]) for _, _, f_, l_, _ in runs]
        r30 = [r for r in lineup_runs(players, win, ranks_by_date=ranks, max_rank=30)
               if r[0] == team]
        rows.append({"team": names.get(team, team), "matches": n,
                     "distinct_lineups_in_window": len(runs),
                     "top30_lineups": len(r30), "segments": segs})
        print("  %-14s matches=%-3d lineups=%-2d (top30: %d)  %s"
              % (names.get(team, team), n, len(runs), len(r30),
                 " ".join(segs[:8])))
    import json as _json
    with open(os.path.join(HERE, "roster_timelines.json"), "w", encoding="utf-8") as fh:
        _json.dump(rows, fh, ensure_ascii=False, indent=1)
    print("ROSTER_TIMELINES_OK -> roster_timelines.json")
    return 0


def cmd_selftest(args):
    ok = True
    dates = ["2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26",
             "2026-02-02", "2026-03-02"]
    A = frozenset(["a", "b", "c", "d", "e"])
    B = frozenset(["a", "b", "c", "d", "z"])
    players = {
        "2026-01-05": {"x": A},
        "2026-01-12": {"x": A},
        "2026-01-19": {"x": A},
        "2026-01-26": {"x": B},
        "2026-02-02": {"x": B},
        "2026-03-02": {"x": B},
    }
    runs = list(lineup_runs(players, dates))
    # the A run is 3 weeks; B splits because the 2026-02-02 -> 2026-03-02 gap
    # exceeds MAX_GAP (the team had left the top 250), so B yields two runs.
    # NOTE: a >MAX_GAP absence splits a run, so B yields 2 then 1.
    if [n for _, _, _, _, n in runs] != [3, 2, 1]:
        print("FAIL runs", [(s, f, l, n) for _, s, f, l, n in runs]); ok = False
    if span_days(runs[0][2], runs[0][3]) != 14:
        print("FAIL span", runs[0]); ok = False

    # non-5-man lineups must be ignored, not treated as a change
    players2 = dict(players)
    players2["2026-01-12"] = {}
    runs2 = list(lineup_runs(players2, dates))
    if [n for _, _, _, _, n in runs2] != [1, 1, 2, 1]:
        print("FAIL short-roster skip", [n for _, _, _, _, n in runs2]); ok = False

    print("ROSTER_CHURN_SELFTEST_%s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("report")
    p.add_argument("--matches", default="csapi_matches_all.jsonl")
    p.add_argument("--out", default="roster_churn.json")
    p.add_argument("--since", default="2025-01-01",
                   help="start of the deeply covered era")
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("timelines")
    p.add_argument("--matches", default="csapi_matches_all.jsonl")
    p.add_argument("--top", type=int, default=15)
    p.set_defaults(func=cmd_timelines)

    p = sub.add_parser("selftest")
    p.set_defaults(func=cmd_selftest)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
