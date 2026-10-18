#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Control arm: ">= 2 top-10 teams" vs the tier-based major definition.

The project's original major criterion was "at least two of the world's top ten
teams participate".  The organisers then adopted an event-tier criterion
(Valve Tier 1 / Liquipedia S-Tier).  This script computes the ORIGINAL criterion
on the same match corpus so the two can be compared on identical ground.

Two facts drive the design (both measured 2026-10-07):

* csapi's per-match `rank` is a CURRENT snapshot, not a point-in-time rank: each
  team had exactly one distinct rank across 11 months (e.g. Spirit always 1,
  Astralis always 20).  Point-in-time ranks therefore come from the weekly
  hltv-ranking archive in ./rankings/.
* csapi splits one tournament across many event names ("IEM Cologne Major 2026",
  "... Stage 1", "... Stage 2").  Events are therefore grouped into families by
  their distinctive-token set, reusing the collector's normaliser.

Usage
-----
  python -X utf8 control_arm.py report   [--matches csapi_matches.jsonl] [--top 10]
  python -X utf8 control_arm.py selftest
"""
import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csapi_collect import core_tokens, event_year, norm_event   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# csapi and hltv-ranking spell a few orgs differently.
NAME_ALIASES = {
    "natus vincere": "navi",
    "natusvincere": "navi",
    "faze clan": "faze",
    "team liquid": "liquid",
    "virtus pro": "virtuspro",
    "virtus.pro": "virtuspro",
    "g2 esports": "g2",
    "mousesports": "mouz",
    "team spirit": "spirit",
    "team vitality": "vitality",
    "team falcons": "falcons",
    "the mongooses": "m80",
    "m80": "m80",
}


def norm_team(name):
    s = norm_event(name).replace(" ", "")
    return NAME_ALIASES.get(s, NAME_ALIASES.get(norm_event(name), s))


# ------------------------------------------------------------------ rankings
def load_rankings(root, limit_dates=6):
    """Return [(date, {norm_team: rank}), ...] sorted by date."""
    out = []
    for path in glob.glob(os.path.join(root, "*", "*.json")):
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
        rank = {}
        for t in d["teams"]:
            rank[norm_team(t.get("name") or "")] = t.get("rank")
        out.append((d["date"], rank))
    out.sort(key=lambda x: x[0])
    return out


def snapshot_for(rankings, date):
    """Latest snapshot on or before `date` (the ranking in force that day)."""
    best = None
    for d, rank in rankings:
        if d <= date:
            best = rank
        else:
            break
    if best is None and rankings:
        best = rankings[0][1]
    return best


# ------------------------------------------------------------------- events
STAGE_WORDS = {"stage", "group", "groups", "playoff", "playoffs", "swiss",
               "elimination", "opening", "split", "final", "finals"}


def family_key(name):
    """Tournament family key.

    Distinctive tokens, except that a sub-event number is ignored when the name
    carries a stage word, so "IEM Cologne Major 2026 Stage 1" and
    "IEM Cologne Major 2026" merge.  A season number is kept ("ESL Pro League
    Season 23" must not merge with Season 24), which is why all digits cannot
    simply be dropped.
    """
    tk = core_tokens(name)
    words = norm_event(name).split()
    if any(w in STAGE_WORDS for w in words):
        tk = {t for t in tk if not (t.isdigit() and len(t) < 4)}
    return frozenset(tk)


def event_families(matches):
    """Group matches into tournament families keyed by distinctive tokens."""
    fam = collections.defaultdict(lambda: {"names": set(), "matches": [],
                                           "teams": set(), "dates": []})
    for m in matches:
        key = family_key(m.get("event") or "")
        f = fam[key]
        f["names"].add(m.get("event") or "")
        f["matches"].append(m)
        f["dates"].append(m.get("date") or "")
        for side in ("team1", "team2"):
            f["teams"].add(m[side]["name"])
    return fam


def evaluate(matches, rankings, top=10):
    fam = event_families(matches)
    rows = []
    unresolved = collections.Counter()
    for key, f in fam.items():
        start = min(d for d in f["dates"] if d) if any(f["dates"]) else ""
        snap = snapshot_for(rankings, start) or {}
        ranked = []
        for t in sorted(f["teams"]):
            r = snap.get(norm_team(t))
            if r is None:
                unresolved[t] += 1
            elif r <= top:
                ranked.append((r, t))
        ranked.sort()
        rows.append({
            "family": sorted(f["names"])[0],
            "names": sorted(f["names"]),
            "year": event_year(start),
            "start": start,
            "end": max(f["dates"]) if f["dates"] else "",
            "n_matches": len(f["matches"]),
            "n_teams": len(f["teams"]),
            "n_top": len(ranked),
            "top_teams": ranked,
            "control_arm": len(ranked) >= 2,
        })
    rows.sort(key=lambda r: (r["start"], r["family"]))
    return rows, unresolved


def cmd_report(args):
    with open(os.path.join(HERE, args.matches), encoding="utf-8") as fh:
        matches = [json.loads(l) for l in fh if l.strip()]
    rankings = load_rankings(os.path.join(HERE, "rankings"))
    print("rankings: %d weekly snapshots (%s .. %s)"
          % (len(rankings), rankings[0][0], rankings[-1][0]))

    rows, unresolved = evaluate(matches, rankings, top=args.top)
    arm = [r for r in rows if r["control_arm"]]
    print("matches=%d  families=%d  control-arm(>=2 top-%d)=%d"
          % (len(matches), len(rows), args.top, len(arm)))
    tot_pairs = sum(r["n_teams"] for r in rows)
    print("unresolved team names: %d distinct / %d name-slots (%.1f%% resolved)"
          % (len(unresolved), tot_pairs,
             100.0 * (1 - sum(unresolved.values()) / max(1, tot_pairs))))
    if unresolved:
        print("  most common unresolved: %s"
              % ", ".join("%s(%d)" % kv for kv in unresolved.most_common(8)))

    print("\n-- control arm (>=2 top-%d at event start) --" % args.top)
    for r in arm:
        print("  %-10s %-44s m=%-3d teams=%-3d top%d=%s"
              % (r["start"], r["family"][:44], r["n_matches"], r["n_teams"],
                 args.top, [t for _, t in r["top_teams"]]))

    # tier-arm set on the SAME csapi event universe
    tier_set = set()
    ov_path = os.path.join(HERE, "csapi_event_overlap.json")
    if os.path.exists(ov_path):
        with open(ov_path, encoding="utf-8") as fh:
            ov = json.load(fh)
        for label in ("stier", "tier1"):
            for e in ov.get(label, {}).get("match", []):
                tier_set.update(e.get("group") or [])
    stats = {"families": len(rows), "control_arm": len(arm)}
    overlap = {"tier_named_events": len(tier_set)}
    ctrl_fam = {frozenset(core_tokens(r["family"])) for r in arm}
    if tier_set:
        tier_keys = {frozenset(core_tokens(e)) for e in tier_set}
        both = ctrl_fam & tier_keys
        overlap.update({
            "both": len(both),
            "control_only": len(ctrl_fam - tier_keys),
            "tier_only": len(tier_keys - ctrl_fam),
        })
        print("\n-- control arm vs tier-based set (same corpus) --")
        print("  control-only: %d" % overlap["control_only"])
        for k in sorted(ctrl_fam - tier_keys):
            print("     %s" % " ".join(sorted(k)))
        print("  tier-only (matched csapi event with <2 top-%d): %d"
              % (args.top, overlap["tier_only"]))
        for k in sorted(tier_keys - ctrl_fam):
            print("     %s" % " ".join(sorted(k)))

    with open(os.path.join(HERE, "control_arm.json"), "w", encoding="utf-8") as fh:
        json.dump({"top_n": args.top, "stats": stats, "overlap": overlap,
                   "rows": rows,
                   "unresolved": dict(unresolved)}, fh,
                  ensure_ascii=False, indent=1)
    print("\nCONTROL_ARM_OK -> control_arm.json")
    return 0


def cmd_selftest(args):
    ok = True

    # 1. team-name normalisation collapses the known spellings
    for a, b in (("Natus Vincere", "NAVI"), ("Team Vitality", "Vitality"),
                 ("Virtus.pro", "Virtus.Pro"), ("G2", "g2"),
                 ("Spirit", "Spirit"), ("MOUZ", "Mouz")):
        if norm_team(a) != norm_team(b):
            print("FAIL norm_team", a, norm_team(a), "!=", b, norm_team(b))
            ok = False

    # 2. snapshot_for picks the latest date <= target
    rk = [("2026-01-05", {"a": 1}), ("2026-01-12", {"a": 2}),
          ("2026-01-19", {"a": 3})]
    if snapshot_for(rk, "2026-01-12")["a"] != 2:
        print("FAIL snapshot exact"); ok = False
    if snapshot_for(rk, "2026-01-15")["a"] != 2:
        print("FAIL snapshot between"); ok = False
    if snapshot_for(rk, "2026-01-01")["a"] != 1:
        print("FAIL snapshot before"); ok = False

    # 3. event families merge stages but not different tournaments
    def mk(i, ev, t1, t2):
        return {"id": i, "event": ev, "date": "2026-06-02",
                "team1": {"name": t1}, "team2": {"name": t2}}
    ms = [mk(1, "IEM Cologne Major 2026 Stage 1", "A", "B"),
          mk(2, "IEM Cologne Major 2026", "C", "D"),
          mk(3, "PGL Astana 2026", "E", "F"),
          mk(4, "IEM Cologne Major 2026 Stage 2", "G", "H"),
          mk(5, "ESL Pro League Season 23", "I", "J"),
          mk(6, "ESL Pro League Season 24", "K", "L")]
    fam = event_families(ms)
    if len(fam) != 4:
        print("FAIL family grouping", len(fam),
              sorted(" ".join(sorted(k)) for k in fam)); ok = False
    merged = [f for f in fam.values() if len(f["names"]) == 3]
    if len(merged) != 1 or len(merged[0]["matches"]) != 3:
        print("FAIL stage merge", [sorted(f["names"]) for f in fam.values()]); ok = False
    if any(len(f["names"]) > 3 for f in fam.values()):
        print("FAIL season merge"); ok = False

    # 4. arm counting uses the point-in-time snapshot, not a team's best rank
    ms2 = [mk(1, "Test Cup 2026", "Spirit", "Astralis")]
    rk2 = [("2026-06-01", {"spirit": 1, "astralis": 11})]
    rows, _ = evaluate(ms2, rk2, top=10)
    if rows[0]["n_top"] != 1 or rows[0]["control_arm"]:
        print("FAIL arm count", rows[0]); ok = False
    rk3 = [("2026-06-01", {"spirit": 1, "astralis": 9})]
    rows2, _ = evaluate(ms2, rk3, top=10)
    if not rows2[0]["control_arm"]:
        print("FAIL arm positive", rows2[0]); ok = False

    # 5. unresolved names are counted, not silently dropped
    rows3, unr = evaluate(ms2, [("2026-06-01", {"spirit": 1})], top=10)
    if unr.get("Astralis") != 1:
        print("FAIL unresolved counter", unr); ok = False

    print("CONTROL_ARM_SELFTEST_%s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("report")
    p.add_argument("--matches", default="csapi_matches.jsonl")
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("selftest")
    p.set_defaults(func=cmd_selftest)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
