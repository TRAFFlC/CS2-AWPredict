#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
issue-014 数据采集：Liquipedia Counter-Strike wikitext → 结构化比赛记录。

只依赖标准库。输入：`action=parse&prop=wikitext&formatversion=2` 的 JSON。

抽取字段（每条比赛）：event/page, date, team1, team2,
  maps=[{map, score1, score2, winner}], series={winner, score1, score2}, hltv_id
未打的图（双方 0 回合的占位）会被丢弃——它们不是「实际打到的图」。

复用：`python -X utf8 liquipedia_matches.py --json playoffs.json --out matches.jsonl`
自证：`python -X utf8 liquipedia_matches.py --selftest`
"""
from __future__ import annotations

import json
import re
import sys

# ---------------------------------------------------------------- brace utils

def _find_balanced(s: str, i: int) -> int:
    """s[i:i+2] == '{{'，返回匹配 '}}' 之后的下标；失败返回 -1。"""
    if not s.startswith("{{", i):
        return -1
    depth = 0
    j = i
    n = len(s)
    while j < n - 1:
        if s.startswith("{{", j):
            depth += 1
            j += 2
            continue
        if s.startswith("}}", j):
            depth -= 1
            j += 2
            if depth == 0:
                return j
            continue
        j += 1
    return -1


def _iter_templates(s: str, name: str):
    """迭代出现的 {{name ...}}，yield 去掉外侧花括号后的 inner（不以 name 开头）。"""
    k = 0
    tag = "{{" + name
    while True:
        i = s.find(tag, k)
        if i < 0:
            return
        after_idx = i + len(tag)
        after = s[after_idx] if after_idx < len(s) else ""
        if after not in ("|", "}", "\n", "\r", " ", "\t", ""):
            k = after_idx
            continue
        j = _find_balanced(s, i)
        if j < 0:
            return
        yield s[after_idx:j - 2]
        k = j


def _split_top(inner: str):
    """按顶层 '|' 切分模板 body（跳过嵌套括号内的 '|'）。"""
    parts = []
    depth = 0
    last = 0
    j = 0
    n = len(inner)
    while j < n:
        if inner.startswith("{{", j):
            depth += 1
            j += 2
            continue
        if inner.startswith("}}", j):
            depth -= 1
            j += 2
            continue
        if inner[j] == "|" and depth == 0:
            parts.append(inner[last:j])
            last = j + 1
        j += 1
    parts.append(inner[last:])
    return parts


def _params(inner: str) -> dict:
    out = {}
    for p in _split_top(inner)[1:]:  # [0] 是模板名
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _clean_team(v: str) -> str:
    """'{{TeamOpponent|spirit}}' → 'spirit'。"""
    v = v.strip()
    m = re.match(r"\{\{\s*TeamOpponent\s*\|\s*([^|}]+)", v)
    if m:
        return m.group(1).strip()
    return re.sub(r"[{}]", "", v).strip()


def _map_score(params: dict, team: int) -> int:
    """把一张图的 t{n}t/t{n}ct 与所有加时的 o{k}t{n}t/o{k}t{n}ct 求和。"""
    pat = re.compile(rf"^(?:t{team}t|t{team}ct|o\d+t{team}t|o\d+t{team}ct)$")
    total = 0
    for k, v in params.items():
        if pat.match(k) and re.fullmatch(r"\d+", v or ""):
            total += int(v)
    return total


def parse_wikitext(wt: str, page: str = "", event: str = ""):
    """返回结构化比赛列表；未打的占位图（0:0）被丢弃。"""
    out = []
    for match_inner in _iter_templates(wt, "Match"):
        p = _params(match_inner)
        if "opponent1" not in p and "opponent2" not in p:
            continue
        rec = {
            "event": event,
            "page": page,
            "date": re.sub(r"\{\{.*?\}\}", "", p.get("date", "")).strip(),
            "team1": _clean_team(p.get("opponent1", "")),
            "team2": _clean_team(p.get("opponent2", "")),
            "maps": [],
            "hltv_id": p.get("hltv", "").strip(),
        }
        for k, v in p.items():
            if re.fullmatch(r"map\d+", k) and v.strip().startswith("{{Map"):
                mp = _params(v.strip()[2:-2])
                if "map" not in mp:
                    continue
                s1, s2 = _map_score(mp, 1), _map_score(mp, 2)
                if s1 == 0 and s2 == 0:
                    continue  # 未打的占位图
                rec["maps"].append({
                    "map": mp["map"],
                    "score1": s1,
                    "score2": s2,
                    "winner": rec["team1"] if s1 > s2 else (rec["team2"] if s2 > s1 else None),
                })
        s1 = sum(1 for m in rec["maps"] if m["winner"] == rec["team1"])
        s2 = sum(1 for m in rec["maps"] if m["winner"] == rec["team2"])
        rec["series"] = {
            "score1": s1, "score2": s2,
            "winner": rec["team1"] if s1 > s2 else (rec["team2"] if s2 > s1 else None),
        }
        if rec["maps"]:
            out.append(rec)
    return out


# ---------------------------------------------------------------- selftest

_SAMPLE = """
{{Match
 |opponent1={{TeamOpponent|spirit}}|opponent2={{TeamOpponent|g2}}
 |date=June 19, 2026 - 15:45 {{Abbr/CEST}} |finished=true
 |map1={{Map|map=Overpass|finished=true
   |t1firstside=ct|t1t=2|t1ct=7|t2t=5|t2ct=8
   |stats=231751}}
 |map2={{Map|map=Dust II|finished=true
   |t1firstside=t|t1t=6|t1ct=6|t2t=6|t2ct=6
   |o1t1t=2|o1t1ct=2|o1t2t=1|o1t2ct=1
   |stats=231755}}
 |hltv=2394998
}}
"""

# 2:0 的 bo3 仍带一个未打的 map3 占位（0:0）——必须被丢弃。
_SAMPLE_UNPLAYED = """
{{Match
 |opponent1={{TeamOpponent|aurora}}|opponent2={{TeamOpponent|betboom}}
 |date=June 18, 2026 - 15:45 |finished=true
 |map1={{Map|map=Nuke|finished=true|t1t=7|t1ct=6|t2t=3|t2ct=3}}
 |map2={{Map|map=Anubis|finished=true|t1t=6|t1ct=7|t2t=4|t2ct=5}}
 |map3={{Map|map=Dust II|finished=false|t1t=0|t1ct=0|t2t=0|t2ct=0}}
 |hltv=2394771
}}
"""


def _selftest() -> int:
    fails = []

    recs = parse_wikitext(_SAMPLE, page="t", event="e")
    if len(recs) != 1:
        fails.append(f"sample1 count={len(recs)}")
    else:
        r = recs[0]
        exp = [("Overpass", 9, 13), ("Dust II", 16, 14)]
        got = [(m["map"], m["score1"], m["score2"]) for m in r["maps"]]
        if got != exp:
            fails.append(f"maps got={got} want={exp}")
        if r["series"] != {"score1": 1, "score2": 1, "winner": None}:
            fails.append(f"series={r['series']}")
        if r["hltv_id"] != "2394998":
            fails.append(f"hltv_id={r['hltv_id']}")
        if r["team1"] != "spirit" or r["team2"] != "g2":
            fails.append(f"teams={r['team1']},{r['team2']}")

    recs2 = parse_wikitext(_SAMPLE_UNPLAYED, page="t2", event="e2")
    if not recs2:
        fails.append("sample2 empty")
    else:
        r2 = recs2[0]
        if len(r2["maps"]) != 2:
            fails.append(f"unplayed map not dropped: {[m['map'] for m in r2['maps']]}")
        if r2["series"] != {"score1": 2, "score2": 0, "winner": "aurora"}:
            fails.append(f"sample2 series={r2['series']}")

    if fails:
        print("LIQUIPEDIA_PARSER_SELFTEST_FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("LIQUIPEDIA_PARSER_SELFTEST_OK")
    return 0


def main(argv):
    if "--selftest" in argv:
        return _selftest()
    jf = argv[argv.index("--json") + 1] if "--json" in argv else None
    out = argv[argv.index("--out") + 1] if "--out" in argv else None
    if not jf:
        print(__doc__)
        return 0
    d = json.load(open(jf, encoding="utf-8"))
    parse = d.get("parse", {})
    wt = parse.get("wikitext", "")
    title = parse.get("title", "")
    event = title.split("/")[0] if title else ""
    recs = parse_wikitext(wt, page=title, event=event)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{title}: {len(recs)} matches, {sum(len(r['maps']) for r in recs)} maps -> {out or 'stdout'}")
    for r in recs:
        ms = " | ".join(f"{m['map']} {m['score1']}:{m['score2']}" for m in r["maps"])
        print(f"  {r['date'][:22]:22s} {r['team1']:>10s} {r['series']['score1']}:{r['series']['score2']} {r['team2']:<10s}  {ms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
