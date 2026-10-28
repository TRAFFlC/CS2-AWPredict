#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用公开赛果重算 `fixtures/team_strength.json` 里的 Elo。

为什么需要这个脚本
------------------
原强度库的 Elo 只吃到 `data/csapi_matches_all.jsonl` 的 2800 场（止 2026-09-05）。
仓库里还有公开增量赛果 `data/groundtruth/new_records_*.jsonl`。本脚本把两批按日期
顺序拼起来重放 Elo，只更新 `rating`，VRS 的 `rank` / `points` 原样保留。

口径
----
* K=24，初始 1500，标准 Elo：`E = 1 / (1 + 10 ** ((Rb-Ra)/400))`。
* 队名按强度库的 `name` + `aliases` 归一（例如 `BetBoom` → `BETBOOM`）。
* 增量语料**不是完整赛程**：脚本把实际最后一场的日期写进 `elo_asof`，
  不假装它等于 VRS 快照日。
* `rank_points` 的胜率斜率不在本脚本范围内，仍沿用 `sources.FITTED` 的旧拟合值。

用法
----
    python pickem/update_elo.py            # 只预览，不写盘
    python pickem/update_elo.py --write    # 写回 fixtures/team_strength.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXTURE = os.path.join(HERE, "fixtures", "team_strength.json")
ORIG = os.path.join(ROOT, "data", "csapi_matches_all.jsonl")
NEW_GLOB = os.path.join(ROOT, "data", "groundtruth", "new_records_*.jsonl")
K = 24.0
START = 1500.0


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def team_name(value):
    if isinstance(value, dict):
        return str(value["name"])
    return str(value)


def load_new(pattern):
    """读全部 `new_records_*.jsonl`，按 id 去重（同一场可能跨批次重复）。"""
    out = {}
    for path in sorted(glob.glob(pattern)):
        for rec in load_jsonl(path):
            key = str(rec.get("id") or "")
            if not key:
                key = "%s|%s|%s" % (rec.get("date"), team_name(rec["team1"]),
                                    team_name(rec["team2"]))
            out[key] = rec
    return list(out.values())


def canon_map(teams):
    """name/aliases（不区分大小写）→ 强度库里的正式队名。"""
    out = {}
    for t in teams:
        out[t["name"].lower()] = t["name"]
    for t in teams:
        for alias in t.get("aliases") or []:
            out.setdefault(str(alias).lower(), t["name"])
    return out


def replay(matches, canon):
    """按给定顺序逐场累计 Elo，返回 {正式队名: rating}。"""
    ratings = {}
    for rec in matches:
        a = team_name(rec["team1"])
        b = team_name(rec["team2"])
        w = team_name(rec["winner"])
        a = canon.get(a.lower(), a)
        b = canon.get(b.lower(), b)
        w = canon.get(w.lower(), w)
        if w not in (a, b):
            raise ValueError("winner=%r 不在对阵 %r vs %r 里（match id=%r）"
                             % (w, a, b, rec.get("id")))
        ra = ratings.get(a, START)
        rb = ratings.get(b, START)
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        sa = 1.0 if w == a else 0.0
        ratings[a] = ra + K * (sa - ea)
        ratings[b] = rb + K * ((1.0 - sa) - (1.0 - ea))
    return ratings


def base_provenance(text):
    """剥掉本脚本上一次生成的段落，避免重复 --write 时 provenance 无限嵌套。"""
    marker = "原 provenance: "
    if text.startswith("rating = 公开语料重算的 Elo：") and marker in text:
        return text.split(marker, 1)[1]
    return text


def build_doc(doc, ratings, corpus):
    """返回写盘用的强度库；corpus 记录语料规模与日期切点。"""
    old_prov = base_provenance(doc.get("_provenance") or "")
    if corpus["n_new"]:
        inc = ("data/groundtruth/new_records_*.jsonl %d 场增量（%s..%s）"
               % (corpus["n_new"], corpus["new_start"], corpus["new_end"]))
    else:
        inc = "无增量赛果"
    gap = ""
    if corpus["elo_asof"] < str(doc.get("asof") or ""):
        gap = ("Elo 切点 %s 早于 VRS 快照 %s，二者之间的赛果未覆盖；"
               % (corpus["elo_asof"], doc.get("asof")))
    new_prov = (
        "rating = 公开语料重算的 Elo：data/csapi_matches_all.jsonl %d 场"
        "（%s..%s）+ %s，K=24、初始 1500、按日期逐场累计；"
        "语料切点 %s（见 elo_asof）。VRS rank/points 仍为 %s 快照，未重估；"
        "%s增量语料不是完整赛程，Elo 仍可能滞后。"
        "rank_points 胜率斜率沿用既有历史拟合值，尚未针对本次 VRS 快照重新校准。"
        "aliases 保留原库值；未出现在语料中的队伍保留原 rating。"
        "原 provenance: %s"
        % (corpus["n_orig"], corpus["orig_start"], corpus["orig_end"], inc,
           corpus["elo_asof"], doc.get("asof"), gap, old_prov)
    )
    teams = doc["teams"]
    counts = dict(doc.get("counts") or {})
    counts["elo"] = sum(1 for t in teams if t.get("rating") is not None)
    counts["both"] = sum(1 for t in teams
                         if t.get("rating") is not None and t.get("points") is not None)
    return {
        "asof": doc["asof"],
        "elo_asof": corpus["elo_asof"],
        "elo_matches": corpus["n"],
        "source": doc.get("source", "elo"),
        "_role": ("公开强度库：队名查找与两种强度来源。rating = 公开语料重算的 Elo"
                  "（截至 %s）；rank/points 来自 VRS 快照（%s）。"
                  "公开侧只查表，不读 data/。" % (corpus["elo_asof"], doc.get("asof"))),
        "_provenance": new_prov,
        "counts": counts,
        "teams": teams,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="用公开赛果重算 pickem 强度库的 Elo")
    ap.add_argument("--write", action="store_true", help="写回 fixture（默认只预览）")
    args = ap.parse_args(argv)

    with open(FIXTURE, encoding="utf-8") as fh:
        doc = json.load(fh)
    canon = canon_map(doc["teams"])

    orig = load_jsonl(ORIG)
    new = load_new(NEW_GLOB)
    if not orig:
        raise SystemExit("原始语料为空：%s" % ORIG)
    orig.sort(key=lambda r: (r["date"], str(r["id"])))
    new.sort(key=lambda r: (r["date"], str(r["id"])))
    matches = orig + new
    ratings = replay(matches, canon)

    changed = []
    for t in doc["teams"]:
        old = t.get("rating")
        if t["name"] in ratings:
            new_val = ratings[t["name"]]
            if old is None or abs(old - new_val) > 1e-9:
                changed.append((t["name"], old, new_val))
            t["rating"] = new_val

    corpus = {
        "n_orig": len(orig), "orig_start": orig[0]["date"], "orig_end": orig[-1]["date"],
        "n_new": len(new),
        "new_start": new[0]["date"] if new else "—",
        "new_end": new[-1]["date"] if new else "—",
        "n": len(matches),
        "elo_asof": max(r["date"] for r in matches),
    }
    out = build_doc(doc, ratings, corpus)
    changed.sort(key=lambda x: -abs(x[2] - (x[1] if x[1] is not None else 0.0)))

    print("语料：%d 场原始（%s..%s）+ %d 场增量（%s..%s）= %d 场"
          % (corpus["n_orig"], corpus["orig_start"], corpus["orig_end"],
             corpus["n_new"], corpus["new_start"], corpus["new_end"], corpus["n"]))
    print("Elo 切点：%s；更新 %d 支；有 Elo %d / 有 VRS %d / 两者都有 %d"
          % (corpus["elo_asof"], len(changed), out["counts"].get("elo"),
             out["counts"].get("vrs"), out["counts"].get("both")))
    for name, old, new_val in changed[:20]:
        old_s = "null" if old is None else "%.2f" % old
        print("  %-22s %8s -> %8.2f  (%+.2f)"
              % (name, old_s, new_val, new_val - (old or 0.0)))

    if not args.write:
        print("\n（预览模式；加 --write 才会写回 %s）" % FIXTURE)
        return 0

    tmp = FIXTURE + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    os.replace(tmp, FIXTURE)
    print("\n已写回：%s" % FIXTURE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
