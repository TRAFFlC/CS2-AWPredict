#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""baseline_predictor.py —— issue-014 的**第一个四项完整基线**（2026-10-10）。

目标（主理人 2026-10-10 定）：**最高分 = 期望分最大化**。本脚本给出四项目标的
端到端基线分，并**按项拆出「丢了多少分」**，作为后续一切改进的参照系。

四项基线（全部时间安全；Q-011 剔除清单生效）：
  1. 获胜方：差分特征的 Logistic 回归（排名 / 积分 / 历史胜率 / 交手）。
  2. 获胜比分：训练集「胜方回合:败方回合」众数，方向跟随第 1 项预测。
  3. 选图：**池内**按训练集图频次取 top-k（k = bo）；另跑「新图池先验」变体对比。
  4. 系列 MVP：预测胜方队中「历史平均 rating 最高」且**近 120 天内为该队出场过**的选手。

用法：python -X utf8 baseline_predictor.py selftest | run
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)  # scorer.py 与本模块同目录（仓库布局）

import split as SPLIT  # noqa: E402
import pool_experiment as PE  # noqa: E402
from scorer import Prediction, score_match, max_total  # noqa: E402

MATCHES = os.path.join(DATA, "csapi_matches_all.jsonl")
TRUTH = os.path.join(DATA, "truth_maps_B.jsonl")
STATS = os.path.join(DATA, "csapi_stats_all.jsonl")
EXCLUDE = os.path.join(HERE, "exclude_band_unknown_ids.txt")
OUT = os.path.join(HERE, "baseline_predictor_results.json")

RECENT_DAYS = 120      # MVP 候选：近 N 天内为该队出过场


def load_jsonl(path):
    out = []
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _days_apart(d1, d2):
    """日期字符串 (YYYY-MM-DD) 的日差，纯手算避免额外依赖。"""
    def num(d):
        y, m, dd = (int(x) for x in d.split("-"))
        return y * 372 + m * 31 + dd
    return num(d1) - num(d2)


def load_stats(path=STATS):
    """matchid -> {team_name: [(player, rating), ...]}"""
    out = {}
    for rec in load_jsonl(path):
        block = None
        for b in (rec.get("overall") or []):
            if b.get("name") == "All":
                block = b
                break
        if block is None:
            continue
        per = {}
        for key in ("team1", "team2"):
            t = block.get(key) or {}
            name = t.get("name")
            if not name:
                continue
            per[name] = [(p.get("name"), p.get("rating")) for p in (t.get("players") or [])]
        if per:
            out[rec.get("matchid")] = per
    return out


def mvp_label(match, stats):
    """Q-008 口径的真值：**胜方队内系列 rating 最高者**。"""
    per = stats.get(match["id"])
    if not per:
        return ""
    ps = per.get(match["winner"]["name"]) or []
    best = None
    for name, rating in ps:
        if name is None or rating is None:
            continue
        if best is None or float(rating) > best[1]:
            best = (name, float(rating))
    return best[0] if best else ""


class Players:
    """选手级时序历史：平均 rating + 最近所属队 + 最近出场日。"""

    def __init__(self):
        self.s = collections.defaultdict(float)
        self.n = collections.defaultdict(int)
        self.last_team = {}
        self.last_date = {}

    def mean(self, p):
        return self.s[p] / self.n[p] if self.n[p] else None

    def best_on(self, team, date, recent=RECENT_DAYS):
        cands = [p for p, t in self.last_team.items()
                 if t == team and self.last_date.get(p) and _days_apart(date, self.last_date[p]) <= recent]
        if not cands:
            return ""
        return max(cands, key=lambda p: (self.mean(p) or 0.0))

    def update(self, date, per_team):
        for team, players in per_team.items():
            for name, rating in players:
                if not name or rating is None:
                    continue
                self.s[name] += float(rating)
                self.n[name] += 1
                self.last_team[name] = team
                self.last_date[name] = date


# --------------------------------------------------------------- 样本构建
def build(all_rows, ranks, eras, by_pool, tl, truths, stats, mvp_truth):
    """按日期分组：先取特征/预测上下文，再统一更新历史（防同日泄漏）。"""
    rows = [r for r in all_rows if isinstance(r.get("best_of"), int) and r["best_of"] in (1, 3, 5)]
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["date"], rows[i]["id"]))
    hist = PE.Hist()
    players = Players()
    samples = []
    i = 0
    while i < len(order):
        d = rows[order[i]]["date"]
        group = []
        while i < len(order) and rows[order[i]]["date"] == d:
            group.append(rows[order[i]])
            i += 1
        for r in group:
            tA, tB = r["team1"]["name"], r["team2"]["name"]
            t = truths.get(r["id"])
            veto = list(t.get("pool") or []) if t else []
            era, pool = PE.era_and_pool(d, veto, eras, by_pool, tl)
            ra, pa = ranks.rank_points(tA, d)
            rb, pb = ranks.rank_points(tB, d)
            feats = {
                "d_rank": rb - ra,
                "d_logp": math.log1p(pa) - math.log1p(pb),
                "d_wr": hist.match_wr(tA) - hist.match_wr(tB),
                "d_h2h": hist.h2h_rate(tA, tB) - hist.h2h_rate(tB, tA),
            }
            samples.append({
                "id": r["id"], "date": d, "era": era, "pool": pool,
                "teamA": tA, "teamB": tB, "winner": r["winner"]["name"],
                "bo": r["best_of"], "feats": feats, "truth": t,
                "y": 1 if r["winner"]["name"] == tA else 0,
                "mvp_cand": {tA: players.best_on(tA, d), tB: players.best_on(tB, d)},
                "mvp_truth": mvp_truth.get(r["id"], ""),
            })
        for r in group:
            tA, tB = r["team1"]["name"], r["team2"]["name"]
            mres = [(m["name"], m["team1_score"], m["team2_score"]) for m in (r.get("maps") or [])]
            hist.update(tA, tB, r["winner"]["name"], mres)
            players.update(d, stats.get(r["id"], {}))
    return samples


# --------------------------------------------------------------- 评估
def evaluate(samples, predict_winner, modal, freq, maps_variant="restrict", use_mvp=True, oracle_winner=False):
    agg = collections.Counter()
    for s in samples:
        t = s["truth"]
        if not t:
            continue
        truth = PE.to_truth(t)
        truth.mvp = s.get("mvp_truth", "")
        winA = (s["winner"] == s["teamA"]) if oracle_winner else (predict_winner(s) == 1)
        pwin = s["teamA"] if winA else s["teamB"]
        k = max(1, min(s["bo"], 5))
        if maps_variant == "restrict":
            pmaps = PE.predict_maps(freq, s["pool"], k)
        elif maps_variant == "newmap":
            unseen = [m for m in s["pool"] if freq.get(m, 0) == 0]
            pmaps = (PE.predict_maps(freq, s["pool"], k - 1) + [unseen[0]]) if (unseen and k >= 1) \
                else PE.predict_maps(freq, s["pool"], k)
        else:
            pmaps = []
        w, l = (2, 1) if s["bo"] == 3 else ((3, 1) if s["bo"] == 5 else (1, 0))
        prows = (w, l) if winA else (l, w)
        mvp = s["mvp_cand"].get(pwin, "") if use_mvp else ""
        pred = Prediction(winner=pwin, score=prows, maps=pmaps, mvp=mvp)
        b = score_match(pred, truth)
        agg["n"] += 1
        agg["max"] += max_total(truth)
        agg["winner"] += b.winner
        agg["score"] += b.score
        agg["maps"] += b.maps
        agg["mvp"] += b.mvp
        agg["total"] += b.total
        agg["winner_max"] += 10
        agg["score_max"] += 10 if s["bo"] in (3, 5) else 0
        agg["maps_max"] += 5 * k
        agg["mvp_max"] += 10
    return dict(agg)


def render(res, title):
    def line(name, got, mx):
        pct = 100.0 * got / mx if mx else 0.0
        return "  %-7s %6d / %6d = %5.1f%%   丢 %d" % (name, got, mx, pct, mx - got)

    L = ["== %s ==" % title, "  n=%d  满分=%d" % (res["n"], res["max"])]
    L.append(line("winner", res["winner"], res["winner_max"]))
    L.append(line("score", res["score"], res["score_max"]))
    L.append(line("maps", res["maps"], res["maps_max"]))
    L.append(line("mvp", res["mvp"], res["mvp_max"]))
    L.append("  TOTAL   %6d / %6d = %5.1f%%" % (
        res["total"], res["max"], 100.0 * res["total"] / res["max"] if res["max"] else 0))
    return "\n".join(L)


def run():
    rows = SPLIT.load(MATCHES)
    exclude = SPLIT.load_exclude(EXCLUDE) if os.path.exists(EXCLUDE) else None
    buckets, meta = SPLIT.split(rows, purge=True, exclude=exclude)
    truths = {t["id"]: t for t in load_jsonl(TRUTH)}
    stats = load_stats()
    eras, by_pool = PE.load_pool_eras()
    tl = json.load(io.open(PE.TIMELINE, encoding="utf-8"))
    ranks = PE.Ranks()

    all_rows = buckets["train"] + buckets["valid"] + buckets["test"]
    mvp_truth = {m["id"]: mvp_label(m, stats) for m in all_rows}
    samples = build(all_rows, ranks, eras, by_pool, tl, truths, stats, mvp_truth)
    by_id = {s["id"]: s for s in samples}
    parts = {k: [by_id[r["id"]] for r in buckets[k] if r["id"] in by_id] for k in ("train", "valid", "test")}

    train_truths = [s["truth"] for s in parts["train"] if s["truth"]]
    modal = PE.modal_scorelines(train_truths)
    freq = PE.map_freq(train_truths)
    predict = PE.fit_predict(parts["train"], None, PE.FEATURES["base"])

    out = {"split": meta["counts"], "excluded": meta.get("excluded_matches", 0),
           "modal_score": {str(k): list(v) for k, v in modal.items()}, "results": {}}
    lines = ["BASELINE_PREDICTOR  split=%s  excluded=%d" % (meta["counts"], meta.get("excluded_matches", 0)),
             "modal scoreline: " + ", ".join("bo%s=%s" % (k, tuple(v)) for k, v in modal.items())]
    for k in ("train", "valid", "test"):
        r_restrict = evaluate(parts[k], predict, modal, freq, "restrict", True)
        r_newmap = evaluate(parts[k], predict, modal, freq, "newmap", True)
        r_nomvp = evaluate(parts[k], predict, modal, freq, "restrict", False)
        r_oracle = evaluate(parts[k], predict, modal, freq, "restrict", True, oracle_winner=True)
        out["results"][k] = {"restrict": r_restrict, "newmap": r_newmap, "no_mvp": r_nomvp, "oracle_winner": r_oracle,
                             "eras": dict(collections.Counter(s["era"] for s in parts[k]))}
        lines.append("")
        lines.append(render(r_restrict, k + "（选图=池内频次 + MVP 基线）"))
        lines.append("  [变体] 新图固定槽: total %d ／ 去掉 MVP: total %d"
                     % (r_newmap["total"], r_nomvp["total"]))
        lines.append("  [边际] 胜方项做到完美(oracle): total %d（+%d）"
                     % (r_oracle["total"], r_oracle["total"] - r_restrict["total"]))
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print("\n".join(lines))
    return out


# --------------------------------------------------------------- selftest
def _selftest():
    fails = []

    def chk(n, got, want):
        if got != want:
            fails.append("%s: got=%r want=%r" % (n, got, want))

    chk("days_apart", _days_apart("2026-03-01", "2026-02-01"), 31)
    chk("days_apart same", _days_apart("2026-02-01", "2026-02-01"), 0)

    p = Players()
    p.update("2026-01-01", {"A": [("x", 1.0), ("y", 1.2)], "B": [("z", 0.9)]})
    chk("mean x", p.mean("x"), 1.0)
    chk("best on A", p.best_on("A", "2026-01-02"), "y")
    chk("best on B", p.best_on("B", "2026-01-02"), "z")
    chk("best on unknown team", p.best_on("Q", "2026-01-02"), "")
    chk("stale candidate excluded", p.best_on("A", "2027-01-01"), "")
    p.update("2026-01-10", {"A": [("x", 3.0)], "B": [("y", 0.5)]})
    chk("mean x after update", p.mean("x"), 2.0)
    chk("x still on A", p.last_team["x"], "A")
    chk("y moved to B", p.last_team["y"], "B")
    chk("best on A is x now", p.best_on("A", "2026-01-11"), "x")

    # MVP 真值：胜方队内最高 rating
    mt = {"winner": {"name": "W"}, "id": 9}
    st = {9: {"W": [("a", 1.1), ("b", 1.4), ("c", 0.9)], "L": [("z", 2.0)]}}
    chk("mvp_label picks winner best", mvp_label(mt, st), "b")
    chk("mvp_label missing stats", mvp_label(mt, {}), "")

    if fails:
        print("BASELINE_SELFTEST_FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("BASELINE_SELFTEST_OK (3 groups)")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "selftest":
        raise SystemExit(_selftest())
    if cmd == "run":
        run()
        raise SystemExit(0)
    print(__doc__)
    raise SystemExit(2)
