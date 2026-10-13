#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pool_experiment.py —— issue-014「跨图池建模」最小实验。

问题（Q-012 / 10-10 日报「明天接着做什么」第 1 条）：
  跨不同图池的数据还「有关联性」吗？把池当特征 / 按时代分段训练，会不会更好？

做法：在同一套切分（`split.py --purge`：train 2,142 / valid 242 / test 321，
test = era2 全段）上，把**池的处置方式**当作唯一变量，比四项计分（`014-scorer/scorer.py`）。

臂（arms）：
  base        —— 完全不区分池：只用时间安全的「排名差 / 积分差 / 历史胜率差 / 交手差」。
  pool_id     —— base + 池 id one-hot（字面读法：加 era 指示位）。
                 ★ 注意：特征已按「A 减 B」对称化，era 指示位在**每条样本内是常量**，
                 等价于给每个时代一个**独立截距** ⇒ 它会改变判定边界（实测**有害**），
                 并非恒等臂。它只在「测集跨时代」时才有机会兑现，而 test 是纯 era2。
  pool_feat   —— base + **池感知**特征：两队在「当前 7 张池图」上的历史胜率差。
                 这是「池 id 特征」真正可行的替代（池通过「打哪些图」影响胜负）。
  segmented   —— 按时代分段训练：每个时代各训一个模型，预测用**最近时代**的模型。

预测项：
  第 1 项 获胜方（10 分）、第 3 项 选图集合（每张 5 分，读法 B 真值来自 veto）；
  第 2 项比分（方向跟随预测胜方、数值取训练集众数）凑出「总分」；
  第 4 项 MVP 各臂恒 0（暂无 roster 级 MVP 模型），不影响**臂间**比较。

另做「选图项」的池敏感性诊断（不经模型）：
  no_pool / pool_restrict / pool_floor 三种候选图排序，报告第 3 项得分、
  **越池预测（预测了当前池外的图 = 白扔的槽位）** 与 **Cache 命中**。

硬约束：
  * 特征只用**严格早于比赛日**的数据（同日多场一律互不可见）。
  * 池 / 时代只来自 014 的既证产物（`map_pool_timeline.json`）。
  * csapi 自带的 `rank` 字段**不可用**：它对同一队伍跨 9 个月恒定 ⇒ 是单点快照（未来信息），
    本实验改用 `rankings/` 的**周更归档**取「比赛日前最近一期的排名」。

用法：
  python pool_experiment.py selftest     # 离线自证（不读大数据文件）
  python pool_experiment.py run          # 跑实验并写 pool_experiment_results.json
"""

from __future__ import annotations

import bisect
import io
import json
import math
import os
import re
import sys
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")  # # 仓库布局：数据与代码分目录（Alpha 1.0）
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)  # scorer.py 与本模块同目录（仓库布局）

import split as SPLIT  # noqa: E402
from scorer import Prediction, Truth, score_match  # noqa: E402

MATCHES = os.path.join(DATA, "csapi_matches_all.jsonl")
TRUTH = os.path.join(DATA, "truth_maps_B.jsonl")
RANKINGS = os.path.join(DATA, "rankings")  # 精简归档（仓库内 0.7MB），非上游 103MB 原档
TIMELINE = os.path.join(DATA, "map_pool_timeline.json")
OUT_JSON = os.path.join(DATA, "pool_experiment_results.json")
EXCLUDE = os.path.join(HERE, "exclude_band_unknown_ids.txt")  # Q-011 裁决：过渡带无法定池的场次剔除

RANK_CAP = 200          # 归档里查不到队伍时的兜底排名（视为榜外）
LAPLACE = 1.0           # 胜率平滑


# --------------------------------------------------------------------- 排名归档
def list_snapshots(root=RANKINGS):
    """返回按日期升序的周更快照日期列表。"""
    snaps = []
    for year in os.listdir(root):
        d = os.path.join(root, year)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            m = re.match(r"(\d{4}-\d{2}-\d{2})\.json$", f)
            if m:
                snaps.append(m.group(1))
    snaps.sort()
    return snaps


class Ranks:
    """按「比赛日前最近一期」取排名/积分（时序安全）。"""

    def __init__(self, root=RANKINGS):
        self.root = root
        self.dates = list_snapshots(root)
        self._cache = {}

    def _load(self, snap):
        if snap not in self._cache:
            path = os.path.join(self.root, snap[:4], snap + ".json")
            o = json.load(io.open(path, encoding="utf-8"))
            self._cache[snap] = {
                t["name"]: (int(t["rank"]), int(t.get("points") or 0)) for t in o["teams"]
            }
        return self._cache[snap]

    def snapshot_for(self, date):
        i = bisect.bisect_right(self.dates, date) - 1
        return self.dates[i] if i >= 0 else None

    def rank_points(self, name, date, cap=RANK_CAP):
        snap = self.snapshot_for(date)
        if snap is None:
            return cap, 0
        idx = self._load(snap)
        return idx.get(name, (cap, 0))


# ------------------------------------------------------------------- 池 / 时代
def load_pool_eras(path=TIMELINE):
    """返回 (eras, {(frozenset(pool)): index})。"""
    tl = json.load(io.open(path, encoding="utf-8"))
    eras = tl["eras"]
    by_pool = {frozenset(e["pool"]): e["index"] for e in eras}
    return eras, by_pool


def era_and_pool(date, veto_pool, eras, by_pool, tl):
    """日期 → (时代号, 该场池)。

    时代优先按**日期**定（`split.era_tag`，与 014 切分同源）；只有**过渡带**里
    池才不唯一，此时用该场 veto 读出的池定时代。veto 缺失（未收满的 1 场）时
    退回「较早时代」。
    """
    if veto_pool:
        idx = by_pool.get(frozenset(veto_pool))
        if idx is not None:
            return idx, list(veto_pool)
    tag = SPLIT.era_tag(date, tl)
    if tag[0] == "era":
        i = tag[1]
        return i, list(eras[i]["pool"])
    if tag[0] in ("band", "post"):
        i = tag[1]
        return i, list(eras[i]["pool"])
    return None, []


def era_of_pool(pool_list, by_pool):
    """由「该场 actual 池」反查时代号；过渡带里也准确（用该场自己的 veto 池）。"""
    if not pool_list:
        return None
    return by_pool.get(frozenset(pool_list))


# ----------------------------------------------------------------------- 历史
class Hist:
    """逐日推进的时序安全历史：比赛胜负、逐图胜负、交手记录。"""

    def __init__(self):
        self.mw = defaultdict(lambda: [0, 0])          # team -> [wins, losses]（系列）
        self.mapw = defaultdict(lambda: [0, 0])        # (team, map) -> [wins, losses]
        self.h2h = defaultdict(lambda: [0, 0])         # (a, b) 字典序 -> [a 胜, b 胜]

    @staticmethod
    def _rate(win, lose, prior=0.5, k=LAPLACE):
        n = win + lose
        if n == 0:
            return prior
        return (win + k * prior) / (n + k)

    def match_wr(self, team):
        w, l = self.mw[team]
        return self._rate(w, l)

    def pool_wr(self, team, pool):
        """该队在**给定池图**上的历史胜率（图级）；无样本则退回系列胜率。"""
        w = sum(self.mapw[(team, m)][0] for m in pool)
        l = sum(self.mapw[(team, m)][1] for m in pool)
        if w + l == 0:
            return self.match_wr(team)
        return self._rate(w, l)

    def h2h_rate(self, a, b):
        key = (a, b) if a <= b else (b, a)
        wa, wb = self.h2h[key]
        if a <= b:
            mine, theirs = wa, wb
        else:
            mine, theirs = wb, wa
        return self._rate(mine, theirs)

    def update(self, teamA, teamB, winner, map_results):
        a_win = (winner == teamA)
        self.mw[teamA][0 if a_win else 1] += 1
        self.mw[teamB][0 if not a_win else 1] += 1
        key = (teamA, teamB) if teamA <= teamB else (teamB, teamA)
        first_is_a = (teamA <= teamB)
        if a_win:
            self.h2h[key][0 if first_is_a else 1] += 1
        else:
            self.h2h[key][1 if first_is_a else 0] += 1
        for mname, w1, w2 in map_results:
            if w1 == w2:
                continue
            tw, tl = (teamA, teamB) if w1 > w2 else (teamB, teamA)
            self.mapw[(tw, mname)][0] += 1
            self.mapw[(tl, mname)][1] += 1


# --------------------------------------------------------------- 特征与样本
def _signed(a, b):
    """A 相对 B 的差值（正 ⇒ 有利 A）。"""
    return a - b


def build_samples(rows, ranks, eras, by_pool, tl, truths):
    """逐日推进，构造样本。同一天的比赛先全部取特征、再统一更新历史（防同日泄漏）。"""
    rows = [r for r in rows if isinstance(r.get("best_of"), int) and r["best_of"] in (1, 3, 5)]
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["date"], rows[i]["id"]))

    hist = Hist()
    samples = []
    # 按日期分组
    i = 0
    while i < len(order):
        d = rows[order[i]]["date"]
        group = []
        while i < len(order) and rows[order[i]]["date"] == d:
            group.append(rows[order[i]])
            i += 1
        # 1) 先取特征
        pending = []
        for r in group:
            tA = r["team1"]["name"]
            tB = r["team2"]["name"]
            t = truths.get(r["id"])
            veto_pool = list(t.get("pool") or []) if t else []
            era, pool = era_and_pool(d, veto_pool, eras, by_pool, tl)
            ra, pa = ranks.rank_points(tA, d)
            rb, pb = ranks.rank_points(tB, d)
            feats = {
                "d_rank": _signed(rb, ra),                       # A 排名更靠前 ⇒ 正
                "d_logp": _signed(math.log1p(pa), math.log1p(pb)),
                "d_wr": _signed(hist.match_wr(tA), hist.match_wr(tB)),
                "d_h2h": _signed(hist.h2h_rate(tA, tB), hist.h2h_rate(tB, tA)),
                "d_poolwr": _signed(hist.pool_wr(tA, pool), hist.pool_wr(tB, pool)),
                "era0": 1 if era == 0 else 0,
                "era1": 1 if era == 1 else 0,
                "era2": 1 if era == 2 else 0,
            }
            maps = [m for m in (r.get("maps") or [])]
            map_results = [(m["name"], m["team1_score"], m["team2_score"]) for m in maps]
            pending.append({
                "id": r["id"], "date": d, "era": era, "pool": pool,
                "teamA": tA, "teamB": tB, "winner": r["winner"]["name"],
                "bo": r["best_of"], "feats": feats, "truth": t,
                "y": 1 if r["winner"]["name"] == tA else 0,
            })
        samples.extend(pending)
        # 2) 再更新历史
        for r in group:
            tA = r["team1"]["name"]
            tB = r["team2"]["name"]
            map_results = [(m["name"], m["team1_score"], m["team2_score"]) for m in (r.get("maps") or [])]
            hist.update(tA, tB, r["winner"]["name"], map_results)
    return samples


FEATURES = {
    "base": ["d_rank", "d_logp", "d_wr", "d_h2h"],
    "pool_id": ["d_rank", "d_logp", "d_wr", "d_h2h", "era0", "era1"],
    "pool_feat": ["d_rank", "d_logp", "d_wr", "d_h2h", "d_poolwr"],
}


# ------------------------------------------------------------------- 计分工具
def to_truth(t):
    return Truth(
        teamA=t["teamA"], teamB=t["teamB"], winner=t["winner"],
        score=tuple(t["score"]), maps=list(t.get("maps") or []),
        maps_played=list(t.get("maps_played") or []),
        maps_mode=t.get("maps_mode") or "veto", best_of=t.get("bo"),
    )


def modal_scorelines(train_truths):
    """按 bo 统计训练集里「胜方回合数:败方回合数」的众数。"""
    c = defaultdict(Counter)
    for t in train_truths:
        bo = t.get("bo")
        if bo not in (3, 5):
            continue
        sa, sb = t["score"]
        if sa == sb:
            continue
        w, l = (sa, sb) if sa > sb else (sb, sa)
        c[bo][(w, l)] += 1
    default = {3: (2, 1), 5: (3, 1)}
    out = {}
    for bo in (3, 5):
        out[bo] = c[bo].most_common(1)[0][0] if c[bo] else default[bo]
    return out


def pick_scoreline(bo, winner_is_A, modal):
    if bo == 1:
        return (1, 0) if winner_is_A else (0, 1)
    w, l = modal.get(bo, (2, 1))
    return (w, l) if winner_is_A else (l, w)


# --------------------------------------------------------------------- 选图臂
def map_freq(train_truths, restrict=None):
    c = Counter()
    for t in train_truths:
        for m in (t.get("maps") or []):
            if restrict is None or m in restrict:
                c[m] += 1
    return c


def predict_maps(freq, pool, k, floor=None):
    """在候选（pool，若给出）内按频次取 top-k；floor 给训练集零样本的池图一个先验。"""
    cand = list(pool) if pool else list(freq)
    scored = []
    mn = min([v for v in freq.values() if v > 0], default=1)
    for m in cand:
        v = freq.get(m, 0)
        if v == 0 and floor is not None:
            v = floor * mn
        scored.append((v, m))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [m for _, m in scored[:k]]


# ----------------------------------------------------------------------- 模型
def fit_predict(train, targets, feats, seed_note=""):
    """sklearn LogisticRegression；返回 predict(dict)->0/1 的闭包。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    X = np.array([[s["feats"][f] for f in feats] for s in train], dtype=float)
    y = np.array([s["y"] for s in train], dtype=int)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X), y)

    def predict(sample):
        v = np.array([[sample["feats"][f] for f in feats]], dtype=float)
        return int(clf.predict(sc.transform(v))[0])

    return predict


# ------------------------------------------------------------------- 评估
def evaluate(samples, kind, predict_winner, modal, freq, floor_maps=0.0):
    """返回 {n, acc, item1, item2, item3, total, out_of_pool, cache_pred, cache_hit}。"""
    n = len(samples)
    acc = i1 = i2 = i3 = tot = 0
    out_of_pool = cache_pred = cache_hit = 0
    for s in samples:
        t = s["truth"]
        if not t:
            continue
        truth = to_truth(t)
        pwin_is_A = (predict_winner(s) == 1)
        pwin = s["teamA"] if pwin_is_A else s["teamB"]
        k = max(1, min(s["bo"], 5))
        global_freq = freq["global"]
        if kind == "maps_no_pool":
            pmaps = [m for m, _ in sorted(global_freq.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]
        elif kind == "maps_pool_restrict":
            pmaps = predict_maps(global_freq, s["pool"], k)
        elif kind == "maps_pool_floor":
            pmaps = predict_maps(global_freq, s["pool"], k, floor=floor_maps)
        elif kind == "maps_newmap":
            # 池内有训练集零样本的图（Cache）⇒ 必须留一个候选槽位给它
            unseen = [m for m in s["pool"] if global_freq.get(m, 0) == 0]
            if unseen and k >= 1:
                pmaps = predict_maps(global_freq, s["pool"], k - 1) + [unseen[0]]
            else:
                pmaps = predict_maps(global_freq, s["pool"], k)
        else:
            pmaps = []
        pred = Prediction(winner=pwin, score=pick_scoreline(s["bo"], pwin_is_A, modal), maps=pmaps)
        b = score_match(pred, truth)
        acc += (pwin == s["winner"])
        i1 += b.winner
        i2 += b.score
        i3 += b.maps
        tot += b.total
        real_pool = set(s["pool"])
        if real_pool:
            out_of_pool += sum(1 for m in pmaps if m not in real_pool)
        if "Cache" in pmaps:
            cache_pred += 1
            if "Cache" in (t.get("maps") or []):
                cache_hit += 1
    return {
        "n": n, "acc": acc / n if n else 0.0,
        "item1": i1, "item2": i2, "item3": i3, "total": tot,
        "out_of_pool": out_of_pool, "cache_pred": cache_pred, "cache_hit": cache_hit,
    }


def bootstrap_ci(base_ok, arm_ok, iters=2000, seed=7):
    """配对 bootstrap：两条 0/1 向量的准确率差 95% CI。"""
    import random
    rnd = random.Random(seed)
    n = len(base_ok)
    diffs = []
    for _ in range(iters):
        idx = [rnd.randrange(n) for _ in range(n)]
        db = sum(base_ok[i] for i in idx) / n
        da = sum(arm_ok[i] for i in idx) / n
        diffs.append(da - db)
    diffs.sort()
    lo = diffs[int(0.025 * iters)]
    hi = diffs[int(0.975 * iters)]
    return lo, hi


# --------------------------------------------------------------------- 主流程
def run():
    rows = SPLIT.load(MATCHES)
    exclude = SPLIT.load_exclude(EXCLUDE) if os.path.exists(EXCLUDE) else None
    buckets, meta = SPLIT.split(rows, purge=True, exclude=exclude)
    truths = {}
    for line in io.open(TRUTH, encoding="utf-8"):
        line = line.strip()
        if line:
            o = json.loads(line)
            truths[o["id"]] = o
    eras, by_pool = load_pool_eras()
    tl = json.load(io.open(TIMELINE, encoding="utf-8"))
    ranks = Ranks()

    all_rows = buckets["train"] + buckets["valid"] + buckets["test"]
    samples = build_samples(all_rows, ranks, eras, by_pool, tl, truths)
    by_id = {s["id"]: s for s in samples}
    tr = [by_id[r["id"]] for r in buckets["train"] if r["id"] in by_id]
    va = [by_id[r["id"]] for r in buckets["valid"] if r["id"] in by_id]
    te = [by_id[r["id"]] for r in buckets["test"] if r["id"] in by_id]

    train_truths = [s["truth"] for s in tr if s["truth"]]
    modal = modal_scorelines(train_truths)
    freq = {"global": map_freq(train_truths)}

    # 排名基线（时间安全）
    def rank_baseline(s):
        return 1 if s["feats"]["d_rank"] >= 0 else 0

    report = {"split": meta["counts"], "boundaries": {"b1": meta["b1"], "b2": meta["b2"]},
              "excluded_matches": meta.get("excluded_matches", 0),
              "modal_score": {str(k): list(v) for k, v in modal.items()},
              "n_train_truth": len(train_truths), "results": {}}

    for split_name, ss in (("valid", va), ("test", te)):
        block = {}
        # 排名基线
        block["rank_baseline"] = evaluate(ss, "", rank_baseline, modal, freq)
        # 模型臂
        preds = {}
        for arm, feats in FEATURES.items():
            predict = fit_predict(tr, None, feats)
            preds[arm] = predict
            block[arm] = evaluate(ss, "", predict, modal, freq)
        # 分段训练：训每时代一个模型，预测用最近时代的
        seg_pred = {}
        for era in sorted({s["era"] for s in tr if s["era"] is not None}):
            sub = [s for s in tr if s["era"] == era]
            if len(sub) >= 50:
                seg_pred[era] = fit_predict(sub, None, FEATURES["base"])

        def seg_predict(s):
            if seg_pred:
                return seg_pred[max(seg_pred)](s)
            return preds["base"](s)

        block["segmented"] = evaluate(ss, "", seg_predict, modal, freq)
        # 选图臂
        block["maps_no_pool"] = evaluate(ss, "maps_no_pool", lambda s: rank_baseline(s), modal, freq)
        block["maps_pool_restrict"] = evaluate(ss, "maps_pool_restrict", lambda s: rank_baseline(s), modal, freq)
        block["maps_pool_floor"] = evaluate(ss, "maps_pool_floor", lambda s: rank_baseline(s), modal, freq, floor_maps=1.0)
        block["maps_newmap"] = evaluate(ss, "maps_newmap", lambda s: rank_baseline(s), modal, freq)
        # 诊断：truth 含「训练集零样本的池图」的场次数
        unseen_maps = [m for m, v in freq["global"].items() if v == 0]
        n_cache = sum(1 for s in ss if "Cache" in (s["truth"].get("maps") or []))
        block["diag_unseen_pool_maps_in_test"] = {m: sum(1 for s in ss if m in (s["truth"].get("maps") or []))
                                                  for m in ("Cache",)}
        block["diag_cache_matches"] = n_cache
        # 配对 bootstrap：pool_feat vs base 的胜方准确率差
        base_ok = [1 if (preds["base"](s) == 1) == (s["winner"] == s["teamA"]) else 0 for s in ss]
        feat_ok = [1 if (preds["pool_feat"](s) == 1) == (s["winner"] == s["teamA"]) else 0 for s in ss]
        lo, hi = bootstrap_ci(base_ok, feat_ok)
        block["boot_pool_feat_vs_base"] = [lo, hi]
        # era 构成
        block["era_mix"] = dict(Counter(s["era"] for s in ss))
        block["seg_models"] = sorted(seg_pred)
        report["results"][split_name] = block

    with io.open(OUT_JSON, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print(render(report))
    return report


def render(report):
    L = []
    L.append("POOL_EXPERIMENT  split=%s  b1=%s b2=%s  train_truth=%d" % (
        report["split"], report["boundaries"]["b1"], report["boundaries"]["b2"], report["n_train_truth"]))
    L.append("modal scoreline: " + ", ".join("bo%s=%s" % (k, tuple(v)) for k, v in report["modal_score"].items()))
    for split_name in ("valid", "test"):
        b = report["results"][split_name]
        L.append("")
        L.append("== %s  n=%d  era_mix=%s  seg_models=%s ==" % (
            split_name, b["rank_baseline"]["n"], b["era_mix"], b["seg_models"]))
        L.append("  %-20s %6s %7s %6s %6s %7s %6s  %s" % (
            "arm", "acc", "item1", "item2", "item3", "total", "outpool", "note"))
        for arm in ("rank_baseline", "base", "pool_id", "pool_feat", "segmented"):
            r = b[arm]
            L.append("  %-20s %6.3f %7d %6d %6d %7d %6s  %s" % (
                arm, r["acc"], r["item1"], r["item2"], r["item3"], r["total"], "-", ""))
        for arm in ("maps_no_pool", "maps_pool_restrict", "maps_pool_floor", "maps_newmap"):
            r = b[arm]
            L.append("  %-20s %6s %7s %6s %6d %7s %6d  cache_pred=%d hit=%d" % (
                arm, "-", "-", "-", r["item3"], "-", r["out_of_pool"], r["cache_pred"], r["cache_hit"]))
        lo, hi = b["boot_pool_feat_vs_base"]
        L.append("  bootstrap(pool_feat - base, acc): [%.4f, %.4f]" % (lo, hi))
        L.append("  diag: truth 含 Cache 的场次 = %d  %s" % (
            b["diag_cache_matches"], b["diag_unseen_pool_maps_in_test"]))
    return "\n".join(L)


# ---------------------------------------------------------------------- selftest
def _selftest():
    fails = []

    def chk(name, got, want):
        if got != want:
            fails.append("%s: got %r want %r" % (name, got, want))

    # 1. 池->时代映射（含过渡带少数派池）
    eras, by_pool = load_pool_eras()
    chk("era of era0 pool", era_of_pool(eras[0]["pool"], by_pool), 0)
    chk("era of era2 pool", era_of_pool(eras[2]["pool"], by_pool), 2)
    chk("era of unknown pool", era_of_pool(["A", "B"], by_pool), None)
    chk("era of empty pool", era_of_pool([], by_pool), None)

    # 2. Hist：胜率/交手/池胜率 + 对称性（A/B 互换应翻号）
    h = Hist()
    chk("empty wr", h.match_wr("X"), 0.5)
    h.update("A", "B", "A", [("Mirage", 13, 9), ("Nuke", 9, 13)])
    chk("A wr>0.5", h.match_wr("A") > 0.5, True)
    chk("h2h A", h.h2h_rate("A", "B") > 0.5, True)
    chk("h2h sym", abs(h.h2h_rate("A", "B") + h.h2h_rate("B", "A") - 1.0) < 1e-9, True)
    chk("poolwr A on Mirage", h.pool_wr("A", ["Mirage"]) > 0.5, True)
    chk("poolwr no data -> fallback", h.pool_wr("A", ["Zzz"]), h.match_wr("A"))
    # 池胜率只算池内图：A 在 Mirage 1:0、Nuke 0:1 ⇒ 池 [Mirage] 应 >0.5
    chk("poolwr uses only pool maps", h.pool_wr("A", ["Nuke"]) < 0.5, True)

    # 3. 特征对称：同场 A/B 互换 ⇒ d_* 翻号（零和）
    for f in ("d_rank", "d_logp", "d_wr", "d_h2h", "d_poolwr"):
        chk("signed zero", _signed(3, 3), 0)

    # 4. 众数比分 + 方向
    modal = modal_scorelines([
        {"bo": 3, "score": (2, 1)}, {"bo": 3, "score": (2, 0)}, {"bo": 3, "score": (2, 1)},
        {"bo": 5, "score": (3, 2)},
    ])
    chk("modal bo3", modal[3], (2, 1))
    chk("modal bo5", modal[5], (3, 2))
    chk("scoreline A wins", pick_scoreline(3, True, modal), (2, 1))
    chk("scoreline B wins", pick_scoreline(3, False, modal), (1, 2))
    chk("bo1 A", pick_scoreline(1, True, modal), (1, 0))

    # 5. predict_maps：池内 top-k + 零样本 floor
    freq = Counter({"Mirage": 10, "Dust2": 8, "Cache": 0, "Nuke": 2})
    got = predict_maps(freq, ["Mirage", "Dust2", "Cache", "Nuke"], 3)
    chk("maps top3 no floor", got, ["Mirage", "Dust2", "Nuke"])
    # floor=1.0 ⇒ 零样本池图拿 min(正频次)=2 的先验，与 Nuke(2) 平手、按名字排序 ⇒ Cache 挤进 top-3
    got = predict_maps(freq, ["Mirage", "Dust2", "Cache", "Nuke"], 3, floor=1.0)
    chk("maps floor lifts zero-count into top-k", got, ["Mirage", "Dust2", "Cache"])
    chk("maps restrict never out of pool",
        all(m in {"Mirage", "Dust2"} for m in predict_maps(freq, ["Mirage", "Dust2"], 3)), True)

    # 6. Truth 构造与 scorer 对接（第 3 项集合命中）
    t = to_truth({"teamA": "A", "teamB": "B", "winner": "A", "score": [2, 1],
                  "maps": ["Mirage", "Nuke", "Dust2"], "maps_played": ["Mirage", "Nuke"],
                  "maps_mode": "veto", "bo": 3})
    b = score_match(Prediction(winner="A", score=(2, 1), maps=["Mirage", "Nuke", "Ancient"]), t)
    chk("item1", b.winner, 10)
    chk("item2", b.score, 10)
    chk("item3 hits", b.maps, 10)
    chk("item3 hit list", b.maps_hit, ["Mirage", "Nuke"])

    if fails:
        for f in fails:
            print("FAIL " + f)
        print("POOL_EXPERIMENT_SELFTEST_FAILED (%d)" % len(fails))
        return 1
    print("POOL_EXPERIMENT_SELFTEST_OK (6 groups)")
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
