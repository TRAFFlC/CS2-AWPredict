"""C：MVP 平局裁决器（不替换 rating 主排序，只在 top-k 之间打破平局）。

背景（实测推翻 v3 的说法）：
  · 真实 MVP 落在**胜方候选集**里 = **97.73%**（train 97.71 / valid 97.92 / test 97.73）
    ⇒ v3 笔记里「候选集天花板 30~35%」是把「候选集上界」与「当前规则水平」混为一谈。
  · 真正的瓶颈是 margin：胜方候选第一 − 第二的 rating 差**中位仅 0.038**，
    86.4% 的场次 ≤0.10，95.1% ≤0.20 ⇒ MVP 基本是「队里两三人里谁这场恰好好一点」。
  · 当前规则（rating + 担当率）条件命中 32.9%；若前二近似等权，随机上限约 50%
    ⇒ **地址空间约 +17pp**。

主理人提的三个信号（Q-019），全部**只在候选之间**使用：
  ① 近期 MVP 频率     「刚获得 mvp，是否意味着状态比较好」
  ② 逐图 rating       「有人不擅长打某些图」——注意：v3 测的是**整体替换**判据（掉 43%），
                     本次是**平局裁决**，用法完全不同
  ③ 近期评分趋势      近 N 场的 rating 斜率
判据：**train 扩展窗口 CV 选、valid 验、1SE 容差、test 不报**。
"""
import io, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_v2 as M
import veto_model as VM
import split as SPLIT
import pool_experiment as PE

TOPK = 3          # 只在前 k 名里裁决
LOOKBACK = 120    # 近期 MVP 频率 / 趋势的回看天数


def build_trend_and_mvp_freq():
    """时序安全地建两本新账：
       mvp_freq[player]  : 最近 LOOKBACK 天内该选手当 MVP 的次数（指数衰减）
       trend[player]    : 最近 LOOKBACK 天该选手 rating 的加权线性斜率
    只在重放语料时顺带累积，不用未来信息。
    """
    rows = SPLIT.load(M.MATCHES)
    exc = SPLIT.load_exclude(M.EXCLUDE) if os.path.exists(M.EXCLUDE) else None
    buckets, _ = SPLIT.split(rows, purge=True, exclude=exc)
    truths = {t["id"]: t for t in M.load_jsonl(M.TRUTH)}
    veto = VM.load_veto()
    stats = M.load_stats()
    ms = M.load_stats_map()
    eras, by_pool = PE.load_pool_eras()
    tl = json.load(io.open(PE.TIMELINE, encoding="utf-8"))

    def pool_of(r):
        t = truths.get(r["id"]) or {}
        return list(t["pool"]) if t.get("pool") else \
            list(PE.era_and_pool(r["date"], [], eras, by_pool, tl)[1])

    split_of = {}
    for b in ("train", "valid", "test"):
        for r in buckets[b]:
            split_of[r["id"]] = b
    usable = [r for r in rows
              if isinstance(r.get("best_of"), int) and r["best_of"] in (1, 3, 5)
              and r["id"] in split_of]
    usable.sort(key=lambda r: (r["date"], r["id"]))

    s, h = M.State(), VM.MapHistory()
    mvp_freq = {}
    hist = {}          # player -> [(date, rating)]
    per_split = {"train": [], "valid": [], "test": []}

    def decay(d0, d1):
        g = VM.days_apart(d1, d0)
        return 0.5 ** (g / LOOKBACK) if g > 0 else 1.0

    def as_pairs(x):
        """`load_stats()` 给的是 {team: [(name, rating), ...]}（元组列表，不是字典）。"""
        if isinstance(x, dict):
            return [(k, v) for k, v in x.items()]
        return [(p, rt) for p, rt in (x or [])]

    def rec(d, stats_row, winner):
        for _team, pl in (stats_row or {}).items():
            for p, rt in as_pairs(pl):
                if rt is None:
                    continue
                hist.setdefault(p, []).append((d, float(rt)))
                hist[p] = hist[p][-40:]
        w = as_pairs((stats_row or {}).get(winner))
        rated = [(p, float(rt)) for p, rt in w if rt is not None]
        if rated:
            best = max(rated, key=lambda x: x[1])
            mvp_freq[best[0]] = mvp_freq.get(best[0], 0.0) + 1.0
        f = 0.5 ** (1.0 / LOOKBACK)
        for k in list(mvp_freq):
            mvp_freq[k] *= f
            if mvp_freq[k] < 1e-3:
                del mvp_freq[k]

    i = 0
    while i < len(usable):
        d = usable[i]["date"]
        s.advance(d)
        h.advance(d)
        grp = []
        while i < len(usable) and usable[i]["date"] == d:
            grp.append(usable[i])
            i += 1
        for r in grp:
            t = truths.get(r["id"]) or {}
            bo = t.get("bo") or r.get("best_of") or 3
            tA, tB = r["team1"]["name"], r["team2"]["name"]
            w = r["winner"]["name"]
            cand = s.lineup(w, d)
            if not cand:
                continue
            opp = tB if w == tA else tA
            opp_elo = s.elo.get(opp, s.elo0)   # player_row 要的是标量
            rows_ = []
            for p in cand:
                row = s.player_row(w, p, d, opp_elo, cand)
                hist_p = hist.get(p, [])
                trend = 0.0
                if len(hist_p) >= 3:
                    xs = np.array([VM.days_apart(d, dd) for dd, _ in hist_p[-8:]], float)
                    ys = np.array([rr for _, rr in hist_p[-8:]], float)
                    if xs.std() > 0:
                        trend = float(np.polyfit(xs, ys, 1)[0])
                rows_.append({
                    "p": p,
                    "adj": row.get("adj", 0.0), "raw": row.get("raw", 0.0),
                    "toprate": row.get("toprate", 0.0), "exp": row.get("exp", 0.0),
                    "days": row.get("days", 999.0),
                    "mvp_f": mvp_freq.get(p, 0.0), "trend": trend,
                })
            mvp_truth = None
            wt = as_pairs((stats.get(r["id"]) or {}).get(w))
            rated = [(p, float(rt)) for p, rt in wt if rt is not None]
            if rated:
                mvp_truth = max(rated, key=lambda x: x[1])[0]
            per_split[split_of[r["id"]]].append({"cand": rows_, "mvp": mvp_truth,
                                                 "bo": bo, "winner": w})
        for r in grp:
            tA, tB = r["team1"]["name"], r["team2"]["name"]
            mres = [(m["name"], m["team1_score"], m["team2_score"])
                    for m in (r.get("maps") or [])]
            t = truths.get(r["id"]) or {}
            rec(d, stats.get(r["id"]) or {}, r["winner"]["name"])
            s.observe(d, tA, tB, r["winner"]["name"], mres,
                      stats.get(r["id"], {}), ms.get(r["id"]))
            h.observe_match(tA, tB, pool_of(r), t.get("maps") or [],
                             veto.get(r["id"]), mres)
    return per_split


def rule_pick(cand, mode, w):
    if not cand:
        return None
    def base(c):
        return c["adj"]
    if mode == "rating":
        return max(cand, key=base)["p"]
    top = sorted(cand, key=lambda c: -base(c))[:TOPK]
    if mode == "rating+mvp_f":
        mx = max(abs(c["mvp_f"]) for c in top) or 1.0
        return max(top, key=lambda c: base(c) + w * c["mvp_f"] / mx)["p"]
    if mode == "rating+trend":
        mx = max(abs(c["trend"]) for c in top) or 1.0
        return max(top, key=lambda c: base(c) + w * c["trend"] / mx)["p"]
    if mode == "rating+mvp_f+trend":
        mx1 = max(abs(c["mvp_f"]) for c in top) or 1.0
        mx2 = max(abs(c["trend"]) for c in top) or 1.0
        return max(top, key=lambda c: base(c) + w * (c["mvp_f"] / mx1 + c["trend"] / mx2))["p"]
    raise ValueError(mode)


def score(rows, mode, w=0.0):
    ok = n = 0
    for s in rows:
        if not s["mvp"] or len(s["cand"]) < 2:
            continue
        n += 1
        ok += int(rule_pick(s["cand"], mode, w) == s["mvp"])
    return ok, n


def main():
    print("== 逐日重放，建「近期 MVP 频率」与「rating 趋势」两本新账 ==")
    PS = build_trend_and_mvp_freq()
    tr, va, te = PS["train"], PS["valid"], PS["test"]
    print("  可用场次：train=%d valid=%d test=%d" % (len(tr), len(va), len(te)))

    base_ok, base_n = score(tr, "rating")
    print("\n== 基线：纯 rating 主排序（现状规则的核心）==")
    print("  train 命中 %d/%d = %.2f%%" % (base_ok, base_n, 100 * base_ok / base_n))
    v_ok, v_n = score(va, "rating")
    print("  valid 命中 %d/%d = %.2f%%" % (v_ok, v_n, 100 * v_ok / v_n))
    print("  ⚠ 这是**候选集内的条件命中**（已天然条件于「胜方猜对」），"
          "与 test 上的 32.9% 同口径但样本集不同")

    print("\n== 加平局裁决器（权重按 train 选、valid 验）==")
    print("%-24s %-8s %14s %14s" % ("规则", "w", "train", "valid"))
    se_tr = math.sqrt(max(base_ok / base_n * (1 - base_ok / base_n), 1e-9) / base_n)
    best = None
    for mode in ("rating+mvp_f", "rating+trend", "rating+mvp_f+trend"):
        for w in (0.05, 0.1, 0.2, 0.4, 0.8):
            ok, n = score(tr, mode, w)
            v2, n2 = score(va, mode, w)
            print("%-24s %-8.2f %13.2f%% %13.2f%%"
                  % (mode, w, 100 * ok / n, 100 * v2 / n2))
            if best is None or ok > best[0]:
                best = (ok, n, mode, w, v2, n2)
    ok, n, mode, w, v2, n2 = best
    se_va = math.sqrt(max(v_ok / v_n * (1 - v_ok / v_n), 1e-9) / v_n)
    print("\n按 train 选出：%s w=%.2f" % (mode, w))
    print("  train %.2f%%（基线 %.2f%%，%+.2fpp，%+.1f SE_train）"
          % (100 * ok / n, 100 * base_ok / base_n, 100 * (ok - base_ok) / n,
             (ok - base_ok) / se_tr))
    print("  valid %.2f%%（基线 %.2f%%，%+.2fpp，%+.1f SE_valid=%.1f 场）"
          % (100 * v2 / n2, 100 * v_ok / v_n, 100 * (v2 - v_ok) / n2,
             (v2 - v_ok) / se_va, se_va * v_n))
    print("\n判读：%s" % ("valid 超 1SE ⇒ 按协议可报 test" if v2 > v_ok + se_va * v_n
                         else "valid 未超 1SE ⇒ test 不报"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
