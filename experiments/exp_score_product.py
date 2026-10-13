"""B：第 2 项（比分）的**乘积分解** —— P(2:0) = P(赢图1) × P(赢图2)

主理人 2026-10-12 的思路（Q-020，从未被测过）：
> 「当然也和选图有关……我几乎必然会预测它们是 2:1 或者 1:2，
>  这说明我认为它们大概率会拿下自己的选图。而非简单根据 elo 进行预测。」

与 v3 试过的 `gap_logit` 的区别（这是本实验的关键）：
  gap_logit 用**队伍级**特征（|d_elo|,|d_elom|,|d_poolwr|）→ OOF 方向准确率仅 56.1%
  本实验用**逐图**强度：p(A 赢图 m) 由两队在该图的胜率经 Bradley-Terry 合成，
  再对「最可能被 pick 的两张图」取**乘积** ⇒ 天然条件在第 3 项的图集预测上。

评估口径：条件于「胜方已对」，只测「通吃 vs 2:1」的判别准确率。
  · 现状基线 `always_win0`（永远押通吃）
  · 随机基线：语料通吃率
判据：**train 扩展窗口 OOF + valid**；test 不报（除非 valid 明确超 1SE）。
"""
import io, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_v2 as M
import veto_model as VM
import split as SPLIT
import pool_experiment as PE

TOPK = 2          # 2:0 时打的是两张 pick（读法B 的第 3 张是未打的决胜图）


def collect(state, mh, rows, truths, veto, eras, by_pool, tl, stats, map_stats):
    """逐日驱动，返回每场「条件于真实胜方」的比分判别特征。"""
    def pool_of(r):
        t = truths.get(r["id"]) or {}
        return list(t["pool"]) if t.get("pool") else \
            list(PE.era_and_pool(r["date"], [], eras, by_pool, tl)[1])

    usable = [r for r in rows
              if isinstance(r.get("best_of"), int) and r["best_of"] in (1, 3, 5)]
    usable.sort(key=lambda r: (r["date"], r["id"]))
    out = []
    i = 0
    while i < len(usable):
        d = usable[i]["date"]
        state.advance(d)
        mh.advance(d)
        group = []
        while i < len(usable) and usable[i]["date"] == d:
            group.append(usable[i])
            i += 1
        for r in group:
            tA, tB = r["team1"]["name"], r["team2"]["name"]
            t = truths.get(r["id"]) or {}
            bo = t.get("bo") or r.get("best_of") or 3
            if bo != 3:
                out.append(None)          # 本实验只做 bo3（2:0 / 2:1）
                continue
            A, B = r["winner"]["name"], (tB if r["winner"]["name"] == tA else tA)
            pool = pool_of(r)
            # 逐图胜率（State 已内建收缩），Bradley-Terry 合成
            pr = {}
            for m in pool:
                aw, bw = state.map_perf(A, m), state.map_perf(B, m)
                pr[m] = (aw, bw)
            # A 的 pick 偏好排序（MapHistory 的 team_pick 份额）
            u = len(pool)
            pref = sorted(pool, key=lambda m: (-mh.team_pick(A, m, u), m))
            top = pref[:TOPK]
            prs = [pr[m][0] / (pr[m][0] + pr[m][1]) if (pr[m][0] + pr[m][1]) > 0 else 0.5
                   for m in top]
            sc = (r["team1"]["score"], r["team2"]["score"])
            sweep = (max(sc) == 2 and min(sc) == 0)
            out.append({"p_prod": prs[0] * prs[1], "p_min": min(prs),
                        "p_mean": sum(prs) / len(prs),
                        "sweep": 1 if sweep else 0, "date": d})
        for r in group:
            tA, tB = r["team1"]["name"], r["team2"]["name"]
            mres = [(m["name"], m["team1_score"], m["team2_score"])
                    for m in (r.get("maps") or [])]
            t = truths.get(r["id"]) or {}
            state.observe(d, tA, tB, r["winner"]["name"], mres,
                          stats.get(r["id"], {}), map_stats.get(r["id"]))
            mh.observe_match(tA, tB, pool_of(r), t.get("maps") or [],
                             veto.get(r["id"]), mres)
    return [x for x in out if x is not None]


def main():
    rows = SPLIT.load(M.MATCHES)
    exc = SPLIT.load_exclude(M.EXCLUDE) if os.path.exists(M.EXCLUDE) else None
    buckets, _ = SPLIT.split(rows, purge=True, exclude=exc)
    truths = {t["id"]: t for t in M.load_jsonl(M.TRUTH)}
    veto = VM.load_veto()
    stats = M.load_stats()
    map_stats = M.load_stats_map()
    eras, by_pool = PE.load_pool_eras()
    tl = json.load(io.open(PE.TIMELINE, encoding="utf-8"))
    s, h = M.State(), VM.MapHistory()
    tr = collect(s, h, buckets["train"], truths, veto, eras, by_pool, tl, stats, map_stats)
    va = collect(s, h, buckets["valid"], truths, veto, eras, by_pool, tl, stats, map_stats)
    n_tr, n_va = len(tr), len(va)
    base_tr = sum(x["sweep"] for x in tr)
    base_va = sum(x["sweep"] for x in va)
    print("bo3 可用：train=%d valid=%d" % (n_tr, n_va))
    print("基线 always_win0（永远押通吃）命中率：train %.2f%%（%d/%d）  valid %.2f%%（%d/%d）"
          % (100 * base_tr / n_tr, base_tr, n_tr, 100 * base_va / n_va, base_va, n_va))

    # ---------- 1) 单变量方向检验 ----------
    print("\n== 1) 单变量：把 P(通吃) 当打分，看方向对不对 ==")
    for k in ("p_prod", "p_min", "p_mean"):
        # 以 train 的中位数定阈值，valid 上检验（不偷看 valid）
        thr = sorted(x[k] for x in tr)[len(tr) // 2]
        ok_va = sum(1 for x in va if (x[k] >= thr) == (x["sweep"] == 1))
        print("  %-8s 阈值(train 中位数)=%.4f  valid 方向命中 %.2f%%（%d/%d）"
              % (k, thr, 100 * ok_va / n_va, ok_va, n_va))
        # train 自身 OOF 风格的中位数阈值命中率
        ok_tr = sum(1 for x in tr if (x[k] >= thr) == (x["sweep"] == 1))
        print("           train 同阈值 %.2f%%" % (100 * ok_tr / n_tr))

    # ---------- 2) 阈值扫描（train 选、valid 验）----------
    print("\n== 2) P(2:0)=p_prod ≥ τ ⇒ 押通吃；否则押 2:1 ==")
    print("%8s %12s %12s %12s" % ("τ", "train 命中", "valid 命中", "valid vs 基线"))
    se_tr = math.sqrt(max(base_tr / n_tr * (1 - base_tr / n_tr), 1e-9) / n_tr)
    best = None
    for tau in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        ok_tr = sum(1 for x in tr if (x["p_prod"] >= tau) == (x["sweep"] == 1))
        ok_va = sum(1 for x in va if (x["p_prod"] >= tau) == (x["sweep"] == 1))
        d = ok_va - base_va
        print("%8.2f %11.2f%% %11.2f%% %+12d" % (tau, 100 * ok_tr / n_tr, 100 * ok_va / n_va, d))
        if best is None or ok_tr > best[1]:
            best = (tau, ok_tr, ok_va)
    tau, ok_tr, ok_va = best
    se_va = math.sqrt(max(base_va / n_va * (1 - base_va / n_va), 1e-9) / n_va)
    print("\n按 train 选出的 τ=%.2f：train %.2f%%（基线 %.2f%%，%+.2fpp）  valid %.2f%%（基线 %.2f%%，%+.2fpp，%.1f×1SE）"
          % (tau, 100 * ok_tr / n_tr, 100 * base_tr / n_tr, 100 * (ok_tr - base_tr) / n_tr,
             100 * ok_va / n_va, 100 * base_va / n_va, 100 * (ok_va - base_va) / n_va,
             (ok_va - base_va) / se_va))
    print("1SE(valid) = %.1f 场" % se_va)
    print("\n判读：%s" % ("**valid 上超过 1SE** ⇒ 按协议可报 test"
                         if ok_va > base_va + se_va else
                         "valid 未超 1SE ⇒ test 不报"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
