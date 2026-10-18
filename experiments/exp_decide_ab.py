"""exp_decide_ab.py —— 「最高分 vs 最高排名」的零风险对照实验

## 为什么有这个脚本（补一个从未被填上的空白）

`decide.py:787-793` 的实盘对照**只对 `plans_score` 计分**：

    realized = 0
    for md, pl in zip(mds, plans_score):      # ← 只有 score 版
        realized += _score_real(md, pl)

⇒ **rank 目标的计划从未被拿到真值上计过分**。所有 rank 侧数字
（E[分] 7,119 / E[rank] 1.59 / 37 场分歧）**全部是模型自模拟，不是回测**。

而主理人早在 2026-10-08 就承诺过要做这件事（`issues/QNA.md:194` 原话：
「最终会根据『最高分数』和『最高排名』两个版本做一次实际测试」）——
至今未做（`issues/QNA.md:322` 措辞是「暂以…为目标」「必要的时候会…测试」，
属**挂起**而非否决）。

本脚本就是那次测试。**零风险**：不联网、不碰生产路径、只读既有语料。

## 三个口径（互相独立，结论要三个都成立才算数）

  MODE_A 整桶：一个 valid/test 桶当成「一整场比赛」，与 decide.analyse() 同口径，
          便于和已记录的 E[分] 7,100→7,119、E[rank] 1.69→1.59 直接对照。
  MODE_B 按赛事 walk-forward：按 event 分组，一届赛事 = 一次决策单元。
          这才是真实用法（一场竞赛 = 全部对局），且每届只用该届的信息。
  CEILING 天花板：oracle = max over 全部候选计划 的真实得分。
          用来判断「两目标之争」在整条可争取空间里占多大比重。

## 用法
    python -X utf8 exp_decide_ab.py --bucket test
    python -X utf8 exp_decide_ab.py --bucket both --events 8
"""
import argparse
import collections
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "2026-10-07", "014-scorer"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "2026-10-07", "014-scorer"))

import decide as D
import model_v2 as M
import pool_experiment as PE


# ------------------------------------------------------------------ 工具
def build(bucket):
    """与 decide.analyse() **完全同一套建模路径**，逐字复刻，不另起炉灶。"""
    prep = M.prepare(verbose=False)
    data = prep["data"]
    wmodel, mmodel, wsel = prep["wmodel"], prep["mmodel"], prep["wsel"]
    for sp in data["train"]:
        sp["pA"] = M.predict_proba_win(wsel, sp)
    for k in ("valid", "test"):
        for sp in data[k]:
            sp["pA"] = M.predict_proba_win(wmodel, sp)
    smodel = D.ScoreModel.fit(data["train"])
    return [D.MatchDist(s, mmodel, smodel) for s in data[bucket]]


def dist_moments(d):
    m = sum(v * k for k, v in d.items())
    v = sum(v * (k - m) ** 2 for k, v in d.items())
    return m, v


_MDIST = {}


def pdist(md, pl):
    """带缓存的 points_dist（坐标下降会反复问同一批计划）。"""
    k = (id(md), pl["winner"], pl["score"], pl["maps"], pl["mvp"])
    if k not in _MDIST:
        _MDIST[k] = md.points_dist(pl)
    return _MDIST[k]


def key_of(pl):
    return (pl["winner"], pl["score"], tuple(pl["maps"]), pl["mvp"])


def descend(mds, field, rounds=3):
    """从 score 解出发做坐标下降（与 decide.analyse() 同一写法）。"""
    plans = [D.choose_score(md) for md in mds]
    mom = []
    for md, pl in zip(mds, plans):
        m, v = dist_moments(pdist(md, pl))
        mom.append((m, v))
    for _ in range(rounds):
        changed = 0
        for i, md in enumerate(mds):
            others = [mom[j] for j in range(len(mom)) if j != i]
            base_mu = sum(x for x, _ in others)
            base_var = sum(y for _, y in others)
            best = None
            for pl in md.plans:
                m, v = dist_moments(pdist(md, pl))
                er = field.expected_rank(base_mu + m, base_var + v)
                if best is None or er < best[0] - 1e-12:
                    best = (er, pl, (m, v))
            if key_of(best[1]) != key_of(plans[i]):
                plans[i] = best[1]
                mom[i] = best[2]
                changed += 1
        if not changed:
            break
    return plans


def realized_items(mds, plans):
    """把计划拿到真值上计分，返回 (总分, 分项 Counter)。**这是 decide.py 缺的那步。**"""
    from scorer import Prediction, score_match
    tot = 0
    items = collections.Counter()
    for md, pl in zip(mds, plans):
        t = md.sample.get("truth")
        if not t:
            continue
        truth = M.to_truth(t, md.sample.get("mvp_truth", ""))
        b = score_match(Prediction(pl["winner"], tuple(pl["score"]),
                                   list(pl["maps"]), pl["mvp"]), truth)
        tot += b.total
        items["winner"] += b.winner
        items["score"] += b.score
        items["maps"] += b.maps
        items["mvp"] += b.mvp
    return tot, items


def sim_moments(mds, plans):
    ms = [dist_moments(pdist(md, pl)) for md, pl in zip(mds, plans)]
    mu = sum(x for x, _ in ms)
    var = sum(y for _, y in ms)
    return mu, math.sqrt(var)


# ------------------------------------------------------------------ 方差诊断
def var_by_item(md, pl):
    """把单场计划的方差按计分项拆开（近似：各项近似独立，用边际分布）。"""
    p = md.p_win
    out = {}
    d = pdist(md, pl)
    m, v = dist_moments(d)
    out["total"] = v
    # 逐项边际
    for label, dist in _item_dists(md, pl):
        mm, vv = dist_moments(dist)
        out[label] = vv
    return out


def _item_dists(md, pl):
    p = md.p_win
    yield ("winner", {10: p[pl["winner"]], 0: 1 - p[pl["winner"]]})
    if md.bo in (3, 5):
        hit = md.score_true[pl["winner"]].get(tuple(pl["score"]), 0.0)
        q = p[pl["winner"]]
        yield ("score", {10: q * hit, 0: 1 - q * hit})
    md_ = collections.defaultdict(float)
    for tset, pr in md.map_true.items():
        md_[5 * sum(1 for m in pl["maps"] if m in tset)] += pr
    yield ("maps", dict(md_))
    win_rows = md.sample["player_rows"].get(pl["winner"]) or []
    mh = (D.MVP_CAL_POOR if D._mvp_richness(win_rows) < D.MVP_RICH_MIN
          else D.MVP_CAL_RICH)
    q = p[pl["winner"]]
    yield ("mvp", {10: q * mh, 0: 1 - q * mh})


def gamble_audit(mds, strict_rel=1e-3):
    """在**真实语料**的候选集上找「用均值换方差」的赌选项。

    ⚠ 两个判据必须分开，否则会误读（2026-10-19 实测踩到）：

      宽松 `sd >= sd_max 且 E < E_max` —— 会命中大量**对称性假象**：
          胜负项方差 100·p(1−p) 对 p↔1−p 对称 ⇒ 押 0.49 的另一方，
          **sd 与热门完全相同**，只是 E 低一点点。这类「赌选项」**不加方差**，
          只是白送均值。valid 上命中的 21 个全部是这一类（sd 完全相等）。

      严格 `sd > sd_max × (1+strict_rel) 且 E < E_max` —— 才是真的把方差买上来了。

    NOTE-014 §12.4 记的「赌选项 = 0 个」是在**单场 72 个候选**上测的，
    结论对那一场成立；本函数扫全量候选，用于给出更一般的口径。
    """
    n_gam = n_strict = n_plans = 0
    xs, ys = [], []
    per = []
    for md in mds:
        mom = [(dist_moments(pdist(md, pl)), pl) for pl in md.plans]
        if not mom:
            continue
        mmax = max(m for (m, _v), _pl in mom)
        vmax = max(v for (_m, v), _pl in mom)
        xs += [m for (m, _v), _pl in mom]
        ys += [math.sqrt(max(v, 0.0)) for (_m, v), _pl in mom]
        g = [pl for (m, v), pl in mom if v >= vmax - 1e-9 and m < mmax - 1e-9]
        st = [pl for (m, v), pl in mom
              if v > vmax * (1.0 + strict_rel) and m < mmax - 1e-9]
        n_plans += len(mom)
        n_gam += len(g)
        n_strict += len(st)
        per.append((len(mom), len(g), len(st)))
    n = len(xs)
    mx = sum(xs) / max(1, n)
    my = sum(ys) / max(1, n)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    corr = cov / (sx * sy) if sx > 0 and sy > 0 else 0.0
    return {"plans_total": n_plans, "gamble_options": n_gam,
            "gamble_strict": n_strict, "strict_rel": strict_rel,
            "corr_E_sd": corr, "per_match": per}


# ------------------------------------------------------------------ 主流程
def paired_bootstrap(mds, ps, pr, trials=20000, seed=20261019):
    """配对自助法：两目标**逐场**差分的抽样分布。

    为什么必须做：rank 版的优势若只来自十几场的 MVP 抖动，
    +50 分完全可能是噪声。不给显著性就报数 = 拿噪声当结论。
    配对（同场两个计划一起抽）能消掉场次难度差异。
    """
    import random
    from scorer import Prediction, score_match
    diff = []
    for md, a, b in zip(mds, ps, pr):
        t = md.sample.get("truth")
        if not t:
            continue
        truth = M.to_truth(t, md.sample.get("mvp_truth", ""))
        sa = score_match(Prediction(a["winner"], tuple(a["score"]),
                                    list(a["maps"]), a["mvp"]), truth).total
        sb = score_match(Prediction(b["winner"], tuple(b["score"]),
                                    list(b["maps"]), b["mvp"]), truth).total
        diff.append(sb - sa)
    n = len(diff)
    if not n:
        return {"n": 0}
    rng = random.Random(seed)
    mean = sum(diff) / n
    boots = []
    for _ in range(trials):
        s = 0.0
        for _ in range(n):
            s += diff[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int(0.025 * trials)]
    hi = boots[int(0.975 * trials)]
    nz = sum(1 for x in diff if x != 0)
    return {"n": n, "mean_diff": mean, "ci95": [lo, hi],
            "changed_matches": nz,
            "p_two_sided": 2.0 * min(sum(1 for x in boots if x <= 0),
                                      sum(1 for x in boots if x >= 0)) / trials,
            "wins": sum(1 for x in diff if x > 0),
            "losses": sum(1 for x in diff if x < 0)}


def var_profile(mds, plans):
    """把总方差按计分项拆开（各项近独立的近似）。"""
    tot = collections.Counter()
    for md, pl in zip(mds, plans):
        for lab, dist in _item_dists(md, pl):
            tot[lab] += dist_moments(dist)[1]
    return dict(tot)


def gamble_detail(mds, limit=12):
    """把审计里找到的赌选项逐个列出来：E、sd、分项方差、以及它在不在候选里。"""
    out = []
    for i, md in enumerate(mds):
        mom = [(dist_moments(pdist(md, pl)), pl) for pl in md.plans]
        if not mom:
            continue
        mmax = max(m for (m, _v), _pl in mom)
        vmax = max(v for (_m, v), _pl in mom)
        best_e = min(mom, key=lambda x: (0 if x[0][0] >= mmax - 1e-9 else 1,))[1]
        for (m, v), pl in mom:
            if v >= vmax - 1e-9 and m < mmax - 1e-9:
                d = var_profile([md], [pl])
                out.append({
                    "idx": i, "teams": [md.A, md.B], "bo": md.bo,
                    "winner": pl["winner"],
                    "p_win": round(md.p_win[pl["winner"]], 4),
                    "mvp": pl["mvp"],
                    "E": round(m, 2), "sd": round(math.sqrt(max(v, 0)), 2),
                    "E_of_best": round(mmax, 2),
                    "sd_of_best": round(math.sqrt(max(vmax, 0)), 2),
                    "var_by_item": {k: round(x, 2) for k, x in d.items()},
                    "best_e_mvp": best_e["mvp"],
                })
    return out[:limit], len(out)


def _print_audit(b, g):
    print("\n[%s] 候选 %d 个 | corr(E,sd)=%+.4f"
          % (b, g["plans_total"], g["corr_E_sd"]))
    print("      宽松判据（sd ≥ 最高E方案 且 E 更低）：%d 个，分布在 %d/%d 场"
          % (g["gamble_options"],
             sum(1 for _, x, _s in g["per_match"] if x), len(g["per_match"])))
    print("      **严格判据（sd 真的更高 >+%.1f%% 且 E 更低）：%d 个，"
          "分布在 %d/%d 场**"
          % (100 * g["strict_rel"], g["gamble_strict"],
             sum(1 for _, _x, s in g["per_match"] if s), len(g["per_match"])))
    if g["gamble_options"] and not g["gamble_strict"]:
        print("      ⇒ 宽松命中的全是**对称性假象**：押 p≈0.5 的另一方，"
              "sd 与热门**完全相同**、只降均值 ⇒ 对方差零贡献")


def run_bucket(bucket, opponents, noise, by_event, max_events):
    mds = build(bucket)
    units = []
    if by_event:
        grp = collections.OrderedDict()
        for md in mds:
            ev = (md.sample.get("truth") or {}).get("event") or "?"
            grp.setdefault(ev, []).append(md)
        if max_events:
            grp = collections.OrderedDict(
                list(sorted(grp.items(), key=lambda kv: -len(kv[1])))[:max_events])
        units = list(grp.values())
    else:
        units = [mds]

    agg = {k: {"n": 0, "pts": 0, "max": 0, "items": collections.Counter(),
               "mu": 0.0, "var": 0.0, "er": 0.0, "diff": 0}
           for k in ("score", "rank", "ceiling")}
    n_diff = n_diff_winner = 0

    for u in units:
        field = D.Field(opponents, noise).fit(u)
        ps = [D.choose_score(md) for md in u]
        pr = descend(u, field)
        pts_s, it_s = realized_items(u, ps)
        pts_r, it_r = realized_items(u, pr)
        # 天花板：oracle 逐场取 max
        pts_c = 0
        it_c = collections.Counter()
        from scorer import Prediction, score_match
        for md in u:
            t = md.sample.get("truth")
            if not t:
                continue
            truth = M.to_truth(t, md.sample.get("mvp_truth", ""))
            best = None
            for pl in md.plans:
                b = score_match(Prediction(pl["winner"], tuple(pl["score"]),
                                           list(pl["maps"]), pl["mvp"]), truth)
                if best is None or b.total > best.total:
                    best = b
            pts_c += best.total
            it_c["winner"] += best.winner
            it_c["score"] += best.score
            it_c["maps"] += best.maps
            it_c["mvp"] += best.mvp
        mx = sum(md.max_points() for md in u)

        mu_s, sd_s = sim_moments(u, ps)
        mu_r, sd_r = sim_moments(u, pr)
        for k, pts, it, mu, sd, pl in (("score", pts_s, it_s, mu_s, sd_s, ps),
                                       ("rank", pts_r, it_r, mu_r, sd_r, pr),
                                       ("ceiling", pts_c, it_c, mx, 0.0, ps)):
            a = agg[k]
            a["n"] += len(u)
            a["pts"] += pts
            a["max"] += mx if k == "ceiling" else mx
            for kk, vv in it.items():
                a["items"][kk] += vv
            a["mu"] += mu
            a["var"] += sd
            a["er"] += field.expected_rank(mu, sd * sd)
        d = sum(1 for a, b in zip(ps, pr) if key_of(a) != key_of(b))
        dw = sum(1 for a, b in zip(ps, pr) if a["winner"] != b["winner"])
        n_diff += d
        n_diff_winner += dw

    return agg, n_diff, n_diff_winner, len(units)


def report(tag, agg, n_diff, n_diff_winner, n_units, opponents, boot=None):
    print("\n" + "=" * 68)
    print("== %s  （%d 场，%d 个决策单元，对手 %d 人）=="
          % (tag, agg["score"]["n"], n_units, opponents))
    print("=" * 68)
    print("%-10s %9s %9s %10s %9s %9s" %
          ("目标", "真值得分", "得分率", "自估E[分]", "自估sd", "模拟名次"))
    for k in ("score", "rank", "ceiling"):
        a = agg[k]
        pct = 100.0 * a["pts"] / max(1, a["max"])
        print("%-10s %9d %8.1f%% %10.0f %9.0f %9.2f" %
              (k, a["pts"], pct, a["mu"], a["var"], a["er"]))
    print("\n分项真实得分（winner / score / maps / mvp）：")
    for k in ("score", "rank", "ceiling"):
        it = agg[k]["items"]
        print("  %-9s %4d / %4d / %4d / %4d   合计 %d"
              % (k, it["winner"], it["score"], it["maps"], it["mvp"],
                 sum(it.values())))
    print("\n两目标分歧：%d 场（其中**胜方不同 %d 场**）"
          % (n_diff, n_diff_winner))
    d = agg["rank"]["pts"] - agg["score"]["pts"]
    print("⇒ rank 版比 score 版真值得分 **%+d 分**（%+.3f 分/场）"
          % (d, d / max(1, agg["score"]["n"])))
    if boot and boot.get("n"):
        print("   配对自助法：均值 %+.3f 分/场，95%% CI [%+.2f, %+.2f]，"
              "双侧 p = %.3f"
              % (boot["mean_diff"], boot["ci95"][0], boot["ci95"][1],
                 boot["p_two_sided"]))
        print("   分歧场上：rank 更赚 %d 场、更亏 %d 场（共 %d 场有差异）"
              % (boot["wins"], boot["losses"], boot["changed_matches"]))
        verdict = ("**统计显著**" if boot["p_two_sided"] < 0.05
                   else "**落在噪声内**（95% CI 含 0）")
        print("   判定：%s" % verdict)
    head = agg["ceiling"]["pts"] - agg["score"]["pts"]
    print("⇒ 天花板 %d 分，比 score 版高 %d ⇒ **两目标之争只占可争取空间的 %.1f%%**"
          % (agg["ceiling"]["pts"], head,
             100.0 * head / max(1, agg["ceiling"]["pts"])))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="test", choices=["valid", "test", "both"])
    ap.add_argument("--opponents", type=int, default=20)
    ap.add_argument("--noise", type=float, default=0.25)
    ap.add_argument("--by-event", action="store_true",
                    help="按赛事切分决策单元（真实用法）")
    ap.add_argument("--events", type=int, default=0, help="只取前 N 个赛事")
    ap.add_argument("--audit", action="store_true", help="只跑赌选项审计")
    ap.add_argument("--detail", action="store_true",
                    help="逐个列出赌选项（需 --audit 同时给出总数）")
    a = ap.parse_args()

    buckets = ["valid", "test"] if a.bucket == "both" else [a.bucket]
    out = {}
    for b in buckets:
        mds = build(b)
        if a.audit and not a.detail:          # --audit 单独用 = 只审计
            g = gamble_audit(mds)
            _print_audit(b, g)
            out.setdefault(b, {})["audit"] = {k: v for k, v in g.items()
                                              if k != "per_match"}
            continue
        tag = "%s%s" % (b, "/按赛事" if a.by_event else "/整桶")
        agg, nd, ndw, nu = run_bucket(b, a.opponents, a.noise, a.by_event, a.events)
        # 整桶口径下补自助法（按赛事模式下逐赛事独立，不合并）
        boot = None
        if not a.by_event:
            field = D.Field(a.opponents, a.noise).fit(mds)
            ps = [D.choose_score(md) for md in mds]
            pr = descend(mds, field)
            boot = paired_bootstrap(mds, ps, pr)
            vs = var_profile(mds, ps)
            vr = var_profile(mds, pr)
        else:
            vs = vr = {}
        out.setdefault(b, {})["by_event" if a.by_event else "whole"] = {
            "n": agg["score"]["n"], "units": nu,
            "realized": {k: agg[k]["pts"] for k in agg},
            "max": agg["score"]["max"],
            "items": {k: dict(agg[k]["items"]) for k in agg},
            "diff": nd, "diff_winner": ndw,
            "bootstrap": boot,
            "var_by_item_score": vs, "var_by_item_rank": vr,
        }
        report(tag, agg, nd, ndw, nu, a.opponents, boot)
        if vs:
            print("\n方差来源分解（Var 按计分项，整桶合计）：")
            print("  %-8s %12s %12s %8s" % ("项", "score 版", "rank 版", "占比"))
            tot_s = sum(vs.values()) or 1.0
            for k in ("winner", "score", "maps", "mvp"):
                print("  %-8s %12.0f %12.0f %7.1f%%"
                      % (k, vs.get(k, 0), vr.get(k, 0), 100.0 * vs.get(k, 0) / tot_s))
            print("  ⇒ 最大方差来源 = **%s**"
                  % max(vs, key=lambda k: vs[k]))

    if a.detail:                # --detail：主流程 + 逐个赌选项明细
        print("\n" + "=" * 68)
        print("== 赌选项审计（全量候选扫描）==")
        print("=" * 68)
        for b in buckets:
            mds = build(b)
            g = gamble_audit(mds)
            _print_audit(b, g)
            out.setdefault(b, {})["audit"] = {k: v for k, v in g.items()
                                              if k != "per_match"}
            if g["gamble_options"]:
                rows, tot = gamble_detail(mds)
                for r in rows:
                    print("   %s vs %s bo%d | 押 %s (p=%.2f) MVP=%s | "
                          "E=%.1f(sd %.1f) vs 最高E=%.1f(sd %.1f)"
                          % (r["teams"][0], r["teams"][1], r["bo"], r["winner"],
                             r["p_win"], r["mvp"], r["E"], r["sd"],
                             r["E_of_best"], r["sd_of_best"]))
                out.setdefault(b, {})["gamble_detail"] = rows

    path = os.path.join(HERE, "exp_decide_ab_results.json")
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print("\n已写 %s" % os.path.basename(path))


if __name__ == "__main__":
    sys.exit(main())