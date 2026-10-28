#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pick'Em 阶段预测器（生产入口）—— issue-015「可交付使用」的那一环。

给定**一个 Major 阶段的队伍名单与种子**，输出该阶段的 **10 格 Pick'Em 推荐**
（2×3:0 + 6×晋级(3:1/3:2) + 2×0:3）与 **P(答对 ≥5)**。

链路：

    roster(JSON) → sources(强度→胜率) → swiss(16 队瑞士轮联合模拟)
                 → pickem(在场景矩阵上最大化 P(≥k)) → 十格 + 过关概率

为什么不能「各格取最优」：十格的正确事件由**同一批瑞士轮结果**驱动、强相关，
目标是**阈值概率**而不是十个独立分类问题 ⇒ 这是组合选择问题，门槛一改最优解就变。
本 CLI 同时给出「独立假设」对照，量化忽略相关性会乐观多少。

用法：

    python predict_stage.py --roster fixtures/example_roster.json
    python predict_stage.py --roster my.json --n 50000 --k 5 --json out.json

退出码：0 成功；1 用法 / roster 非法 / 读文件失败。

roster 格式见 README「roster 格式」一节。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pickem as PK   # noqa: E402
import sources as S   # noqa: E402
import swiss as SW    # noqa: E402
import uncertain as UN  # noqa: E402

# 优化器预算（本 CLI 的**生产取值**）。
# 背景（2026-10-15，穷举对拍，见 issue-015 §8 第九条）：`pickem.optimize` 是
# 「随机重启 + 首次改进爬山」，**不保证最优**。在 12 队可穷举小实例上实测：
# 默认 `restarts=8` 有 1/26 实例落在次优（该实例 0.3167 vs 穷举 0.3233，差 0.0067），
# 而 `restarts>=24` 在同一实例上换 5 个优化种子全部达最优。
# 真实 Budapest 实例（n=20000）在 restarts ∈ {8,16,48,200,800} 下结果**完全一致**，
# 故提高预算**不改变已公布数字**，只是把平坦实例上的稳健性买回来。
OPT_RESTARTS = 24


def load_roster(path):
    """读 roster 并校验。返回 (roster, team_names, seeds)。非法即抛 ValueError。"""
    with open(path, encoding="utf-8") as fh:
        roster = json.load(fh)
    return validate_roster(roster)


def validate_roster(roster):
    """校验一个**已在内存里**的 roster dict（CLI 与 Web 入口共用）。

    返回 (roster, team_names, seeds)。非法即抛 ValueError——**非法一律拒绝，不出结果**。
    """
    if not isinstance(roster, dict):
        raise ValueError("roster 应为 JSON 对象")
    teams = roster.get("teams") or []
    kind = roster.get("source", "rank_points")
    # matrix 源没有逐队标量强度；它给的是两两胜率矩阵（外部模型的桥）
    key = None if kind == "matrix" else ("rating" if kind == "elo" else "points")
    problems = []
    if len(teams) != 16:
        problems.append("roster 必须恰好 16 队，实得 %d" % len(teams))
    names = [t.get("name") for t in teams]
    if len(set(names)) != len(names):
        problems.append("队名有重复")
    for t in teams:
        if not t.get("name") or t.get("seed") is None:
            problems.append("每队必须有 name 与 seed")
            break
        if key and t.get(key) is None:
            problems.append("source=%s 时每队必须有 %s" % (kind, key))
            break
    if not problems and kind == "matrix":
        problems.extend(S.check_matrix(roster.get("matrix"), names))
    if not problems:
        seeds = {t["name"]: int(t["seed"]) for t in teams}
        if sorted(seeds.values()) != list(range(1, 17)):
            problems.append("seed 必须是 1..16 的排列，实得 %r" % sorted(seeds.values()))
    if problems:
        raise ValueError("; ".join(problems))
    return roster, names, seeds


# ---------------------------------------------------------------- 队名 → roster
# 公开强度库（`fixtures/team_strength.json`）：**预计算**好每队强度，玩家只填队名。
# 为什么要有它：公开工具不读 data/，但「填 16 个队名就出十格」要求强度是现成的。
# 库由私有侧一次性生成（provenance 写在库内），公开侧只查表。
STRENGTH_LIB = os.path.join(HERE, "fixtures", "team_strength.json")
DEFAULT_RATING = 1500.0        # Elo 兜底值：与 sources.EloSource 的 default 一致
DEFAULT_POINTS = 250.0         # rank_points 兜底值：低于 VRS 榜尾（355），但不是 0 分悬崖


def load_strength_library(path=STRENGTH_LIB):
    """读公开强度库 → {"asof", "source", "_role", "_provenance", "counts",
    "teams": [队记录...], "index": {名/别名(小写): 记录}}。非法即 ValueError。"""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    raw = doc.get("teams") or []
    if not raw:
        raise ValueError("强度库没有 teams（%s）" % path)
    teams, index = [], {}
    for t in raw:
        name = (t.get("name") or "").strip()
        if not name:
            raise ValueError("强度库里有一条没有 name")
        rec = {"name": name, "rating": t.get("rating"), "rank": t.get("rank"),
               "points": t.get("points"),
               "aliases": [a for a in (t.get("aliases") or []) if a]}
        if name.lower() in index:
            raise ValueError("强度库队名重复：%s" % name)
        index[name.lower()] = rec
        teams.append(rec)
    for rec in teams:                      # 别名索引（不覆盖正式名）
        for a in rec["aliases"]:
            index.setdefault(a.lower(), rec)
    out = dict(doc)
    out.update({"teams": teams, "index": index})
    return out


def resolve_team(query, index):
    """按名解析：精确（忽略大小写）→ 别名（忽略大小写）→ **唯一**子串。

    返回 (规范名, 记录)；解析不了返回 (None, None)。歧义子串也返回 None（不猜）。
    """
    q = (query or "").strip()
    if not q:
        return None, None
    rec = index.get(q.lower())
    if rec is not None:
        return rec["name"], rec
    hits = {r["name"]: r for key, r in index.items() if q.lower() in key}
    if len(hits) == 1:
        name, r = next(iter(hits.items()))
        return name, r
    return None, None


def build_roster(names, lib=None, source="rank_points", stage=None, date=None):
    """**16 个队名 + 强度库 → roster**（种子 = 输入顺序 1..16）。

    返回 (roster, warnings)。契约：
      * 恰好 16 个名字，否则 ValueError（不猜、不补）；
      * 名字按 精确/别名/唯一子串 解析到库里的**规范名**（大小写不敏感）；
      * 库里有定义 → 取 rating（source=elo）或 points（source=rank_points）；
      * 库里**没有**（或该字段缺失）→ 用默认强度兜底，并**在 warnings 里点名**，
        让「结果不可信」这件事可见（静默兜底是本项目明令禁止的失败模式）；
      * 同一队出现两次 → ValueError（否则会静默退化成 15 队）。

    输出一定过 `validate_roster`（种子 1..16 排列、16 队、字段齐）。
    """
    if not isinstance(names, (list, tuple)):
        raise ValueError("teams 应为 16 个队名的数组")
    names = [str(t).strip() for t in names]
    if len(names) != 16:
        raise ValueError("需要恰好 16 个队名，实得 %d" % len(names))
    if any(not t for t in names):
        raise ValueError("有 %d 个位置是空的（16 个队名都要填）" % sum(1 for t in names if not t))
    if lib is None:
        lib = load_strength_library()
    else:
        lib = load_strength_library(lib) if isinstance(lib, str) else lib
    index = lib.get("index")
    if index is None:
        raise ValueError("强度库缺 index")
    if source not in ("elo", "rank_points"):
        raise ValueError("source 只支持 elo / rank_points（实得 %r）" % source)

    teams, warnings, seen = [], [], {}
    for i, raw in enumerate(names, start=1):
        name, rec = resolve_team(raw, index)
        known = rec is not None
        if not known:                       # 库里查无此队：仍建 roster，但必须点名
            name = raw
            rec = {"name": raw, "rating": None, "rank": None, "points": None}
            warnings.append("⚠ 队伍 %r 不在强度库里（拼错？）：按默认强度 %s 兜底，"
                            "该队结果不可信"
                            % (raw, DEFAULT_RATING if source == "elo" else DEFAULT_POINTS))
        key = name.lower()
        if key in seen:
            raise ValueError("第 %d 位 %r 与第 %d 位的 %r 是同一支队"
                             % (i, raw, seen[key], name))
        seen[key] = i
        if source == "elo":
            val = rec.get("rating")
            if val is None:
                val = DEFAULT_RATING
                if known:                   # 已在库、只是缺这一列（有 VRS 无 Elo）
                    warnings.append("⚠ 队伍 %s 缺 Elo（库里只有 VRS 排名）：按 %s 兜底"
                                    % (name, DEFAULT_RATING))
            row = {"name": name, "seed": i, "rating": float(val)}
        else:
            val = rec.get("points")
            if val is None:
                val = DEFAULT_POINTS
                if known:
                    warnings.append("⚠ 队伍 %s 缺排名 points：按 %s 兜底"
                                    % (name, DEFAULT_POINTS))
            row = {"name": name, "seed": i, "points": float(val)}
        if rec.get("rank") is not None:
            row["rank"] = int(rec["rank"])
        teams.append(row)

    roster = {"stage": stage or ("自填阶段（强度库 %s）" % lib.get("asof", "未知日期")),
              "source": source, "teams": teams}
    if date:
        roster["date"] = date
    validate_roster(roster)                # 出口再验一遍：非法一律拒绝，不出结果
    return roster, warnings


def naive_rank_picks(seeds):
    """naive rank 序基线（issue §7 待办 1）：按种子**直填**——
    3:0 取种子 1-2、晋级取种子 3-8、0:3 取种子 15-16。零建模，纯排序。
    """
    order = sorted(seeds, key=lambda t: seeds[t])
    return {"p30": order[:2], "padv": order[2:8], "p03": order[-2:]}


def grid_stats(picks, marg, masks, idx, k):
    """评估一组十格：P(≥k) + 期望命中格数 + 按类别的期望命中分解。

    分解项是「所选格子的边际概率之和」，用于回答「增益来自哪几个格子」。
    """
    per = {c: sum(marg[t][c] for t in picks[c]) for c in ("p30", "padv", "p03")}
    return {"p_ge_k": PK.p_at_least(picks["p30"], picks["padv"], picks["p03"], masks, k, idx),
            "expected_hits": sum(per.values()),
            "expected_by_cat": per}


def predict(roster, n=20000, k=5, seed=20261013, uncertain=False, tie_break="seed"):
    """跑完整链路，返回结果 dict（可 JSON 化）。

    uncertain=True 时改跑 **σ 后验臂**（`uncertain.py`）：每队抽 θ ~ N(μ, σ) 再跑
    完整瑞士轮。默认 False ⇒ 与既有行为逐字节一致（0.2106 复现命令不变）。
    非法 roster（缺 exposure/games、rank_points 源）由 `build_posterior` 抛 ValueError。
    """
    names = [t["name"] for t in roster["teams"]]
    seeds = {t["name"]: int(t["seed"]) for t in roster["teams"]}

    src, kind, coefs = S.source_from_roster(roster)
    if kind == "elo":
        strength = {t: src.rating(t) for t in names}
    elif kind == "rank_points":
        strength = {t: src.pts(t) for t in names}
    else:                       # matrix：用「对场胜率」当可读的替代列（不参与计算）
        strength = {t: src.strength(t) for t in names}
    post = None
    if uncertain:
        post = UN.build_posterior(roster)          # 非法即 ValueError → CLI rc=1
        scenes, stages = UN.simulate_uncertain(
            names, seeds, [post["teams"][t]["mu"] for t in names],
            [post["teams"][t]["sigma"] for t in names],
            coefs["bo1"], coefs["bo3"], n, seed, tie_break=tie_break)
    else:
        pwin = S.fast_pwin(src, names)
        scenes, stages = SW.simulate_scenes(names, seeds, pwin, n, seed=seed,
                                            tie_break=tie_break)
    C = PK.build_cat_matrix(scenes, names)
    marg = PK.marginals(C, names)

    picks, p_k, evals = PK.optimize(C, names, k=k, seed=seed, restarts=OPT_RESTARTS)
    ind = PK.independent_p_at_least(PK.pick_probs(picks, marg), k)

    masks = (C == PK.CAT30, C == PK.CATADV, C == PK.CAT03)
    idx = {t: i for i, t in enumerate(names)}
    grid_p = {kk: PK.p_at_least(picks["p30"], picks["padv"], picks["p03"], masks, kk, idx)
              for kk in (5, 6, 7)}

    baselines = {}
    for label, cand in (("optimizer", picks), ("naive_rank", naive_rank_picks(seeds)),
                        ("hot", PK.heuristic_picks(marg))):
        baselines[label] = dict(grid_stats(cand, marg, masks, idx, k), picks=cand)

    rep = {
        "stage": roster.get("stage", "(未命名阶段)"),
        "date": roster.get("date"),
        "n": n, "seed": seed, "k": k,
        "source": kind, "slopes": coefs,
        "arm": "uncertain" if uncertain else "point",
        "tie_break": tie_break,
        "baselines": baselines,
        "schedule": stages,
        "marginals": marg,
        "strength": strength,
        "seeds": seeds,
        "picks": picks,
        "p_ge_k": p_k,
        "grid_p": grid_p,
        "independent_p_ge_k": ind,
        "evals": evals,
    }
    if post is not None:
        rep["uncertain"] = post
    return rep


def report(rep):
    print("== Pick'Em 阶段预测：%s ==" % rep["stage"])
    if rep.get("date"):
        print("   阶段日期 %s" % rep["date"])
    if rep["source"] == "matrix":
        print("   强度源 matrix（两两胜率直给）｜ bo1 折算 α=%.2f（bo3 概率直给）｜ 模拟 %d 次（seed=%d）"
              % (rep["slopes"]["bo1"], rep["n"], rep["seed"]))
    else:
        print("   强度源 %s ｜ 斜率 bo1=%.4f / bo3=%.4f ｜ 模拟 %d 次（seed=%d）"
              % (rep["source"], rep["slopes"]["bo1"], rep["slopes"]["bo3"], rep["n"], rep["seed"]))
    print("   赛程校验：bo1=%d / bo3=%d（规则要求 20 / 13）"
          % (rep["schedule"]["bo1"], rep["schedule"]["bo3"]))
    print("   同 Buchholz 平局键：%s"
          % ("均匀随机抽签（--random-ties）" if rep.get("tie_break") == "random"
             else "种子升序（默认，可复现已公布数字）"))
    u = rep.get("uncertain")
    if u:
        fit = u.get("fit")
        nblend = sum(1 for v in u["teams"].values() if v["blended"])
        print("   臂：σ 不确定性传播 —— σE 分桶 E=0→120 / E1-2→80 / E≥3→40（Elo 单位），"
              "E<%d 的 %d 队均值回归 rank 先验%s"
              % (u["blend_e"], nblend,
                 ("（rating ~ %.1f %+.4f·ln(1+rank)）" % (fit["a"], fit["b"]))
                 if fit else "（rank 点不足，未拟合）"))
    print()
    print("逐队最终战绩分布（模拟）：")
    marg = rep["marginals"]
    if u:
        print("  %-4s %-20s %9s %7s %9s %4s %8s %8s %8s"
              % ("种子", "队伍", "强度", "μ", "σ", "E", "P(3:0)", "P(晋级)", "P(0:3)"))
        for t in sorted(marg, key=lambda x: rep["seeds"][x]):
            m, v = marg[t], u["teams"][t]
            print("  %-4d %-20s %9.1f %7.1f %9.1f %4s %8.3f %8.3f %8.3f"
                  % (rep["seeds"][t], t, rep["strength"][t], v["mu"], v["sigma"],
                     v["exposure"] if v["exposure"] is not None else "-",
                     m["p30"], m["padv"], m["p03"]))
        print("     （回归队 %s）" % ", ".join(sorted(t for t in u["teams"]
                                              if u["teams"][t]["blended"])))
    elif rep["source"] == "matrix":
        print("  %-4s %-20s %10s %8s %8s %8s"
              % ("种子", "队伍", "对场胜率", "P(3:0)", "P(晋级)", "P(0:3)"))
        for t in sorted(marg, key=lambda x: rep["seeds"][x]):
            m = marg[t]
            print("  %-4d %-20s %10.3f %8.3f %8.3f %8.3f"
                  % (rep["seeds"][t], t, rep["strength"][t], m["p30"], m["padv"], m["p03"]))
    else:
        print("  %-4s %-20s %10s %8s %8s %8s" % ("种子", "队伍", "强度", "P(3:0)", "P(晋级)", "P(0:3)"))
        for t in sorted(marg, key=lambda x: rep["seeds"][x]):
            m = marg[t]
            print("  %-4d %-20s %10.1f %8.3f %8.3f %8.3f"
                  % (rep["seeds"][t], t, rep["strength"][t], m["p30"], m["padv"], m["p03"]))
    picked = rep["picks"]
    print()
    print("十格推荐（最大化 P(≥%d)）：" % rep["k"])
    for key, label in (("p30", "3:0 "), ("padv", "晋级"), ("p03", "0:3 ")):
        cells = ["%s(%.2f)" % (t, marg[t][key])
                 for t in sorted(picked[key], key=lambda x: -marg[x][key])]
        print("  %s  %s" % (label, "  ".join(cells)))
    print("    （按边际概率降序显示；JSON 的 picks 保留优化器原始顺序，属同一集合）")
    print()
    print("过关概率：")
    print("  P(命中≥5) = %.4f   P(≥6) = %.4f   P(≥7) = %.4f"
          % (rep["grid_p"][5], rep["grid_p"][6], rep["grid_p"][7]))
    print("  独立假设 P(≥%d) = %.4f ⇒ 相关性修正 %+.4f"
          % (rep["k"], rep["independent_p_ge_k"], rep["p_ge_k"] - rep["independent_p_ge_k"]))
    print("  （优化器评估次数 %d）" % rep["evals"])
    b = rep.get("baselines")
    if b:
        print()
        print("基线对照（同一批场景）：")
        for label, name in (("optimizer", "优化器   "), ("naive_rank", "naive 序 "),
                            ("hot", "热门基线 ")):
            v = b[label]
            print("  %s P(≥%d)=%.4f  期望命中格数=%.2f"
                  % (name, rep["k"], v["p_ge_k"], v["expected_hits"]))
        for lab, nm in (("naive_rank", "naive 序 "), ("hot", "热门基线 ")):
            dd = {c: b["optimizer"]["expected_by_cat"][c] - b[lab]["expected_by_cat"][c]
                  for c in ("p30", "padv", "p03")}
            print("  优化器 − %s（期望命中分解）：3:0 %+.2f ｜ 晋级 %+.2f ｜ 0:3 %+.2f"
                  % (nm.strip(), dd["p30"], dd["padv"], dd["p03"]))
        if all(set(b["optimizer"]["picks"][c]) == set(b["hot"]["picks"][c])
               for c in ("p30", "padv", "p03")):
            print("  ⚠ 本实例：热门基线（逐格取边际最大）与优化器选出**同一批格子**"
                  "（仅组内顺序不同）⇒ 搜索未带来增益，见 issue-015 §7 待办 1")
    print()
    print("注：P(≥k) 是**该十格**在模拟分布下的过关概率，不是模型的整体准确率；")
    print("    十格强相关，故它不等于十格边际概率的独立组合（对照行给出差值）。")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Major 阶段 Pick'Em 十格预测")
    ap.add_argument("--roster", required=True, help="roster JSON 路径（source=elo / rank_points / matrix）")
    ap.add_argument("--n", type=int, default=20000, help="模拟场景数（默认 20000）")
    ap.add_argument("--k", type=int, default=5, help="优化目标过关线（默认 5）")
    ap.add_argument("--seed", type=int, default=20261013, help="随机种子")
    ap.add_argument("--json", dest="json_out", help="把完整结果写入该路径")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--uncertain", action="store_true",
                    help="σ 不确定性传播臂（冷门队；需每队 exposure/games[/rank] 或 sigma，仅 source=elo）")
    ap.add_argument("--random-ties", action="store_true",
                    help="同 Buchholz 并列改为均匀随机抽签（issue §5.4；默认种子升序）")
    a = ap.parse_args(argv)

    try:
        roster, _names, _seeds = load_roster(a.roster)
        rep = predict(roster, n=a.n, k=a.k, seed=a.seed, uncertain=a.uncertain,
                      tie_break="random" if a.random_ties else "seed")
    except (OSError, ValueError) as exc:
        print("[ERR] roster 无效：%s" % exc, file=sys.stderr)
        return 1
    if not a.quiet:
        report(rep)
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=1)
        print("\n[pickem] 结果写入 %s" % a.json_out)
    print("PICKEM_STAGE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
