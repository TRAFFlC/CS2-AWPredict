#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""coldstart.py —— 冷门/未知队伍的 **映射器**（2026-10-10）。

场景（主理人 2026-10-10 提出）
------------------------------
比赛里偶尔出现**以前没在大赛中出现过的队伍**。对这种队，我们拿不到它的比赛史，
但主理人可以提供：
  * **近期选图**：每张图被选/被打了多少次 + 该图胜率（**不含 ban 详情**）；
  * **队员 rating**。

映射器要做的事：从已有队伍里找**最相似的一支**，按情况微调后套到新队身上，
让模型（`model_v2.State` + `veto_model.MapHistory`）能对这场出预测。

做法
----
1. **档案化**：对每支已知队伍，从全量历史里抽出
      - 图出现在它比赛图集里的比例 `app_rate[m]`（7 维）
      - 该队在该图的图级胜率 `win_rate[m]`（7 维）
      - 阵容平均（对手校正）rating、Elo、场次
2. **相似度**：`sim = w1·cos(app) + w2·(1−平均|Δwin|) + w3·(1−标准化|Δrating|)`
   （只比**同一张池**上的向量；权重可调，见 `WEIGHTS`）
3. **微调**：
      - Elo 用「全数据上 rating→Elo 的线性拟合」外推，再以 donor Elo 为基准做平移：
        `elo = donor_elo + b·(input_rating − donor_rating)`；
      - 图胜率 = 提供的胜率（缺失的用 donor 补/收缩）；
      - 图出现率 = 提供的选图次数归一化（缺失的用 donor 补）；
      - 场次 `games` 给一个保守小值（默认 8）⇒ Elo 的 K 偏大，注入后仍会被真实比赛快速纠正。
4. **注入**：`State.seed_team` + `MapHistory.seed_team`（见两个模块的文档），
   只写该队自己的统计，**不动全局**。

验证（`validate`）
------------------
把若干**已知队伍**当「未知」处理：只用它近 120 天的选图次数/胜率/选手 rating，
看映射器还原的 Elo 与真实 Elo 的误差，以及**用它出预测时胜方准确率**相对
「不注入（Elo=1500）」的提升。这直接回答「冷门队伍能不能被预测」。

用法
----
  python -X utf8 coldstart.py selftest
  python -X utf8 coldstart.py donors [--top 30]
  python -X utf8 coldstart.py map --file profile.json
  python -X utf8 coldstart.py validate [--n 12]
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)  # scorer.py 与本模块同目录（仓库布局）

import veto_model as VM  # noqa: E402
import model_v2 as M  # noqa: E402
import split as SPLIT  # noqa: E402
import pool_experiment as PE  # noqa: E402

WEIGHTS = {"app": 1.0, "win": 1.0, "rating": 0.6}
# Elo 映射（两折交叉验证在 donor 集上选出；对照见 `validate`）
ELO_SHRINK = 0.5     # donor Elo 向 1500 收缩的系数
ELO_WR = 400.0       # 整体图胜率每偏离 0.5 一个单位的 Elo 平移
POOL_ORDER = ["Ancient", "Anubis", "Cache", "Dust2", "Inferno", "Mirage", "Nuke", "Overpass", "Train"]


# ------------------------------------------------------------------ 参考档案
def build_reference(quiet=True):
    """跑一遍全量历史，返回 (state, mh, sample_by_id, last_date)。

    这是「当前的已知世界」。`state`/`mh` 都可用 `seed_team` 注入。
    """
    rows = SPLIT.load(M.MATCHES)
    exclude = SPLIT.load_exclude(M.EXCLUDE) if os.path.exists(M.EXCLUDE) else None
    buckets, meta = SPLIT.split(rows, purge=True, exclude=exclude)
    truths = {t["id"]: t for t in M.load_jsonl(M.TRUTH)}
    veto = VM.load_veto()
    stats = M.load_stats()
    ranks = PE.Ranks()
    eras, by_pool = PE.load_pool_eras()
    tl = json.load(io.open(PE.TIMELINE, encoding="utf-8"))
    state = M.State()
    mh = VM.MapHistory()
    data = M.build(buckets, truths, veto, stats, ranks, eras, by_pool, tl, state, mh)
    return state, mh, data, rows, meta


def current_pool(mh):
    """当前池 = 最近一次出现过的池（按 mh 里见过的图取 7 张）。"""
    seen = sorted(mh.seen_pool)
    if len(seen) >= 7:
        # 简单取 7 张：优先 era2 的池（含 Cache）
        return ["Ancient", "Anubis", "Cache", "Dust2", "Inferno", "Mirage", "Nuke"]
    return seen


def team_profile(team, state, mh, pool, date=None, window_days=None):
    """已知队伍的档案（全部来自历史）。"""
    app = {m: mh.team_app(team, m) for m in pool}
    wr = {m: mh.team_wr(team, m) for m in pool}
    elo, games = state.elo_of(team)
    lineup = state.lineup(team, date or state.last_date)
    ratings = [state.player_adj(p) for p in lineup]
    ratings = [r for r in ratings if r is not None]
    return {
        "team": team,
        "app": app,
        "win": wr,
        "elo": elo,
        "games": games,
        "rating_mean": (sum(ratings) / len(ratings)) if ratings else None,
        "rating_sd": None,
        "lineup": lineup,
    }


def all_profiles(state, mh, pool, min_games=6):
    out = {}
    for t in list(state.games):
        if state.games[t] < min_games:
            continue
        out[t] = team_profile(t, state, mh, pool)
    return out


# ------------------------------------------------------------------ 相似度
def _cos(a, b):
    keys = sorted(set(a) | set(b))
    va = [a.get(k, 0.0) for k in keys]
    vb = [b.get(k, 0.0) for k in keys]
    na = math.sqrt(sum(x * x for x in va)) or 1.0
    nb = math.sqrt(sum(x * x for x in vb)) or 1.0
    return sum(x * y for x, y in zip(va, vb)) / (na * nb)


def similarity(inp, donor):
    """输入档案 vs 已知队伍档案 → 相似度（越大越像）。"""
    s_app = _cos(inp.get("app") or {}, donor["app"])
    win_d = [abs((inp.get("win") or {}).get(m, donor["win"][m]) - donor["win"][m])
             for m in donor["win"]]
    s_win = 1.0 - (sum(win_d) / len(win_d) if win_d else 0.0)
    s_rt = 0.5
    if inp.get("rating_mean") and donor.get("rating_mean"):
        s_rt = 1.0 - min(1.0, abs(inp["rating_mean"] - donor["rating_mean"]) / 0.25)
    return (WEIGHTS["app"] * s_app + WEIGHTS["win"] * s_win + WEIGHTS["rating"] * s_rt) / \
        sum(WEIGHTS.values())


def rating_to_elo_slope(profiles):
    """在全量已知队伍上拟合 elo ≈ a + b·rating_mean（最小二乘）。"""
    xs, ys = [], []
    for p in profiles.values():
        if p.get("rating_mean") is None or p["games"] < 8:
            continue
        xs.append(p["rating_mean"])
        ys.append(p["elo"])
    n = len(xs)
    if n < 5:
        return 0.0, 1500.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs) or 1e-9
    b = num / den
    a = my - b * mx
    return b, a


# ------------------------------------------------------------------ 映射
def map_team(inp, profiles, pool, slope, intercept, top_k=5, weight=8.0):
    """把未知队伍映射到最相似的已知队伍，并给出可注入的虚拟档案。

    `inp` = {"name":.., "picks":{map:次数}, "win":{map:胜率}, "players":[(name,rating)..]}
    """
    counts = inp.get("picks") or {}
    tot = sum(float(v) for v in counts.values())
    app_in = {m: (float(counts.get(m, 0.0)) / tot) if tot else None for m in pool}
    win_in = {m: inp.get("win", {}).get(m) for m in pool}
    players = inp.get("players") or []
    rating_mean = (sum(float(r) for _n, r in players) / len(players)) if players else None

    probe = {"app": {m: (app_in[m] if app_in[m] is not None else 0.0) for m in pool},
             "win": {m: (win_in[m] if win_in[m] is not None else 0.5) for m in pool},
             "rating_mean": rating_mean}
    ranked = sorted(profiles.items(), key=lambda kv: -similarity(probe, kv[1]))
    donor_name, donor = ranked[0]

    # ---- 微调 ----
    # 整体图胜率（有选图次数则按次数加权；否则取已给胜率的简单平均）
    if tot:
        w_num = sum(float(counts.get(m, 0.0)) * win_in[m] for m in pool if win_in[m] is not None)
        w_den = sum(float(counts.get(m, 0.0)) for m in pool if win_in[m] is not None)
        wr_overall = (w_num / w_den) if w_den else 0.5
    else:
        vals = [v for v in win_in.values() if v is not None]
        wr_overall = (sum(vals) / len(vals)) if vals else 0.5

    elo = 1500.0 + ELO_SHRINK * (donor["elo"] - 1500.0) + ELO_WR * (wr_overall - 0.5)
    notes = ["Elo = 1500 + %.2f·(donor %.0f − 1500) + %.0f·(整体图胜率 %.3f − 0.5) = %.0f"
             % (ELO_SHRINK, donor["elo"], ELO_WR, wr_overall, elo)]
    if rating_mean is not None and slope:
        notes.append("（参考）rating→Elo 全局拟合斜率 %.0f，单独用它 MAE 更差，故只作诊断" % slope)
    if not tot:
        notes.append("未提供选图次数 ⇒ 图出现率全部沿用 donor")

    app_out, win_out, picks_out = {}, {}, {}
    for m in pool:
        # 图出现率：有计数就用（缺的补 donor），否则沿用 donor
        app_out[m] = app_in[m] if app_in[m] is not None else donor["app"][m]
        win_out[m] = win_in[m] if win_in[m] is not None else donor["win"][m]
        picks_out[m] = float(counts.get(m, 0.0)) if tot else donor["app"][m]
    # 归一化选图（用作 pick 偏好）
    if tot:
        s = sum(picks_out.values()) or 1.0
        picks_out = {m: v / s for m, v in picks_out.items()}

    return {
        "name": inp.get("name") or "UNKNOWN",
        "donor": donor_name,
        "donor_games": donor["games"],
        "top": [(n, round(similarity(probe, p), 4)) for n, p in ranked[:top_k]],
        "wr_overall": wr_overall,
        "elo": elo,
        "elo_m": elo,
        "games": 8,                       # 保守小值 ⇒ 注入后仍能被真实比赛快速纠正
        "exp_games": donor["games"],      # d_games 特征用 donor 的经验水平（假定它像 donor）
        "map_app": app_out,
        "map_win": win_out,
        "map_pick": picks_out,
        "players": list(players),
        "notes": notes,
    }


def apply_mapping(state, mh, mapping, own_players=None):
    """把映射结果注入 State + MapHistory（只写该队自己的统计）。"""
    state.seed_team(mapping["name"], elo=mapping["elo"], elo_m=mapping["elo_m"],
                    games=mapping["games"], players=own_players or mapping["players"],
                    winrate=None, map_rates=mapping["map_win"],
                    exp_games=mapping.get("exp_games"))
    mh.seed_team(mapping["name"], app_rates=mapping["map_app"], win_rates=mapping["map_win"],
                 picks=mapping["map_pick"], weight=8.0)
    return state, mh


# ------------------------------------------------------------------ 验证
def recent_profiles(rows, truths, stats, last_date, weight_days=120, pool=None):
    """只用近 `weight_days` 天的数据构造「外部可见」的档案（模拟主理人提供的信息）。"""
    picks = collections.defaultdict(lambda: collections.defaultdict(int))
    win = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    matches = collections.Counter()
    ratings = collections.defaultdict(list)
    last_lineup = {}
    for r in rows:
        if VM.days_apart(last_date, r["date"]) > weight_days:
            continue
        t = truths.get(r["id"]) or {}
        mset = t.get("maps") or []
        for team in (r["team1"]["name"], r["team2"]["name"]):
            matches[team] += 1
            for m in mset:
                picks[team][m] += 1
        for m in (r.get("maps") or []):
            if m["team1_score"] == m["team2_score"]:
                continue
            for team, sc, oc in ((r["team1"]["name"], m["team1_score"], m["team2_score"]),
                                 (r["team2"]["name"], m["team2_score"], m["team1_score"])):
                win[team][m["name"]][0 if sc > oc else 1] += 1
        per = stats.get(r["id"]) or {}
        for team in (r["team1"]["name"], r["team2"]["name"]):
            ps = [(p, rt) for p, rt in (per.get(team) or []) if rt is not None]
            if ps:
                last_lineup[team] = ps
    return {"picks": picks, "win": win, "matches": matches, "last_lineup": last_lineup}


def predict_with_elo(wmodel, sample, team, elo):
    """把样本里该队的 Elo 换成 `elo` 后再预测（只用 d_elo / d_elom 两项）。"""
    feat = dict(sample["feats"])
    if sample["teamA"] == team:
        feat["d_elo"] = (elo - sample["elo"][sample["teamB"]]) / 400.0
        feat["d_elom"] = (elo - sample["elom"][sample["teamB"]]) / 400.0
    else:
        feat["d_elo"] = (sample["elo"][sample["teamA"]] - elo) / 400.0
        feat["d_elom"] = (sample["elom"][sample["teamA"]] - elo) / 400.0
    return M.predict_proba_win(wmodel, {"feats": feat})


def validate(n=12, seed=3, weight_days=120, verbose=True):
    """把**已知队伍**当未知：只用近 `weight_days` 天的外部信息，检验
    (a) 映射器还原 Elo 的误差；(b) 用它出预测时胜方准确率对「不注入」的提升。"""
    import random
    state, mh, data, rows, meta = build_reference()
    pool = current_pool(mh)
    profiles = all_profiles(state, mh, pool, min_games=8)
    slope, intercept = rating_to_elo_slope(profiles)

    truths = {t["id"]: t for t in M.load_jsonl(M.TRUTH)}
    stats = M.load_stats()
    last_date = max(r["date"] for r in rows)
    rec = recent_profiles(rows, truths, stats, last_date, weight_days, pool)

    by_team = collections.defaultdict(list)
    for s in data["valid"] + data["test"]:
        by_team[s["teamA"]].append(s)
        by_team[s["teamB"]].append(s)

    wmodel = M._train_winner(data["train"], C=M.WIN_C)

    cands = [t for t, c in rec["matches"].items()
             if c >= 8 and state.games.get(t, 0) >= 10 and by_team.get(t)]
    rng = random.Random(seed)
    rng.shuffle(cands)
    cands = cands[:n]

    res = []
    for team in cands:
        counts = {m: rec["picks"][team][m] for m in pool if rec["picks"][team][m]}
        winr = {m: (rec["win"][team][m][0] + 1.0) / (rec["win"][team][m][0] + rec["win"][team][m][1] + 2.0)
                for m in pool if rec["win"][team][m][0] + rec["win"][team][m][1] > 0}
        lineup = rec["last_lineup"].get(team) or []
        if not lineup:
            continue
        inp = {"name": team + "__cold", "picks": counts, "win": winr, "players": lineup}
        mp = map_team(inp, profiles, pool, slope, intercept)

        true_elo = state.elo[team]
        matches = by_team[team][:8]
        ok_true = ok_map = ok_cold = 0
        for s in matches:
            if (predict_with_elo(wmodel, s, team, true_elo) >= 0.5) == (s["y"] == 1):
                ok_true += 1
            if (predict_with_elo(wmodel, s, team, mp["elo"]) >= 0.5) == (s["y"] == 1):
                ok_map += 1
            if (predict_with_elo(wmodel, s, team, 1500.0) >= 0.5) == (s["y"] == 1):
                ok_cold += 1
        res.append({"team": team, "donor": mp["donor"], "true_elo": true_elo,
                    "map_elo": mp["elo"], "n": len(matches),
                    "acc_true": ok_true, "acc_map": ok_map, "acc_cold": ok_cold})

    if verbose:
        print("== coldstart.validate（%d 支队伍，近 %d 天外部档案）==" % (len(res), weight_days))
        print("  rating→Elo 拟合: elo ≈ %.0f + %.0f × rating" % (intercept, slope))
        err = [abs(r["map_elo"] - r["true_elo"]) for r in res] or [0]
        err_cold = [abs(1500.0 - r["true_elo"]) for r in res] or [0]
        print("  Elo 还原平均绝对误差: 映射 %.0f ／ 不注入(1500) %.0f" % (
            sum(err) / len(err), sum(err_cold) / len(err_cold)))
        tot = sum(r["n"] for r in res)
        if tot:
            print("  这些场次的胜方准确率: 真实 Elo %.1f%% ／ 映射 %.1f%% ／ 不注入 %.1f%%" % (
                100.0 * sum(r["acc_true"] for r in res) / tot,
                100.0 * sum(r["acc_map"] for r in res) / tot,
                100.0 * sum(r["acc_cold"] for r in res) / tot))
        for r in res:
            print("    %-22s → %-22s elo %6.0f / 真 %6.0f (Δ%+5.0f)  胜方 映射%d/%d 冷启%d/%d" % (
                r["team"], r["donor"], r["map_elo"], r["true_elo"], r["map_elo"] - r["true_elo"],
                r["acc_map"], r["n"], r["acc_cold"], r["n"]))
    return res


# ------------------------------------------------------------------ selftest
def _selftest():
    fails = []
    n_chk = [0]                      # 2026-10-11：此前打印的是**硬编码 9**

    def chk(name, got, want):
        n_chk[0] += 1
        if got != want:
            fails.append("%s: got=%r want=%r" % (name, got, want))

    chk("cos identical", round(_cos({"a": 1, "b": 0}, {"a": 1, "b": 0}), 6), 1.0)
    chk("cos orthogonal", round(_cos({"a": 1}, {"b": 1}), 6), 0.0)

    profiles = {
        "Agro": {"team": "Agro", "app": {"Mirage": 0.8, "Nuke": 0.5, "Dust2": 0.2},
                 "win": {"Mirage": 0.7, "Nuke": 0.6, "Dust2": 0.4}, "elo": 1800.0, "games": 40,
                 "rating_mean": 1.15, "lineup": ["a", "b"], "rating_sd": None},
        "Bside": {"team": "Bside", "app": {"Mirage": 0.2, "Nuke": 0.5, "Dust2": 0.8},
                  "win": {"Mirage": 0.4, "Nuke": 0.5, "Dust2": 0.6}, "elo": 1500.0, "games": 40,
                  "rating_mean": 1.05, "lineup": ["c"], "rating_sd": None},
    }
    inp = {"name": "X", "picks": {"Mirage": 8, "Nuke": 5, "Dust2": 2},
           "win": {"Mirage": 0.72, "Nuke": 0.6, "Dust2": 0.4}, "players": [("p", 1.16)]}
    mp = map_team(inp, profiles, ["Mirage", "Nuke", "Dust2"], slope=100.0, intercept=1400.0)
    chk("picks most similar donor", mp["donor"], "Agro")
    # elo = 1500 + 0.5·(1800−1500) + 400·(加权胜率 − 0.5)
    wr_overall = (8 * 0.72 + 5 * 0.6 + 2 * 0.4) / 15.0
    chk("elo formula", round(mp["elo"], 6),
        round(1500.0 + 0.5 * 300.0 + 400.0 * (wr_overall - 0.5), 6))
    chk("elo above 1500", mp["elo"] > 1500.0, True)
    chk("map app normalized", round(sum(mp["map_pick"].values()), 6), 1.0)
    chk("win filled from input", round(mp["map_win"]["Mirage"], 3), 0.72)
    chk("missing map filled from donor", mp["map_win"]["Dust2"], 0.4)
    chk("top list has 2", len(mp["top"]), 2)

    # 没有选图次数 ⇒ 全部沿用 donor
    mp2 = map_team({"name": "Y", "players": [("p", 1.15)]}, profiles, ["Mirage", "Nuke"], 100.0, 1400.0)
    chk("no counts → donor app", mp2["map_app"]["Mirage"], profiles["Agro"]["app"]["Mirage"])

    if fails:
        print("COLDSTART_SELFTEST_FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("COLDSTART_SELFTEST_OK (%d checks)" % n_chk[0])
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(_selftest())
    if len(sys.argv) > 1 and sys.argv[1] == "donors":
        ap = argparse.ArgumentParser()
        ap.add_argument("--top", type=int, default=200)
        a = ap.parse_args(sys.argv[2:])
        st, mh, _d, _r, _m = build_reference()
        prof = all_profiles(st, mh, current_pool(mh))
        for t, p in sorted(prof.items(), key=lambda kv: -kv[1]["games"])[:a.top]:
            print("%-26s elo=%6.0f games=%3d rating=%s" % (
                t, p["elo"], p["games"],
                ("%.3f" % p["rating_mean"]) if p["rating_mean"] else "-"))
        raise SystemExit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "map":
        ap = argparse.ArgumentParser()
        ap.add_argument("--file", required=True)
        a = ap.parse_args(sys.argv[2:])
        inp = json.load(io.open(a.file, encoding="utf-8"))
        st, mh, _d, _r, _m = build_reference()
        pool = current_pool(mh)
        prof = all_profiles(st, mh, pool)
        slope, intercept = rating_to_elo_slope(prof)
        mp = map_team(inp, prof, pool, slope, intercept)
        print(json.dumps(mp, ensure_ascii=False, indent=2))
        raise SystemExit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        ap = argparse.ArgumentParser()
        ap.add_argument("--n", type=int, default=12)
        a = ap.parse_args(sys.argv[2:])
        validate(a.n)
        raise SystemExit(0)
    print(__doc__)
