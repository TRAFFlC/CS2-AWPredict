# -*- coding: utf-8 -*-
"""exp_mapinfo.py —— 2026-10-11（会话三·续）：**选手级图偏好**能不能给第 1 项加分？

信息源：csapi `/matches/{id}/stats` 的 **`by_map`** —— 每场每个打过的图都有逐选手 rating。
此前 `model_v2.load_stats()` 只读 `overall`，把这块**整块丢掉**（本脚本是首次使用）。

本脚本只读语料 + 私有区，输出三件事：
  1. **覆盖率诊断**：预测时点上有多少 (队, 图) 单元有观测（还有多少是「回退到总体水平」）；
  2. **选型网格**：把新特征组放进 `model_v2.WIN_FEATURE_SETS`，用 **train 的扩展窗口 CV** 选，
     再用 **train+valid CV** 复核（与 2026-10-11 笔记 §3.2 同一套协议）；
  3. **图级信号**：只用**过去**的图偏好判断「某张已打的图谁赢」，单独测它的方向准确率
     （≈50% 就说明这份信息对图结果没有增益）。

用法：python -X utf8 exp_mapinfo.py
"""
import io
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)  # scorer.py 与本模块同目录（仓库布局）
os.chdir(HERE)

import model_v2 as M  # noqa: E402

NEW = ["...+poolperf", "...+poolperf+best", "all13+perf"]


def main():
    prep = M.prepare(verbose=False)
    data = prep["data"]
    print("split", prep["meta"]["counts"])

    # ---------- 1. 覆盖率 ----------
    for bucket in ("train", "valid", "test"):
        cells = len(data[bucket]) and sum(len(s.get("perf") or {}) for s in data[bucket])
        n_pool = sum(1 for s in data[bucket] for _m in (s.get("perf") or {}))
        nz_pool = sum(1 for s in data[bucket] if abs(s["feats"]["d_poolperf"]) > 1e-12)
        nz_best = sum(1 for s in data[bucket] if abs(s["feats"]["d_bestperf"]) > 1e-12)
        nz_sprd = sum(1 for s in data[bucket] if abs(s["feats"]["d_perfspread"]) > 1e-12)
        n = max(1, len(data[bucket]))
        print("  %-5s 场次 %4d  池内(队,图)单元 %6d  d_poolperf 非零 %4d (%.0f%%)  "
              "d_bestperf %4d  d_perfspread %4d"
              % (bucket, len(data[bucket]), cells, nz_pool, 100.0 * nz_pool / n, nz_best, nz_sprd))

    # ---------- 2. 选型网格（train 扩展窗口 CV）----------
    print("\n== train 扩展窗口 CV（选型用）==")
    rows = []
    for fname, feats in M.WIN_FEATURE_SETS:
        for C in (0.02, 0.05, 0.1, 0.3, 1.0):
            rows.append((M.cv_acc(data["train"], feats, C), fname, C, len(feats)))
    rows.sort(key=lambda r: (-r[0], r[3], r[2]))
    print("  %-24s %-6s %-4s %s" % ("特征组", "C", "#f", "cv(train)"))
    for cv, fname, C, nf in rows[:10]:
        print("  %-24s %-6s %-4d %.4f" % (fname, C, nf, cv))
    print("  --- 新特征组 ---")
    for cv, fname, C, nf in rows:
        if fname in NEW:
            print("  %-24s %-6s %-4d %.4f" % (fname, C, nf, cv))

    def best_of(name):
        r = [x for x in rows if x[1] == name]
        return max(r) if r else None

    champ = best_of("...+games")
    print("\n  [对照] 现行选定 %s C=%s cv=%.4f" % (champ[1], champ[2], champ[0]))
    for nm in NEW:
        b = best_of(nm)
        print("  [新]   %-18s C=%-5s cv=%.4f  Δ=%+.4f" % (nm, b[2], b[0], b[0] - champ[0]))

    # ---------- 3. train+valid CV 复核 ----------
    print("\n== train+valid 扩展窗口 CV（复核用，n=%d）==" % (len(data["train"]) + len(data["valid"])))
    both = data["train"] + data["valid"]
    checks = [("...+games", champ[2]), ("...+games", 0.05), ("...+form+rest+games", 0.02)]
    for nm in NEW:
        checks.append((nm, best_of(nm)[2]))
    for fname, feats in M.WIN_FEATURE_SETS:
        if fname in [c[0] for c in checks]:
            for C in [c[1] for c in checks if c[0] == fname]:
                print("  %-24s C=%-6s cv(train+val)=%.4f" % (fname, C, M.cv_acc(both, feats, C)))

    # ---------- 4. 图级信号（只用过去的图偏好）----------
    print("%s== 图级方向准确率（sign(map_perf 差) 预测该图谁赢）==" % chr(10))
    rows = M.SPLIT.load(M.MATCHES)
    mapwin = {}
    for r in rows:
        t1 = r["team1"]["name"]
        t2 = r["team2"]["name"]
        for m in (r.get("maps") or []):
            s1, s2 = m.get("team1_score"), m.get("team2_score")
            if s1 is None or s2 is None or s1 == s2:
                continue
            mapwin[(r["id"], m["name"])] = t1 if s1 > s2 else t2
    for bucket in ("valid", "test"):
        hit = tot = miss_data = 0
        hit_elo = tot_elo = 0
        for s in data[bucket]:
            t = s.get("truth") or {}
            perf = s.get("perf") or {}
            for m in (t.get("maps_played") or []):
                w = mapwin.get((s["id"], m))
                if w is None or m not in perf:
                    miss_data += 1
                    continue
                pa, pb = perf[m]
                d = pa - pb
                d_elo = s["feats"]["d_elo"]
                if d == 0:
                    miss_data += 1
                    continue
                pred = s["teamA"] if d > 0 else s["teamB"]
                hit += (pred == w)
                tot += 1
                # 同子集上的**对照**：系列级 Elo 方向的符号（不含新信息）
                if abs(d_elo) > 1e-12:
                    pred_e = s["teamA"] if d_elo > 0 else s["teamB"]
                    hit_elo += (pred_e == w)
                    tot_elo += 1
        if tot:
            se = math.sqrt(0.25 / tot)
            print("  %-5s n=%4d  图偏好 %.1f%%   ／ 同子集 Elo 方向 %.1f%%（n=%d）  "
                  "（±%.1f 个 1SE；50%% 为无信息）  缺数据 %d"
                  % (bucket, tot, 100.0 * hit / tot,
                     100.0 * hit_elo / max(1, tot_elo), tot_elo, 100.0 * se, miss_data))
        else:
            print("  %-5s 无可判样本（缺逐图胜方）" % bucket)


if __name__ == "__main__":
    main()
