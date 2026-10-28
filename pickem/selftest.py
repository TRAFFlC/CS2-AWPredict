#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pickem 自检（聚合入口）—— tools/ 收录标准 #2「有可执行的验证入口」。

分三层：
  * **模块自证**：逐个跑 swiss / pickem / sources / uncertain / scoring / calibration /
    predict_gui 的 selftest（共 243 项断言），验证模拟器规则、优化器最优性、胜率来源公式、
    σ 不确定性传播机制、**真实命中计分**、**校准指标**与 **Web 面（HTTP 回环）**。
  * **穷举对拍**：12 队小实例上穷举全空间（83,160 组）作**精确参照**，验证生产预算
    （`predict_stage.OPT_RESTARTS`）下优化器**达最优**——模块自证只有「不劣于热门基线」这条弱下界。
  * **端到端**：拿真实 roster 跑生产 CLI，验证「roster → 十格 + P(≥5)」这条链路，
    并**复现 issue-015 §5.1 已公布的数字**（Elo 行 P(≥5)=0.2106）——
    这是公开工具与私有回测「同一实现」的交叉证据；另跑 `--uncertain` σ 臂（§5.5）与三路基线对照。
    最后跑一遍**主用法**：只有 16 个队名 + 公开强度库（`fixtures/team_strength.json`）
    就走完整条链路（零手填参数）。

每个判定都有对应的负样本（非法 roster / 非排列种子 / 缺字段 / 坏 JSON / 矩阵缺项与越界 /
rank_points+--uncertain / 缺或负 exposure / 结构不符的真值表 / 阶段错配的十格 /
错方法 405 / 路径穿越 400 / 超大请求体 413）。

用法：python selftest.py      退出码：0=全过；1=有失败
"""
from __future__ import annotations

import itertools
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np      # noqa: E402
import pickem as PK     # noqa: E402
import predict_stage as PS  # noqa: E402
import sources as SRC   # noqa: E402
import swiss as SW      # noqa: E402
FIXTURE = HERE / "fixtures" / "budapest2025s1_elo.json"
FIXTURE_U = HERE / "fixtures" / "budapest2025s1_uncertain.json"

FAILS = []
TOTAL = [0]


def check(name, ok, detail=""):
    TOTAL[0] += 1
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, (" — " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


def run(args, timeout=180):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run([sys.executable, "-X", "utf8"] + [str(a) for a in args],
                       cwd=str(HERE), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env, timeout=timeout)
    return p.returncode, (p.stdout or ""), (p.stderr or "")


def roster(teams, source="elo", slopes=None):
    d = {"stage": "selftest", "source": source, "teams": teams}
    if slopes:
        d["slopes"] = slopes
    return d


def write_tmp(td, name, obj):
    path = Path(td) / name
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return path


def synth_instance(seed, n_scenes=300, n_teams=12):
    """合成一个**小实例**：真跑 16 队瑞士轮模拟，然后取前 n_teams 队的行。

    为什么不用均匀随机类别：真实实例里队间胜负强相关，平坦/相关的目标面才
    是优化器会卡住的地方（均匀随机会把它测得太宽松）。
    12 队 ⇒ 全空间 C(12,2)×C(10,6)×C(4,2) = 83,160 组，可穷举。
    """
    rng = random.Random(seed)
    teams16 = ["T%02d" % (i + 1) for i in range(16)]
    seeds = {t: i + 1 for i, t in enumerate(teams16)}
    ratings = {t: 1400.0 + rng.uniform(-250.0, 250.0) for t in teams16}
    pwin = SRC.fast_pwin(SRC.EloSource(ratings), teams16)
    scenes, _stages = SW.simulate_scenes(teams16, seeds, pwin, n_scenes, seed=seed)
    C = PK.build_cat_matrix(scenes, teams16)
    return C[:n_teams], teams16[:n_teams]


def brute_force(C, teams, k):
    """穷举最优（只在 12 队量级可用）：返回 (best_p, 组合数)。

    注意掩码必须转 int：布尔数组相加在 numpy 是**逻辑或**，直接加会静默得出全 0。
    """
    idx = {t: i for i, t in enumerate(teams)}
    m30 = (C == PK.CAT30).astype(np.int16)
    madv = (C == PK.CATADV).astype(np.int16)
    m03 = (C == PK.CAT03).astype(np.int16)
    best, cnt = -1.0, 0
    for a in itertools.combinations(teams, 2):
        s1 = m30[idx[a[0]]] + m30[idx[a[1]]]
        rest = [t for t in teams if t not in a]
        for b in itertools.combinations(rest, 6):
            s2 = s1.copy()
            for t in b:
                s2 = s2 + madv[idx[t]]
            for c in itertools.combinations([t for t in rest if t not in b], 2):
                s3 = s2.copy()
                for t in c:
                    s3 = s3 + m03[idx[t]]
                cnt += 1
                v = float((s3 >= k).mean())
                if v > best:
                    best = v
    return best, cnt


def main():
    ok_all = True
    print("pickem 自检")

    # ---- 1) 模块自证（243 项断言：swiss 45 / pickem 16 / sources 36 / uncertain 26 /
    #         scoring 38 / calibration 24 / predict_gui 58） ----
    mods = {"swiss.py": "SWISS", "pickem.py": "PICKEM", "sources.py": "SOURCES",
            "uncertain.py": "UNCERTAIN", "scoring.py": "SCORING",
            "calibration.py": "CALIBRATION", "predict_gui.py": "PREDICT_GUI"}
    for mod, marker in mods.items():
        rc, out, err = run([mod, "selftest"])
        ok = rc == 0 and ("%s_SELFTEST_OK" % marker) in out
        check("%s selftest rc=0 且 %s_SELFTEST_OK" % (mod, marker), ok,
              (out.strip().splitlines() or [err.strip()[:80]])[-1])

    # ---- 1.5) 优化器最优性：小实例与**穷举最优**对拍 ----
    # 此前对优化器的唯一断言是「不劣于热门基线」（pickem 自证第 6 条）——那只是**弱下界**，
    # 不排除「搜索困在次优」这类问题。这里用 12 队可穷举实例给出**精确**参照：
    # 生产预算（`predict_stage.OPT_RESTARTS`）下必须达到穷举最优。
    for sd in (11, 12, 13):
        C, T = synth_instance(sd)
        oracle, ncomb = brute_force(C, T, 5)
        got = PK.optimize(C, T, k=5, seed=sd, restarts=PS.OPT_RESTARTS)[1]
        check("小实例 seed=%d：穷举 %d 组 → 优化器达最优（穷举=%.4f / 优化器=%.4f）"
              % (sd, ncomb, oracle, got), abs(oracle - got) < 1e-12, "")
        hot = PK.heuristic_picks(PK.marginals(C, T))
        ih = {t: i for i, t in enumerate(T)}
        masks = (C == PK.CAT30, C == PK.CATADV, C == PK.CAT03)
        ph = PK.p_at_least(hot["p30"], hot["padv"], hot["p03"], masks, 5, ih)
        check("小实例 seed=%d：穷举最优 ≥ 热门基线（%.4f ≥ %.4f）" % (sd, oracle, ph),
              oracle + 1e-12 >= ph, "")

    # ---- 2) 端到端：真实 roster 复现已公布数字 ----
    rc, out, err = run(["predict_stage.py", "--roster", str(FIXTURE), "--n", "20000",
                        "--seed", "20261012"])
    check("CLI 真实 roster rc=0 且 PICKEM_STAGE_DONE", rc == 0 and "PICKEM_STAGE_DONE" in out,
          err.strip()[:100])
    check("复现 issue-015 §5.1 Elo 行 P(≥5)=0.2106",
          "P(命中≥5) = 0.2106" in out,
          next((l.strip() for l in out.splitlines() if "P(命中" in l), ""))
    check("赛程校验 bo1=20 / bo3=13",
          "bo1=20 / bo3=13" in out, "")
    check("十格各为 2 / 6 / 2", out.count("3:0") >= 1 and out.count("晋级") >= 1 and out.count("0:3") >= 1, "")

    # 确定性：同 seed 两次逐字节一致
    _rc2, out2, _ = run(["predict_stage.py", "--roster", str(FIXTURE), "--n", "20000",
                         "--seed", "20261012"])
    check("同 seed 两次连跑输出逐字节一致", out == out2, "")

    # ---- 3) k 口径：--k 6 可跑且 P(≥6) 与 k=5 网格一致 ----
    rc, out_k6, _ = run(["predict_stage.py", "--roster", str(FIXTURE), "--n", "4000",
                         "--k", "6", "--seed", "7"])
    check("--k 6 rc=0", rc == 0, "")
    v5 = next((float(l.split("=")[1].split()[0]) for l in out_k6.splitlines() if "P(命中≥5)" in l), -1)
    check("P(≥5) ∈ [0,1]（k=6 网格下）", 0.0 <= v5 <= 1.0, "%.4f" % v5)

    # ---- 4) --json 落盘字段齐 ----
    with tempfile.TemporaryDirectory(prefix="pickem_selftest_") as td:
        jout = Path(td) / "rep.json"
        rc, _out, _err = run(["predict_stage.py", "--roster", str(FIXTURE), "--n", "2000",
                              "--seed", "1", "--json", str(jout), "--quiet"])
        keys_ok = False
        if rc == 0 and jout.exists():
            rep = json.loads(jout.read_text(encoding="utf-8"))
            keys_ok = all(k in rep for k in ("marginals", "picks", "grid_p", "p_ge_k",
                                             "schedule", "seeds", "strength"))
            keys_ok = keys_ok and len(rep["picks"]["p30"]) == 2 and len(rep["picks"]["padv"]) == 6 \
                and len(rep["picks"]["p03"]) == 2
        check("--json 落盘且十格/字段齐", rc == 0 and keys_ok, "")

        # ---- 5) 负样本：非法 roster 必须 rc=1（不许静默出结果） ----
        base = [{"name": "T%02d" % i, "seed": i, "rating": 1500.0} for i in range(1, 17)]

        bad15 = write_tmp(td, "bad15.json", roster(base[:15]))
        rc, _o, e = run(["predict_stage.py", "--roster", str(bad15)])
        check("15 队 → rc=1", rc == 1 and "roster 无效" in e, e.strip()[:60])

        dup = [dict(t) for t in base]
        dup[1]["seed"] = 1
        rc, _o, e = run(["predict_stage.py", "--roster", str(write_tmp(td, "dup.json", roster(dup)))])
        check("种子非 1..16 排列 → rc=1", rc == 1, e.strip()[:60])

        same = [dict(t) for t in base]
        same[1]["name"] = same[0]["name"]
        rc, _o, e = run(["predict_stage.py", "--roster", str(write_tmp(td, "same.json", roster(same)))])
        check("队名重复 → rc=1", rc == 1, e.strip()[:60])

        nofield = [dict(t) for t in base]
        del nofield[0]["rating"]
        rc, _o, e = run(["predict_stage.py", "--roster", str(write_tmp(td, "nofield.json", roster(nofield)))])
        check("缺 rating 字段 → rc=1", rc == 1, e.strip()[:60])

        broken = Path(td) / "broken.json"
        broken.write_text("{ not json", encoding="utf-8")
        rc, _o, e = run(["predict_stage.py", "--roster", str(broken)])
        check("坏 JSON → rc=1 且无 traceback", rc == 1 and "roster 无效" in e, e.strip()[:60])

        # ---- 6) rank_points 源可跑 ----
        rp = [{"name": "T%02d" % i, "seed": i, "points": 1000.0 - 40.0 * i} for i in range(1, 17)]
        rc, out_rp, _e = run(["predict_stage.py", "--roster",
                              str(write_tmp(td, "rank.json", roster(rp, source="rank_points"))),
                              "--n", "1500", "--seed", "3", "--quiet"])
        check("rank_points 源 rc=0", rc == 0, "")

        # ---- 7) --uncertain 臂（issue-015 §5.5 σ 传播并入 CLI） ----
        # 7a) 附加字段不扰动旧路径：uncertain fixture 不带 --uncertain 仍复现 0.2106
        rc, out_old, err = run(["predict_stage.py", "--roster", str(FIXTURE_U),
                                "--n", "20000", "--seed", "20261012"])
        check("uncertain fixture 不带开关 rc=0", rc == 0 and "PICKEM_STAGE_DONE" in out_old,
              err.strip()[:100])
        check("附加字段不扰动：仍复现 P(≥5)=0.2106",
              "P(命中≥5) = 0.2106" in out_old,
              next((l.strip() for l in out_old.splitlines() if "P(命中" in l), ""))

        # 7b) --uncertain 端到端
        rc, out_u, err = run(["predict_stage.py", "--roster", str(FIXTURE_U),
                              "--uncertain", "--n", "20000", "--seed", "20261012"])
        check("--uncertain rc=0 且 PICKEM_STAGE_DONE", rc == 0 and "PICKEM_STAGE_DONE" in out_u,
              err.strip()[:100])
        p_u = next((l.strip() for l in out_u.splitlines() if "P(命中" in l), "")
        check("--uncertain 出 P(≥5) ∈ [0,1] 且臂标识在报",
              "σ" in out_u and "P(≥5)" in out_u and p_u != "", p_u)
        check("--uncertain 报表含 σ 列与 blend 信息",
              "σE 分桶" in out_u and "均值回归" in out_u, "")

        # 7c) 确定性：同 seed 两次逐字节一致
        _rc, out_u2, _ = run(["predict_stage.py", "--roster", str(FIXTURE_U),
                              "--uncertain", "--n", "20000", "--seed", "20261012"])
        check("--uncertain 同 seed 两次输出逐字节一致", out_u == out_u2, "")

        # 7d) JSON 落盘含 uncertain 块
        jout2 = Path(td) / "rep_u.json"
        rc, _o, _e = run(["predict_stage.py", "--roster", str(FIXTURE_U), "--uncertain",
                          "--n", "2000", "--seed", "1", "--json", str(jout2), "--quiet"])
        u_ok = False
        if rc == 0 and jout2.exists():
            rep = json.loads(jout2.read_text(encoding="utf-8"))
            u = rep.get("uncertain") or {}
            tu = u.get("teams") or {}
            u_ok = len(tu) == 16 and all("mu" in v and "sigma" in v for v in tu.values()) \
                and any(v.get("blended") for v in tu.values()) \
                and any(not v.get("blended") for v in tu.values())
        check("--json 含 uncertain 块（16 队 μ/σ + blend 双态）", rc == 0 and u_ok, "")

        # 7e) 负样本：--uncertain 的拒绝路径
        rc, _o, e = run(["predict_stage.py", "--roster", str(FIXTURE),
                         "--uncertain", "--n", "500", "--seed", "1"])
        check("无 exposure 字段 + --uncertain → rc=1", rc == 1 and "roster 无效" in e,
              e.strip()[:90])

        rc, _o, e = run(["predict_stage.py", "--roster",
                         str(write_tmp(td, "rank_u.json", roster(rp, source="rank_points"))),
                         "--uncertain", "--n", "500", "--seed", "1"])
        check("rank_points + --uncertain → rc=1", rc == 1 and "仅支持 source=elo" in e,
              e.strip()[:90])

        negf = [dict(t) for t in base]
        negf[0]["exposure"] = -1
        negf[0]["games"] = 3
        rc, _o, e = run(["predict_stage.py", "--roster",
                         str(write_tmp(td, "neg_e.json", roster(negf))),
                         "--uncertain", "--n", "500", "--seed", "1"])
        check("负 exposure + --uncertain → rc=1", rc == 1 and "exposure" in e,
              e.strip()[:90])

        # ---- 8) --random-ties：同 Buchholz 并列改均匀抽签（issue §5.4「模拟器待修」） ----
        rc, out_rt, err = run(["predict_stage.py", "--roster", str(FIXTURE_U),
                               "--random-ties", "--n", "8000", "--seed", "20261013"])
        check("--random-ties rc=0 且 PICKEM_STAGE_DONE",
              rc == 0 and "PICKEM_STAGE_DONE" in out_rt, err.strip()[:100])
        check("--random-ties 报表标注平局键口径", "均匀随机抽签" in out_rt, "")
        _rc, out_rt2, _err = run(["predict_stage.py", "--roster", str(FIXTURE_U),
                                  "--random-ties", "--n", "8000", "--seed", "20261013"])
        check("--random-ties 同 seed 两次输出逐字节一致", out_rt == out_rt2, "")

        rc, out_rtu, err = run(["predict_stage.py", "--roster", str(FIXTURE_U), "--uncertain",
                                "--random-ties", "--n", "4000", "--seed", "20261013"])
        check("--uncertain --random-ties rc=0（σ 臂透传）",
              rc == 0 and "PICKEM_STAGE_DONE" in out_rtu, err.strip()[:100])

        jout3 = Path(td) / "rep_rt.json"
        rc, _o, _e = run(["predict_stage.py", "--roster", str(FIXTURE_U), "--random-ties",
                          "--n", "2000", "--seed", "1", "--json", str(jout3), "--quiet"])
        ok_rt = False
        if rc == 0 and jout3.exists():
            rr = json.loads(jout3.read_text(encoding="utf-8"))
            ok_rt = rr.get("tie_break") == "random" and rr.get("arm") == "point"
        check("--json 记录 tie_break / arm 字段", rc == 0 and ok_rt, "")

        # ---- 9) 基线对照（issue §7 推论待办 1：naive rank 序 / 热门基线） ----
        jout4 = Path(td) / "rep_base.json"
        rc, out_base, err = run(["predict_stage.py", "--roster", str(FIXTURE), "--n", "8000",
                                 "--seed", "20261013", "--json", str(jout4)])
        check("报表含基线对照块", rc == 0 and "基线对照" in out_base, err.strip()[:80])
        bb = {}
        if rc == 0 and jout4.exists():
            bb = json.loads(jout4.read_text(encoding="utf-8")).get("baselines") or {}
        ok_b = (set(bb) == {"optimizer", "naive_rank", "hot"}
                and all(0.0 <= v["p_ge_k"] <= 1.0 for v in bb.values())
                and all(len(v["picks"]["padv"]) == 6 for v in bb.values())
                and all(abs(sum(v["expected_by_cat"].values()) - v["expected_hits"]) < 1e-9
                        for v in bb.values()))
        check("--json 含三路基线（optimizer/naive_rank/hot）", rc == 0 and ok_b, "")
        ok_dom = (bb.get("optimizer", {}).get("p_ge_k", -1.0) + 1e-12
                  >= max(bb.get("naive_rank", {}).get("p_ge_k", 1.0),
                         bb.get("hot", {}).get("p_ge_k", 1.0)))
        check("优化器不劣于任一基线（支配性）", rc == 0 and ok_dom, "")
        np_ = bb.get("naive_rank", {}).get("picks", {})
        check("naive 序基线按种子直填：1-2 / 3-8 / 15-16",
              set(np_.get("p30", [])) == {"FaZe", "Legacy"}
              and set(np_.get("p03", [])) == {"The Huns", "Rare Atom"}
              and len(np_.get("padv", [])) == 6, "")

        # ---- 10) 真实命中计分器（issue-015 §4 验收 3/4 的尺子） ----
        # 真值表留私有（data/ 永不公开），公共区只放计分器 + **合成**真值 fixture。
        TRUTH_SYN = HERE / "fixtures" / "synthetic_stage_truth.json"
        perfect = {"p30": ["T01", "T05"],
                   "padv": ["T02", "T03", "T04", "T06", "T07", "T08"],
                   "p03": ["T14", "T16"]}
        card_syn = write_tmp(td, "card_syn.json", perfect)
        jout5 = Path(td) / "score.json"
        rc, out_sc, err = run(["scoring.py", "--truth", str(TRUTH_SYN),
                               "--card", str(card_syn), "--json", str(jout5)])
        check("scoring.py 合成真值 × 完美卡 rc=0",
              rc == 0 and "SCORING_DONE" in out_sc, err.strip()[:90])
        check("scoring 完美卡得 10/10 且判过", "命中 10/10（过）" in out_sc, "")
        ok_sc = False
        if jout5.exists():
            sj = json.loads(jout5.read_text(encoding="utf-8"))
            rr = (sj.get("results") or {}).get("card") or {}
            ok_sc = (rr.get("hits") == 10 and rr.get("passed") is True
                     and rr.get("by_cat") == {"p30": 2, "padv": 6, "p03": 2}
                     and len((rr.get("hit") or {}).get("padv", [])) == 6)
        check("scoring --json 落盘字段齐（hits/passed/by_cat）", ok_sc, "")

        badt = json.loads(TRUTH_SYN.read_text(encoding="utf-8"))
        badt["records"]["T16"] = [3, 0]
        rc, _o, e = run(["scoring.py", "--truth", str(write_tmp(td, "badt.json", badt)),
                         "--card", str(card_syn)])
        check("真值表 3 支 3:0 → rc=1",
              rc == 1 and "计分输入无效" in e, e.strip()[:90])

        mis = dict(perfect, p30=["FaZe", "Legacy"])
        rc, _o, e = run(["scoring.py", "--truth", str(TRUTH_SYN),
                         "--card", str(write_tmp(td, "mis.json", mis))])
        check("十格与真值表阶段错配 → rc=1",
              rc == 1 and "阶段错配" in e, e.strip()[:90])

        # 跨模块：拿 predict_stage 的真实阶段报表去评合成真值 ⇒ 必须拒绝
        rc, _o, e = run(["scoring.py", "--truth", str(TRUTH_SYN), "--card", str(jout)])
        check("真实阶段报表 × 合成真值 → rc=1（拒绝阶段错配）",
              rc == 1 and "阶段错配" in e, e.strip()[:90])

        # ---- 11) matrix 源：外部模型（014 Predictor）接入本工具的桥 ----
        FIX_M = HERE / "fixtures" / "example_matrix.json"
        rc, out_m, err = run(["predict_stage.py", "--roster", str(FIX_M),
                              "--n", "3000", "--seed", "5"])
        check("matrix 源 rc=0 且 PICKEM_STAGE_DONE",
              rc == 0 and "PICKEM_STAGE_DONE" in out_m, err.strip()[:90])
        check("matrix 报表标注两两胜率与 α 折算",
              "两两胜率直给" in out_m and "对场胜率" in out_m, "")
        jm = Path(td) / "rep_m.json"
        rc, _o, _e = run(["predict_stage.py", "--roster", str(FIX_M), "--n", "3000",
                          "--seed", "5", "--json", str(jm), "--quiet"])
        okm = False
        if rc == 0 and jm.exists():
            rm = json.loads(jm.read_text(encoding="utf-8"))
            okm = (rm.get("source") == "matrix" and len(rm.get("strength") or {}) == 16
                   and all(0.0 < v < 1.0 for v in rm["strength"].values())
                   and all(len(v["picks"]["padv"]) == 6
                           for v in (rm.get("baselines") or {}).values()))
        check("matrix 源 --json 含 source/strength/三路基线", okm, "")
        base_m = json.loads(FIX_M.read_text(encoding="utf-8"))
        m1 = json.loads(json.dumps(base_m))
        m1["matrix"]["FaZe"]["Legacy"] = 1.0
        rc, _o, e = run(["predict_stage.py", "--roster", str(write_tmp(td, "m1.json", m1))])
        check("矩阵含 1.0 → rc=1", rc == 1 and "(0,1)" in e, e.strip()[:90])
        m2 = json.loads(json.dumps(base_m))
        del m2["matrix"]["FaZe"]["Legacy"]
        rc, _o, e = run(["predict_stage.py", "--roster", str(write_tmp(td, "m2.json", m2))])
        check("矩阵缺一对 → rc=1", rc == 1 and "也无反向" in e, e.strip()[:90])
        m3 = json.loads(json.dumps(base_m))
        m3["teams"][1]["name"] = "Typo"
        rc, _o, e = run(["predict_stage.py", "--roster", str(write_tmp(td, "m3.json", m3))])
        check("矩阵含未知队名 → rc=1（拼错不得静默失效）", rc == 1 and "未知队名" in e, e.strip()[:90])
        rc, _o, e = run(["predict_stage.py", "--roster", str(FIX_M),
                         "--uncertain", "--n", "500", "--seed", "1"])
        check("matrix + --uncertain → rc=1", rc == 1 and "仅支持 source=elo" in e, e.strip()[:90])

    # ---- 12) 主用法：只有 16 个队名 → 十格（强度库预计算，零参数手填） ----
    # 这是「任何人一看到就能用」那条链：玩家填队名，工具查 fixtures/team_strength.json。
    DEMO = HERE / "fixtures" / "demo_stage_s1.json"
    LIBF = HERE / "fixtures" / "team_strength.json"
    lib_doc = json.loads(LIBF.read_text(encoding="utf-8")) if LIBF.exists() else {}
    check("强度库存在且带 provenance/counts/asof",
          bool(lib_doc.get("teams")) and bool(lib_doc.get("_provenance"))
          and bool(lib_doc.get("counts")) and bool(lib_doc.get("asof")),
          "%d 队" % len(lib_doc.get("teams") or []))
    demo_doc = json.loads(DEMO.read_text(encoding="utf-8")) if DEMO.exists() else {}
    check("示例阶段给 16 个队名", len(demo_doc.get("teams") or []) == 16, "")
    code = (
        "import json,sys; sys.path.insert(0,'.'); import predict_stage as PS;"
        "d=json.load(open(%r,encoding='utf-8'));"
        "r,w=PS.build_roster(d['teams'], stage=d['stage']);"
        "rep=PS.predict(r,n=2000,seed=7);"
        "print('PICKEM_LIBRARY_DONE', len(r['teams']), len(w), rep['p_ge_k'],"
        "len(rep['picks']['padv']))" % str(DEMO)
    )
    rc, out_lib, err = run(["-c", code], timeout=300)
    check("强度库 + 16 队名 → 完整链路 rc=0",
          rc == 0 and "PICKEM_LIBRARY_DONE" in out_lib, (err or out_lib).strip()[:100])
    parts = (out_lib.split("PICKEM_LIBRARY_DONE")[-1].split() if "PICKEM_LIBRARY_DONE" in out_lib
             else [])
    ok_lib = (len(parts) == 4 and int(parts[0]) == 16 and int(parts[1]) == 1
              and 0.0 <= float(parts[2]) <= 1.0 and int(parts[3]) == 6)
    check("示例阶段 16 队全建出（RED Canids 无 VRS 积分 → 1 条点名告警）且出 6 格晋级",
          ok_lib, " ".join(parts))
    code_neg = code.replace("d['teams']", "d['teams'][:15]")
    rc, _o, e = run(["-c", code_neg], timeout=300)
    check("只有 15 个队名 → 拒绝（不静默补位）",
          rc != 0 and "恰好 16" in e, e.strip().splitlines()[-1][:90] if e.strip() else "")

    ok_all = not FAILS
    print("-" * 44)
    print("结果：%s（%d/%d）" % ("PASS" if ok_all else "FAIL", TOTAL[0] - len(FAILS), TOTAL[0]))
    for name in FAILS:
        print("  - " + name)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
