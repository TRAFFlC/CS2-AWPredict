"""MVP 标定修复后的 exp_points 偏差复核。

问题：`decide.py:points_dist` 曾把 `mvp_probs` 的原始值（≈0.96）计入 MVP 项期望，
而 70 场实测命中率只有 0.171（随机基线 0.20）⇒ 单场 E[分] 系统性虚高。
修复：MVP_CAL_PROB = 0.20（选谁都一样）。

本脚本在**单个 Predictor 进程**内跑完全部场次（每次 subprocess 重新建模太慢），
对比「修复后的 E[分]」与「真实得分」，看偏差是否收敛。
"""
import io, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decide as D
import predict as PR

DAYS = ("2026-09-08", "2026-09-09", "2026-09-10",
        "2026-09-11", "2026-09-12", "2026-09-13")


def load():
    res = []
    for d in DAYS:
        # 09-08 必须用**重新生成**的 fissure_2026-09-08_results.json：
        # 旧的 fissure_s3_results.json 里 my_score 是字符串 "2:0"，
        # tuple() 后成 ('2','0') ⇒ 比分项恒 0 ⇒ E[分] 假性偏低 7~11。
        f = "fissure_%s_results.json" % d
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            res += json.load(io.open(p, encoding="utf-8"))
    return res


def main():
    res = load()
    pr = PR.Predictor(quiet=True)
    print("MVP_CAL_PROB = %.2f（随机基线，胜方 5 选 1）\n" % D.MVP_CAL_PROB)
    tot_new = 0.0
    tot_old = 0.0
    rows = []
    for x in res:
        bo = 5 if len(x["maps"]) == 5 else 3
        try:
            s = pr.live_sample(x["A"], x["B"], bo, x["maps"][0] and None)
        except Exception as e:                                  # noqa: BLE001
            print("  跳过 %s vs %s: %s" % (x["A"], x["B"], e))
            continue
        if not s:
            print("  跳过 %s vs %s: 无样本" % (x["A"], x["B"]))
            continue
        md = D.MatchDist(s, pr.mmodel, pr.smodel)
        pl = {"winner": x["my_winner"],
              "score": tuple(x["my_score"]),
              "maps": tuple(x["my_maps"]),
              "mvp": x["my_mvp"]}
        # 修复后
        mu_new = md.moments(pl)[0]
        # 修复前（还原原始 MVP 期望）
        saved = D.MVP_CAL_PROB
        try:
            mh = md.mvp_true[pl["winner"]].get(pl["mvp"], 0.0)
            q = md.p_win[pl["winner"]]
            base_new = mu_new
            mu_old = mu_new - q * saved * 10 + q * mh * 10
        finally:
            pass
        tot_new += mu_new
        tot_old += mu_old
        rows.append((x["A"], x["B"], mu_old, mu_new, x["total"]))
    n = len(rows)
    act = sum(r[4] for r in rows)
    mx = sum(55 if len(x["maps"]) == 5 else 45 for x in res)
    print("  修复前 E[分] 合计 %.1f  ⇒ 偏差 %+.1f 分 (%+.1fpp)"
          % (tot_old, tot_old - act, 100 * (tot_old - act) / mx))
    print("  修复后 E[分] 合计 %.1f  ⇒ 偏差 %+.1f 分 (%+.1fpp)"
          % (tot_new, tot_new - act, 100 * (tot_new - act) / mx))
    print("  实际得分       %.1f / %d" % (act, mx))
    print("\n  逐场（修复前 → 修复后 → 实际）")
    for A, B, o, nn, t in rows:
        print("    %-11s %-11s %5.1f → %5.1f  | 实际 %3d  差 %+5.1f"
              % (A, B, o, nn, t, nn - t))
    eo = np.array([r[2] for r in rows])
    en = np.array([r[3] for r in rows])
    aa = np.array([r[4] for r in rows])
    print("\n  平均绝对误差：修复前 %.2f  →  修复后 %.2f"
          % (np.mean(np.abs(eo - aa)), np.mean(np.abs(en - aa))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
