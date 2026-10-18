"""量一件事：模型的选图概率有没有**超过逻辑上界**。

## 为什么问这个而不是「改动能不能提高总分」

2026-07-19 BetBoom/FURIA 那场 bo5：模型给 Inferno 0.268 并排进 5 张预测，
实际 BetBoom 首 ban 了它 ⇒ 白丢 5 分。主理人指出「所有人都知道 BetBoom 不打
Inferno，只有模型认为会」。

这句话指向的不是「准确率差 2.5%」，而是**模型断言了一个逻辑上为假的事**：

    P(这张图进读法B图集) ≤ P(至少有一支队会打它)
                       ≤ 1 − (1 − 队A上场率)(1 − 队B上场率)

BetBoom 对 Inferno 的上场率 = **0.001**（账本实测），FURIA = 0.118
⇒ 上界 ≈ 0.119。**模型说 0.268 = 上界的 2.2 倍。**

这不是「偏了一点」，是**越界**。所以该测的是**越界率**，不是分数增量。
分数是有噪的统计量；「声称一支从不打某图的队会打它」是可直接判真假的事实性错误。

用法：python -X utf8 exp_map_bounds.py
"""
import collections
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）
sys.path.insert(0, HERE)
import veto_model as VM


def bound_of(row, g_app):
    """P(至少一队会打这张图) 的保守上界（两队上场率独立近似）。"""
    a, b = row.get("a_app", 0.0), row.get("b_app", 0.0)
    a = min(1.0, max(0.0, a))
    b = min(1.0, max(0.0, b))
    return 1.0 - (1.0 - a) * (1.0 - b)


def main():
    rows = [json.loads(l) for l in
            io.open(os.path.join(DATA, "csapi_matches_all.jsonl"), encoding="utf-8")
            if l.strip()][:400]
    print("取语料前 %d 场建 MapHistory（走 observe_match，账本即真实历史）\n" % len(rows))

    POOL7 = ["Ancient", "Anubis", "Cache", "Dust2", "Inferno", "Mirage", "Nuke"]
    TRUTHS = {t["id"]: t for t in
              (json.loads(l) for l in io.open(
                  os.path.join(DATA, "truth_maps_B.jsonl"),
                  encoding="utf-8") if l.strip())}
    hist = VM.MapHistory()
    for d in sorted({r["date"] for r in rows}):
        hist.advance(d)
    for r in rows:
        # csapi 的 maps 是 [{name,..},..] ⇒ 取名字；读法B 图集优先（更贴近计分口径）
        t = TRUTHS.get(r["id"]) or {}
        maps = t.get("maps") or [m.get("name") for m in (r.get("maps") or [])]
        hist.observe_match(r["team1"]["name"], r["team2"]["name"],
                           POOL7, [m for m in maps if m in POOL7])

    viol, tot, worst = [], 0, []
    for r in rows[:len(rows) - 50]:          # 留出 50 场给历史积累
        tA, tB = r["team1"]["name"], r["team2"]["name"]
        feats = hist.features_for(tA, tB, POOL7, 3, date=r["date"])
        for f in feats:
            bnd = bound_of(f, hist.g_app(f["map"]))
            tot += 1
            # 「越界」的判据：用**某队几乎不打**作为硬事实（阈值 5%）
            weak = min(f.get("a_app", 1.0), f.get("b_app", 1.0))
            if weak < 0.05 and bnd < 0.15:
                worst.append((tA, tB, f["map"], weak, bnd,
                              f.get("a_pick", 0), f.get("b_pick", 0)))

    print("样本 %d 个 (场×图) 对" % tot)
    print("其中「至少一队上场率 < 5%% ⇒ 上界 < 15%%」的极端情形：%d 个" % len(worst))
    if worst:
        print("\n%-22s %-22s %-9s %8s %8s %8s %8s"
              % ("队A", "队B", "图", "a_app", "b_app", "上界", "min_pick"))
        for tA, tB, m, weak, bnd, ap, bp in worst[:12]:
            print("%-22s %-22s %-9s %8.3f %8s %8.3f %8.3f"
                  % (tA, tB, m, weak, "-", bnd, min(ap, bp)))
        bymap = collections.Counter(w[2] for w in worst)
        print("\n按图统计：%s" % dict(bymap))
        print("\n⇒ 这些图**任一队都几乎不打**，其进图集的概率应当被硬性压到 ~0；")
        print("   模型仍给它们排进 top-5，就是白丢分。")

    # 用真实训练好的模型看 BetBoom/FURIA 那场的具体越界幅度
    print("\n--- 具体案例：BetBoom vs FURIA bo5（主理人报的那场）---")
    import predict as PR
    pr = PR.Predictor(quiet=True)
    s = pr.live_sample("BetBoom", "FURIA", 5, None)
    p = pr.predict("BetBoom", "FURIA", bo=5)
    for f in sorted(s["map_rows"], key=lambda r: -p["map_probs"].get(r["map"], 0)):
        m = f["map"]
        bnd = bound_of(f, 0.0)
        a_app = f.get("a_app", 0.0) if s["teamA"] == "BetBoom" else f.get("b_app", 0.0)
        b_app = f.get("b_app", 0.0) if s["teamA"] == "BetBoom" else f.get("a_app", 0.0)
        bnd = 1.0 - (1.0 - min(1, a_app)) * (1.0 - min(1, b_app))
        pr_model = p["map_probs"].get(m, 0.0)
        flag = "  ← 越界！" if pr_model > bnd + 1e-9 else ""
        print("  %-9s 模型 %5.1f%%   上界 %5.1f%%  (BetBoom %.3f / FURIA %.3f)%s"
              % (m, 100 * pr_model, 100 * bnd, a_app, b_app, flag))


if __name__ == "__main__":
    sys.exit(main())