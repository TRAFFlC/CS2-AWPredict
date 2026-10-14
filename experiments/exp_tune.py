"""三项调优实验（依据 43 场真实盲预测的失分结构）。

失分结构（41 场，模型口径 845/1865 = 45.31%）：
  胜方 250/410 (61.0%)   可回收 160 分
  比分 160/410 (39.0%)   可回收 250 分
  选图 385/615 (62.6%)   可回收 230 分   ← 唯一有信号的项
  MVP   50/410 (12.2%)   可回收 360 分   ← 但低于随机 20%

实验 A：胜方项 —— 模型 / rank / **概率融合**
  先把 rank 转成概率 p_rank = σ(a·(rank_B − rank_A) + b)，**只在 train 上拟合**，
  再做 p = w·p_model + (1−w)·p_rank，在 valid 上选 w，在 test 上只报一次。
  泄漏说明：rank 是抓取时的快照，理论上可能含本场结果的信息；
  但它对「模型 / rank / 融合」三方是**同等条件**，横向可比。

实验 B：MVP 项 —— argmax / 次选 / 末选
  实测 argmax 只有 12~17%，随机基线 20%。若 argmax 真的劣于随机，
  那么「次选」或任何非 argmax 的规则都该更好。

实验 C：选图项 —— 当前 vs 加权
  这是唯一有信号的项，看还有没有可挖的空间。
"""
import io, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.0）
sys.path.insert(0, HERE)
import model_v2 as M

from sklearn.linear_model import LogisticRegression


def load_rank():
    rk = {}
    for l in io.open(os.path.join(DATA, "csapi_matches_all.jsonl"), encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if r["team1"].get("rank") and r["team2"].get("rank"):
            rk[str(r["id"])] = (int(r["team1"]["rank"]), int(r["team2"]["rank"]))
    return rk


def acc(probs, ys):
    p = np.asarray(probs)
    return float(np.mean((p >= 0.5) == (np.asarray(ys) >= 0.5)))


def brier(probs, ys):
    return float(np.mean((np.asarray(probs) - np.asarray(ys)) ** 2))


def main():
    rk = load_rank()
    prep = M.prepare(verbose=False)
    data = prep["data"]
    print("=" * 72)
    print("实验 A：胜方项 —— 模型 / rank / 融合")
    print("=" * 72)

    def xy(split):
        X, Y = [], []
        for t in data[split]:
            v = rk.get(str(t.get("id")))
            if not v:
                continue
            X.append(v[1] - v[0])           # rank 差：正 = A 队排名数字更大（更弱）
            Y.append(float(t["y"]))
        return np.array(X), np.array(Y)

    # p_rank 只用 train 拟合
    Xtr, Ytr = xy("train")
    lr = LogisticRegression(C=1e6, max_iter=3000).fit(Xtr.reshape(-1, 1), Ytr)
    print("  p_rank = σ(%.3f·(rank_B − rank_A) %+.3f)" % (lr.coef_[0][0], lr.intercept_[0]))

    def probs(split, model):
        P = []
        for t in data[split]:
            v = rk.get(str(t.get("id")))
            if not v:
                continue
            pm = M.predict_proba_win(model, t)
            pr = float(lr.predict_proba(np.array([[v[1] - v[0]]]))[0][1])
            P.append((pm, pr, float(t["y"])))
        return P

    tr = probs("train", prep["wmodel"])
    va = probs("valid", prep["wsel"])
    te = probs("test", prep["wmodel"])
    print("\n  n: train=%d valid=%d test=%d\n" % (len(tr), len(va), len(te)))
    print("  %-16s %10s %10s %10s" % ("方案", "valid", "test", "test Brier"))
    print("  " + "-" * 50)

    def report(tag, f):
        a_v = acc([f(a, b) for a, b, _ in va], [y for _, _, y in va])
        a_t = acc([f(a, b) for a, b, _ in te], [y for _, _, y in te])
        b_t = brier([f(a, b) for a, b, _ in te], [y for _, _, y in te])
        print("  %-16s %9.1f%% %9.1f%% %10.4f" % (tag, 100 * a_v, 100 * a_t, b_t))
        return a_t, b_t

    report("模型", lambda m, r: m)
    report("rank", lambda m, r: r)
    best = None
    rows = []
    for w in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
        f = (lambda ww: (lambda m, r: ww * m + (1 - ww) * r))(w)
        a_v = acc([f(a, b) for a, b, _ in va], [y for _, _, y in va])
        a_t = acc([f(a, b) for a, b, _ in te], [y for _, _, y in te])
        rows.append((w, a_v, a_t))
        print("  融合 w=%.1f      %9.1f%% %9.1f%%" % (w, 100 * a_v, 100 * a_t))
    bw = max(rows, key=lambda z: z[1])
    print("\n  valid 最优 w=%.1f ⇒ test %.1f%%（对比模型 %.1f%%、rank %.1f%%）"
          % (bw[0], 100 * bw[2], 100 * report.__defaults__ if False else 0, 0))
    base_t = acc([a for a, _, _ in te], [y for _, _, y in te])
    rank_t = acc([b for _, b, _ in te], [y for _, _, y in te])
    print("  模型 test %.1f%%   rank test %.1f%%   ⇒ 融合 %+.1fpp vs 模型"
          % (100 * base_t, 100 * rank_t, 100 * (bw[2] - base_t)))

    print("\n" + "=" * 72)
    print("实验 B：MVP 项选择规则（语料 test）")
    print("=" * 72)
    import decide as D
    pr = None
    try:
        import predict as PR
        pr = PR.Predictor(quiet=True)
    except Exception as e:                                    # noqa: BLE001
        print("  跳过：%s" % e)
    if pr is not None:
        cnt = {"argmax": 0, "2nd": 0, "last": 0}
        tot = 0
        for t in data["test"]:
            s = t
            w = s["winner"] if s.get("winner") in (s.get("teamA"), s.get("teamB")) else None
            if not w:
                continue
            rows_p = (s.get("player_rows") or {}).get(w) or []
            if len(rows_p) < 2:
                continue
            mp = D.mvp_probs(rows_p)
            if not mp:
                continue
            order = sorted(mp, key=lambda k: -mp[k])
            tv = s.get("mvp_truth", "")
            cnt["argmax"] += int(order[0] == tv)
            cnt["2nd"] += int(order[1] == tv)
            cnt["last"] += int(order[-1] == tv)
            tot += 1
        print("  n=%d  候选池=胜方选手数" % tot)
        for k in ("argmax", "2nd", "last"):
            print("    %-6s %3d/%d = %5.1f%%" % (k, cnt[k], tot, 100 * cnt[k] / max(1, tot)))
        print("    随机基线 ≈ 20%%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
