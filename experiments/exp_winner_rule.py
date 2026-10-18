"""胜方项三种取法的无泄漏对照：模型 / rank / 融合。

背景修正（2026-10-12）：
  csapi 的 `rank` 是**抓取时（≈2026-09-05）的单一快照**，对全部 2,800 场冻结不变
  （实测 179 支队中 rank 随月份变化的 = 0 支）。
  ⇒ 语料上「rank 70.3% vs 模型 67.2%」**无效**：旧比赛被未来实力标注了。
  ⇒ 只有 **43 场真实对局（09-08 ~ 09-20，快照仅领先 3~15 天）** 的对照是无泄漏的。

p_rank 的构造不能用全语料拟合的 logistic（实测系数仅 0.002，近乎常数，
等于没区分度）。改为**只用语料最近的一段拟合**——那里快照 rank 才真正新鲜，
这也正是线上使用时的真实情形。

三种取法：
  模型      p = p_model
  rank      p = p_rank
  融合      p = w·p_model + (1−w)·p_rank
另外加一条**降级规则**：模型置信度极高时允许覆盖 rank（单边检验）。
"""
import io, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.0）
sys.path.insert(0, HERE)

from sklearn.linear_model import LogisticRegression

BO5_KEY = ("Vitality", "Aurora")
DAYS = ("2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-12",
        "2026-09-13", "2026-09-17", "2026-09-18", "2026-09-19", "2026-09-20")


def load_real():
    res = []
    for d in DAYS:
        p = os.path.join(HERE, "fissure_%s_results.json" % d)
        if not os.path.exists(p):
            continue
        for x in json.load(io.open(p, encoding="utf-8")):
            if (x["A"], x["B"]) == BO5_KEY:
                continue
            res.append(x)
    p = os.path.join(DATA, "groundtruth", "fissure_2026-09-20-bo5_results.json")
    if os.path.exists(p):
        res += json.load(io.open(p, encoding="utf-8"))
    return res


def fit_p_rank(recent_n=400):
    """只用语料最近 recent_n 场拟合 p_rank = σ(a·(rank_A − rank_B) + b)。"""
    rk = {}
    for l in io.open(os.path.join(DATA, "csapi_matches_all.jsonl"), encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if r["team1"].get("rank") and r["team2"].get("rank"):
            rk[str(r["id"])] = (int(r["team1"]["rank"]), int(r["team2"]["rank"]))
    rows = []
    for l in io.open(os.path.join(HERE, "records_v1.jsonl"), encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        v = rk.get(str(r["id"]))
        if not v:
            continue
        # y = team1 是否获胜（records 里 winner 是队名）
        y = 1.0 if r["winner"] == r["team1"]["name"] else 0.0
        rows.append((str(r["date"]), v[0] - v[1], y))
    rows.sort(key=lambda z: z[0])
    rows = rows[-recent_n:]
    X = np.array([r[1] for r in rows]).reshape(-1, 1)
    Y = np.array([r[2] for r in rows])
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(X, Y)
    print("  p_rank 用语料最近 %d 场拟合： σ(%.4f·(rank_A−rank_B) %+.3f)"
          % (len(rows), lr.coef_[0][0], lr.intercept_[0]))
    print("  训练段 rank 热门准确率 %.1f%%"
          % (100 * np.mean([(r[1] < 0) == (r[2] > 0.5) for r in rows])))
    return lr


def main():
    lr = fit_p_rank()
    res = load_real()
    print("\n=== 43 场真实对局（无泄漏）· 三种取法 ===\n")

    def prank(x):
        return float(lr.predict_proba(np.array([[x["my_rank"]["A"] - x["my_rank"]["B"]]]))[0][1])

    def score(pick):
        return sum(1 for x in res if pick(x) == x["winner"])

    variants = [
        ("模型", lambda x: x["my_winner"]),
        ("rank", lambda x: x["A"] if x["my_rank"]["A"] < x["my_rank"]["B"] else x["B"]),
    ]
    for w in (0.5, 0.4, 0.3, 0.2):
        variants.append(("融合 w=%.1f" % w,
                         (lambda ww: lambda x: x["A"] if (ww * x["my_p"] / 100.0
                                    + (1 - ww) * prank(x)) > 0.5 else x["B"])(w)))
    for thr in (0.70, 0.75, 0.80):
        variants.append(("模型>%.2f 覆盖" % thr,
                         (lambda tt: lambda x: x["my_winner"]
                          if x["my_p"] / 100.0 > tt
                          else (x["A"] if x["my_rank"]["A"] < x["my_rank"]["B"] else x["B"]))(thr)))

    print("  %-18s %10s %10s" % ("规则", "胜方命中", "准确率"))
    print("  " + "-" * 42)
    for tag, f in variants:
        k = score(f)
        print("  %-18s %7d/43 %9.1f%%" % (tag, k, 100 * k / len(res)))

    # 概率层面的比较（不是只看 argmax）
    pm = np.array([x["my_p"] / 100.0 if x["my_winner"] == x["A"]
                   else 1 - x["my_p"] / 100.0 for x in res])
    pr = np.array([prank(x) if x["A"] == (x["A"] if x["my_rank"]["A"] < x["my_rank"]["B"]
                                          else x["B"]) else 1 - prank(x) for x in res])
    y = np.array([1.0 if x["my_winner"] == x["winner"] else 0.0 for x in res])
    print("\n  概率校准（Brier，越小越好）：")
    print("    模型   %.4f" % np.mean((pm - y) ** 2))
    print("    rank   %.4f" % np.mean((pr - y) ** 2))
    for w in (0.5, 0.4, 0.3):
        print("    融合%.1f %.4f" % (w, np.mean((w * pm + (1 - w) * pr - y) ** 2)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
