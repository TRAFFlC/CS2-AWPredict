"""实验：把「对手在该图的池内弱度排名」做成显式特征，选图项能否受益？

依据 H1（时序安全，n=8,609 个 veto 动作）：pick 的「对手在该图胜率百分位」= 0.4299
（z = −11.5），而 pick 的「自身频率百分位」= 0.4896（z = −2.0，无效应）。
但已训练模型里 `b_wr` 只有 +0.0860（19/39），且与 app/ban/pick/rank 家族共线
⇒ 单系数不能证明边际效应，需要做**定向实验**。

做法：**只 monkey-patch** `MapHistory.features` 追加两列，不改 veto_model.py 源码：
    a_wr_rank  该队在池内按「该图胜率」的名次（1 = 最强）
    b_wr_rank  对手在池内按「该图胜率」的名次（1 = 对手最强）
并加一个定向交互项 `opp_weak` = -b_wr_rank（对手在该图越弱 → 越该被选）。

判据：**valid 上的选图得分**（与 veto_model.run 同一口径，同一 walk-forward 构造）。
test 只在 valid 明确涨且超过噪声时才报一次。
"""
import io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import veto_model as VM
import split as SPLIT
import model_v2 as M
import pool_experiment as PE

EXTRA = ["a_wr_rank", "b_wr_rank", "opp_weak", "a_wr_gap"]
_orig_features = VM.MapHistory.features


def features_ext(self, teamA, teamB, pool, bo=3, first=None):
    rows = _orig_features(self, teamA, teamB, pool, bo, first)
    ra = self._pool_ranks(teamA, pool, lambda t, m: self.team_wr(t, m))
    rb = self._pool_ranks(teamB, pool, lambda t, m: self.team_wr(t, m))
    for r in rows:
        m = r["map"]
        a, b = ra[m], rb[m]
        r["a_wr_rank"] = a
        r["b_wr_rank"] = b
        r["opp_weak"] = -b                      # 对手越弱（rank 越大）→ 值越大
        r["a_wr_gap"] = a - b                   # 己方强于对手的程度
    return rows


def build_all():
    rows = SPLIT.load(VM.MATCHES)
    exc = SPLIT.load_exclude(VM.EXCLUDE) if os.path.exists(VM.EXCLUDE) else None
    buckets, _ = SPLIT.split(rows, purge=True, exclude=exc)
    truths = {t["id"]: t for t in VM.load_jsonl(VM.TRUTH)}
    veto = VM.load_veto()
    eras, by_pool = PE.load_pool_eras()
    tl = json.load(io.open(PE.TIMELINE, encoding="utf-8"))

    def pool_of(r):
        t = truths.get(r["id"]) or {}
        if t.get("pool"):
            return list(t["pool"])
        return list(PE.era_and_pool(r["date"], [], eras, by_pool, tl)[1])

    hist = VM.MapHistory()
    out = {}
    for b in ("train", "valid"):
        out[b] = VM.build_dataset(buckets[b], truths, veto, hist, pool_of)
    return out


def run_arm(name, feats, data, Cs=(0.03, 0.1, 0.3, 1.0, 3.0)):
    best = None
    for C in Cs:
        m = VM.train(data["train"], features=feats, C=C)
        ev = VM.evaluate(m, data["valid"])
        if best is None or ev["points"] > best[1]["points"]:
            best = (C, ev, m)
    C, ev, m = best
    print("  %-26s C=%-5s valid 选图 = %4d 分  命中率 %.1f%%"
          % (name, C, ev["points"], 100 * ev["hit_rate"]))
    return C, ev, m


def main():
    print("== 构造 walk-forward 样本（train / valid）==")
    VM.MapHistory.features = features_ext        # ← 只在这里 patch
    data = build_all()
    n_extra_ok = all(k in data["valid"][0]["rows"][0] for k in EXTRA)
    print("  train=%d valid=%d 场   附加列就位=%s"
          % (len(data["train"]), len(data["valid"]), n_extra_ok))

    base = list(VM.FEATURES3)
    arms = [
        ("FEATURES3（现生产，39 维）", base),
        ("+ 己方胜率池内名次", base + ["a_wr_rank"]),
        ("+ 对手胜率池内名次", base + ["b_wr_rank"]),
        ("+ 定向 opp_weak（对手弱→选）", base + ["opp_weak"]),
        ("+ 强弱差 a_wr_gap", base + ["a_wr_gap"]),
        ("+ 全部四列", base + EXTRA),
    ]
    print("\n== valid 对照（同一 walk-forward 构造、同一 C 网格）==")
    res = {}
    for nm, f in arms:
        res[nm] = run_arm(nm, f, data)

    b = res["FEATURES3（现生产，39 维）"][1]["points"]
    print("\n== 相对现生产的 valid 增益 ==")
    for nm, _ in arms[1:]:
        d = res[nm][1]["points"] - b
        print("  %-30s %+5d 分（%+.1f%%）" % (nm, d, 100.0 * d / b))
    print("\nvalid 总场次 %d，满分 %d 分 ⇒ 1 张图的命中差 = 5 分"
          % (len(data["valid"]), len(data["valid"]) * 15))
    return 0


if __name__ == "__main__":
    sys.exit(main())
