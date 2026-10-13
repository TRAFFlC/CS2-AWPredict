"""实验：给第 1 项（胜方）加「图级别强度」（map-level Elo）。

动机：
  · v3 对第 1 项只证明了「选型噪声 ≥ 模型差异」（67.2 vs 69.4 在 1SE ±1.6pp 内），
    **没有**像第 3 项那样做「把 test 也拿去训练」的泄漏上界证明 ⇒ 信息未必用尽。
  · 现状的 `d_poolwr` 是「该队在整个 7 图池上的胜率」，很粗；主理人指出
    「打自己不舒服的地图会让面对某些对手总弱势」⇒ 强度应按图区分。
  · H1 已实测：选图行为**确实**按图区分强弱（对手在该图胜率百分位 z=−11.5）。

做法：给 State 增加一个**图级别 Elo 账本** `mapelo[(team,map)]`，
      只用该图上的胜负更新（半衰期衰减与主 Elo 同源），导出三个特征：
        d_mapelo      预测图集上该图 Elo 差（按图集概率加权）
        d_mapelo_max  图集内最强图上的差（取 max）
        d_mapelo_worst 图集内最弱图上的差（取 min）
判据：**train 扩展窗口 CV**（与 v3 的胜方选型同口径）+ **valid**。test 只在明确超 1SE 时报。
"""
import collections, io, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_v2 as M
import veto_model as VM
import split as SPLIT
import pool_experiment as PE

K_MAP_ELO = 24.0          # 图级 Elo 的 K（比主 Elo 小：单图的样本量更少）
EL0 = 1500.0


class MapElo:
    """图级别 Elo 账本。与 State 主 Elo 同样的「先衰减再更新」时序。"""

    def __init__(self, half_life=120.0, k=K_MAP_ELO, el0=EL0):
        self.hl, self.k, self.el0 = half_life, k, el0
        self.e = collections.defaultdict(lambda: el0)
        self.n = collections.Counter()
        self.last_date = None

    def advance(self, date):
        if self.last_date is not None:
            d = VM.days_apart(date, self.last_date)
            if d > 0:
                f = 0.5 ** (d / self.hl)
                for kk in self.e:
                    self.e[kk] = self.el0 + (self.e[kk] - self.el0) * f
        self.last_date = date

    def rating(self, team, m):
        return self.e[(team, m)]

    def observe(self, teamA, teamB, map_results):
        for m, s1, s2 in map_results:
            if s1 == s2:
                continue
            a, b = self.e[(teamA, m)], self.e[(teamB, m)]
            ea = 1.0 / (1.0 + 10 ** (-(a - b) / 400.0))
            kk = self.k / (1.0 + self.n[(teamA, m)] / 60.0)   # 样本多则步长小
            kk_b = self.k / (1.0 + self.n[(teamB, m)] / 60.0)
            self.e[(teamA, m)] = a + kk * (1.0 - ea)
            self.e[(teamB, m)] = b + kk_b * (0.0 - (1.0 - ea))
            self.n[(teamA, m)] += 1
            self.n[(teamB, m)] += 1


def collect(state, mh, rows, truths, veto, eras, by_pool, tl, stats):
    """逐日驱动，同时喂主 State 与图级 Elo；返回每场的图级特征。"""
    me = MapElo()

    def pool_of(r):
        t = truths.get(r["id"]) or {}
        return list(t["pool"]) if t.get("pool") else \
            list(PE.era_and_pool(r["date"], [], eras, by_pool, tl)[1])

    usable = [r for r in rows
              if isinstance(r.get("best_of"), int) and r["best_of"] in (1, 3, 5)]
    usable.sort(key=lambda r: (r["date"], r["id"]))
    map_stats = M.load_stats_map()          # 提到循环外（原来在循环里每次重读 14MB）
    out = []
    i = 0
    while i < len(usable):
        d = usable[i]["date"]
        state.advance(d)
        mh.advance(d)
        me.advance(d)
        group = []
        while i < len(usable) and usable[i]["date"] == d:
            group.append(usable[i])
            i += 1
        pend = []
        for r in group:
            tA, tB = r["team1"]["name"], r["team2"]["name"]
            t = truths.get(r["id"]) or {}
            pool = pool_of(r)
            bo = t.get("bo") or r.get("best_of") or 3
            diffs = {m: me.rating(tA, m) - me.rating(tB, m) for m in pool}
            vals = sorted(diffs.values())
            y = (r["winner"]["name"] == tA)
            ea, _ = state.elo_of(tA)
            eb, _ = state.elo_of(tB)
            pend.append({
                "y": y,
                "d_elo": (ea - eb) / 400.0,
                "d_mapelo_mean": sum(diffs.values()) / len(diffs),
                "d_mapelo_max": vals[-1],
                "d_mapelo_min": vals[0],
                "d_mapelo_spread": vals[-1] - vals[0],
                "win": 1.0 if y else 0.0,
            })
        out.extend(pend)
        for r in group:
            tA, tB = r["team1"]["name"], r["team2"]["name"]
            mres = [(m["name"], m["team1_score"], m["team2_score"])
                    for m in (r.get("maps") or [])]
            t = truths.get(r["id"]) or {}
            state.observe(d, tA, tB, r["winner"]["name"], mres,
                          stats.get(r["id"], {}), map_stats.get(r["id"]))
            mh.observe_match(tA, tB, pool_of(r), t.get("maps") or [],
                             veto.get(r["id"]), mres)
            me.observe(tA, tB, mres)
    return out


def main():
    rows = SPLIT.load(M.MATCHES)
    exc = SPLIT.load_exclude(M.EXCLUDE) if os.path.exists(M.EXCLUDE) else None
    buckets, _ = SPLIT.split(rows, purge=True, exclude=exc)
    truths = {t["id"]: t for t in M.load_jsonl(M.TRUTH)}
    veto = VM.load_veto()
    stats = M.load_stats()
    eras, by_pool = PE.load_pool_eras()
    tl = json.load(io.open(PE.TIMELINE, encoding="utf-8"))

    print("== 收集 train/valid 的图级 Elo 特征 ==")
    s, h = M.State(), VM.MapHistory()
    feats = collect(s, h, buckets["train"], truths, veto, eras, by_pool, tl, stats)
    print("  train %d 场" % len(feats))
    tr = feats
    va = collect(s, h, buckets["valid"], truths, veto, eras, by_pool, tl, stats)
    print("  valid %d 场" % len(va))

    # ---------- 单变量方向检验 ----------
    print("\n== 单变量方向检验（valid）==")
    for k in ("d_mapelo_mean", "d_mapelo_max", "d_mapelo_min", "d_mapelo_spread"):
        pos = sum(1 for f in va if (f[k] > 0) == (f["y"] == 1))
        acc = max(pos, len(va) - pos) / len(va)
        print("  %-20s 方向命中率 %.2f%%（正号=该队 Elo 更高）" % (k, 100 * acc))

    # ---------- 训练扩展窗口 CV（与 v3 同口径）+ valid ----------
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def mat(rows_, feats):
        X = np.array([[f[k] for k in feats] for f in rows_], dtype=float)
        y = np.array([int(f["y"]) for f in rows_])
        return X, y

    folds = M.CV_FOLDS if hasattr(M, "CV_FOLDS") else (0.40, 0.55, 0.70, 0.85, 1.00)
    print("\n== train 扩展窗口 CV + valid（同 v3 判据）==")
    print("  folds=%s" % (folds,))
    NEW = ["d_mapelo_mean", "d_mapelo_max", "d_mapelo_min", "d_mapelo_spread"]
    for name, feats in (("d_mapelo_mean", ["d_mapelo_mean"]),
                        ("d_mapelo_max", ["d_mapelo_max"]),
                        ("d_mapelo_max+min", ["d_mapelo_max", "d_mapelo_min"]),
                        ("mean+max+min+spread", NEW)):
        n_val = 0
        correct = 0
        Xtr, ytr = mat(tr[:int(len(tr) * folds[0])], feats)
        sc0 = StandardScaler().fit(Xtr)
        clf0 = LogisticRegression(max_iter=4000, C=0.1).fit(sc0.transform(Xtr), ytr)
        for f in range(1, len(folds)):
            lo, hi = int(len(tr) * folds[f - 1]), int(len(tr) * folds[f])
            if hi <= lo:
                continue
            X, y = mat(tr[lo:hi], feats)
            p = clf0.predict(sc0.transform(X))
            correct += int((p == y).sum())
            n_val += len(y)
        Xv, yv = mat(va, feats)
        pv = clf0.predict(sc0.transform(Xv))
        vacc = (pv == yv).mean()
        print("  %-22s CV=%.4f (n=%d)   valid=%.2f%%" % (name, correct / n_val, n_val, 100 * vacc))
    print("\n  参照：v3 报的主胜方模型 CV=0.6233；胜方正确率 valid 约 62%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
