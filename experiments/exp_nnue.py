"""NNUE 风格实验：稀疏查表 + 差分累加 + 共享隐向量 + 双头。

为什么是这个形态（而不是"上深层网络"）：
  棋类 NNUE 的可迁移内核是 ①稀疏特征查表 ②累加成单一隐向量 ③共享隐向量喂多头。
  第 ④点（增量累加器）是为搜索期高频重算服务的，本项目预测低频，不适用。
  而本机只有 numpy + sklearn（无 PyTorch），训练样本仅 2,087 场 ⇒ 必须低秩、强正则。

模型：
    s      = w_s · x                                标量特征（与逻辑回归同一套）
    t      = (U[A] - U[B]) · (V_pool + M_pool)      队伍主效应 ⊗ 图池效应（矩阵分解）
    z      = s + sum(t) + b                         **共享隐向量**
    p_win  = sigmoid(a_w · z + b_w)                 头 1：第 1 项（胜方）
    p_sw   = sigmoid(a_s · z + b_s)                 头 2：第 2 项（是否通吃 2:0）

超参（极小网格，全部按 1SE 规则判）：k / lr / l2 / epochs
判据：**与 model_v2.cv_acc 完全相同的 train 扩展窗口 OOF**；test 只在明确超 1SE 时报。
"""
import io, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_v2 as M
import veto_model as VM
import split as SPLIT
import pool_experiment as PE

SCALARS = ["d_elo", "d_elom", "d_rank", "d_logp", "d_wr", "d_h2h", "d_poolwr",
           "d_lineup", "d_churn", "d_form", "d_rest", "d_games"]


# ---------------------------------------------------------------- 数据装配
def build_index():
    """按 (date,id) 顺序重走语料，取出每场的 (tA, tB, pool)，
    与 model_v2.build() 的样本顺序**逐位对齐**（同一过滤、同一排序、同一逐日两趟）。"""
    rows = SPLIT.load(M.MATCHES)
    exc = SPLIT.load_exclude(M.EXCLUDE) if os.path.exists(M.EXCLUDE) else None
    buckets, _ = SPLIT.split(rows, purge=True, exclude=exc)
    truths = {t["id"]: t for t in M.load_jsonl(M.TRUTH)}
    eras, by_pool = PE.load_pool_eras()
    tl = json.load(io.open(PE.TIMELINE, encoding="utf-8"))

    def pool_of(r):
        t = truths.get(r["id"]) or {}
        return list(t["pool"]) if t.get("pool") else \
            list(PE.era_and_pool(r["date"], [], eras, by_pool, tl)[1])

    out = {"train": [], "valid": []}
    for b in ("train", "valid"):
        sel = [r for r in buckets[b]
               if isinstance(r.get("best_of"), int) and r["best_of"] in (1, 3, 5)]
        sel.sort(key=lambda r: (r["date"], r["id"]))
        out[b] = [((r["team1"]["name"], r["team2"]["name"]), pool_of(r), r["id"])
                  for r in sel]
    return out


def assemble(samples, idx, team2i, map2i, n_maps=7):
    X = np.array([[s["feats"].get(f, 0.0) for f in SCALARS] for s in samples],
                 dtype=np.float64)
    A = np.empty(len(samples), dtype=np.int64)
    B = np.empty(len(samples), dtype=np.int64)
    P = np.zeros((len(samples), n_maps), dtype=np.int64)
    PM = np.zeros((len(samples), n_maps), dtype=np.float64)
    for i, ((ta, tb), pool, _) in enumerate(idx):
        A[i] = team2i[ta]
        B[i] = team2i[tb]
        for j, m in enumerate(pool[:n_maps]):
            P[i, j] = map2i.get(m, 0)
            PM[i, j] = 1.0
    y = np.array([int(s["y"]) for s in samples])
    ysw = np.array([1 if (s["truth"].get("score") or [9, 9])[0] >
                    (s["truth"].get("score") or [0, 0])[1] and
                    sum(s["truth"].get("score") or [0, 0]) == 2 else 0
                    for s in samples]) if "truth" in samples[0] else None
    return X, A, B, P, PM, y, ysw


# ---------------------------------------------------------------- 模型
class NNUE:
    def __init__(self, n_scalar, n_team, n_map, k=4, seed=0,
                 use_scalars=True, use_emb=True):
        rg = np.random.default_rng(seed)
        self.k = k
        self.use_scalars = use_scalars
        self.use_emb = use_emb
        self.ws = np.zeros(n_scalar) if use_scalars else rg.normal(0, 0.1, n_scalar)
        self.b = 0.0
        s = 0.1 if use_emb else 0.0
        self.U = rg.normal(0, s, (n_team, k))
        self.V = rg.normal(0, s, (n_map, k))
        self.Mm = rg.normal(0, s, (n_map, k))
        self.aw, self.bw = 1.0, 0.0
        self.as_, self.bs = 1.0, 0.0

    def _params(self):
        return [self.ws, self.U, self.V, self.Mm,
                np.array([self.b, self.aw, self.bw, self.as_, self.bs])]

    def forward(self, X, A, B, P, PM):
        s = (X @ self.ws) if self.use_scalars else 0.0
        dU_ab = (self.U[A] - self.U[B])[:, None, :]           # (n,1,k)
        Wp = (self.V[P] + self.Mm[P]) * PM[:, :, None]        # (n,7,k)
        t = (dU_ab * Wp).sum(axis=(1, 2)) if self.use_emb else 0.0
        z = s + t + self.b
        return z, 1.0 / (1.0 + np.exp(-(self.aw * z + self.bw))), \
            1.0 / (1.0 + np.exp(-(self.as_ * z + self.bs)))

    def fit(self, X, A, B, P, PM, y, epochs=300, lr=0.05, l2=1e-3, seed=0):
        """全批量 Adam（本机只有 numpy；2k 样本逐样本 SGD 太慢，且全批量更稳）。"""
        params = [self.ws, self.U, self.V, self.Mm,
                  np.array([self.b, self.aw, self.bw, self.as_, self.bs])]
        mW = [np.zeros_like(p) for p in params]
        vW = [np.zeros_like(p) for p in params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t_ in range(1, epochs + 1):
            s = (X @ self.ws) if self.use_scalars else 0.0
            dU_ab = self.U[A] - self.U[B]                      # (n,k)
            Wp = (self.V[P] + self.Mm[P]) * PM[:, :, None]    # (n,7,k)
            t = (dU_ab[:, None, :] * Wp).sum(axis=(1, 2)) if self.use_emb else 0.0
            z = s + t + self.b
            pw = 1.0 / (1.0 + np.exp(-(self.aw * z + self.bw)))
            e = pw - y
            gz = e * self.aw
            gws = X.T @ gz + l2 * self.ws
            gU = np.zeros_like(self.U)
            gV = np.zeros_like(self.V)
            gM = np.zeros_like(self.Mm)
            coef = (gz[:, None] * PM)[:, :, None]             # (n,7,1)
            np.add.at(gU, A, (coef * Wp).sum(axis=1))
            np.add.at(gU, B, -(coef * Wp).sum(axis=1))
            contrib = coef * dU_ab[:, None, :]                 # (n,7,k)
            np.add.at(gV, P, contrib)
            np.add.at(gM, P, contrib)
            gh = np.array([gz.sum() + l2 * self.b, (e * z).sum(),
                           e.sum(), (e * z).sum(), e.sum()])
            grads = [gws, gU, gV, gM, gh]
            for j in range(len(params)):
                mW[j] = b1 * mW[j] + (1 - b1) * grads[j]
                vW[j] = b2 * vW[j] + (1 - b2) * grads[j] ** 2
                params[j] -= lr * (mW[j] / (1 - b1 ** t_)) / \
                    (np.sqrt(vW[j] / (1 - b2 ** t_)) + eps)
            self.ws, self.U, self.V, self.Mm = params[0], params[1], params[2], params[3]
            self.b, self.aw, self.bw, self.as_, self.bs = params[4]
        return self

    def pred(self, X, A, B, P, PM):
        return self.forward(X, A, B, P, PM)[1]


def cv_nnue(train, k, lr, l2, epochs, folds, n_team, n_map,
             use_scalars=True, use_emb=True, scale=None):
    """与 model_v2.cv_acc 同口径的扩展窗口 OOF。
    ⚠ 实体维度必须用**全量**队伍/图数，不能用训练折里的 max()——验证折可能
    含有训练折没见过的队伍（这正是嵌入表「冷启动」的地方）。"""
    correct = n_val = 0
    for fi in range(1, len(folds)):
        end = int(len(train[0]) * folds[fi])
        prev = int(len(train[0]) * folds[fi - 1])
        if end - prev < 20 or prev < 100:
            continue
        tr = tuple(a[:prev] for a in train)
        va = tuple(a[prev:end] for a in train)
        xs = tr[0] if scale is None else tr[0] * scale
        xv = va[0] if scale is None else va[0] * scale
        m = NNUE(train[0].shape[1], n_team, n_map, k=k,
                 use_scalars=use_scalars, use_emb=use_emb)
        m.fit(xs, tr[1], tr[2], tr[3], tr[4], tr[5], epochs=epochs, lr=lr, l2=l2)
        p = m.pred(xv, va[1], va[2], va[3], va[4])
        y = va[5]
        correct += int(((p >= 0.5).astype(int) == y).sum())
        n_val += len(y)
    return correct / n_val, n_val


def main():
    prep = M.prepare(verbose=False)
    data = prep["data"]
    idx = build_index()
    print("样本：train=%d valid=%d   参照现生产 CV=%.4f（%s, C=%s）"
          % (len(data["train"]), len(data["valid"]),
             M.cv_acc(data["train"], list(prep["winfeats"]), prep["C"]),
             prep["win_name"], prep["C"]))

    teams = sorted({t for b in ("train", "valid")
                    for (ta, tb), _, _ in idx[b] for t in (ta, tb)})
    maps = sorted({m for b in ("train", "valid")
                   for _, pool, _ in idx[b] for m in pool})
    team2i = {t: i for i, t in enumerate(teams)}
    map2i = {m: i for i, m in enumerate(maps)}
    print("实体维度：队伍 %d，图 %d，标量特征 %d" % (len(teams), len(maps), len(SCALARS)))

    tr = assemble(data["train"], idx["train"], team2i, map2i)
    va = assemble(data["valid"], idx["valid"], team2i, map2i)
    folds = M.CV_FOLDS
    print("\n== NNUE 网格（train 扩展窗口 OOF，与 cv_acc 同 folds=%s）==" % (folds,))
    print("%-34s %8s" % ("配置", "CV"))
    best = None
    for k in (2, 4, 8):
        for lr in (0.03, 0.08):
            for l2 in (1e-2, 1e-1):
                acc, n = cv_nnue(tr, k, lr, l2, 260, folds, len(teams), len(maps))
                print("%-34s %8.4f" % ("k=%d lr=%.2f l2=%.0e" % (k, lr, l2), acc))
                if best is None or acc > best[0]:
                    best = (acc, k, lr, l2)
    base = M.cv_acc(data["train"], list(prep["winfeats"]), prep["C"])
    n_val = int(len(data["train"]) * (folds[-1] - folds[0]))
    se = math.sqrt(max(base * (1 - base), 1e-9) / n_val)
    print("\n参照（现生产）CV=%.4f   1SE=%.4f" % (base, se))
    print("NNUE 最佳：CV=%.4f（k=%d lr=%.2f l2=%.0e）  相对 %+.4f  ⇒ %s"
          % (best[0], best[1], best[2], best[3], best[0] - base,
             "★ 超 1SE" if best[0] > base + se else "未超 1SE ⇒ 不采纳、test 不报"))

    # valid 上的直观对照
    m = NNUE(tr[0].shape[1], len(teams), len(maps), k=best[1])
    m.fit(tr[0], tr[1], tr[2], tr[3], tr[4], tr[5], epochs=260, lr=best[2], l2=best[3])
    pv = m.pred(va[0], va[1], va[2], va[3], va[4])
    acc = ((pv >= 0.5).astype(int) == va[5]).mean()
    plin = M.predict_proba_win(prep["wmodel"], data["valid"][0]) if False else None
    print("\nvalid 命中率：NNUE=%.2f%%（n=%d）" % (100 * acc, len(va[5])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
