"""NNUE 真形重测：稀疏索引 → 嵌入表 → **加法**累加 → 头。

上一版 `exp_nnue.py` 用的是**乘积**（矩阵分解）t = (U[A]-U[B])·(V+M)，
这是**偏离 NNUE** 的：真 NNUE 的累加器是 z = Σᵢ vᵢ·emb[idᵢ]（**加法**），
梯度 ∂L/∂emb = vᵢ 恒非零。我那个乘积形式在零初始化下梯度恒为 0（嵌入被锁死），
实测已证实（零初始化列对 l2 完全不敏感）。

本版严格照 NNUE：
  · 实体 = {12 个标量特征槽} ∪ {队伍} ∪ {(队伍, 图)}   —— 稀疏索引表
  · 每样本的稀疏输入 = [(idx, value), ...]，value = 特征值 / +1(A队) / −1(B队)
  · 累加器 z = Σ value · emb[idx]        ← **加法**
  · 头：p = sigmoid(a·z + b)
参数量：n_slot = 12 + n_team + n_(team,map) 对（只登记实际出现过的组合）
"""
import math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_v2 as M
import exp_nnue as E

SCALARS = E.SCALARS


def build_sparse(tr, t2i, pair2i, n_scalar, width=40):
    """把 (X, A, B, P, PM) 摊成 NNUE 的稀疏 (idx, val) 对。"""
    n = len(tr[0])
    idx = np.zeros((n, width), dtype=np.int64)
    val = np.zeros((n, width), dtype=np.float64)
    for i in range(n):
        k = 0
        for j, v in enumerate(tr[0][i]):
            if k >= width:
                break
            idx[i, k] = j
            val[i, k] = v
            k += 1
        idx[i, k] = n_scalar + tr[1][i]; val[i, k] = 1.0; k += 1
        idx[i, k] = n_scalar + tr[2][i]; val[i, k] = -1.0; k += 1
        for j in range(tr[3].shape[1]):
            if tr[4][i, j] == 0 or k + 2 > width:
                continue
            pa, pb = pair2i[(tr[1][i], tr[3][i, j])], pair2i[(tr[2][i], tr[3][i, j])]
            idx[i, k] = pa; val[i, k] = 1.0; k += 1
            idx[i, k] = pb; val[i, k] = -1.0; k += 1
    return idx, val


class NNUEAdd:
    def __init__(self, n_slot, k=8, seed=0, zero_init=False):
        rg = np.random.default_rng(seed)
        self.k = k
        self.E = np.zeros((n_slot, k)) if zero_init else rg.normal(0, 0.05, (n_slot, k))
        self.a = np.ones(k)
        self.b = 0.0

    def acc(self, idx, val):
        E = self.E[idx]                       # (n, width, k)
        return (E * val[:, :, None]).sum(axis=1)      # (n, k)  ← **加法累加**

    def pred(self, idx, val):
        return 1.0 / (1.0 + np.exp(-(self.acc(idx, val) @ self.a + self.b)))

    def fit(self, idx, val, y, epochs=400, lr=0.05, l2=1e-3, l2_emb=None,
            l2_scalar=1e-2):
        l2_emb = l2 if l2_emb is None else l2_emb
        n_slot = self.E.shape[0]
        mask_scalar = np.zeros(n_slot, dtype=bool)
        mask_scalar[:len(SCALARS)] = True
        E, a, b = self.E, self.a, self.b
        mE, vE = np.zeros_like(E), np.zeros_like(E)
        ma, va = np.zeros_like(a), np.zeros_like(a)
        mb = vb = 0.0
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t_ in range(1, epochs + 1):
            acc = (E[idx] * val[:, :, None]).sum(axis=1)
            p = 1.0 / (1.0 + np.exp(-(acc @ a + b)))
            e = p - y
            gz = e[:, None] * a[None, :]                    # dL/dacc (n,k)
            gE = np.zeros_like(E)
            contrib = gz[:, None, :] * val[:, :, None]      # (n,width,k)
            np.add.at(gE, idx, contrib)
            reg = np.where(mask_scalar, l2_scalar, l2_emb)[:, None] * E
            gE = gE + reg
            ga = (gz * acc).sum(axis=0)
            gb = e.sum()
            mE = b1 * mE + (1 - b1) * gE; vE = b2 * vE + (1 - b2) * gE ** 2
            ma = b1 * ma + (1 - b1) * ga; va = b2 * va + (1 - b2) * ga ** 2
            mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb ** 2
            E -= lr * (mE / (1 - b1 ** t_)) / (np.sqrt(vE / (1 - b2 ** t_)) + eps)
            a -= lr * (ma / (1 - b1 ** t_)) / (np.sqrt(va / (1 - b2 ** t_)) + eps)
            b -= lr * (mb / (1 - b1 ** t_)) / (np.sqrt(vb / (1 - b2 ** t_)) + eps)
        self.E, self.a, self.b = E, a, b
        return self


def main():
    prep = M.prepare(verbose=False)
    data = prep["data"]
    idx = E.build_index()
    teams = sorted({t for x in ("train", "valid") for (ta, tb), _, _ in idx[x]
                    for t in (ta, tb)})
    maps = sorted({m for x in ("train", "valid") for _, pool, _ in idx[x] for m in pool})
    t2i = {t: i for i, t in enumerate(teams)}
    m2i = {m: i for i, m in enumerate(maps)}
    tr = E.assemble(data["train"], idx["train"], t2i, m2i)
    va = E.assemble(data["valid"], idx["valid"], t2i, m2i)
    pairs = set()
    for arr in (tr, va):
        A, B, P, PM = arr[1], arr[2], arr[3], arr[4]
        for i in range(len(A)):
            for j in range(P.shape[1]):
                if PM[i, j] == 0:
                    continue
                pairs.add((int(A[i]), int(P[i, j])))
                pairs.add((int(B[i]), int(P[i, j])))
    pairs = sorted(pairs)
    pair2i = {p: i for i, p in enumerate(pairs)}
    n_slot = len(SCALARS) + len(teams) + len(pairs)
    print("稀疏槽：标量 %d + 队伍 %d + (队伍,图) 对 %d = %d   嵌入 k×%d"
          % (len(SCALARS), len(teams), len(pairs), n_slot, 8))
    print("参数量：k=8 → %d（对照：逻辑回归 6 维 = 7 个参数）" % (n_slot * 8 + 9))
    print("每队样本：train %.1f 场、首折 %.1f 场；每个(队伍,图)对 %.1f 场"
          % (2 * len(tr[0]) / len(teams), 2 * int(len(tr[0]) * M.CV_FOLDS[0]) / len(teams),
             2 * len(tr[0]) / len(pairs)))

    I, V = build_sparse(tr, t2i, pair2i, len(SCALARS))
    Iv, Vv = build_sparse(va, t2i, pair2i, len(SCALARS))
    folds = M.CV_FOLDS
    base = M.cv_acc(data["train"], list(prep["winfeats"]), prep["C"])
    n_val = int(len(data["train"]) * (folds[-1] - folds[0]))
    se = math.sqrt(max(base * (1 - base), 1e-9) / n_val)
    print("\n参照：逻辑回归 CV=%.4f  1SE=%.4f" % (base, se))

    def cv(k, lr, l2_emb, epochs):
        corr = nv = 0
        for fi in range(1, len(folds)):
            end, prev = int(len(I) * folds[fi]), int(len(I) * folds[fi - 1])
            if end - prev < 20 or prev < 100:
                continue
            m = NNUEAdd(n_slot, k=k)
            m.fit(I[:prev], V[:prev], tr[5][:prev], epochs=epochs, lr=lr, l2_emb=l2_emb)
            p = m.pred(I[prev:end], V[prev:end])
            corr += int(((p >= 0.5).astype(int) == tr[5][prev:end]).sum())
            nv += end - prev
        return corr / nv

    print("\n== 加法 NNUE 网格（train 扩展窗口 OOF，同 folds）==")
    print("%-30s %8s" % ("配置", "CV"))
    best = None
    for k in (4, 8, 16):
        for l2e in (1e-3, 1e-2, 1e-1):
            a = cv(k, 0.05, l2e, 500)
            print("%-30s %8.4f" % ("k=%d l2_emb=%.0e" % (k, l2e), a))
            if best is None or a > best[0]:
                best = (a, k, l2e)
    print("\n加法 NNUE 最佳 CV=%.4f（k=%d l2_emb=%.0e）" % best)
    print("相对逻辑回归 %+.4f（1SE=%.4f）⇒ %s"
          % (best[0] - base, se,
             "★ 超过 1SE" if best[0] > base + se else "未超 1SE ⇒ 不采纳、test 不报"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
