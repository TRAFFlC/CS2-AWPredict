"""判别：嵌入到底是「携带信号但没学好」还是「根本不携带信号」？

两个决定性对照：
  A. **零初始化** —— 去掉随机初始化的混淆。若从零初始化训练后仍低于标量基线，
     说明嵌入是**主动有害**的。
  B. **嵌入专属 L2 扫描** —— l2_emb → ∞ 时嵌入被压成 0，模型退化为「只标量」。
     · 若 CV **单调上升**到标量基线且无内部峰值 ⇒ 嵌入是**纯噪声**，最优解就是「关掉它」
     · 若存在**内部峰值**高于标量基线 ⇒ 嵌入**确实携带信号**，只是需要更强正则
这比看 epoch 曲线可靠得多。
"""
import math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_v2 as M
import exp_nnue as E


class NNUE2(E.NNUE):
    """在原模型上加：零初始化选项 + 嵌入专属 L2。"""

    def __init__(self, *a, zero_init=False, **kw):
        super().__init__(*a, **kw)
        if zero_init:
            self.U[:] = 0.0
            self.V[:] = 0.0
            self.Mm[:] = 0.0

    def fit2(self, X, A, B, P, PM, y, epochs, lr, l2, l2_emb):
        params = [self.ws, self.U, self.V, self.Mm,
                  np.array([self.b, self.aw, self.bw, self.as_, self.bs])]
        mW = [np.zeros_like(p) for p in params]
        vW = [np.zeros_like(p) for p in params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t_ in range(1, epochs + 1):
            s = X @ self.ws
            dU = self.U[A] - self.U[B]
            Wp = (self.V[P] + self.Mm[P]) * PM[:, :, None]
            z = s + (dU[:, None, :] * Wp).sum(axis=(1, 2)) + self.b
            pw = 1.0 / (1.0 + np.exp(-(self.aw * z + self.bw)))
            e = pw - y
            gz = e * self.aw
            gU = np.zeros_like(self.U)
            gV = np.zeros_like(self.V)
            gM = np.zeros_like(self.Mm)
            coef = (gz[:, None] * PM)[:, :, None]
            np.add.at(gU, A, (coef * Wp).sum(axis=1))
            np.add.at(gU, B, -(coef * Wp).sum(axis=1))
            contrib = coef * dU[:, None, :]
            np.add.at(gV, P, contrib)
            np.add.at(gM, P, contrib)
            grads = [X.T @ gz + l2 * self.ws, gU + l2_emb * self.U, gV + l2_emb * self.V,
                     gM + l2_emb * self.Mm,
                     np.array([gz.sum() + l2 * self.b, (e * z).sum(), e.sum(),
                             (e * z).sum(), e.sum()])]
            for j in range(len(params)):
                mW[j] = b1 * mW[j] + (1 - b1) * grads[j]
                vW[j] = b2 * vW[j] + (1 - b2) * grads[j] ** 2
                params[j] -= lr * (mW[j] / (1 - b1 ** t_)) / \
                    (np.sqrt(vW[j] / (1 - b2 ** t_)) + eps)
            self.ws, self.U, self.V, self.Mm = params[:4]
            self.b, self.aw, self.bw, self.as_, self.bs = params[4]
        return self


def cv(tr, k, lr, l2, l2_emb, epochs, folds, n_team, n_map, zero_init):
    corr = nv = 0
    for fi in range(1, len(folds)):
        end, prev = int(len(tr[0]) * folds[fi]), int(len(tr[0]) * folds[fi - 1])
        if end - prev < 20 or prev < 100:
            continue
        a = [x[:prev] for x in tr]
        b = [x[prev:end] for x in tr]
        m = NNUE2(a[0].shape[1], n_team, n_map, k=k, zero_init=zero_init)
        m.fit2(a[0], a[1], a[2], a[3], a[4], a[5], epochs, lr, l2, l2_emb)
        p = m.pred(b[0], b[1], b[2], b[3], b[4])
        corr += int(((p >= 0.5).astype(int) == b[5]).sum())
        nv += len(b[5])
    return corr / nv


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
    folds = M.CV_FOLDS
    base = M.cv_acc(data["train"], list(prep["winfeats"]), prep["C"])
    n_val = int(len(data["train"]) * (folds[-1] - folds[0]))
    se = math.sqrt(max(base * (1 - base), 1e-9) / n_val)
    print("参照：逻辑回归 CV=%.4f   1SE=%.4f   n_val=%d" % (base, se, n_val))
    print("实体：队伍 %d  图 %d   每队场次 ≈ %.1f（首个训练折 ≈ %.1f）"
          % (len(teams), len(maps),
             2 * len(tr[0]) / len(teams),
             2 * int(len(tr[0]) * folds[0]) / len(teams)))

    # 只标量基线（本实现）
    scal = cv(tr, 4, 0.08, 1e-2, 1e9, 1000, folds, len(teams), len(maps), False)
    print("\n本实现「只标量」基线（嵌入被 L2 压死）CV=%.4f" % scal)

    print("\n== B. 嵌入专属 L2 扫描（k=4, lr=0.08, ep=1000）==")
    print("%12s %10s %10s" % ("l2_emb", "随机初始化", "零初始化"))
    rows = []
    for l2e in (0.0, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        r1 = cv(tr, 4, 0.08, 1e-2, l2e, 1000, folds, len(teams), len(maps), False)
        r2 = cv(tr, 4, 0.08, 1e-2, l2e, 1000, folds, len(teams), len(maps), True)
        rows.append((l2e, r1, r2))
        print("%12.0e %10.4f %10.4f" % (l2e, r1, r2))

    best = max(rows, key=lambda r: max(r[1], r[2]))
    print("\n判读：")
    print("  嵌入最优点 l2_emb=%.0e → CV=%.4f（随机初始化 %.4f / 零初始化 %.4f）"
          % (best[0], best[1], max(best[1], best[2]), max(best[1], best[2])))
    print("  「只标量」基线 = %.4f" % scal)
    print("  ⇒ %s" % ("存在**内部峰值**高于只标量基线 ⇒ 嵌入携带信号，"
                      "可继续调正则" if max(best[1], best[2]) > scal + se / 2 else
                      "CV 随 l2_emb 增大**单调趋近只标量基线** ⇒ 嵌入是**纯噪声**，"
                      "最优解就是关掉它"))
    for l2e, r1, r2 in rows:
        print("     l2_emb=%-8.0e CV=%.4f  相对只标量 %+.4f" % (l2e, r1, r1 - scal))
    return 0


if __name__ == "__main__":
    sys.exit(main())
