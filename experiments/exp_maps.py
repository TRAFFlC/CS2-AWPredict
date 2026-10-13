#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_maps.py —— 第 3 项（BP 选图）诊断与选型实验（2026-10-11）。

关心三件事：
  1. 现有 per-map 逻辑回归还差多少？（用「在全部数据上训练」的**泄漏上界**估计）
  2. 换模型族（GBM）能不能涨？
  3. 加特征（排名类 / 策略交互类）能不能涨？

评测口径与 `veto_model.evaluate` 一致（集合命中，每张 5 分）。
"""
from __future__ import annotations

import collections
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import veto_model as VM  # noqa: E402
import split as SPLIT  # noqa: E402
import pool_experiment as PE  # noqa: E402


def build_all(half_life=VM.HALF_LIFE):
    rows = SPLIT.load(VM.MATCHES)
    exclude = SPLIT.load_exclude(VM.EXCLUDE)
    buckets, meta = SPLIT.split(rows, purge=True, exclude=exclude)
    truths = {t["id"]: t for t in VM.load_jsonl(VM.TRUTH)}
    veto = VM.load_veto()
    eras, by_pool = PE.load_pool_eras()
    tl = json.load(io.open(PE.TIMELINE, encoding="utf-8"))

    def pool_of(r):
        t = truths.get(r["id"]) or {}
        vp = t.get("pool")
        if vp:
            return list(vp)
        _era, pool = PE.era_and_pool(r["date"], [], eras, by_pool, tl)
        return list(pool)

    hist = VM.MapHistory(half_life=half_life)
    out = {}
    for b in ("train", "valid", "test"):
        out[b] = VM.build_dataset(buckets[b], truths, veto, hist, pool_of)
    return out, meta


def expand(samples, feats):
    X = np.array([[f[fn] for fn in feats] for s in samples for f in s["rows"]], dtype=float)
    y = np.array([v for s in samples for v in s["y"]], dtype=int)
    return X, y


def ev(model, samples, feats):
    from sklearn.preprocessing import StandardScaler  # noqa: F401
    if model["kind"] == "logit":
        X = np.array([[f[fn] for fn in feats] for s in samples for f in s["rows"]], dtype=float)
        p = model["clf"].predict_proba(model["sc"].transform(X))[:, 1]
    else:
        X = np.array([[f[fn] for fn in feats] for s in samples for f in s["rows"]], dtype=float)
        p = model["clf"].predict_proba(X)[:, 1]
    idx = 0
    hits = slots = 0
    slot_hits = collections.Counter()
    for s in samples:
        k = max(1, min(int(s["bo"] or 3), 5))
        n = len(s["rows"])
        prob = p[idx:idx + n]
        idx += n
        order = sorted(range(n), key=lambda i: (-prob[i], s["rows"][i]["map"]))
        truth = set(f["map"] for f, v in zip(s["rows"], s["y"]) if v)
        for j, i in enumerate(order[:k]):
            slots += 1
            if s["rows"][i]["map"] in truth:
                hits += 1
                slot_hits[j] += 1
    return {"points": 5 * hits, "hits": hits, "slots": slots,
            "rate": hits / slots if slots else 0.0, "per_slot": dict(slot_hits)}


def fit_logit(samples, feats, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, y = expand(samples, feats)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=4000, C=C).fit(sc.transform(X), y)
    return {"kind": "logit", "sc": sc, "clf": clf}


def fit_gbm(samples, feats, depth=3, lr=0.05, n=150):
    from sklearn.ensemble import GradientBoostingClassifier
    X, y = expand(samples, feats)
    clf = GradientBoostingClassifier(max_depth=depth, learning_rate=lr,
                                     n_estimators=n, random_state=0).fit(X, y)
    return {"kind": "gbm", "clf": clf}


def main():
    data, meta = build_all()
    print("split", meta["counts"])
    base = VM.FEATURES
    cands = {
        "base": base,
    }
    if hasattr(VM, "FEATURES2"):
        cands["v2"] = VM.FEATURES2
    if hasattr(VM, "FEATURES3"):
        cands["v3"] = VM.FEATURES3
    for name, feats in cands.items():
        print("")
        print("== 特征组 %s（%d 个特征）==" % (name, len(feats)))
        for C in (0.03, 0.1, 0.3, 1.0, 3.0):
            m = fit_logit(data["train"], feats, C)
            v = ev(m, data["valid"], feats)
            print("  logit C=%-5s valid=%d(%.1f%%)" % (C, v["points"], 100 * v["rate"]))
        best = None
        for C in (0.03, 0.1, 0.3, 1.0, 3.0):
            v = ev(fit_logit(data["train"], feats, C), data["valid"], feats)
            if best is None or v["points"] > best[0]:
                best = (v["points"], C)
        C = best[1]
        m = fit_logit(data["train"], feats, C)
        t = ev(m, data["test"], feats)
        print("  -> valid 选 C=%s；test=%d(%.1f%%) 槽位=%d %s"
              % (C, t["points"], 100 * t["rate"], t["slots"], t["per_slot"]))
        # 管道口径：最终模型用 train+valid 重拟合（= model_v2.prepare 的协议，见 2026-10-11 笔记 §5.2）
        mp = fit_logit(data["train"] + data["valid"], feats, C)
        tp = ev(mp, data["test"], feats)
        print("  [管道口径 train+valid 拟合] test=%d(%.1f%%)" % (tp["points"], 100 * tp["rate"]))
        # 泄漏上界：train+valid+test 一起训练
        allx = data["train"] + data["valid"] + data["test"]
        ml = fit_logit(allx, feats, C)
        tl = ev(ml, data["test"], feats)
        print("  [泄漏上界] test=%d(%.1f%%) —— 特征/模型族的极限" % (tl["points"], 100 * tl["rate"]))
        g = fit_gbm(data["train"], feats)
        gv = ev(g, data["valid"], feats)
        gt = ev(g, data["test"], feats)
        print("  GBM valid=%d(%.1f%%) test=%d(%.1f%%)" % (gv["points"], 100 * gv["rate"],
                                                          gt["points"], 100 * gt["rate"]))


if __name__ == "__main__":
    main()
