#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 **train+valid 的扩展窗口 CV**（最靠近 test 的可用口径）复核第 1 项选型。（用完即删）"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_v2 as M  # noqa: E402
import exp_roster as ER  # noqa: E402

ctx = ER.load_all()
data, _st = ER.build_all(ctx, M.RHO, M.RHO_G)
tv = data["train"] + data["valid"]
print("train+valid n =", len(tv))
rows = []
for C in (0.02, 0.05, 0.1, 0.3, 1.0):
    for fname, feats in M.WIN_FEATURE_SETS:
        cv_tv = M.cv_acc(tv, feats, C)
        cv_tr = M.cv_acc(data["train"], feats, C)
        rows.append((cv_tv, cv_tr, fname, C, len(feats)))
rows.sort(reverse=True)
print("%-22s %-5s %4s %10s %10s" % ("features", "C", "#f", "cv(train+val)", "cv(train)"))
for cv_tv, cv_tr, fname, C, nf in rows[:22]:
    print("%-22s %-5s %4d %10.4f %10.4f" % (fname, C, nf, cv_tv, cv_tr))
# 关键对照
print("")
for key in ("elo2", "...+games", "elo2+rank+h2h+lineup", "all13"):
    hits = [r for r in rows if r[2] == key]
    if hits:
        print("best for %-22s -> C=%s cv(train+val)=%.4f" % (key, hits[0][3], hits[0][0]))
