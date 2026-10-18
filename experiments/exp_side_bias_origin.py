# -*- coding: utf-8 -*-
"""exp_side_bias_origin.py —— team1 胜率 61.6% 是**真实效应**还是**抓取约定**？

判据：
  (1) 「先 veto / 先选图方」的历史胜率 —— 若接近 61%，则是真实的 first-pick 优势；
      若接近 50%，那 61.6% 就与图池无关，是纯数据约定（不可兑现）。
  (2) team1 胜率按月/按 best_of 是否稳定 —— 稳定 ⇒ 系统性；剧烈波动 ⇒ 采样。
  (3) veto_order 的首动作方与 team1 的一致率。

用法：  python -X utf8 exp_side_bias_origin.py
"""
from __future__ import annotations

import collections
import io
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RECS = os.path.join(HERE, "records_v1.jsonl")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def name_of(side):
    return side.get("name") if isinstance(side, dict) else side


def first_action_team(r):
    for step in (r.get("veto_order") or []):
        if isinstance(step, (list, tuple)) and len(step) >= 3 and step[2]:
            return step[2]
        if isinstance(step, dict) and (step.get("team") or step.get("name")):
            return step.get("team") or step.get("name")
    return None


def main():
    rows = [json.loads(l) for l in io.open(RECS, encoding="utf-8") if l.strip()]
    print("语料 %d 场（%s .. %s）"
          % (len(rows), min(r["date"] for r in rows), max(r["date"] for r in rows)))

    # (1) 先 veto / 先选图方
    fcnt, fwin = collections.Counter(), collections.Counter()
    same_as_t1 = c2 = miss = 0
    for r in rows:
        ft = first_action_team(r)
        t1, t2 = name_of(r.get("team1")), name_of(r.get("team2"))
        w = name_of(r.get("winner"))
        if ft not in (t1, t2):
            miss += 1
            continue
        fcnt[ft] += 1
        fwin[ft] += int(w == ft)
        if ft == t1:
            same_as_t1 += 1
        else:
            c2 += 1
    tot, tt = sum(fwin.values()), sum(fcnt.values())
    lo, hi = wilson(tot, tt)
    print("\n[1] 先 veto/先选图方的胜率 = %d/%d = %.1f%%  Wilson [%.1f, %.1f]"
          % (tot, tt, 100.0 * tot / max(1, tt), 100 * lo, 100 * hi))
    print("    ⇒ %s" % ("**显著偏离 50%**（真实 first-pick 优势）"
                        if hi < 0.5 or lo > 0.5 else "与 50% 无法区分（无真实优势）"))
    print("    veto 首动作方 == team1 : %d 场，== team2 : %d 场（不可用 %d）"
          % (same_as_t1, c2, miss))
    print("    ⇒ team1 与「先 veto 方」的一致率 %.1f%%"
          % (100.0 * same_as_t1 / max(1, same_as_t1 + c2)))

    # (2) team1 胜率按月
    d, w = collections.Counter(), collections.Counter()
    for r in rows:
        m = r["date"][:7]
        d[m] += 1
        w[m] += int(name_of(r.get("winner")) == name_of(r.get("team1")))
    tt, ww = sum(d.values()), sum(w.values())
    lo, hi = wilson(ww, tt)
    print("\n[2] team1 总胜率 = %d/%d = %.1f%%  Wilson [%.1f, %.1f]  %s"
          % (ww, tt, 100.0 * ww / tt, 100 * lo, 100 * hi,
             "显著偏离 50%" if hi < 0.5 or lo > 0.5 else "与 50% 无异"))
    print("    按月：")
    for m in sorted(d):
        print("      %s  %3d/%3d = %5.1f%%" % (m, w[m], d[m], 100.0 * w[m] / d[m]))

    # (3) 按 best_of
    bb, bw = collections.Counter(), collections.Counter()
    for r in rows:
        bo = r.get("best_of")
        bb[bo] += 1
        bw[bo] += int(name_of(r.get("winner")) == name_of(r.get("team1")))
    print("\n[3] team1 胜率 按 best_of：")
    for bo in sorted(bb, key=str):
        print("      bo=%-4s %3d/%3d = %5.1f%%" % (bo, bw[bo], bb[bo], 100.0 * bw[bo] / bb[bo]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())