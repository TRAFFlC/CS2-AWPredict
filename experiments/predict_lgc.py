"""Logitech G Connect 2026（2026-09-17 ~ 09-24）盲预测 · 35 场。

赛制：09-17~09-19 为 bo1（30 场），随后 bo3 4 场，09-24 bo5 决赛 1 场。
满分：30×25 + 4×45 + 1×55 = 985。

**bo1 的计分结构值得单独说**（与 bo3/bo5 完全不同）：
  第 2 项比分在 bo1 **整项不计**（裁决 Q-007），满分只有 25。
  第 3 项选图 = **7 张图里蒙 1 张**（基准 ~14%），权重 5/25 = 20%。
  第 4 项 MVP = 10/25 = **40%**，而实测命中率 ~0.14~0.31（`decide.MVP_CAL_PROB`）。
  ⇒ **bo1 的期望得分远低于 bo3/bo5，这是计分结构决定的，不是模型变差。**

队伍可预测性（csapi 语料覆盖，见 coverage）：
  ✅ 可进模型 9 支：B8 #13 / Metizport #60 / M80 #40 / GamerLegion #28 /
     Inner Circle #16 / Liquid #17 / INFINITE #47 / 3DMAX #33 / EYEBALLERS #35
  ⚠ Luminosity #44 仅 **2 场**、BBL #53 仅 **1 场**
  🔴 **NiP 完全不在语料中**（1/5 的 bo1 对局涉及）⇒ 无 Elo 也无 rank

用法：
  python -X utf8 predict_lgc.py
"""
import io
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）
sys.path.insert(0, HERE)
from predict_fissure import ALIAS, coverage, resolve   # noqa: E402

# (日期, 简称A, 简称B, bo)
SCHED = [
    ("2026-09-17", "b8", "metizport", 1),
    ("2026-09-17", "m80", "nip", 1),
    ("2026-09-17", "gamerlegion", "luminosity", 1),
    ("2026-09-17", "luminosity", "metizport", 1),
    ("2026-09-17", "b8", "gamerlegion", 1),
    ("2026-09-17", "m80", "gamerlegion", 1),
    ("2026-09-17", "b8", "nip", 1),
    ("2026-09-17", "b8", "luminosity", 1),
    ("2026-09-17", "nip", "metizport", 1),
    ("2026-09-17", "gamerlegion", "nip", 1),
    ("2026-09-17", "m80", "metizport", 1),
    ("2026-09-18", "luminosity", "m80", 1),
    ("2026-09-18", "metizport", "gamerlegion", 1),
    ("2026-09-18", "b8", "m80", 1),
    ("2026-09-18", "luminosity", "m80", 1),
    ("2026-09-18", "innercircle", "bbl", 1),
    ("2026-09-18", "liquid", "infinite", 1),
    ("2026-09-18", "3dmax", "eyeballers", 1),
    ("2026-09-18", "bbl", "infinite", 1),
    ("2026-09-18", "innercircle", "liquid", 1),
    ("2026-09-18", "innercircle", "3dmax", 1),
    ("2026-09-18", "eyeballers", "liquid", 1),
    ("2026-09-18", "innercircle", "infinite", 1),
    ("2026-09-18", "3dmax", "bbl", 1),
    ("2026-09-19", "liquid", "3dmax", 1),
    ("2026-09-19", "bbl", "eyeballers", 1),
    ("2026-09-19", "bbl", "liquid", 1),
    ("2026-09-19", "eyeballers", "infinite", 1),
    ("2026-09-19", "eyeballers", "innercircle", 1),
    ("2026-09-19", "3dmax", "infinite", 1),
    ("2026-09-21", "luminosity", "bbl", 3),
    ("2026-09-21", "3dmax", "m80", 3),
    ("2026-09-22", "luminosity", "b8", 3),
    ("2026-09-23", "3dmax", "innercircle", 3),
    ("2026-09-24", "3dmax", "luminosity", 5),
]


def main():
    idx, seen = coverage()
    mx = {1: 25, 3: 45, 5: 55}
    print("== Logitech G Connect 2026 · 35 场盲预测 ==")
    n1 = sum(1 for s in SCHED if s[3] == 1)
    n3 = sum(1 for s in SCHED if s[3] == 3)
    n5 = sum(1 for s in SCHED if s[3] == 5)
    print("   bo1 %d 场 / bo3 %d 场 / bo5 %d 场   满分 %d"
          % (n1, n3, n5, n1 * 25 + n3 * 45 + n5 * 55))
    print("   ⚠ bo1 比分项整项不计（满分 25）；选图 7 选 1、MVP 占 40%\n")
    out, fail = [], []
    for date, a, b, bo in SCHED:
        A, B = ALIAS[a], ALIAS[b]
        da, db = resolve(A, idx, seen), resolve(B, idx, seen)
        r = subprocess.run([sys.executable, "-X", "utf8", "predict.py", "match",
                            "--A", A, "--B", B, "--bo", str(bo)],
                           capture_output=True, cwd=HERE)
        txt = r.stdout.decode("utf-8", "replace")
        if "胜方:" not in txt:
            fail.append((date, A, B, (r.stderr.decode("utf-8", "replace")
                                      or txt)[-160:].strip()))
            print("  %s  %-22s vs %-22s  **无法预测**" % (date, A, B))
            continue
        w = re.search(r"胜方:\s*(.+?)\s+P=([\d.]+)%", txt)
        sc = re.search(r"比分:\s*(\d):(\d+)", txt)
        mp = re.search(r"选图:\s*(.+)", txt)
        mv = re.search(r"MVP\s*:\s*(\S+)", txt)
        ep = re.search(r"期望得分\s*([\d.]+)", txt)
        exp = float(ep.group(1)) if ep else 0.0
        flags = []
        for q, d in ((A, da), (B, db)):
            if d is None or d["n"] < 10:
                flags.append("🔴%s(语料 %d 场)" % (q, d["n"] if d else 0))
        out.append({"date": date, "A": A, "B": B, "bo": bo,
                    "winner": w.group(1), "p_win": float(w.group(2)),
                    "score": "%d:%d" % (int(sc.group(1)), int(sc.group(2))) if sc else "",
                    "maps": mp.group(1).strip() if mp else "",
                    "mvp": mv.group(1) if mv else "",
                    "exp": exp, "flags": flags})
        print("  %s  %-22s %s %-22s P=%.1f%%  %-5s %-30s %-12s %5.1f %s"
              % (date, A, "vs", B, float(w.group(2)),
                 out[-1]["score"] if bo > 1 else "-", out[-1]["maps"],
                 out[-1]["mvp"], exp, ("⚠ " + ";".join(flags)) if flags else ""))
    json.dump(out, open(os.path.join(DATA, "groundtruth", "lgc_predictions.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    tot = sum(o["exp"] for o in out)
    full = sum(mx[o["bo"]] for o in out)
    print("\n  成功 %d 场，失败 %d 场" % (len(out), len(fail)))
    for f in fail:
        print("    ✗ %s %s vs %s → %s" % f)
    print("\n  合计期望 %.1f / %d = %.2f%%" % (tot, full, 100 * tot / full))
    for bo in (1, 3, 5):
        g = [o for o in out if o["bo"] == bo]
        if g:
            e = sum(o["exp"] for o in g)
            f2 = mx[bo] * len(g)
            print("    bo%d  %2d 场  期望 %6.1f / %3d = %.1f%%"
                  % (bo, len(g), e, f2, 100 * e / f2))
    print("\n已写 lgc_predictions.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
