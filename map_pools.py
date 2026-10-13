#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""issue-014 地图池的**时变时代划分**（map pool eras）——从数据实测，不照抄记忆。

为什么必须有这个：
- 第 3 项（选图）的**候选集必须是「当时的 7 张」**，不能用今天的池去后验。
- 「某队在某张图上很强」这类特征，只在**该图在池内**的时段可迁移。换图后该维度
  要么消失（Train / Overpass 离池），要么从零开始（Anubis / Cache 进池）。
- 跨池直接混训会稀释这类维度（主理人 2026-10-08 提醒：「跨不同图池的数据也有关联性、
  但逻辑可能被减弱，数据量小的时候模型可能学不到」）。故切分与特征都要显式按池时代分段。

两个来源、精度不同：
1. **veto（精确）**：HLTV 的 7 步 veto 会把**没打的图也列出来**，故给出该场确切池。
2. **实际打的图（有偏）**：某图若在窗口内一场没打就看不见它。
   故只用来定「某图的可见窗口」，**不用它直接定池**。但它能做两件定池做不了的事：
   - `vanish_evidence`：「末现之后」的 0 次是不是真离池（泊松尾部）。
   - `entry_evidence`：「首现之前」的 0 次是不是真没进池（对称的尾部检验）。

★ 实测反常识一（2026-10-08，461 场 veto 证据）：
**池不是日期的全局函数。** 2026-01-13 .. 2026-02-03 期间，同一天的不同赛事拿到不同的池
（例：2026-01-18 的 CCT Europe Series 13 是含 Anubis 的新池，同日的 Roman Imperium Cup IV /
European Pro League Series 4 仍是含 Train 的旧池；EPL S4 一路到 2026-02-03 都还是旧池）。
→ 故本工具的输出分两层：**逐场池（权威，有 veto 时）** 与 **时代标签（粗分段，用于切分与报告）**，
并在「过渡带」里显式列出少数派场次，避免把粗标签当成事实。

★ 实测反常识二（必须留在明面上）：**On 2026-06-25 之后 Overpass 一场没打**（其后 464 场、
  按末现前 60 天的出现率 0.252 期望应出现 117 次 → 观测 0 次）；而 **Cache 直到 2026-08-20
  才首次出现**（此前 2,540 场、0 次）。两者按「7 张池」不能同时成立，
  ⇒ **B|C 换图日无法只靠 played 定出**，必须等 veto 覆盖 2026-03-09 之后（见覆盖率缺口）。
  本工具**不做猜测**，只把证据摆出来。

用法：
  python -X utf8 map_pools.py report                    # 时代划分 + 证据 + 不确定性
  python -X utf8 map_pools.py timeline out.json         # 导出机器可读时间线（供 split.py）
  python -X utf8 map_pools.py probes 2026-08-02 ...     # 关键日期的**可证候选图集**
  python -X utf8 map_pools.py selftest
"""
from __future__ import annotations
import io, json, math, sys, collections

POOL_SIZE = 7          # CS2 active duty 是 7 张
MIN_ERA_N = 5          # 证据少于该场次的段标 transient（多为个别赛事的独立配置）
IMP_P = 0.01           # 泊松尾概率阈值：低于此即判「该图已离池 / 才进池」而非「恰好没被选」
RATE_WINDOW = 60       # 估计某图出现率时回看的自然日数
MIN_RATE_N = 20        # 估率的样本少于该场次 → 不出「离池/进池」结论（防一拍定案）


# ------------------------------------------------------------------ 载入
def load_matches(path="csapi_matches_all.jsonl"):
    out = {}
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["id"]] = r
    return out


def load_veto(path="csapi_veto.jsonl"):
    """id → 最好的 veto 记录（好记录优先；同类取后写的）。"""
    out = {}
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        prev = out.get(r["id"])
        if prev is None or (r.get("veto_ok") and not prev.get("veto_ok")):
            out[r["id"]] = r
    return out


# ------------------------------------------------------------------ 基本量
def pool_key(maps):
    return "/".join(sorted(set(maps)))


def veto_pool(v):
    """从一条 veto 记录取该场的池（去重、排序）。"""
    return tuple(sorted({s[3] for s in (v.get("veto") or [])}))


def _days(a, b):
    """(b - a) 的天数；只用于同一 YYYY-MM-DD 口径。"""
    import datetime
    fa = datetime.date(*map(int, a.split("-")))
    fb = datetime.date(*map(int, b.split("-")))
    return (fb - fa).days


def map_windows(matches):
    """图 → (首现, 末现, 场次)。只看**实际打的图**，故是「可见窗口」而非池窗口。"""
    w = {}
    for r in matches.values():
        d = r.get("date")
        if not d:
            continue
        for m in (r.get("maps") or []):
            n = m.get("name")
            if not n:
                continue
            if n not in w:
                w[n] = [d, d, 0]
            e = w[n]
            e[0] = min(e[0], d)
            e[1] = max(e[1], d)
            e[2] += 1
    return {k: tuple(v) for k, v in w.items()}


def per_date_pools(matches, vetos):
    """日期 → Counter(池)；只计 veto_ok 的场次。"""
    pdp = collections.defaultdict(collections.Counter)
    for mid, v in vetos.items():
        if not v.get("veto_ok"):
            continue
        d = (matches.get(mid) or {}).get("date")
        if not d:
            continue
        pdp[d][pool_key(veto_pool(v))] += 1
    return dict(pdp)


def observed_pools(matches, vetos):
    """池 → {dates, n, ids, lo, hi}（按 id 排序后聚合）。"""
    pools = {}
    for mid, v in sorted(vetos.items()):
        if not v.get("veto_ok"):
            continue
        d = (matches.get(mid) or {}).get("date")
        if not d:
            continue
        k = pool_key(veto_pool(v))
        e = pools.setdefault(k, {"dates": set(), "n": 0, "ids": []})
        e["dates"].add(d)
        e["n"] += 1
        e["ids"].append(mid)
    for k, e in pools.items():
        e["dates"] = sorted(e["dates"])
        e["lo"], e["hi"] = e["dates"][0], e["dates"][-1]
    return pools


def transition_dates(pdp, threshold=2):
    """同日出现 ≥ threshold 个不同池的日期 → 说明池不是全局函数。"""
    out = []
    for d in sorted(pdp):
        c = pdp[d]
        if len(c) >= threshold:
            out.append((d, dict(c)))
    return out


def dominant_series(pdp):
    """日期 → 该日多数派的池；平票记 None（该日无法归入任一段）。"""
    ser = []
    for d in sorted(pdp):
        top = pdp[d].most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            ser.append((d, None))
        else:
            ser.append((d, top[0][0]))
    return ser


def era_segments(series):
    """把 (日期, 池) 序列压成连续段 [(起, 止, 池, 0, 0, 天数)]。

    `None` 是独立取值（会切断段），以免一个平票日被吞进相邻段里。
    第 4/5 位（证据场次、少数派场次）由 annotate_segments 回填。
    """
    if not series:
        return []
    out = []
    cur = [series[0][0], series[0][0], series[0][1], 0, 0, 0]
    for d, p in series:
        if p == cur[2]:
            cur[1] = d
            cur[5] += 1
        else:
            out.append(tuple(cur))
            cur = [d, d, p, 0, 0, 1]
    out.append(tuple(cur))
    return out


def annotate_segments(segs, pdp):
    """给每段补上证据场次与「属于别的池」的场次（平票段自身不算少数派）。"""
    out = []
    for lo, hi, key, _, _, nd in segs:
        n = 0
        minority = 0
        for d in pdp:
            if lo <= d <= hi:
                for k, c in pdp[d].items():
                    if k == key:
                        n += c
                    elif key is not None:
                        minority += c
        out.append((lo, hi, key, n, minority, nd))
    return out


def pure_eras(segs, min_n=MIN_ERA_N):
    """只保留「有明确池且证据足够」的段；平票日与瞬时段交给过渡带报告。"""
    return [s for s in segs if s[2] is not None and s[3] >= min_n]


def minority_dates(pdp, eras):
    """每个时代**区间内**出现「非本时代池」的日期 → 过渡带的确切位置。"""
    out = []
    for lo, hi, key, _, _, _ in eras:
        for d in sorted(pdp):
            if lo <= d <= hi:
                other = {k: c for k, c in pdp[d].items() if k != key}
                if other:
                    out.append((d, other))
    return out


def transition_band(pdp, segs, eras):
    """过渡带 = 过渡日 ∪ 时代内少数派日，取最小..最大日期；含带内少数派场次数。"""
    td = transition_dates(pdp)
    rd = minority_dates(pdp, eras)
    ds = sorted(set([d for d, _ in td]) | set([d for d, _ in rd]))
    if not ds:
        return None
    n_minor = sum(sum(other.values()) for _, other in rd)
    return {"first": ds[0], "last": ds[-1], "dates": ds, "n_dates": len(ds),
            "n_minority_matches": n_minor, "n_transition_dates": len(td)}


def boundary_spans(matches, eras, pdp):
    """相邻**纯时代**之间的日期空隙——真实换图日必落在其中（保守区间）。"""
    dates = sorted({r.get("date") for r in matches.values() if r.get("date")})
    ev = sorted(pdp)
    spans = []
    for a, b in zip(eras, eras[1:]):
        lo, hi = a[1], b[0]
        between = [d for d in dates if lo < d < hi]
        spans.append({"old_pool": a[2], "new_pool": b[2], "old_hi": lo, "new_lo": hi,
                      "gap_matches": len(between),
                      "gap_dates": (between[0], between[-1]) if between else None,
                      "veto_evidence_dates": len([d for d in ev if lo <= d <= hi])})
    return spans


def coverage_gaps(matches, vetos, min_days=1):
    """无 veto 证据的时段（按有比赛的日子切）——这些日子只能由 played 窗口界定。"""
    dates = sorted({r.get("date") for r in matches.values() if r.get("date")})
    ev = set(per_date_pools(matches, vetos))
    gaps, run = [], []
    for d in dates:
        if d in ev:
            if len(run) >= min_days:
                gaps.append((run[0], run[-1], len(run)))
            run = []
        else:
            run.append(d)
    if len(run) >= min_days:
        gaps.append((run[0], run[-1], len(run)))
    return gaps


def _rate_tail(windows, per_match, name, side, rate_window=RATE_WINDOW):
    """共用：用「地图确实在池内」的那侧 rate_window 天估出现率，再算另一侧的期望与 0 次概率。"""
    first, last, total = windows[name]
    if side == "after":                      # 率取自末现*之前*的 rate_window 天，看末现之后
        base = [s for (d, s) in per_match if d <= last and _days(d, last) < rate_window]
        other = [(d, s) for (d, s) in per_match if d > last]
        lo, hi = last, (other[0][0] if other else None)
    else:                                    # 率取自首现*之后*的 rate_window 天，看首现之前
        base = [s for (d, s) in per_match if first <= d <= last and _days(first, d) < rate_window]
        other = [(d, s) for (d, s) in per_match if d < first]
        lo, hi = (other[-1][0] if other else None), first
    hit = sum(1 for s in base if name in s)
    rate = (hit / len(base)) if base else 0.0
    exp = rate * len(other)
    p = math.exp(-exp) if exp > 0 else 1.0
    return {"first": first, "last": last, "played": total, "rate": round(rate, 4),
            "rate_n": len(base), "n_other": len(other), "expected": round(exp, 2),
            "p_zero": p, "window": (lo, hi)}


def vanish_evidence(matches, windows, rate_window=RATE_WINDOW):
    """「某图末现之后不再出现」是不是真离池？泊松尾部。

    基准率 = 末现前 rate_window 天内的出现率；期望 = 率 × 末现之后的场次数。
    观测 0 次的概率 < IMP_P ⇒ 判「已离池」（而不是「恰好没被选」）。
    """
    per_match = sorted((r.get("date"), {m.get("name") for m in (r.get("maps") or []) if m.get("name")})
                       for r in matches.values() if r.get("date"))
    out = {}
    for name in windows:
        e = _rate_tail(windows, per_match, name, "after", rate_window)
        e["left_pool"] = bool(e["rate_n"] >= MIN_RATE_N and e["expected"] > 0
                              and e["p_zero"] < IMP_P)
        out[name] = e
    return out


def entry_evidence(matches, windows, rate_window=RATE_WINDOW):
    """某图**首现之前**是否已在池？与 vanish_evidence 对称的尾部检验。"""
    per_match = sorted((r.get("date"), {m.get("name") for m in (r.get("maps") or []) if m.get("name")})
                       for r in matches.values() if r.get("date"))
    out = {}
    for name in windows:
        e = _rate_tail(windows, per_match, name, "before", rate_window)
        e["new_entry"] = bool(e["rate_n"] >= MIN_RATE_N and e["expected"] > 0
                             and e["p_zero"] < IMP_P)
        out[name] = e
    return out


def candidate_pool(date, a):
    """给定日期，返回**可证明**的候选图集（第 3 项「选图」的候选空间）。

    构造（每一步都可证，不做猜测）：
      ① 基线 = 该日期所属纯时代的池；落在过渡带则取相邻两时代池的**并集**（8 张，需二选一）；
         落在最后一个时代之后则取最后一个时代的池。
      ② 加入：played 窗口覆盖该日期的图（当天被打过 ⇒ 该日必在池）。
      ③ 剔除：已证离池（left_pool 且 last < date）或已证未进池（new_entry 且 first > date）。
         **例外（2026-10-08 修）**：日期落在**纯时代之内**时不做剔除 —— 时代的池本身就是逐场
         veto 的直接证据，比「played 首末现」派生证据强。实测反例：era#2（2026-07-21 起）的池含
         Cache，而 Cache 的 **played 首现是 08-20**；按派生证据会把 07-21..08-19 的 Cache 误判成
         「未进池」，于是 2026-07-30 被报成「6 张 + Cache 已证不可用」，与该时代 257 场 veto 直接冲突。
    返回 dict：provable / forbidden / n / band / basis / complete(n == POOL_SIZE)。
    n < POOL_SIZE ⇒ 有图的归属未定（必须等 veto 覆盖）；n > POOL_SIZE ⇒ 过渡带，需逐场定。
    """
    eras = a["segments"]
    if not eras:
        return {"provable": [], "forbidden": [], "n": 0, "band": False,
                "basis": "no-era", "complete": False}
    inside = [e for e in eras if e[0] <= date <= e[1]]
    band = False
    if inside:
        base, basis = set(inside[0][2].split("/")), "era#%d" % eras.index(inside[0])
    else:
        before = [e for e in eras if e[1] < date]
        after = [e for e in eras if e[0] > date]
        if before and after:
            base = set(before[-1][2].split("/")) | set(after[0][2].split("/"))
            band = True
            basis = "band(era#%d ∪ era#%d)" % (eras.index(before[-1]), eras.index(after[0]))
        elif before:
            base = set(before[-1][2].split("/"))
            basis = "era#%d+（末时代之后）" % eras.index(before[-1])
        else:
            base, basis = set(eras[0][2].split("/")), "era#0-（首时代之前）"
    provable = set(base)
    for m, w in a["windows"].items():
        if w[0] <= date <= w[1]:
            provable.add(m)
    forbidden = set()
    if not inside:
        # 只在**时代之外**（过渡带 / 末时代之后 / 首时代之前）才用「played 首末现」派生证据剔除：
        # 时代之内的池是直接 veto 证据，派生证据不得推翻它（见 docstring 例外）。
        for m, e in a["vanished"].items():
            if e["left_pool"] and e["last"] < date:
                forbidden.add(m)
        for m, e in a["entered"].items():
            if e["new_entry"] and e["first"] > date:
                forbidden.add(m)
    provable -= forbidden
    return {"provable": sorted(provable), "forbidden": sorted(forbidden),
            "n": len(provable), "band": band, "basis": basis,
            "complete": len(provable) == POOL_SIZE}


# ------------------------------------------------------------------ 汇总
def _veto_span(matches, vetos):
    ds = sorted((matches.get(i) or {}).get("date") for i, v in vetos.items()
                if v.get("veto_ok") and (matches.get(i) or {}).get("date"))
    return (ds[0], ds[-1]) if ds else (None, None)


def analyze(matches=None, vetos=None):
    matches = load_matches() if matches is None else matches
    vetos = load_veto() if vetos is None else vetos
    pdp = per_date_pools(matches, vetos)
    segs_all = annotate_segments(era_segments(dominant_series(pdp)), pdp)
    eras = pure_eras(segs_all)
    win = map_windows(matches)
    return {
        "windows": win,
        "pools": observed_pools(matches, vetos),
        "per_date_pools": pdp,
        "segments_all": segs_all,
        "segments": eras,
        "band": transition_band(pdp, segs_all, eras),
        "transitions": transition_dates(pdp),
        "boundary_spans": boundary_spans(matches, eras, pdp),
        "coverage_gaps": coverage_gaps(matches, vetos),
        "n_veto_ok": sum(1 for v in vetos.values() if v.get("veto_ok")),
        "veto_span": _veto_span(matches, vetos),
        "vanished": vanish_evidence(matches, win),
        "entered": entry_evidence(matches, win),
    }


def timeline(matches=None, vetos=None):
    """机器可读时间线。split.py 用它把「池时代」接进切分。"""
    a = analyze(matches, vetos)
    eras = []
    for i, (lo, hi, key, n, minority, nd) in enumerate(a["segments"]):
        maps = key.split("/")
        eras.append({
            "index": i, "start": lo, "end": hi, "pool": maps, "maps": len(maps),
            "n_veto": n, "n_minority": minority, "n_dates": nd,
            "suspect_size": len(maps) != POOL_SIZE,
        })
    return {
        "kind": "map_pool_timeline",
        "veto_span": list(a["veto_span"]),
        "n_veto_ok": a["n_veto_ok"],
        "pool_size_expected": POOL_SIZE,
        "eras": eras,
        "band": a["band"],
        "transitions": [{"date": d, "pools": c} for d, c in a["transitions"]],
        "boundary_spans": a["boundary_spans"],
        "coverage_gaps": [{"start": x[0], "end": x[1], "days": x[2]} for x in a["coverage_gaps"]],
        "map_windows": {k: {"first": v[0], "last": v[1], "played": v[2]}
                        for k, v in sorted(a["windows"].items())},
        "vanished": a["vanished"],
        "entered": a["entered"],
    }


# ------------------------------------------------------------------ 报告
def report(matches=None, vetos=None, quiet=False):
    a = analyze(matches, vetos)
    if quiet:
        return a
    W = a["windows"]
    print("=== 1. 图的可见窗口（来自**实际打的图**；只定「何时看得见」，不直接定池）===")
    print("%-10s %7s %-12s %-12s" % ("map", "场次", "首现", "末现"))
    for n in sorted(W, key=lambda x: -W[x][2]):
        print("%-10s %7d %-12s %-12s" % (n, W[n][2], W[n][0], W[n][1]))
    print()
    print("=== 2. 观测到的池（来自 veto；覆盖 %s .. %s，%d 场）===" %
          (a["veto_span"][0], a["veto_span"][1], a["n_veto_ok"]))
    for k in sorted(a["pools"], key=lambda x: a["pools"][x]["lo"]):
        e = a["pools"][k]
        tag = "" if len(k.split("/")) == POOL_SIZE else "   ⚠ 不是 %d 张" % POOL_SIZE
        print("  %s .. %s  n=%3d  %s%s" % (e["lo"], e["hi"], e["n"], k, tag))
    print()
    print("=== 3. 池时代（纯池段：有明确池且证据 ≥ %d 场）===" % MIN_ERA_N)
    tot = sum(s[3] for s in a["segments"]) or 1
    for lo, hi, key, n, minority, nd in a["segments"]:
        flag = "" if len(key.split("/")) == POOL_SIZE else "   ⚠ 只有 %d 张" % len(key.split("/"))
        print("  [#%d] %s .. %s  n=%3d (%4.1f%%)  日期 %d 天%s" %
              (a["segments"].index((lo, hi, key, n, minority, nd)), lo, hi, n,
               100.0 * n / tot, nd, flag))
        print("       %s" % " / ".join(key.split("/")))
        if minority:
            print("       ↳ 段内另有 %d 场属于**别的池**（见第 4/5 节）" % minority)
    if a["band"]:
        b = a["band"]
        print("  ── 过渡带：%s .. %s（%d 个含冲突的日子，%d 场少数派）──" %
              (b["first"], b["last"], b["n_dates"], b["n_minority_matches"]))
    print()
    print("=== 4. 过渡/重叠带明细：同一天出现多个池 → **池不是日期的全局函数** ===")
    if not a["transitions"]:
        print("  （无）")
    for d, c in a["transitions"]:
        n_bad = len(c) - 1
        print("  %s  （%d 个池并存）" % (d, len(c)))
        for k, n in sorted(c.items(), key=lambda x: -x[1]):
            print("      n=%2d  %s" % (n, k))
        if n_bad:
            pass
    print()
    print("=== 5. 相邻时代之间的边界不确定区间（真实换图日必落在其中）===")
    if not a["boundary_spans"]:
        print("  （相邻纯时代少于 2 个）")
    for s in a["boundary_spans"]:
        print("  旧池最晚证据 %s  →  新池最早证据 %s    空隙内 %d 场" %
              (s["old_hi"], s["new_lo"], s["gap_matches"]))
        if s["gap_dates"]:
            print("      空隙内有比赛的日子：%s .. %s" % s["gap_dates"])
        print("      旧池 %s" % s["old_pool"])
        print("      新池 %s" % s["new_pool"])
    print()
    for title, key, true_key in (("6. 离池证据：末现之后 0 次是不是真离池", "vanished", "left_pool"),
                                 ("7. 入池证据：首现之前 0 次是不是真没进池", "entered", "new_entry")):
        print("=== %s（泊松尾部，阈值 p < %g）===" % (title, IMP_P))
        print("%-10s %-12s %-12s %6s %8s %10s %12s %s" %
              ("map", "首现", "末现", "基准率", "对侧场次", "期望出现", "观测 0 次概率", "判定"))
        rows = sorted(a[key].items(), key=lambda x: x[1]["last"])
        for n, e in rows:
            if e["n_other"] == 0:
                continue
            print("%-10s %-12s %-12s %6.3f %8d %10.2f %12.3g %s" %
                  (n, e["first"], e["last"], e["rate"], e["n_other"], e["expected"],
                   e["p_zero"], "是" if e[true_key] else "不足以判定"))
        print()
    print("=== 8. 无 veto 证据的时段（这些日子**只能**由 played 窗口界定，池归属未定）===")
    for lo, hi, k in a["coverage_gaps"]:
        print("  %s .. %s  （%d 个有比赛的日子）" % (lo, hi, k))
    print()
    print("MAP_POOLS_REPORT_DONE eras=%d band=%s veto=%d gaps=%d" %
          (len(a["segments"]), (a["band"] or {}).get("first", "-"), a["n_veto_ok"],
           len(a["coverage_gaps"])))
    return a


# ------------------------------------------------------------------ 自证
def _M(id, date, maps=()):
    return {"id": id, "date": date, "maps": [{"name": m} for m in maps]}


def _selftest():
    fails = []

    def chk(n, got, want):
        if got != want:
            fails.append("%s: got=%r want=%r" % (n, got, want))

    # 夹具：旧池 A=Apple/Banana/Cherry（01-01..01-05，5 场），新池 B=Apple/Banana/Durian（01-07..01-11，5 场）
    A = ["Apple", "Banana", "Cherry"]
    B = ["Apple", "Banana", "Durian"]
    matches, vetos = {}, {}

    def add(mid, date, played, pool=None):
        matches[mid] = _M(mid, date, played)
        vetos[mid] = {"id": mid, "date": date, "veto_ok": bool(pool),
                      "veto": ([[1, "re", "T", m] for m in pool] if pool else None)}

    for i, d in enumerate(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]):
        add(100 + i, d, ["Apple", "Banana"], A)
    for i, d in enumerate(["2026-01-07", "2026-01-08", "2026-01-09", "2026-01-10", "2026-01-11"]):
        add(200 + i, d, ["Apple"], B)
    for i in range(3):                              # 瞬时段：证据 3 场 < MIN_ERA_N → 不进时代
        add(300 + i, "2026-01-12", ["Apple", "Elderberry"], ["Apple", "Elderberry"])
    add(999, "2026-01-13", [])                      # 撞限流页的坏记录

    # 1) veto_ok=False 不计入任何池
    pools = observed_pools(matches, vetos)
    chk("pools.keys", sorted(pools),
        ["Apple/Banana/Cherry", "Apple/Banana/Durian", "Apple/Elderberry"])
    chk("pools.A.n", pools["Apple/Banana/Cherry"]["n"], 5)
    chk("pools.B.span", (pools["Apple/Banana/Durian"]["lo"], pools["Apple/Banana/Durian"]["hi"]),
        ("2026-01-07", "2026-01-11"))
    chk("veto_pool.dedup", veto_pool({"veto": [[1, "re", "T", "A"], [2, "re", "T", "A"]]}), ("A",))

    # 2) 日期→池；坏记录所在日无证据
    pdp = per_date_pools(matches, vetos)
    chk("pdp.no_bad_date", "2026-01-13" in pdp, False)
    chk("pdp.n_dates", len(pdp), 11)

    # 3) 重叠检测：给 01-08 混入一场旧池 → 必须被抓到
    add(250, "2026-01-08", ["Cherry"], A)
    pdp2 = per_date_pools(matches, vetos)
    chk("transitions.dates", [d for d, _ in transition_dates(pdp2)], ["2026-01-08"])
    chk("transitions.mix", sorted(dict(transition_dates(pdp2))["2026-01-08"].items()),
        [("Apple/Banana/Cherry", 1), ("Apple/Banana/Durian", 1)])
    chk("dominant.tie_none", dict(dominant_series(pdp2))["2026-01-08"], None)
    chk("segs.tie_breaks", [x[2] for x in era_segments(dominant_series(pdp2))],
        ["Apple/Banana/Cherry", "Apple/Banana/Durian", None, "Apple/Banana/Durian",
         "Apple/Elderberry"])
    # 2 场新池 + 1 场旧池 → 多数派取新池
    add(251, "2026-01-08", ["Apple"], B)
    pdp3 = per_date_pools(matches, vetos)
    chk("dominant.majority", dict(dominant_series(pdp3))["2026-01-08"], "Apple/Banana/Durian")

    # 4) 段压缩 + 少数派计数；pure_eras 丢掉证据不足的段
    segs_all = annotate_segments(era_segments(dominant_series(pdp3)), pdp3)
    chk("segs.pools", [x[2] for x in segs_all],
        ["Apple/Banana/Cherry", "Apple/Banana/Durian", "Apple/Elderberry"])
    chk("segs.first", (segs_all[0][0], segs_all[0][1], segs_all[0][3]), ("2026-01-01", "2026-01-05", 5))
    chk("segs.second", (segs_all[1][0], segs_all[1][1], segs_all[1][3], segs_all[1][4]),
        ("2026-01-07", "2026-01-11", 6, 1))
    chk("pure_eras.keeps", len(pure_eras(segs_all)), 2)
    # 瞬时段必须「在 segments_all 里、但不在 pure_eras 里」——真正的集成路径
    chk("pure_eras.small_dropped",
        (any(x[2] == "Apple/Elderberry" for x in segs_all),
         any(x[2] == "Apple/Elderberry" for x in pure_eras(segs_all))), (True, False))
    chk("pure_eras.small_n", [x[3] for x in segs_all if x[2] == "Apple/Elderberry"], [3])
    chk("pure_eras.drops_small", pure_eras(annotate_segments(
        era_segments([("2026-02-01", "Apple"), ("2026-02-02", "Banana")]), pdp3)), [])

    # 5) 过渡带：01-08 是过渡日；01-13 有比赛但只有坏记录 → 无池证据，不算过渡日
    band = transition_band(pdp3, segs_all, pure_eras(segs_all))
    chk("band.span", (band["first"], band["last"]), ("2026-01-08", "2026-01-08"))
    chk("band.n_minority", band["n_minority_matches"], 1)
    chk("band.no_band", transition_band({}, [], []), None)
    md = minority_dates(pdp3, pure_eras(segs_all))
    chk("minority_dates.n", len(md), 1)
    chk("minority_dates.date", md[0][0], "2026-01-08")

    # 6) 边界空隙：纯时代 A 止于 01-05，B 起于 01-07 → 空隙里只有 01-06（无比赛）
    bs = boundary_spans(matches, pure_eras(segs_all), pdp3)
    chk("bspans.n", len(bs), 1)
    chk("bspans.gap", (bs[0]["gap_matches"], bs[0]["gap_dates"]), (0, None))
    chk("bspans.pools", (bs[0]["old_pool"], bs[0]["new_pool"]),
        ("Apple/Banana/Cherry", "Apple/Banana/Durian"))

    # 7) 覆盖缺口：01-13 有比赛但只有坏记录 → 本身就是一个缺口
    chk("gaps.baseline", coverage_gaps(matches, vetos), [("2026-01-13", "2026-01-13", 1)])
    v2 = dict(vetos)
    v2.pop(202)                      # 抹掉 01-09 的 veto
    chk("gaps.after_pop", coverage_gaps(matches, v2),
        [("2026-01-09", "2026-01-09", 1), ("2026-01-13", "2026-01-13", 1)])

    # 8) 离池判定（泊松尾部）——用专门夹具，别依赖上面的时代夹具
    m3 = {}
    for i in range(25):
        m3[i] = _M(i, "2026-01-01", ["Apple", "Cherry"])   # 估率样本 25 场 >= MIN_RATE_N
    for i in range(25, 400):
        m3[i] = _M(i, "2026-01-20", ["Apple"])
    m3[400] = _M(400, "2026-01-20", ["Apple", "Fig"])      # Fig 稀有但仍在池
    m3[401] = _M(401, "2026-02-01", ["Apple"])
    v3 = vanish_evidence(m3, map_windows(m3))
    chk("vanish.cherry.after", v3["Cherry"]["n_other"], 377)          # 375 + 1 + 1
    chk("vanish.cherry.left", v3["Cherry"]["left_pool"], True)
    chk("vanish.fig.not_left", v3["Fig"]["left_pool"], False)         # 稀有图不得误判
    chk("vanish.fig.p", v3["Fig"]["p_zero"] > IMP_P, True)
    chk("vanish.rate_n_guard", (v3["Cherry"]["rate_n"] >= MIN_RATE_N), True)
    chk("vanish.apple.last", (v3["Apple"]["n_other"], v3["Apple"]["left_pool"]), (0, False))
    m4 = {"a": _M("a", "2026-01-01", ["Apple"]), "b": _M("b", "2026-01-01", ["Apple"])}
    v4 = vanish_evidence(m4, map_windows(m4))
    chk("vanish.no_window_after", (v4["Apple"]["n_other"], v4["Apple"]["left_pool"]), (0, False))

    # 9) 入池判定（与 8 对称）：首现之前 400 场 0 次 → 判新进池
    m5 = {}
    for i in range(400):
        m5[i] = _M(i, "2026-01-01", ["Apple", "Fig"])
    for i in range(400, 425):
        m5[i] = _M(i, "2026-01-20", ["Apple", "Cherry"])   # 估率样本 26 场 >= MIN_RATE_N
    m5[500] = _M(500, "2026-01-21", ["Apple", "Cherry", "Melon"])
    e5 = entry_evidence(m5, map_windows(m5))
    chk("entry.cherry.before", e5["Cherry"]["n_other"], 400)
    chk("entry.cherry.rate", (e5["Cherry"]["rate"], e5["Cherry"]["rate_n"]), (1.0, 26))
    chk("entry.cherry.new", e5["Cherry"]["new_entry"], True)
    chk("entry.melon.new", e5["Melon"]["new_entry"], False)   # 估率样本 1 场 < MIN_RATE_N
    chk("entry.melon.rate_n", e5["Melon"]["rate_n"], 1)
    chk("entry.fig.at_start", (e5["Fig"]["n_other"], e5["Fig"]["new_entry"]), (0, False))

    # 10) timeline 结构
    tl = timeline(matches, vetos)
    chk("tl.kind", tl["kind"], "map_pool_timeline")
    chk("tl.eras.n", len(tl["eras"]), 2)
    chk("tl.pool_size", tl["pool_size_expected"], 7)
    chk("tl.suspect_size", tl["eras"][0]["suspect_size"], True)   # 夹具只有 3 张
    chk("tl.eras0.start", tl["eras"][0]["start"], "2026-01-01")
    chk("tl.transition_recorded", [t["date"] for t in tl["transitions"]], ["2026-01-08"])
    chk("tl.band", (tl["band"]["first"], tl["band"]["last"]), ("2026-01-08", "2026-01-08"))
    chk("tl.entered.keys", "Cherry" in tl["entered"], True)
    # map_windows 只来自**实际打的图**：Durian 只在 veto 池里 → 不应出现在窗口表
    chk("tl.map_windows.keys", sorted(tl["map_windows"]),
        ["Apple", "Banana", "Cherry", "Elderberry"])
    chk("tl.map_windows.no_veto_only_map", "Durian" in tl["map_windows"], False)
    chk("tl.veto_span", tl["veto_span"], ["2026-01-01", "2026-01-12"])
    chk("tl.json_roundtrip", len(json.dumps(tl, ensure_ascii=False)) > 100, True)

    # 11) 空输入不得崩
    chk("empty.segments", era_segments([]), [])
    chk("empty.pools", observed_pools({}, {}), {})
    chk("empty.timeline_eras", timeline({}, {})["eras"], [])
    chk("empty.vanish", vanish_evidence({}, {}), {})
    chk("empty.entry", entry_evidence({}, {}), {})
    chk("empty.dominant", dominant_series({}), [])
    chk("empty.coverage", coverage_gaps({}, {}), [])
    chk("empty.report", report({}, {}, quiet=True)["n_veto_ok"], 0)

    # 12) _days 正确
    chk("days.basic", _days("2026-01-01", "2026-01-31"), 30)
    chk("days.leap", _days("2026-02-28", "2026-03-01"), 1)

    # 13) candidate_pool：基线 ± 可证的进出池
    mp_a = analyze(matches, vetos)
    c1 = candidate_pool("2026-01-03", mp_a)          # 落在纯时代 A 内
    chk("cand.in_era.n", c1["n"], 3)
    chk("cand.in_era.basis", (c1["basis"], c1["band"]), ("era#0", False))
    c2 = candidate_pool("2026-01-06", mp_a)          # 两时代之间的空隙 → 并集
    chk("cand.between.band", (c2["band"], c2["n"]), (True, 4))
    chk("cand.between.basis", c2["basis"], "band(era#0 ∪ era#1)")
    c3 = candidate_pool("2026-01-13", mp_a)          # 末时代之后
    chk("cand.after.basis", c3["basis"], "era#1+（末时代之后）")
    chk("cand.no_era", candidate_pool("2026-01-01", analyze({}, {}))["basis"], "no-era")
    # 已证离池的图必须从候选里剔除
    mm, vv = {}, {}
    for i in range(30):                       # 30 场 → 估率样本 >= MIN_RATE_N
        mm[i] = _M(i, "2026-01-01", ["Apple", "Fig"])
        vv[i] = {"id": i, "date": "2026-01-01", "veto_ok": True,
                 "veto": [[1, "re", "T", m] for m in ["Apple", "Fig"]]}
    for i in range(30, 60):
        mm[i] = _M(i, "2026-01-15", ["Apple"])
        vv[i] = {"id": i, "date": "2026-01-15", "veto_ok": True,
                 "veto": [[1, "re", "T", m] for m in ["Apple", "Banana"]]}
    a6 = analyze(mm, vv)
    chk("cand.left_pool_evidence", a6["vanished"]["Fig"]["left_pool"], True)
    c4 = candidate_pool("2026-01-20", a6)
    chk("cand.forbids_left_map", ("Fig" in c4["forbidden"], "Fig" in c4["provable"]), (True, False))
    chk("cand.after_base", c4["provable"], ["Apple", "Banana"])

    # 14) ★ 纯时代内的池不得被 played 派生证据推翻（2026-10-08 修的真实反例形状）
    #     构造：03-01..03-20 的 veto 池含 Cache（但**没打过** Cache），Cache 的 played 首现 04-01。
    #     旧实现会把 03-10 报成「Cache 已证未进池」→ 2 张，与时代池直接冲突。
    mc, vc = {}, {}
    for i in range(60):
        d = "2026-03-01" if i < 30 else "2026-03-20"
        mc[i] = _M(i, d, ["Apple", "Banana"])
        vc[i] = {"id": i, "date": d, "veto_ok": True,
                 "veto": [[1, "re", "T", x] for x in ["Apple", "Banana", "Cache"]]}
    for i in range(60, 120):
        mc[i] = _M(i, "2026-04-01", ["Apple", "Cache"])
        vc[i] = {"id": i, "date": "2026-04-01", "veto_ok": True,
                 "veto": [[1, "re", "T", x] for x in ["Apple", "Cache", "Cherry"]]}
    a7 = analyze(mc, vc)
    chk("cand.era_cache_entry_evidence", a7["entered"]["Cache"]["new_entry"], True)
    chk("cand.era_cache_first_played", a7["entered"]["Cache"]["first"], "2026-04-01")
    c5 = candidate_pool("2026-03-10", a7)          # 落在纯时代内
    chk("cand.era_not_overridden", ("Cache" in c5["provable"], "Cache" in c5["forbidden"]), (True, False))
    chk("cand.era_n", c5["n"], 3)
    # 反向对照：同一天挪到**两时代之间**（03-25）→ 剔除必须恢复生效
    c6 = candidate_pool("2026-03-25", a7)
    chk("cand.band_still_forbids", ("Cache" in c6["forbidden"], c6["band"]), (True, True))

    if fails:
        print("MAP_POOLS_SELFTEST_FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("MAP_POOLS_SELFTEST_OK (14 groups)")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "selftest":
        raise SystemExit(_selftest())
    if cmd == "report":
        report()
        raise SystemExit(0)
    if cmd == "probes":
        a = analyze()
        for d in sys.argv[2:]:
            c = candidate_pool(d, a)
            print("%s  候选 %d 张  band=%s  依据=%s" % (d, c["n"], c["band"], c["basis"]))
            print("   可用: %s" % " / ".join(c["provable"]))
            if c["forbidden"]:
                print("   已证不可用: %s" % " / ".join(c["forbidden"]))
            if c["n"] != POOL_SIZE:
                print("   ⚠ %s" % ("过渡带：需在相邻两池间二选一" if c["n"] > POOL_SIZE
                                   else "证据不足：还差 %d 张未定" % (POOL_SIZE - c["n"])))
        raise SystemExit(0)
    if cmd == "timeline":
        path = sys.argv[2] if len(sys.argv) > 2 else "map_pool_timeline.json"
        tl = timeline()
        with io.open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(tl, ensure_ascii=False, indent=2) + "\n")
        print("TIMELINE_OK eras=%d -> %s" % (len(tl["eras"]), path))
        raise SystemExit(0)
    print(__doc__)
