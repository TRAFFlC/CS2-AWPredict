#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""issue-014 数据集**时间切分**（防泄漏）——只产出清单，不改动原始数据。

三条硬原则：
1. **按时间切，绝不随机切**。特征只能用该局之前的信息，随机切会让未来信息漏进训练。
2. **赛事完整（event-complete）**：一场赛事不得跨越两个切分。否则同一赛事的条件（同一版本、
   同一图池、同一批队伍的状态）会从训练泄漏到验证/测试，把分数抬高成假象。
   → 边界不直接采用给定的日期，而是**吸附到赛事边界**：取「最后一个在边界日之前结束的赛事」的结束日。
3. **切分清单绑数据指纹**：manifest 记录输入文件的 sha256 与各切分的场次/赛事/队伍统计，
   使结论可复现、可审计。

用法：
  python -X utf8 split.py report                        # 按默认边界切并打印报告
  python -X utf8 split.py manifest split_manifest.json  # 写清单
  python -X utf8 split.py selftest                      # 自证
  python -X utf8 split.py manifest out.json --purge --exclude ids.txt   # 剔除清单内的场次（Q-011 裁决）

"""
from __future__ import annotations
import io, json, sys, hashlib, collections

# 0000-00-00 作为「无上界/无下界」哨兵，避免 None 判断散落各处
LO, HI = "0000-00-00", "9999-12-31"

# 默认边界（依 2800 场按月分布选定：76.5% / 10.0% / 13.5%）。可被命令行覆盖。
DEFAULT_B1 = "2026-06-01"   # train | valid
DEFAULT_B2 = "2026-08-01"   # valid | test


def load(path="csapi_matches_all.jsonl"):
    rows = []
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def sha256(path):
    h = hashlib.sha256()
    with io.open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_exclude(path):
    """读「剔除清单」：每行一个 match id（# 开头为注释、空行忽略）。

    用途（2026-10-10 主理人对 Q-011 的裁决）：**过渡带里无法定池的场次从样本中去除**。
    清单是独立的数据产物（带出处与计数），使剔除可审计、可复跑；
    `split()` 默认不剔除，只有显式传入时才生效。
    """
    ids = set()
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.add(int(line.split()[0]))
    return ids


def event_spans(rows):
    """event → (min_date, max_date, n)。"""
    sp = {}
    for r in rows:
        k = r.get("event") or "?"
        d = r.get("date") or LO
        if k not in sp:
            sp[k] = [d, d, 0]
        s = sp[k]
        s[0] = min(s[0], d); s[1] = max(s[1], d); s[2] += 1
    return {k: tuple(v) for k, v in sp.items()}


def snap_boundary(rows, boundary):
    """把边界日吸附到「干净的切点」：该日**不落在任何赛事的区间内**。

    早期实现只取「最后一个在 boundary 前结束的赛事」的结束日，这是错的——
    该日可能正落在**另一个更长赛事**的区间中间，于是那个赛事照样跨切分。
    （真实语料上由此漏掉了 BLAST Bounty 2026 Season 2 Finals 等赛事，见反向对照。）
    现改为：按日期从近到远试每个「赛事结束日」，取第一个满足「无任何赛事满足 start <= d < end」的。

    若找不到干净切点，返回 LO（该切分为空，由调用方报错）。
    """
    spans = list(event_spans(rows).values())
    for d in sorted({mx for (_, mx, _) in spans if mx < boundary}, reverse=True):
        if all(not (mn <= d < mx) for (mn, mx, _) in spans):
            return d
    return LO


def purge_straddlers(rows, boundaries):
    """剔除「区间横跨任一边界日」的赛事（整场剔除），返回 (kept, dropped_events)。"""
    spans = event_spans(rows)
    bad = {k for k, (mn, mx, _) in spans.items() if any(mn <= b < mx for b in boundaries)}
    kept = [r for r in rows if (r.get("event") or "?") not in bad]
    return kept, sorted(bad)


def assign(rows, b1, b2):
    """按 (date <= b1) / (b1 < date <= b2) / (date > b2) 三档归类。"""
    out = {"train": [], "valid": [], "test": []}
    for r in rows:
        d = r.get("date") or LO
        out["train" if d <= b1 else ("valid" if d <= b2 else "test")].append(r)
    return out


def split(rows, b1=DEFAULT_B1, b2=DEFAULT_B2, snap=True, purge=False, exclude=None):
    """切分并返回 (buckets, meta)。

    两种保障「赛事不跨切分」的策略（互斥，由 purge 选择）：
      - purge=False（默认，无损）：**移动边界**吸附到日历允许的干净切点。不丢数据，
        但线上联赛跨月时边界会被拉早（本语料实测：valid|test 从 8-01 拉早到 6-25）。
      - purge=True（保边界）：先**整场剔除**横跨边界的赛事，再吸附。保住原定边界与更均衡的
        valid/test 比例，代价是本语料实测损失 86 场（3.1%，7 项三线线上联赛）。
    """
    raw = {"b1": b1, "b2": b2}
    dropped = []
    n_before = len(rows)
    if exclude:
        rows = [r for r in rows if r.get("id") not in exclude]
    n_excluded = n_before - len(rows)
    if purge:
        rows, dropped = purge_straddlers(rows, [b1, b2])
    if snap:
        b1 = snap_boundary(rows, b1)
        b2 = snap_boundary(rows, b2)
        if b2 <= b1:
            raise ValueError("吸附后边界非法：b1=%s b2=%s" % (b1, b2))
    buckets = assign(rows, b1, b2)
    meta = {"requested": raw, "b1": b1, "b2": b2, "snapped": snap, "purge": purge,
            "excluded_matches": n_excluded,
            "dropped_events": dropped, "dropped_matches": len(rows) - sum(len(v) for v in buckets.values()) if purge else 0,
            "counts": {k: len(v) for k, v in buckets.items()}}
    return buckets, meta


def check_leak(buckets, spans_notes=True):
    """泄漏检查。返回 (errors, infos)。"""
    errs, infos = [], []
    def rng(k):
        ds = [r.get("date") or LO for r in buckets[k]]
        return (min(ds), max(ds)) if ds else (None, None)
    tr, va, te = rng("train"), rng("valid"), rng("test")
    # L1 时间单调：后一段必须严格晚于前一段
    if tr[1] and va[0] and not (tr[1] < va[0]):
        errs.append("train 最大日 %s 未早于 valid 最小日 %s" % (tr[1], va[0]))
    if va[1] and te[0] and not (va[1] < te[0]):
        errs.append("valid 最大日 %s 未早于 test 最小日 %s" % (va[1], te[0]))
    # L2 赛事不得跨切分（吸附后应当自动成立；仍显式验证，防吸附失效）
    ev = {k: set(r.get("event") for r in buckets[k]) for k in buckets}
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        both = ev[a] & ev[b]
        if both:
            errs.append("赛事跨切分 %s∩%s: %s" % (a, b, sorted(both)[:5]))
    # L3 信息：测试集里训练从未见过的队伍（真正的泛化面）
    teams = {k: set() for k in buckets}
    for k in buckets:
        for r in buckets[k]:
            teams[k].add((r.get("team1") or {}).get("name"))
            teams[k].add((r.get("team2") or {}).get("name"))
    for k in ("valid", "test"):
        new = teams[k] - teams["train"]
        infos.append("%s 中训练未见过的队伍: %d / %d" % (k, len(new), len(teams[k])))
    return errs, infos


def summarize(buckets):
    """每档的场次/赛事/队伍/bo 构成 + 需补 veto 的场次。"""
    BO_FULL = {1: 1, 2: 3, 3: 3, 5: 5, 7: 5}
    out = {}
    for k, rows in buckets.items():
        ev, tm = set(), set()
        bo = collections.Counter(); need = 0
        for r in rows:
            ev.add(r.get("event"))
            tm.add((r.get("team1") or {}).get("name")); tm.add((r.get("team2") or {}).get("name"))
            bo[r.get("best_of")] += 1
            played = {m.get("name") for m in (r.get("maps") or []) if m.get("name")}
            full = BO_FULL.get(r.get("best_of"))
            if full and len(played) < full:
                need += 1
        ds = [r.get("date") or LO for r in rows]
        out[k] = {"matches": len(rows), "events": len(ev), "teams": len(tm),
                  "date_range": (min(ds), max(ds)) if ds else (None, None),
                  "bo": dict(sorted(bo.items())), "need_veto": need}
    return out


def report(buckets, meta, src="csapi_matches_all.jsonl", quiet=False, tl=None):
    """打印报告并返回 (ok, errs, infos)。quiet=True 时只判断不打印（自证用）。"""
    emit = (lambda *a: None) if quiet else print
    errs, infos = check_leak(buckets)
    pinfo, perrs, pnotes = (pool_diagnostics(buckets, tl, meta) if tl else (None, [], []))
    errs = errs + perrs
    infos = infos + pnotes
    for k in ("train", "valid", "test"):
        if not buckets[k]:
            errs.append("%s 为空——该切分无法评测" % k)
    summ = summarize(buckets)
    emit("输入        : %s" % src)
    emit("请求边界    : train|valid=%s  valid|test=%s" % (meta["requested"]["b1"], meta["requested"]["b2"]))
    emit("吸附后边界  : train|valid=%s  valid|test=%s" % (meta["b1"], meta["b2"]))
    emit()
    emit("%-6s %8s %7s %6s %24s %6s %s" % ("档", "场次", "赛事", "队伍", "日期范围", "待补", "占比"))
    tot = sum(len(v) for v in buckets.values()) or 1
    for k in ("train", "valid", "test"):
        s = summ[k]
        emit("%-6s %8d %7d %6d %24s %6d %.1f%%" %
              (k, s["matches"], s["events"], s["teams"],
               "%s..%s" % s["date_range"], s["need_veto"], 100.0 * s["matches"] / tot))
    emit()
    for k in ("train", "valid", "test"):
        emit("  %-6s bo 构成: %s" % (k, summ[k]["bo"]))
    emit()
    if pinfo:
        band = tl.get("band") or {}
        emit("=== 池时代构成（%d 个已证时代；过渡带 %s .. %s）===" %
             (len(tl["eras"]), band.get("first", "-"), band.get("last", "-")))
        tags = sorted({t for v in pinfo["crosstab"].values() for t in v})
        emit("%-6s %s" % ("档", " ".join("%-12s" % t for t in tags)))
        for k in ("train", "valid", "test"):
            emit("%-6s %s" % (k, " ".join("%-12s" % ("%d/%d" % (
                pinfo["crosstab"][k].get(t, {}).get("matches", 0),
                pinfo["crosstab"][k].get(t, {}).get("events", 0))) for t in tags)))
        emit("        （每格 = 场次/赛事数）")
        for k in ("train", "valid", "test"):
            c = pinfo["candidates"].get(k)
            if not c:
                continue
            emit("  %-6s 可证候选图集: 起点 %d 张 [%s]" % (k, c["start_n"], ", ".join(c["start"])))
            emit("  %-6s             终点 %d 张 [%s]   （%s → %s）" %
                 ("", c["end_n"], ", ".join(c["end"]), c["basis_lo"], c["basis_hi"]))
            flo, fhi = c["forbidden_lo"], c["forbidden_hi"]
            if flo or fhi:
                # 必须**分段**打印：区间首日已证不可用的图（如 Cache 之于 test 起点）在区间末日可能是可用的，
                # 取并集写成一个「已证不可用」会与本档的「终点 N 张 [...]」自相矛盾。
                emit("  %-6s             已证不可用: 起点 [%s]  终点 [%s]" %
                     ("", ", ".join(flo) or "无", ", ".join(fhi) or "无"))
        for k in ("valid", "test"):
            if k not in pinfo["new_maps"]:
                continue
            v = pinfo["new_maps"][k]
            emit("  %-6s 训练集从未出现的图（跨池新维度，无历史可学）: %s" %
                 (k, ", ".join(v) if v else "无"))
        emit()
    for i in infos:
        emit("  INFO", i)
    if errs:
        emit()
        for e in errs:
            emit("  ERROR", e)
        emit("SPLIT_LEAK_CHECK_FAIL")
        return False, errs, infos
    emit("SPLIT_LEAK_CHECK_OK (时间单调 + 无赛事跨切分)")
    return True, errs, infos


def manifest(buckets, meta, path, src="csapi_matches_all.jsonl", tl=None):
    ok, errs, infos = report(buckets, meta, src, tl=tl)
    pinfo = pool_diagnostics(buckets, tl, meta)[0] if tl else None
    m = {"source": src, "source_sha256": sha256(src),
         "boundaries": {"requested": meta["requested"], "snapped": {"b1": meta["b1"], "b2": meta["b2"]},
                        "policy": ("event-complete + purge（先整场剔除跨边界赛事，再吸附）" if meta.get("purge")
                        else "event-complete + move（移动边界吸附到干净切点，不丢数据）"),
                        "dropped_events": meta.get("dropped_events", []),
                        "dropped_matches": meta.get("dropped_matches", 0),
                        "excluded_matches": meta.get("excluded_matches", 0)},
         "counts": {k: len(v) for k, v in buckets.items()},
         "summary": summarize(buckets), "leak_errors": errs, "notes": infos,
         "pool_timeline": ({"eras": tl["eras"], "band": tl.get("band"),
                            "crosstab": pinfo["crosstab"], "candidates": pinfo["candidates"],
                            "new_maps": pinfo["new_maps"]} if pinfo else None),
         "ok": ok,
         "sample_ids": {k: [r["id"] for r in buckets[k][:10]] for k in buckets}}
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(m, ensure_ascii=False, indent=2) + "\n")
    print("\nmanifest -> %s (ok=%s, sha256=%s)" % (path, ok, m["source_sha256"][:12]))
    return m


# ------------------------------------------------------------------ 图池时代
DEFAULT_TIMELINE = "map_pool_timeline.json"


def load_timeline(path=DEFAULT_TIMELINE):
    """读 map_pools.py 导出的时间线；缺失则返回 None（此时切分退化为纯时间切分）。"""
    try:
        with io.open(path, encoding="utf-8") as f:
            tl = json.load(f)
    except (IOError, OSError, ValueError):
        return None
    return tl if isinstance(tl, dict) and tl.get("kind") == "map_pool_timeline" else None


def era_tag(date, tl):
    """日期 → 池标签 (kind, i[, j])。

      ("era",  i)     落在**已证纯时代** i 内（池已知，7 张）
      ("band", i, j)  落在时代 i 与 j 之间的过渡带（池按赛事而定，必须逐场，不能按日推）
      ("post", i)     在**最后一个已证时代** i 之后（veto 未覆盖，池只部分可知）
      ("pre",)        在第一个已证时代之前
    """
    eras = tl["eras"]
    for e in eras:
        if e["start"] <= date <= e["end"]:
            return ("era", e["index"])
    before = [e for e in eras if e["end"] < date]
    after = [e for e in eras if e["start"] > date]
    if before and after:
        return ("band", before[-1]["index"], after[0]["index"])
    if before:
        return ("post", before[-1]["index"])
    return ("pre",)


def era_name(tag):
    if tag[0] == "era":
        return "era%d" % tag[1]
    if tag[0] == "band":
        return "band%d|%d" % (tag[1], tag[2])
    if tag[0] == "post":
        return "post%d" % tag[1]
    return "pre"


def era_crosstab(buckets, tl):
    """每档 × 每个池时代：场次 / 赛事数。"""
    out = {}
    for k, rows in buckets.items():
        per = {}
        for r in rows:
            tag = era_name(era_tag(r.get("date") or LO, tl))
            e = per.setdefault(tag, {"matches": 0, "events": set()})
            e["matches"] += 1
            e["events"].add(r.get("event"))
        out[k] = {t: {"matches": v["matches"], "events": len(v["events"])}
                  for t, v in sorted(per.items())}
    return out


def pool_diagnostics(buckets, tl, meta=None):
    """跨池诊断：候选图集、跨池新增图、边界是否落在过渡带。返回 (info, errs, notes)。

    只用**可证**的东西：candidate_pool 的 provable 集；缺 1 张时报「未定」而不是瞎猜。
    """
    errs, notes = [], []
    info = {"crosstab": era_crosstab(buckets, tl), "candidates": {}, "new_maps": {},
            "band": tl.get("band")}
    try:
        import map_pools as mp
    except ImportError:
        notes.append("未能 import map_pools，跳过候选图集分析")
        mp = None
    if mp is not None:
        a = mp.analyze()
        for k in ("train", "valid", "test"):
            rows = buckets.get(k) or []
            if not rows:
                continue
            ds = sorted(r.get("date") or LO for r in rows)
            c_lo, c_hi = mp.candidate_pool(ds[0], a), mp.candidate_pool(ds[-1], a)
            info["candidates"][k] = {"start": c_lo["provable"], "end": c_hi["provable"],
                                     "start_n": c_lo["n"], "end_n": c_hi["n"],
                                     "basis_lo": c_lo["basis"], "basis_hi": c_hi["basis"],
                                     "forbidden_lo": c_lo["forbidden"],
                                     "forbidden_hi": c_hi["forbidden"]}
        train_maps = set(info["candidates"].get("train", {}).get("start", [])) |             set(info["candidates"].get("train", {}).get("end", []))
        for k in ("valid", "test"):
            c = info["candidates"].get(k)
            if not c:
                continue
            cand = set(c["start"]) | set(c["end"])
            info["new_maps"][k] = sorted(cand - train_maps)
            if len(cand) < 7:
                notes.append("%s 的可证候选只有 %d 张（缺 %d 张未定）——依赖 veto 覆盖补齐"
                             % (k, len(cand), 7 - len(cand)))
    # 边界不得落在过渡带内：过渡带的池按赛事而定，按日切会把两种池混进两个档
    band = tl.get("band")
    if band:
        b1, b2 = meta.get("b1"), meta.get("b2") if meta else (None, None)
        for nm, d in (("train|valid", b1), ("valid|test", b2)):
            if not d:
                continue
            if band["first"] <= d <= band["last"]:
                errs.append("边界 %s=%s 落在过渡带 %s..%s 内：两侧池归属按赛事不同，切分不干净"
                            % (nm, d, band["first"], band["last"]))
            else:
                inside = [x for x in tl["eras"] if x["start"] <= d <= x["end"]]
                if inside:
                    notes.append("边界 %s=%s 落在已证时代 era%d 内部" % (nm, d, inside[0]["index"]))
    return info, errs, notes


# ------------------------------------------------------------------ selftest
def _mk(id, date, event, t1="A", t2="B", bo=3, nmaps=2):
    return {"id": id, "date": date, "event": event, "best_of": bo,
            "team1": {"name": t1}, "team2": {"name": t2},
            "maps": [{"name": "M%d" % i} for i in range(nmaps)]}


def _fixture():
    r = []
    for i, d in enumerate(["2025-10-01", "2025-10-02", "2025-10-03"]):
        r.append(_mk(100 + i, d, "E1"))
    for i, d in enumerate(["2025-10-05", "2025-10-06", "2025-10-08"]):
        r.append(_mk(200 + i, d, "E2"))
    for i, d in enumerate(["2025-11-01", "2025-11-02", "2025-11-04"]):
        r.append(_mk(300 + i, d, "E3", t1="C"))
    for i, d in enumerate(["2025-12-01", "2025-12-02"]):
        r.append(_mk(400 + i, d, "E4", t1="D"))
    return r


def _selftest():
    fails = []

    def chk(n, got, want):
        if got != want:
            fails.append("%s: got=%r want=%r" % (n, got, want))

    rows = _fixture()
    # 1) 边界吸附到赛事边界
    chk("snap.b1", snap_boundary(rows, "2025-10-20"), "2025-10-08")
    chk("snap.b2", snap_boundary(rows, "2025-11-15"), "2025-11-04")
    # 2) 吸附后切分正确
    b, meta = split(rows, "2025-10-20", "2025-11-15")
    chk("split.train", len(b["train"]), 6)
    chk("split.valid", len(b["valid"]), 3)
    chk("split.test", len(b["test"]), 2)
    chk("split.snapped_b1", meta["b1"], "2025-10-08")
    # 3) 吸附后无泄漏
    errs, infos = check_leak(b)
    chk("leak.ok", errs, [])
    chk("leak.info_present", len(infos), 2)
    # 4) **不吸附时必须能抓到赛事跨切分**（否则检查形同虚设）
    b2, _ = split(rows, "2025-10-06", "2025-11-15", snap=False)
    errs2, _ = check_leak(b2)
    chk("leak.detects_straddle", any("赛事跨切分" in e for e in errs2), True)
    # 5) 空档必须报错
    b3, meta3 = split(rows, "2025-10-20", "2026-06-01")
    ok3, errs3, _ = report(b3, meta3, src=__file__, quiet=True)
    chk("empty.test_is_error", any("test 为空" in e for e in errs3), True)
    chk("empty.ok_false", ok3, False)
    # 6) summarize 统计正确
    s = summarize(b)
    chk("summ.train_matches", s["train"]["matches"], 6)
    chk("summ.valid_teams", s["valid"]["teams"], 2)   # E3 只有队伍 B/C
    chk("summ.test_need_veto", s["test"]["need_veto"], 2)   # bo3 只打 2 图 → 两张都缺
    # 7) 指纹可算
    chk("sha256.len", len(sha256(__file__)), 64)

    # 8) **切点不得落在某个赛事的区间中间**（真实语料上曝光过的回归夹具）
    #    E_LONG 横跨 10-07..10-21，故候选切点 10-10（E2 的结束日）必须被拒绝，退到 10-03
    rows3 = _fixture() + [_mk(500, "2025-10-07", "E_LONG"), _mk(501, "2025-10-21", "E_LONG")]
    chk("snap.avoids_span", snap_boundary(rows3, "2025-10-19"), "2025-10-03")
    b4, _ = split(rows3, "2025-10-19", "2025-11-15")
    errs4, _ = check_leak(b4)
    chk("snap.no_straddle_after", errs4, [])

    # 9) purge 策略：整场剔除跨边界赛事后无泄漏，且该赛事一场不剩
    b5, meta5 = split(rows3, "2025-10-19", "2025-11-15", purge=True)
    chk("purge.dropped", meta5["dropped_events"], ["E_LONG"])
    chk("purge.no_E_LONG", any((r.get("event") == "E_LONG") for k in b5 for r in b5[k]), False)
    errs5, _ = check_leak(b5)
    chk("purge.no_leak", errs5, [])

    # 10) 图池时代层：标签 / 交叉表 / 边界落在过渡带必须报错
    tl = {"kind": "map_pool_timeline",
          "eras": [{"index": 0, "start": "2025-10-01", "end": "2025-10-20",
                    "pool": ["A", "B", "C", "D", "E", "F", "G"], "maps": 7, "n_veto": 10},
                   {"index": 1, "start": "2025-10-25", "end": "2025-11-20",
                    "pool": ["A", "B", "C", "D", "E", "F", "H"], "maps": 7, "n_veto": 10}],
          "band": {"first": "2025-10-21", "last": "2025-10-23", "dates": ["2025-10-21"],
                   "n_dates": 3, "n_minority_matches": 2, "n_transition_dates": 1},
          "coverage_gaps": [], "veto_span": ["2025-10-01", "2025-11-20"]}
    chk("era.in_era0", era_tag("2025-10-05", tl), ("era", 0))
    chk("era.in_era1", era_tag("2025-11-01", tl), ("era", 1))
    chk("era.band", era_tag("2025-10-22", tl), ("band", 0, 1))
    chk("era.post", era_tag("2025-12-01", tl), ("post", 1))
    chk("era.pre", era_tag("2025-09-01", tl), ("pre",))
    chk("era.names", [era_name(era_tag(d, tl)) for d in
                      ("2025-10-05", "2025-10-22", "2025-12-01", "2025-09-01")],
        ["era0", "band0|1", "post1", "pre"])
    chk("era.load_missing", load_timeline("__no_such_timeline__.json"), None)
    chk("era.load_rejects_other", load_timeline(__file__), None)   # 不是 timeline 结构
    b6, meta6 = split(_fixture(), "2025-10-19", "2025-11-15", snap=False)
    ctab = era_crosstab(b6, tl)
    for k in ("train", "valid", "test"):
        chk("era.ctab.%s" % k, sum(v["matches"] for v in ctab[k].values()), len(b6[k]))
    # 边界落在过渡带内必须报错（过渡带的池按赛事而定，按日切不干净）
    _i, perrs, _n = pool_diagnostics(b6, tl, {"b1": "2025-10-22", "b2": "2025-11-15"})
    chk("era.band_boundary_error", any("过渡带" in e for e in perrs), True)
    _i2, perrs2, _n2 = pool_diagnostics(b6, tl, {"b1": "2025-10-05", "b2": "2025-11-15"})
    chk("era.ok_boundary", perrs2, [])
    # report(quiet) 必须在带 timeline 时把 pool 错误算进 ok
    ok6, errs6, _ = report(b6, meta6, src=__file__, quiet=True, tl=tl)
    chk("era.report_quiet_ok", (ok6, errs6), (True, []))
    ok7, errs7, _ = report(b6, dict(meta6, b1="2025-10-22"), src=__file__, quiet=True, tl=tl)
    chk("era.report_quiet_fail", (ok7, any("过渡带" in e for e in errs7)), (False, True))

    # 11) 剔除清单：只剔给定 id，且 meta 记录数量（Q-011 过渡带去样）
    ex = {rows[0]["id"], rows[1]["id"]}
    b7, meta7 = split(rows, "2025-10-20", "2025-11-15", exclude=ex)
    chk("exclude.count", meta7["excluded_matches"], 2)
    chk("exclude.total", sum(len(v) for v in b7.values()), sum(len(v) for v in b.values()) - 2)
    chk("exclude.no_exclude_means_zero", split(rows, "2025-10-20", "2025-11-15")[1]["excluded_matches"], 0)

    if fails:
        print("SPLIT_SELFTEST_FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("SPLIT_SELFTEST_OK (11 groups)")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "selftest":
        raise SystemExit(_selftest())
    rows = load()
    use_purge = "--purge" in sys.argv
    b1 = sys.argv[sys.argv.index("--b1") + 1] if "--b1" in sys.argv else DEFAULT_B1
    b2 = sys.argv[sys.argv.index("--b2") + 1] if "--b2" in sys.argv else DEFAULT_B2
    tl = None
    if "--no-timeline" not in sys.argv:
        tl_path = (sys.argv[sys.argv.index("--timeline") + 1] if "--timeline" in sys.argv
                   else DEFAULT_TIMELINE)
        tl = load_timeline(tl_path)
        if tl is None:
            print("WARN 未找到可用的图池时间线 %s —— 退化为纯时间切分" % tl_path)
    exclude = load_exclude(sys.argv[sys.argv.index("--exclude") + 1]) if "--exclude" in sys.argv else None
    b, meta = split(rows, b1, b2, purge=use_purge, exclude=exclude)
    if cmd == "report":
        ok, _, _ = report(b, meta, tl=tl)
        raise SystemExit(0 if ok else 1)
    if cmd == "manifest":
        m = manifest(b, meta, sys.argv[2] if len(sys.argv) > 2 else "split_manifest.json", tl=tl)
        raise SystemExit(0 if m["ok"] else 1)
    print(__doc__)
