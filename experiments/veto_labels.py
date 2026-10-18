#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""issue-014 第 3 项（选图）**读法 B** 的真值构建器。

读法 B（2026-10-08 用户裁决 Q-010）：真值图集 = veto 后全部图，**含「已定但实际没打」的决胜图**。
两个来源拼起来：
  - csapi `maps[*].name`  → 实际打过的图
  - HLTV veto（`csapi_veto.jsonl`）→ `picked`（=实际打的）与 `leftover`（=决胜图）
逐场交叉校验（2026-10-08 定稿，三态）：
  - **硬不变式**：csapi 打过的图必须落在 `picks ∪ {decider}` 内，否则标 suspicious；
  - **缺口语义**：csapi 少一张时，少的**不一定是没打的决胜图**——bo3 打成 2:1 时少的那张是**打过**的
    （实测 58 场 / 59 个图位；另 1,399 个图位才是真的没打）。判据用**序列分之和**（bo3 → 2 或 3，
    bo5 → 3/4/5），但 **bo1 的比分字段实测是回合分**（13:11 / 19:16），故只在「最大分 == 取胜所需局数」
    时启用，且**只在 csapi 有缺口时**用它补，绝不推翻 csapi 已列满的证据。
  - `gap_state` 记录三态：`none` / `unplayed`（缺的就是没打的）/ `played_missing`（缺的是打过的）
    / `ambiguous`（缺口内无法判定）。

降级：没有可用 veto 时只能退回读法 A（只算实际打过的图），并显式标 `maps_mode="played"`——
**与读法 B 的分数不可直接比较**（bo3 打成 2:0 时上限 40 而非 45）。

用法：
  python -X utf8 veto_labels.py coverage                  # 覆盖率与质量报告
  python -X utf8 veto_labels.py emit out.jsonl            # 导出逐场真值
  python -X utf8 veto_labels.py selftest                  # 自证
"""
from __future__ import annotations
import io, json, sys, collections

# bo → 应有的图数（含未打的决胜图）。2 与 7 是 csapi 的怪异取值，按邻近值处理并标记。
BO_FULL = {1: 1, 3: 3, 5: 5}
BO_ALIAS = {2: 3, 7: 5}
BO_SUSPECT = {2, 7, 25}


def load_matches(path="csapi_matches_all.jsonl"):
    out = {}
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[r["id"]] = r            # 同 id 后者覆盖（merge 去重后应唯一）
    return out


def load_veto(path="csapi_veto.jsonl"):
    """id → 最优 veto 记录。好记录（veto_ok）优先；同类取最后一行。"""
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


def played_of(row):
    return [m["name"] for m in (row.get("maps") or []) if m.get("name")]


def build_truth(row, veto_rec=None):
    """→ dict（含 maps / maps_mode / 校验标志 / 图池）。"""
    bo_raw = row.get("best_of")
    bo = BO_ALIAS.get(bo_raw, bo_raw)
    full = BO_FULL.get(bo)
    played = played_of(row)
    played_set = set(played)
    t = {
        "id": row["id"], "date": row.get("date"), "event": row.get("event"),
        "bo": bo, "bo_raw": bo_raw, "bo_suspect": bo_raw in BO_SUSPECT,
        "teamA": (row.get("team1") or {}).get("name"),
        "teamB": (row.get("team2") or {}).get("name"),
        "score": ((row.get("team1") or {}).get("score"), (row.get("team2") or {}).get("score")),
        "winner": (row.get("winner") or {}).get("name"),
        "maps_played": played, "full": full,
        "maps": list(played), "maps_mode": "played", "decider": None,
        "pool": None, "xcheck": None, "suspicious": False, "notes": [],
        "maps_unplayed": [], "maps_played_true": None, "gap_state": None,
    }
    if full is None:
        t["suspicious"] = True
        t["notes"].append("未知 bo=%r，无法判定应有图数" % (bo_raw,))
        return t

    v = veto_rec if (veto_rec and veto_rec.get("veto_ok")) else None
    if v:
        picks = [s[3] for s in v["veto"] if s[1] == "pi"]
        decider = v.get("decider")
        t["pool"] = sorted({s[3] for s in v["veto"]})
        t["decider"] = decider
        P = picks + ([decider] if decider else [])
        t["maps"] = list(P)
        t["maps_mode"] = "veto"
        # 实测不变式：**bo3 的 veto 是 2 pick + 1 leftover；bo5 是 4 pick + 1 leftover**
        # （原稿按「picks == 实际打的图」判，bo5 会误报——已由 2025-11-09 FURIA/Vitality
        #   等 2 场暴露）。正确判据是「实际打的 ⊆ picks」且「picks ∪ {decider} 恰好满图数」。
        # bo5 打成 3:0 时第 4 个 pick 与 leftover 都没打 —— 这两张正是读法 B 要补的图。
        #
        # 2026-10-08 追加第三态：csapi 的图少一张时，**少的不一定是没打的决胜图**。
        # 实测 bo3 打成 2:1 时 csapi 只列 2 张，第 3 张是**打过**的（实例 2397058 / 2397043 /
        # 2397031 / 2396829）。判据用**序列分之和**（bo3 → 2 或 3；bo5 → 3/4/5），
        # 但 **bo1 的比分字段实测是回合分**（13:11 / 19:16），故只在「最大分 == 取胜所需局数」
        # 时启用；且**只在 csapi 有缺口时**用它补，不推翻 csapi 已列满的证据。
        gap = [m for m in dict.fromkeys(P) if m not in played_set]
        # 硬不变式：csapi 打过的图必须全部落在「picks ∪ 决胜图」里。
        # 注意 **csapi 图数据为空**（实测 2395959：bo3 一张图都没给）时不能算「不一致」——
        # 那是「无法校验」，单列一态，别把数据缺陷误报成与 veto 冲突。
        if not played_set:
            t["xcheck"] = None
            t["suspicious"] = True
            t["notes"].append("csapi 图数据为空，无法交叉校验")
        else:
            t["xcheck"] = (played_set <= set(P))
            if not t["xcheck"]:
                t["suspicious"] = True
                t["notes"].append("csapi 实际打的图不在 veto 的图集（picks ∪ 决胜图）内")
        s1, s2 = t["score"]
        wins_needed = (full + 1) // 2
        usable = (isinstance(s1, int) and isinstance(s2, int) and max(s1, s2) == wins_needed)
        sc = (s1 + s2) if usable else None
        in_pool = [m for m in dict.fromkeys(P) if m in played_set]
        if not gap and len(P) == full and len(set(P)) == full:
            t["gap_state"] = "none"
            t["maps_played_true"] = in_pool
            t["maps_unplayed"] = []
            t["notes"].append("系列打满，无未打的图")
        elif sc is None or sc > full or len(in_pool) > sc:
            # 比分不可用（bo1 回合分等）→ 维持旧语义：缺的就是没打的
            t["gap_state"] = "unplayed"
            t["maps_played_true"] = in_pool
            t["maps_unplayed"] = gap
        else:
            n_fill = sc - len(in_pool)
            if n_fill == 0:
                t["gap_state"] = "unplayed"
                t["maps_played_true"] = in_pool
                t["maps_unplayed"] = gap
            elif n_fill >= len(gap):
                t["gap_state"] = "played_missing"
                t["maps_played_true"] = in_pool + gap
                t["maps_unplayed"] = []
                t["notes"].append("csapi 漏了已打过的图，按系列分补齐：%s" % ", ".join(gap))
            elif not in_pool:
                # csapi 一张都没给（数据为空）→ 按 **veto 顺序规则**补：先打 picks，再打决胜图，
                # 故前 sc 张即实际打过的（bo5 3:0 时第 4 个 pick 与决胜图都没打，与此一致）。
                t["gap_state"] = "played_by_order"
                t["maps_played_true"] = list(P[:sc])
                t["maps_unplayed"] = list(P[sc:])
                t["notes"].append("csapi 无图数据，按 veto 顺序规则取前 %d 张为已打" % sc)
            else:
                t["gap_state"] = "ambiguous"
                t["maps_played_true"] = None
                t["maps_unplayed"] = []
                t["notes"].append("缺口内无法判定哪 %d 张打过（csapi 缺 %d 张、系列分只差 %d 张）"
                                  % (n_fill, len(gap), n_fill))
        # 决胜图不可能在「没打到决胜图」的系列里出场——这类记录自相矛盾
        if decider and decider in played_set and sc is not None and sc < full:
            t["suspicious"] = True
            t["notes"].append("csapi 把决胜图列为已打，但系列分 %d < 满图数 %d" % (sc, full))
        if len(set(P)) != full:
            t["suspicious"] = True
            t["notes"].append("图集大小 %d != 应有 %d" % (len(set(P)), full))
        return t

    # 没有可用 veto
    if len(played_set) == full:
        t["maps_mode"] = "veto"        # 已完整（系列打满），无需 veto 也满足读法 B
        t["gap_state"] = "no_veto_complete"
        t["maps_played_true"] = list(dict.fromkeys(played_set))
        t["notes"].append("系列打满，csapi 图集本身已完整")
    else:
        t["notes"].append("缺 veto → 降级读法 A（只算实际打过的图）")
    return t


# ------------------------------------------------------------------ 报告
def coverage(matches=None, vetos=None, quiet=False):
    matches = matches if matches is not None else load_matches()
    vetos = vetos if vetos is not None else load_veto()
    rows = [build_truth(r, vetos.get(r["id"])) for r in matches.values()]
    by_mode = collections.Counter(r["maps_mode"] for r in rows)
    need = [r for r in rows if r["full"] and len(set(r["maps_played"])) < r["full"]]
    got = [r for r in need if r["maps_mode"] == "veto"]
    sus = [r for r in rows if r["suspicious"]]
    xbad = [r for r in rows if r["xcheck"] is False]
    pool = collections.Counter()
    for r in rows:
        if r["pool"]:
            pool[tuple(r["pool"])] += 1
    if not quiet:
        print("matches          :", len(rows))
        print("需要补 veto 的场次:", len(need))
        print("已走读法 B 的场次 :", got.__len__(), "(%.1f%%)" % (100.0 * len(got) / max(1, len(need))))
        print("maps_mode 分布    :", dict(by_mode))
        print("cross-check 失败  :", len(xbad))
        gs = collections.Counter(r["gap_state"] for r in rows if r["maps_mode"] == "veto")
        print("缺口构成(读法 B)  :", dict(sorted(gs.items(), key=lambda x: str(x[0]))))
        drop = [r for r in rows if r["gap_state"] == "played_missing"]
        print("csapi 漏了已打过的图:", len(drop), "场（maps_played 由 veto 补齐）",
              [r["id"] for r in drop][:6])
        nu = collections.Counter(len(r["maps_unplayed"]) for r in rows if r["maps_mode"] == "veto")
        print("读法 B 下未打的图数:", dict(sorted(nu.items())))
        print("suspicious        :", len(sus), [r["id"] for r in sus][:8])
        print("bo 取值分布       :", dict(collections.Counter(r["bo_raw"] for r in rows)))
        print("疑似 bo 异常      :", sum(1 for r in rows if r["bo_suspect"]))
        print()
        print("--- 图池（按 veto 的 7 张，出现 ≥1 次的组合）---")
        for k, c in pool.most_common(12):
            print("  %4d  %s" % (c, " / ".join(k)))
        print()
        print("--- 已收集到的决胜图（前 12）---")
        for r in rows:
            if r["decider"]:
                print("  %s bo%s %s: played=%s + decider=%s" %
                      (r["id"], r["bo"], r["date"], r["maps_played"], r["decider"]))
                if r is not rows[-1] and len([x for x in rows[:rows.index(r) + 1] if x["decider"]]) >= 12:
                    break
    return {"rows": rows, "need": need, "got": got, "sus": sus, "xbad": xbad, "pool": pool}


def emit(path):
    matches, vetos = load_matches(), load_veto()
    n = 0
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in matches.values():
            t = build_truth(r, vetos.get(r["id"]))
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
            n += 1
    print("EMIT_OK %d rows -> %s" % (n, path))


# ------------------------------------------------------------------ selftest
PICKABLE = ["Ancient", "Anubis", "Dust2", "Inferno", "Mirage", "Nuke", "Cache"]


def _mk(id, bo, played, s1=2, s2=0):
    return {"id": id, "best_of": bo, "date": "2026-01-01", "event": "E",
            "team1": {"name": "A", "score": s1}, "team2": {"name": "B", "score": s2},
            "winner": {"name": "A" if s1 > s2 else "B"},
            "maps": [{"name": m} for m in played]}


def _vk(id, bo, picks, decider):
    """造一条形状合法的 7 步 veto：2 ban / 2 pick / 2 ban / leftover。"""
    rest = [m for m in PICKABLE if m not in picks and m != decider][:4]
    seq = [[1, "re", "X", rest[0]], [2, "re", "Y", rest[1]],
           [3, "pi", "X", picks[0]], [4, "pi", "Y", picks[1]],
           [5, "re", "X", rest[2]], [6, "re", "Y", rest[3]],
           [7, "L", None, decider]]
    return {"id": id, "bo": bo, "veto": seq, "decider": decider, "veto_ok": True,
            "picks_match": True, "played": list(picks)}


def _selftest():
    fails = []

    def chk(name, got, want):
        if got != want:
            fails.append("%s: got=%r want=%r" % (name, got, want))

    # 1) veto 路径：bo3 2:0 → 真值含未打的决胜图（读法 B 的核心）
    t = build_truth(_mk(1, 3, ["Dust2", "Mirage"]), _vk(1, 3, ["Dust2", "Mirage"], "Inferno"))
    chk("veto.mode", t["maps_mode"], "veto")
    chk("veto.n_maps", len(set(t["maps"])), 3)
    chk("veto.has_decider", "Inferno" in t["maps"], True)
    chk("veto.xcheck", t["xcheck"], True)
    chk("veto.not_suspicious", t["suspicious"], False)

    # 2) 无 veto → 降级读法 A（必须显式标注，不能与读法 B 混算）
    t = build_truth(_mk(2, 3, ["Dust2", "Mirage"]))
    chk("degrade.mode", t["maps_mode"], "played")
    chk("degrade.n_maps", len(t["maps"]), 2)
    chk("degrade.decider", t["decider"], None)

    # 3) 系列打满 → 图集本身已完整
    t = build_truth(_mk(3, 3, ["Dust2", "Mirage", "Inferno"]))
    chk("full.mode", t["maps_mode"], "veto")
    chk("full.n_maps", len(t["maps"]), 3)

    # 4) bo1 恒完整
    t = build_truth(_mk(4, 1, ["Dust2"]))
    chk("bo1.mode", t["maps_mode"], "veto")
    chk("bo1.n_maps", len(t["maps"]), 1)

    # 5) bo5 打 3 图且无 veto → 降级
    t = build_truth(_mk(5, 5, ["Dust2", "Mirage", "Inferno"]))
    chk("bo5deg.mode", t["maps_mode"], "played")
    chk("bo5deg.full", t["full"], 5)

    # 6) veto_ok=False 的记录必须被忽略
    bad = dict(_vk(6, 3, ["Dust2", "Mirage"], "Inferno")); bad["veto_ok"] = False
    chk("badrec.mode", build_truth(_mk(6, 3, ["Dust2", "Mirage"]), bad)["maps_mode"], "played")

    # 7) picked 与实际打的图不符 → suspicious，但仍给出决胜图
    t = build_truth(_mk(7, 3, ["Dust2", "Anubis"]), _vk(7, 3, ["Dust2", "Mirage"], "Inferno"))
    chk("xbad.xcheck", t["xcheck"], False)
    chk("xbad.suspicious", t["suspicious"], True)
    chk("xbad.decider", t["decider"], "Inferno")

    # 8) 决胜图本场实际打过 → 留提示
    t = build_truth(_mk(8, 3, ["Dust2", "Mirage", "Inferno"]),
                    _vk(8, 3, ["Dust2", "Mirage"], "Inferno"))
    chk("playeddec.note", any("系列打满" in n for n in t["notes"]), True)
    chk("playeddec.n_maps", len(set(t["maps"])), 3)
    chk("playeddec.unplayed", t["maps_unplayed"], [])

    # 8b) bo5 的 veto 是 4 pick + 1 leftover：打成 3:0 时**两张**没打（第 4 pick + leftover）
    #     —— 这条是修正判据前会误报的真实回归夹具（取自 2025-11-09 FURIA vs Vitality）
    v5 = {"id": 11, "bo": 5, "veto_ok": True, "decider": "Mirage",
          "veto": [[1, "re", "A", "Nuke"], [2, "re", "B", "Train"],
                   [3, "pi", "A", "Ancient"], [4, "pi", "B", "Inferno"],
                   [5, "pi", "A", "Overpass"], [6, "pi", "B", "Dust2"],
                   [7, "L", None, "Mirage"]]}
    t = build_truth(_mk(11, 5, ["Inferno", "Overpass", "Ancient"], s1=0, s2=3), v5)
    chk("bo5veto.mode", t["maps_mode"], "veto")
    chk("bo5veto.n_maps", len(set(t["maps"])), 5)
    chk("bo5veto.xcheck", t["xcheck"], True)
    chk("bo5veto.unplayed", sorted(t["maps_unplayed"]), ["Dust2", "Mirage"])
    chk("bo5veto.not_suspicious", t["suspicious"], False)

    # 9) load_veto 去重：坏记录不得盖掉好记录
    import tempfile, os
    p = os.path.join(tempfile.gettempdir(), "_vl_test.jsonl")
    with io.open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps(_vk(9, 3, ["Dust2", "Mirage"], "Inferno")) + "\n")
        f.write(json.dumps({"id": 9, "veto_ok": False, "veto": None}) + "\n")
    chk("dedupe.good_wins", load_veto(p)[9]["veto_ok"], True)
    os.remove(p)

    # 10) csapi 怪异 bo（25）被标记且不崩
    t = build_truth(_mk(10, 25, ["Dust2"]))
    chk("weird.suspect", t["bo_suspect"], True)
    chk("weird.suspicious", t["suspicious"], True)

    # 11) ★第三态：csapi 漏了**打过**的图。取自实测 2397058（Nuclear TigeRES 2:1 Nemesis）：
    #     picks=[Ancient, Dust2]，decider=Inferno，而 csapi 只给 ['Dust2','Inferno']
    #     —— 缺的那张 Ancient 是打过的一张 pick，不是没打的决胜图。
    t = build_truth(_mk(12, 3, ["Dust2", "Inferno"], s1=2, s2=1),
                    _vk(12, 3, ["Ancient", "Dust2"], "Inferno"))
    chk("drop.xcheck", t["xcheck"], True)
    chk("drop.gap_state", t["gap_state"], "played_missing")
    chk("drop.not_suspicious", t["suspicious"], False)
    chk("drop.unplayed", t["maps_unplayed"], [])
    chk("drop.played_true_n", len(set(t["maps_played_true"] or [])), 3)
    chk("drop.truth_n", len(set(t["maps"])), 3)

    # 12) 反向对照：同一形状但系列分是 2:0（没打到决胜图）→ 缺口必须仍判成「没打的」
    t = build_truth(_mk(13, 3, ["Dust2", "Inferno"], s1=2, s2=0),
                    _vk(13, 3, ["Dust2", "Inferno"], "Ancient"))
    chk("keep.unplayed_state", t["gap_state"], "unplayed")
    chk("keep.unplayed_list", t["maps_unplayed"], ["Ancient"])
    chk("keep.played_true_n", len(set(t["maps_played_true"] or [])), 2)

    # 13) 自相矛盾：决胜图被列为已打，但系列分说没打满 → suspicious
    t = build_truth(_mk(14, 3, ["Dust2", "Mirage", "Inferno"], s1=2, s2=0),
                    _vk(14, 3, ["Dust2", "Mirage"], "Inferno"))
    chk("contra.suspicious", t["suspicious"], True)

    # 14) csapi 图数据为空（实测 2395959）：xcheck 必须为 None（无法校验，而非不一致），
    #     且按 veto 顺序规则补出「已打」——bo3 2:0 时前 2 张（两个 pick）为已打。
    t = build_truth(_mk(15, 3, [], s1=2, s2=0), _vk(15, 3, ["Inferno", "Cache"], "Ancient"))
    chk("empty.xcheck", t["xcheck"], None)
    chk("empty.gap_state", t["gap_state"], "played_by_order")
    chk("empty.played_true_n", len(set(t["maps_played_true"] or [])), 2)
    chk("empty.unplayed", t["maps_unplayed"], ["Ancient"])
    chk("empty.suspicious", t["suspicious"], True)

    if fails:
        print("VETO_LABELS_SELFTEST_FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("VETO_LABELS_SELFTEST_OK (14 groups + bo5 regression)")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "coverage"
    if cmd == "selftest":
        raise SystemExit(_selftest())
    if cmd == "coverage":
        coverage()
    elif cmd == "emit":
        emit(sys.argv[2] if len(sys.argv) > 2 else "truth_maps_B.jsonl")
    else:
        print(__doc__)
