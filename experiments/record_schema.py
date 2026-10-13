"""match_record —— 增量更新的规范单场记录（[P0]-1 第 A 步）

为什么需要它：`state.observe()` / `mh.observe_match()` 依赖 **4 组**输入，
而此前语料把它们分散在 4 个文件里，且没有任何格式把它们绑成"可增量消费的单场"：

  ① 逐图比分        → elo_m(分差更新) / mapw / tw / tp        [csapi matches[].maps[]]
  ② 完整阵容+rating → p_sum/p_w/p_adj/p_play/p_top/t_level_* [csapi stats.overall]
  ③ 逐图逐选手rating→ pm_sum / pm_w                          [csapi stats.by_map]
  ④ veto 顺序       → tban / tpick / gban / gpick / nban/npick [csapi veto]

EPL S24 那 41 场只满足"系列胜负 + 读法 B 图集"，所以增量只能更新 6 类账本。
本模块定义能喂满全部账本的 schema，并从现有语料导出，作为增量路径的**燃料**。

字段（全部 required，除 veto_order 外缺失即降级并在 validate 里记账）：
  id / date / event / best_of / team1 / team2 / winner
  maps_full_B   读法 B 全图集（bo 张，含 veto 已定但未打的决胜图）
  maps_played   实际打过的（通吃场次可能只有张数、缺名单）
  map_results   [{map, s1, s2}]            ← ①
  rosters       {team: [{name, rating}]}  ← ②
  by_map        {map: {team: [{name,rating}]}} ← ③
  veto_order    [[step, "re"|"pi"|"L", team|null, map]] ← ④
  pool          该场生效的 7 张图池
"""
import io, json, os, sys

SCHEMA_VERSION = "match_record/1"

REQUIRED = ["id", "date", "best_of", "team1", "team2", "winner", "maps_full_B"]
FUEL = ["map_results", "rosters", "by_map", "veto_order"]   # 四组「燃料」
CANON_MAPS = None   # 不写死图池：池随时点变，见 map_pool_timeline.json


def validate(rec, strict=True):
    """返回 (ok, problems, fuel_present)。fuel_present 是 4 组燃料的到位情况。"""
    p = []
    for k in REQUIRED:
        if k not in rec or rec[k] in (None, ""):
            p.append("缺字段 %s" % k)
    if p:
        return False, p, {}
    bo = rec["best_of"]
    if bo not in (1, 3, 5):
        p.append("best_of 非法 %r" % bo)
    t1, t2 = rec["team1"], rec["team2"]
    for t in (t1, t2):
        if not isinstance(t, dict) or "name" not in t or "score" not in t:
            p.append("team 需 {name, score}，实得 %r" % (t,))
    if not isinstance(rec.get("date"), str) or len(rec["date"]) != 10:
        p.append("date 需 YYYY-MM-DD，实得 %r" % rec.get("date"))
    n_b = (bo + 1) // 2
    if max(t1.get("score", 0), t2.get("score", 0)) < n_b:
        p.append("胜方分数 %d < bo%d 需要的 %d"
                 % (max(t1.get("score", 0), t2.get("score", 0)), bo, n_b))
    if t1.get("score", 0) + t2.get("score", 0) > bo:
        p.append("总打图数 %d 超过 bo%d（bo%d 最多 %d 张）"
                 % (t1.get("score", 0) + t2.get("score", 0), bo, bo, bo))
    mf = rec.get("maps_full_B") or []
    if bo == 3 and len(mf) != 3:
        p.append("读法 B：bo3 需 3 张图，实得 %d" % len(mf))
    if bo == 5 and len(mf) != 5:
        p.append("读法 B：bo5 需 5 张图，实得 %d" % len(mf))
    mp = rec.get("maps_played")
    if mp is not None and len(mp) > len(mf):
        p.append("maps_played(%d) 多于 maps_full_B(%d)" % (len(mp), len(mf)))

    fuel = {}
    fuel["map_results"] = bool(rec.get("map_results")) and \
        len(rec["map_results"]) == t1["score"] + t2["score"]
    fuel["rosters"] = bool(rec.get("rosters")) and \
        all(k in rec["rosters"] for k in (t1["name"], t2["name"]))
    fuel["by_map"] = bool(rec.get("by_map"))
    fuel["veto_order"] = bool(rec.get("veto_order"))
    if strict and not all(fuel.values()):
        missing = [k for k, v in fuel.items() if not v]
        p.append("燃料不全：%s（该场只能部分增量更新）" % ", ".join(missing))
    return (not p), p, fuel


def _veto_of(rec):
    """构造 observe_match 认识的 veto_rec；无 veto_order 时返回 None。

    ⚠ 语义（主理人 2026-10-12 纠正）：bo3 真实顺序是 ban,ban,pick,pick,ban,ban
    ⇒ 读法 B 的 3 张图里第 3 张是 **decider**，不是 pick。详见 veto_rec_build。
    """
    vo = rec.get("veto_order")
    if not vo:
        return None
    import veto_rec_build as VRB
    _ms, vr = VRB.build_veto_rec(vo, rec.get("best_of", 3),
                                 rec["team1"]["name"], rec["team2"]["name"])
    return vr


def to_ingest_row(rec):
    """把 match_record 转成 `state_snapshot.ingest()` 认识的行。

    ⚠ 契约细节：`State.observe()` 的 `per_team_players` 与 `map_stats` 要的是
    **元组列表** `[(name, rating), ...]`（`model_v2.py:474` / `:570` 都在解包二元组），
    不是 `{name: rating}` 字典。`match_results` 要的是 `[{"name","team1_score","team2_score"}]`。
    """
    mr = [{"name": m["map"], "team1_score": m["s1"], "team2_score": m["s2"]}
          for m in (rec.get("map_results") or [])]
    stats = {}
    ros = rec.get("rosters") or {}
    if ros:
        stats = {k: [(p["name"], p.get("rating")) for p in v] for k, v in ros.items()}
    by_map = rec.get("by_map")
    ms = None
    if by_map:
        ms = {mp: {t: [(p["name"], p.get("rating")) for p in pl]
                   for t, pl in teams.items()} for mp, teams in by_map.items()}
    return {"id": rec["id"], "date": rec["date"], "best_of": rec["best_of"],
            "team1": {"name": rec["team1"]["name"], "score": rec["team1"]["score"]},
            "team2": {"name": rec["team2"]["name"], "score": rec["team2"]["score"]},
            "winner": {"name": rec["winner"]},
            "maps": mr, "_rosters": stats, "_by_map": ms,
            # `_maps` / `_veto`：主理人手工赛果**不在 truths 表里**，
            # state_snapshot.ingest 会优先取这两个字段（否则整批选图信息丢失）。
            # veto_rec 的构造见 veto_rec_build.build_veto_rec —— bo3 的第 3 张图
            # 是 decider（ban,ban,pick,pick,ban,ban 之后剩下的），**不是 pick**。
            "_maps": rec.get("maps_full_B") or [],
            "_veto": _veto_of(rec)}


def selftest():
    n = 0
    ok_rec = {
        "id": 1, "date": "2026-10-03", "event": "T", "best_of": 3,
        "team1": {"name": "A", "score": 2}, "team2": {"name": "B", "score": 0},
        "winner": "A",
        "maps_full_B": ["Dust2", "Mirage", "Nuke"],
        "maps_played": ["Dust2", "Mirage"],
        "map_results": [{"map": "Dust2", "s1": 13, "s2": 9},
                        {"map": "Mirage", "s1": 16, "s2": 14}],
        "rosters": {"A": [{"name": "p1", "rating": 1.2}, {"name": "p2", "rating": 0.9}],
                    "B": [{"name": "p3", "rating": 1.0}]},
        "by_map": {"Dust2": {"A": [{"name": "p1", "rating": 1.3}],
                             "B": [{"name": "p3", "rating": 0.8}]}},
        "veto_order": [[1, "re", "B", "Ancient"], [2, "re", "A", "Cache"],
                       [3, "pi", "B", "Dust2"], [4, "pi", "A", "Mirage"],
                       [5, "re", "B", "Overpass"], [6, "re", "A", "Anubis"],
                       [7, "L", None, "Nuke"]],
        "pool": ["Ancient", "Anubis", "Cache", "Dust2", "Inferno", "Mirage", "Nuke"],
    }
    ok, pr, fu = validate(ok_rec)
    assert ok and not pr, pr
    assert all(fu.values()), fu
    n += 1
    # 燃料不全 ⇒ 非 strict 通过、strict 失败并点名缺哪几组
    poor = {k: v for k, v in ok_rec.items() if k not in ("map_results", "by_map", "veto_order")}
    ok2, pr2, fu2 = validate(poor, strict=False)
    assert ok2, pr2
    assert fu2["rosters"] and not any(fu2[k] for k in ("map_results", "by_map", "veto_order")), fu2
    ok3, pr3, _ = validate(poor, strict=True)
    assert not ok3 and any("燃料不全" in x for x in pr3), pr3
    n += 1
    # 读法 B 张数校验
    bad = dict(ok_rec, maps_full_B=["Dust2", "Mirage"])
    ok4, pr4, _ = validate(bad, strict=False)
    assert not ok4 and any("读法 B" in x for x in pr4), pr4
    n += 1
    # bo5 张数
    b5 = dict(ok_rec, best_of=5,
              team1={"name": "A", "score": 3}, team2={"name": "B", "score": 0},
              maps_full_B=["Ancient", "Dust2", "Mirage", "Nuke", "Cache"],
              maps_played=["Ancient", "Dust2", "Mirage"],
              map_results=[{"map": m, "s1": 13, "s2": 9}
                           for m in ("Ancient", "Dust2", "Mirage")])
    b5 = {k: v for k, v in b5.items() if k != "veto_order"}
    ok5, pr5, _ = validate(b5, strict=False)
    assert ok5, pr5
    n += 1
    # 非法分数：bo3 不可能打成 2:2
    b6 = dict(ok_rec, team2={"name": "B", "score": 2})
    ok6, pr6, _ = validate(b6, strict=False)
    assert not ok6 and any("总打图数" in x for x in pr6), pr6
    n += 1
    # 转 ingest 行
    row = to_ingest_row(ok_rec)
    assert row["maps"][0] == {"name": "Dust2", "team1_score": 13, "team2_score": 9}
    assert row["_rosters"]["A"] == [("p1", 1.2), ("p2", 0.9)], row["_rosters"]["A"]
    assert row["_by_map"]["Dust2"]["B"] == [("p3", 0.8)], row["_by_map"]["Dust2"]["B"]
    assert row["winner"] == {"name": "A"}
    n += 1
    print("RECORD_SCHEMA_SELFTEST_OK (%d checks)" % n)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        sys.exit(selftest())
    sys.exit(2)
