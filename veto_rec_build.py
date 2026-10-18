"""把主理人给的人工 veto 记录转成 `MapHistory.observe_match` 认识的 `veto_rec`。

⚠⚠ **主理人 2026-10-12 的关键纠正**：列出的三张图**不是三个 pick**。
bo3 的真实顺序是：

    ban, ban, pick, pick, ban, ban → 剩下的 1 张 = **decider**（自动，非任何人 pick）

⇒ 读法 B 的 3 张图里，**第 1、2 张是 pick，第 3 张是 decider**。
   我先前按「三张都是 pick」处理是错的（本会话已犯过多次此类假设错误）。

`veto_rec` 契约（`veto_model.observe_match`）：
    {"order": [[step, kind, team, map], ...]}
      step : 0 起的序号
      kind : "re"(ban/removal) | "pi"(pick)
      team : 队伍名；**None 表示非该队动作**（如系统自动的 decider）

设计取舍：
  * **执行方**：`observe_match` 的 pick 分支带 `and team` 判空 ——
    **team=None 会让整步被跳过、什么都不记**（2026-10-12 实测：team=None 时
    gpick 净新增 = 0）。所以执行方不能留空。
    * **显式给定优先**（2026-10-19 主理人裁决）：`map_set` 若是
      `[[step, kind, team|null, map], ...]`（`record_schema` 文档里的正规写法），
      就**原样用主理人给的执行方**，不再推断。实测反例：Thunderpick WC 2026 的
      13 场 bo3 里有 7 场「pick1=败方 / 双方连 pick」，交替推断与事实**相反**；
      `_apply_mapfix.py` 头注里也记了 GamerLegion 选第 1 张的同一类反例。
    * 仅当只给了**纯图名列表**时，才退回「pick1 = team1、pick2 = team2」交替推断
      —— 那是规则推导而非猜测，但同队连 pick 时 `tpick[队][图]` 的归因会错。
      ⇒ 归因正确性依赖此假设，`diff_ledgers` 只能证明账本自洽、证不了标签对。
  * decider **不产生任何 veto 记录**（`observe_match` 只认 `re`/`pi`，
    `"L"` 步会被自然忽略），但仍进 `map_set`（它确实上场了，
    `app/tapp` 的出现率应该算它）。
  * **真实 CS2 veto 顺序**（2026-10-19 按主理人 bo5 决赛数据纠正）：
        bo3 : ban,ban,pick,pick,ban,ban            ⇒ 2 pick + 1 decider
        bo5 : ban,ban,pick,pick,ban,ban,pick,pick,ban,ban
                                                   ⇒ **4 pick + 1 decider**
    ⚠ 本文件此前把 bo5 写成「5 张全是 pick」（`picks = ms`），是错的：
      那会把 decider 记成 team1 的 pick，凭空多出 1 次 tpick/gpick/npick。
      主理人 2026-10-19 给的 BETBOOM vs FURIA 决赛（4 pick + decider Cache）
      是该 bug 的实测反证。

⇒ 显式形式：pick 与 ban 步**都记**（主理人给了就是事实）；
  纯图名列表形式：只记 pick 步，ban 不记（没给 ban 信息，记 `re` 等于编造）。
"""
from __future__ import annotations

# 真实 veto 规则：读法 B 的前 N 张是 pick，其余是 decider（见模块 docstring）
N_PICK = {3: 2, 5: 4}


def split_pick_decider(map_set, bo):
    """给报告用：返回 (picks, deciders)。

    显式形式按主理人标的 `kind` 切（`"pi"` = pick，其余含 `"L"` = decider）；
    纯图名列表按 `N_PICK[bo]` 的张数切。
    """
    ms = list(map_set or [])
    if is_explicit(ms):
        return ([x[3] for x in ms if x[1] == "pi"],
                [x[3] for x in ms if x[1] != "pi"])
    n = N_PICK.get(bo, 0)
    return ms[:n], ms[n:]


def is_explicit(map_set):
    """True = `[[step, kind, team|null, map], ...]`（含执行方）；False = 纯图名列表。"""
    ms = list(map_set or [])
    return bool(ms) and all(isinstance(x, (list, tuple)) and len(x) >= 4
                            for x in ms)


# pick 的执行方交替规则：pick1=先手方(team1)，pick2=后手方(team2)，再交替
def _pick_teams(teamA, teamB, n_pick):
    return [teamA if i % 2 == 0 else teamB for i in range(n_pick)]


def build_veto_rec(map_set, bo, teamA=None, teamB=None):
    """返回 (map_set_for_app, veto_rec)。

    map_set_for_app：进 `app/tapp` 的完整图集（读法 B 全部图，含 decider）
    veto_rec：`{"order": [[step, kind, team, map], ...]}`。两种输入：

      * 显式 `[[step, "re"|"pi"|"L", team|null, map], ...]` → **原样透传**，
        执行方用主理人给的值（含 `"re"` ban 步）。
      * 纯图名列表 `["Dust2", "Mirage", "Nuke"]` → 按 `N_PICK[bo]` 切出 pick 段，
        执行方交替推断，**ban 步不记**（无信息，记等于编造）。
    """
    ms = list(map_set or [])
    if not ms:
        return [], None
    if is_explicit(ms):
        return ([x[3] for x in ms],
                {"order": [[int(x[0]), x[1], x[2], x[3]] for x in ms]})
    picks = ms[:N_PICK.get(bo, 0)]
    if not picks:
        return ms, None
    teams = _pick_teams(teamA, teamB, len(picks)) if teamA and teamB else \
        [None] * len(picks)
    order = [[i, "pi", teams[i], m] for i, m in enumerate(picks)]
    return ms, {"order": order}


def selftest():
    n = 0
    # bo3：2 pick + decider（交替推断路径，行为不变）
    ms, vr = build_veto_rec(["Dust2", "Mirage", "Nuke"], 3, "A", "B")
    assert ms == ["Dust2", "Mirage", "Nuke"] and len(vr["order"]) == 2, (ms, vr)
    assert [x[3] for x in vr["order"]] == ["Dust2", "Mirage"], vr
    assert [x[2] for x in vr["order"]] == ["A", "B"], vr
    n += 1
    # bo5：4 pick + 1 decider —— decider **不得**进 veto_rec（曾被误记成 team1 的 pick）
    ms, vr = build_veto_rec(["Anubis", "Dust2", "Nuke", "Mirage", "Cache"], 5, "A", "B")
    assert len(ms) == 5, ms
    assert [x[3] for x in vr["order"]] == ["Anubis", "Dust2", "Nuke", "Mirage"], vr
    assert "Cache" not in [x[3] for x in vr["order"]], vr
    assert [x[2] for x in vr["order"]] == ["A", "B", "A", "B"], vr
    n += 1
    # 显式形式：同队连 pick（交替推断会错的那类）必须原样保留
    exp = [[0, "pi", "FURIA", "Ancient"], [1, "pi", "FURIA", "Mirage"],
           [2, "L", None, "Nuke"]]
    ms, vr = build_veto_rec(exp, 3, "FURIA", "BETBOOM")
    assert ms == ["Ancient", "Mirage", "Nuke"], ms
    assert vr["order"] == exp, vr
    assert [x[2] for x in vr["order"]] == ["FURIA", "FURIA", None], vr
    n += 1
    # 显式形式：ban 步要记（observe_match 只认 "re"/"pi"，"L" 自然被忽略）
    exp5 = [[1, "re", "A", "Overpass"], [2, "re", "B", "Inferno"],
            [3, "pi", "A", "Anubis"], [4, "pi", "B", "Dust2"],
            [5, "re", "A", "Train"], [6, "re", "B", "Overpass2"],
            [7, "pi", "A", "Nuke"], [8, "pi", "B", "Mirage"], [9, "L", None, "Cache"]]
    _ms, vr = build_veto_rec(exp5, 5, "A", "B")
    kinds = [x[1] for x in vr["order"]]
    assert kinds.count("pi") == 4 and kinds.count("re") == 4, kinds
    n += 1
    # 空输入
    assert build_veto_rec([], 3, "A", "B") == ([], None)
    assert build_veto_rec(None, 3, "A", "B") == ([], None)
    n += 1
    # split_pick_decider 与 build_veto_rec 口径一致
    ms = ["Anubis", "Dust2", "Nuke", "Mirage", "Cache"]
    p, d = split_pick_decider(ms, 5)
    assert (len(p), len(d)) == (4, 1) and d == ["Cache"], (p, d)
    p, d = split_pick_decider(["Dust2", "Mirage", "Nuke"], 3)
    assert (p, d) == (["Dust2", "Mirage"], ["Nuke"]), (p, d)
    assert split_pick_decider([[0, "pi", "A", "Dust2"], [1, "L", None, "Nuke"]],
                              3) == (["Dust2"], ["Nuke"])
    n += 1
    print("VETO_REC_BUILD_SELFTEST_OK (%d checks)" % n)
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        sys.exit(selftest())
    sys.exit(2)
