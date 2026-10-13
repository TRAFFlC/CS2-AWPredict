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
  * **执行方**：主理人未给每步由谁执行。但 CS 的 veto 规则决定了 pick1/pick2
    由**先后手**交替执行（pick1 = 先手方），且 `observe_match` 的 pick 分支
    带 `and team` 判空 —— **team=None 会让整步被跳过、什么都不记**
    （2026-10-12 实测：team=None 时 gpick 净新增 = 0）。
    故按「pick1 = team1、pick2 = team2」记录执行方；这是规则推导而非猜测，
    但若某赛事用非标准顺序（如双方连 pick），`tpick[队][图]` 的归因会错。
    ⇒ 归因正确性依赖此假设，`diff_ledgers` 只能证明账本自洽、证不了标签对。
  * decider **不产生任何 veto 记录**，但仍进 `map_set`（它确实上场了，
    `app/tapp` 的出现率应该算它）。
  * bo5 的 5 张全是 pick，执行方按 pick,pick,ban,ban,pick,pick 交替 ⇒
    team1,team2,team1,team2,team1。

⇒ 返回的 veto_rec 含 pick 步且带执行方，ban 步**不记**（主理人没给 ban 信息，
  记 `re` 等于编造）。
"""
from __future__ import annotations

# pick 的执行方交替规则：pick1=先手方(team1)，pick2=后手方(team2)，再交替
def _pick_teams(teamA, teamB, n_pick):
    return [teamA if i % 2 == 0 else teamB for i in range(n_pick)]


def build_veto_rec(map_set, bo, teamA=None, teamB=None):
    """返回 (map_set_for_app, veto_rec)。

    map_set_for_app：进 `app/tapp` 的完整图集（读法 B 全部图，含 decider）
    veto_rec：只含 kind=="pi" 的 pick 步（ban 无信息，不记）
    """
    ms = list(map_set or [])
    if bo == 3:
        picks, _decider = ms[:2], ms[2:]
    elif bo == 5:
        picks = ms
    else:                                   # bo1：无 veto
        picks = []
    if not picks:
        return ms, None
    teams = _pick_teams(teamA, teamB, len(picks)) if teamA and teamB else \
        [None] * len(picks)
    order = [[i, "pi", teams[i], m] for i, m in enumerate(picks)]
    return ms, {"order": order}


def split_pick_decider(map_set, bo):
    """给报告用：返回 (picks, deciders)。"""
    ms = list(map_set or [])
    if bo == 3:
        return ms[:2], ms[2:]
    if bo == 5:
        return ms, []
    return [], ms
