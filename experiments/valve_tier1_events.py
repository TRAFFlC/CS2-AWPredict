#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
issue-014：Liquipedia 赛事清单表（渲染 HTML）→ 结构化赛事清单。

同一套 `{{TournamentsList}}` 模板在不同页面上给出**两种列布局**（均已实测 @2026-10-07）：

  A) `Valve Tier 1 Events`（10 格/行）：
     [0]Tier [1]游戏图标 [2]赛事图标 [3]赛事名 [4]日期 [5]奖金 [6]地点 [7]队伍数 [8]冠军 [9]亚军
  B) `S-Tier Tournaments/Post 2023`（9 格/行）：
     [0]游戏图标 [1]赛事图标 [2]赛事名 [3]日期 [4]奖金 [5]地点 [6]队伍数 [7]冠军 [8]亚军

→ 因此**不能写死列号**。做法：先定位「日期格」，再相对它取后续 5 格；
  赛事名取日期格前第一个非空文本格；tier 取第 0 格（仅当形如 `X-Tier` 时）；
  页面路径优先取「anchor 的 title 恰好等于赛事名」的那个链接，避免误取 Tier 链接。

⚠ 四个已踩过的坑（1/2/4 由自证夹具抓出，3 由真实页面暴露）：
  1. 赛事名在**单独一格**里，图标格按文本剥标签得空串 → 曾对真实页静默返回 0 条。
  2. 列数不是常数（10 vs 9）→ 写死列号会把日期起整列错位。
  3. 布局 A 第 0 格是 `<a href=/counterstrike/S-Tier_Tournaments>`，**误当成赛事页**。
  4. 延期行只有 7 格且日期是 `Postponed` → 门槛设成 8 会把它们整行丢掉（静默漏数据）。

自证：python -X utf8 valve_tier1_events.py --selftest
用法：python -X utf8 valve_tier1_events.py --html <page.html> --out <events.jsonl>
"""
from __future__ import annotations

import html as _html
import json
import re
import sys

_MIN_CELLS = 7
_DATE_RE = re.compile(r"(?:19|20)\d\d")
_DATE_WORDS = ("Postponed", "TBD", "Cancelled", "Canceled")
_GAME_TITLES = ("Counter-Strike 2", "Counter-Strike: Global Offensive", "Counter-Strike")


def _text(cell: str) -> str:
    return _html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", cell))).strip()


def _anchors(cell: str):
    return re.findall(r"<a\b[^>]*>", cell)


def _page_for(cells, stop: int, name: str) -> str:
    """优先取 title 与赛事名一致的链接；否则取第一个非游戏、非 Tier 页的链接。"""
    for c in cells[:stop]:
        for a in _anchors(c):
            mt = re.search(r'title="([^"]+)"', a)
            mh = re.search(r'href="(/counterstrike/[^"]+)"', a)
            if mh and mt and _html.unescape(mt.group(1)) == name:
                return mh.group(1)
    for c in cells[:stop]:
        for a in _anchors(c):
            mh = re.search(r'href="(/counterstrike/[^"]+)"', a)
            mt = re.search(r'title="([^"]+)"', a)
            if not mh:
                continue
            title = _html.unescape(mt.group(1)) if mt else ""
            if title in _GAME_TITLES or "Tier_Tournaments" in mh.group(1):
                continue
            return mh.group(1)
    return ""


def _is_date(text: str) -> bool:
    if not text:
        return False
    if any(text.startswith(w) for w in _DATE_WORDS):
        return True
    return bool(_DATE_RE.search(text)) and bool(re.match(r"^[A-Z][a-z]{2}\b", text))


def parse_rows(html: str):
    """返回赛事记录列表（布局无关）。"""
    out = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        if len(cells) < _MIN_CELLS:
            continue
        t = [_text(c) for c in cells]
        di = next((i for i, x in enumerate(t) if _is_date(x)), None)
        if di is None or di + 5 >= len(t):
            continue
        name = next((t[j] for j in range(di - 1, -1, -1) if t[j]), "")
        if not name:
            for c in cells[:di]:
                for a in _anchors(c) + re.findall(r"<img\b[^>]*>", c):
                    mt = re.search(r'(?:title|alt)="([^"]+)"', a)
                    if mt and mt.group(1) not in _GAME_TITLES:
                        name = _html.unescape(mt.group(1))
                        break
                if name:
                    break
        if not name:
            continue
        out.append({
            "tier": t[0] if re.fullmatch(r"[SABCD]-Tier", t[0] or "") else "",
            "tournament": name,
            "page": _page_for(cells, di, name),
            "date": t[di],
            "prize": t[di + 1],
            "location": t[di + 2],
            "participants": t[di + 3],
            "winner": t[di + 4],
            "runner_up": t[di + 5],
        })
    return out


def year_of(date_str: str):
    m = _DATE_RE.search(date_str or "")
    return m.group(0) if m else ""


# ------------------------------------------------------------------ selftest
_LAYOUT_A = """
<table><tbody>
<tr><th>Tier</th><th>G</th><th></th><th>Tournament</th><th>Date</th><th>Prize Pool</th><th>Location</th><th>P#</th><th>Winner</th><th>Runner-up</th></tr>
<tr>
  <td><a href="/counterstrike/S-Tier_Tournaments" title="S-Tier Tournaments">S-Tier</a></td>
  <td><a href="/counterstrike/Counter-Strike_2" title="Counter-Strike 2"><img alt="Counter-Strike 2" src="x.png"/></a></td>
  <td><a href="/counterstrike/BLAST/Bounty/2025/Spring" title="BLAST Bounty Spring 2025"><img alt="BLAST Bounty Spring 2025" src="y.png"/></a></td>
  <td><a href="/counterstrike/BLAST/Bounty/2025/Spring" title="BLAST Bounty Spring 2025">BLAST Bounty Spring 2025</a></td>
  <td>Jan 23&#8211;26, 2025</td><td>$482,500</td><td>&#160;Copenhagen</td><td>8</td><td>Team Spirit</td><td>Eternal Fire</td>
</tr>
<tr><td>malformed</td><td>x</td></tr>
</tbody></table>
"""

_LAYOUT_B = """
<table><tbody>
<tr><th>G</th><th></th><th>Tournament</th><th>Date</th><th>Prize Pool</th><th>Location</th><th>P#</th><th>Winner</th><th>Runner-up</th></tr>
<tr>
  <td><a href="/counterstrike/Counter-Strike_2" title="Counter-Strike 2"><img alt="Counter-Strike 2" src="x.png"/></a></td>
  <td><a href="/counterstrike/Intel_Extreme_Masters/2025/Katowice" title="Intel Extreme Masters Katowice 2025"><img alt="Intel Extreme Masters Katowice 2025" src="y.png"/></a></td>
  <td><a href="/counterstrike/Intel_Extreme_Masters/2025/Katowice" title="Intel Extreme Masters Katowice 2025">Intel Extreme Masters Katowice 2025</a></td>
  <td>Jan 29 &#8211; Feb 09, 2025</td><td>$1,250,000</td><td>Katowice</td><td>24</td><td>Team Vitality</td><td>Team Spirit</td>
</tr>
</tbody></table>
"""

# 延期行：只有 7 格，日期是 Postponed —— 不能被静默丢弃
_POSTPONED = """
<table><tbody><tr>
  <td><a href="/counterstrike/FISSURE/Playground/5" title="FISSURE Playground #5">FISSURE Playground #5</a></td>
  <td>Postponed</td><td>$1,000,000</td><td>Shenzhen</td><td>16</td><td>TBD</td><td>TBD</td>
</tr></tbody></table>
"""

_EXPECT_A = {
    "tier": "S-Tier", "tournament": "BLAST Bounty Spring 2025",
    "page": "/counterstrike/BLAST/Bounty/2025/Spring", "date": "Jan 23–26, 2025",
    "prize": "$482,500", "location": "Copenhagen", "participants": "8",
    "winner": "Team Spirit", "runner_up": "Eternal Fire",
}
_EXPECT_B = {
    "tier": "", "tournament": "Intel Extreme Masters Katowice 2025",
    "page": "/counterstrike/Intel_Extreme_Masters/2025/Katowice",
    "date": "Jan 29 – Feb 09, 2025", "prize": "$1,250,000", "location": "Katowice",
    "participants": "24", "winner": "Team Vitality", "runner_up": "Team Spirit",
}


def _selftest() -> int:
    fails = []
    for label, sample, expect, n in (
        ("A(10格)", _LAYOUT_A, _EXPECT_A, 1),
        ("B(9格)", _LAYOUT_B, _EXPECT_B, 1),
    ):
        recs = parse_rows(sample)
        if len(recs) != n:
            fails.append(f"布局{label} count={len(recs)} 应为 {n}")
            continue
        for k, v in expect.items():
            if recs[0].get(k) != v:
                fails.append(f"布局{label} {k}: got={recs[0].get(k)!r} want={v!r}")
        if year_of(recs[0]["date"]) != "2025":
            fails.append(f"布局{label} year_of={year_of(recs[0]['date'])}")

    late = parse_rows(_POSTPONED)
    if len(late) != 1:
        fails.append(f"延期行应被解析出 1 条，实际 {len(late)}")
    else:
        if late[0]["date"] != "Postponed" or late[0]["runner_up"] != "TBD":
            fails.append(f"延期行字段错: {late[0]}")
        if year_of(late[0]["date"]) != "":
            fails.append("延期行不应有年份")

    if fails:
        print("TOURNAMENTS_LIST_PARSER_SELFTEST_FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("TOURNAMENTS_LIST_PARSER_SELFTEST_OK (布局A + 布局B + 延期行)")
    return 0


def main(argv):
    if "--selftest" in argv:
        return _selftest()
    hf = argv[argv.index("--html") + 1] if "--html" in argv else None
    out = argv[argv.index("--out") + 1] if "--out" in argv else None
    if not hf:
        print(__doc__)
        return 0
    recs = parse_rows(open(hf, encoding="utf-8").read())
    noyear = [r for r in recs if not year_of(r["date"])]
    if out:
        with open(out, "w", encoding="utf-8") as f:
            for r in recs:
                r["year"] = year_of(r["date"])
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    byyear = {}
    for r in recs:
        k = year_of(r["date"]) or "(无年份)"
        byyear[k] = byyear.get(k, 0) + 1
    print(f"{len(recs)} events -> {out or 'stdout'}")
    print("按年份：", ", ".join(f"{k}:{v}" for k, v in sorted(byyear.items())))
    if noyear:
        print(f"⚠ {len(noyear)} 条无年份（未开赛/延期），须显式处理：")
        for r in noyear:
            print(f"   - {r['tournament']} | date={r['date']!r}")
    for r in recs[:3]:
        print(f"  {r['tier']:7s} {r['date']:24s} {r['tournament'][:42]:42s} P={r['participants']:>3s} W={r['winner']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
