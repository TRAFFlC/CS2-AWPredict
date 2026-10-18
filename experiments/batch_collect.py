#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
issue-014 批量采集器：赛事清单 → 比赛数据集（按 Liquipedia ToU 限速 + 缓存 + 可中断续跑）。

⚠ 限速是**实测校正过**的。`https://liquipedia.net/api-terms-of-use`（读于 2026-10-07）：
  * 所有请求 ≤ 1/2 秒；**`action=parse` ≤ 1/30 秒**（资源密集）；
  * 自定义 User-Agent + gzip；尽量复用缓存；
  * **违规会临时封 IP（需 CAPTCHA），反复触发可能永久封禁。**
  第一版按 2s 间隔跑 parse，实测立刻吃到 **HTTP 429**——故本版把两种限速分开。

⚠ 子页不是免费的：`Perfect World/Major/2024/Shanghai` 实测有 **31 个子页**，
  其中绝大多数是预选赛（Americas / Asia-Pacific / Europe Qualifier …）。
  按 30s/parse 全抓会不可承受，故只保留**正赛阶段页**（allow-list），并用 `--max-subpages` 上限兜底。

用法：
  python -X utf8 batch_collect.py --events tier1.jsonl stier_events.jsonl --years 2024 --limit 1 --out probe.jsonl
  python -X utf8 batch_collect.py --selftest
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import liquipedia_matches as LP

API = "https://liquipedia.net/counterstrike/api.php"
UA = "another-issue014-research/1.0 (local research; contact: user)"
CACHE_DIR = "cache"
SLEEP_QUERY = 2.0    # ToU: ≤1 req / 2s
SLEEP_PARSE = 30.0   # ToU: ≤1 req / 30s
MAX_CONSECUTIVE_FAILS = 3
MAX_SUBPAGES_DEFAULT = 8

# 正赛阶段页的关键词（命中任一即保留）；预选赛页不含这些词
SUBPAGE_ALLOW = ("stage", "playoff", "group", "final", "swiss", "round robin", "showmatch", "main event")
SUBPAGE_DENY_SEGMENTS = ("open", "closed")  # 按整段匹配：避免 "/open" 误伤 "/Opening Stage"

STATS = {"net": 0, "cache": 0, "pages": 0, "req429": 0, "matches": 0}


def sleep_for(params: dict, override: float | None = None) -> float:
    """按动作选择应等待的秒数（parse 昂贵，30s；其余 2s）。"""
    if override is not None:
        return override
    return SLEEP_PARSE if params.get("action") == "parse" else SLEEP_QUERY


def select_subpages(subs, max_subpages: int = MAX_SUBPAGES_DEFAULT, extra_allow=()):
    """从 apprefix 返回的全部子页里挑出正赛阶段页（去掉预选赛/公开赛）。"""
    allow = tuple(SUBPAGE_ALLOW) + tuple(k.lower() for k in extra_allow)
    out = []
    for s in subs:
        segs = [seg.strip().lower() for seg in s.split("/")]
        if any("qualifier" in seg for seg in segs):
            continue
        if segs and segs[-1] in SUBPAGE_DENY_SEGMENTS:
            continue
        if any(a in s.lower() for a in allow):
            out.append(s)
    return out[:max_subpages]


def _cache_path(key: str) -> str:
    safe = urllib.parse.quote(key, safe="")[:180]
    return os.path.join(CACHE_DIR, safe + ".json")


def api_get(params: dict, cache_key: str, sleep_override: float | None = None):
    """带缓存 + 429 退避的 API GET。返回 (payload, from_cache)。"""
    cp = _cache_path(cache_key)
    if os.path.exists(cp):
        STATS["cache"] += 1
        return json.load(open(cp, encoding="utf-8")), True

    url = API + "?" + urllib.parse.urlencode(params)
    attempt = 0
    while True:
        attempt += 1
        req = urllib.request.Request(url, headers={
            "User-Agent": UA, "Accept-Encoding": "gzip", "Accept": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
            payload = json.loads(raw.decode("utf-8"))
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            STATS["net"] += 1
            time.sleep(sleep_for(params, sleep_override))
            return payload, False
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            STATS["req429"] += 1
            ra = e.headers.get("Retry-After") if e.headers else None
            wait = float(ra) if (ra and ra.isdigit()) else min(60.0 * attempt, 300.0)
            print(f"    ⚠ 429 限流（第 {attempt} 次）→ 退避 {wait:.0f}s")
            if attempt >= MAX_CONSECUTIVE_FAILS:
                raise RuntimeError(
                    "连续 429：为避免触发临时/永久 IP 封禁而主动中止。"
                    "请等待后重跑（缓存已保留，会从断点继续）。"
                )
            time.sleep(wait)


def wikitext(title: str, sleep_override: float | None = None):
    d, _ = api_get({
        "action": "parse", "page": title, "prop": "wikitext",
        "format": "json", "formatversion": "2",
    }, f"parse:{title}", sleep_override)
    return None if "error" in d else d.get("parse", {}).get("wikitext", "")


def subpages(title: str, sleep_override: float | None = None):
    d, _ = api_get({
        "action": "query", "list": "allpages", "apprefix": title,
        "aplimit": "100", "format": "json", "formatversion": "2",
    }, f"sub:{title}", sleep_override)
    return [p["title"] for p in d.get("query", {}).get("allpages", [])]


def dedup_titles(events):
    """按 wiki 标题去重（同一赛事可能同时出现在 Tier1 与 S-Tier 清单里）。"""
    seen, out = set(), []
    for ev in events:
        page = ev.get("page") or ""
        if not page.startswith("/counterstrike/"):
            continue
        title = page[len("/counterstrike/"):]
        if title in seen:
            continue
        seen.add(title)
        out.append((title, ev))
    return out


def collect_events(events, out_path: str, limit=None, sleep_override=None,
                   max_subpages=MAX_SUBPAGES_DEFAULT, dump_every=True):
    uniq = dedup_titles(events)
    rows, tried = [], 0
    for title, ev in uniq:
        if limit is not None and tried >= limit:
            break
        tried += 1
        all_subs = [t for t in subpages(title, sleep_override) if t != title]
        stages = select_subpages(all_subs, max_subpages)
        found = 0
        for t in [title] + stages:
            wt = wikitext(t, sleep_override)
            STATS["pages"] += 1
            if not wt:
                continue
            for r in LP.parse_wikitext(wt, page=t, event=ev.get("tournament", "")):
                r["event_tier"] = ev.get("tier", "")
                r["event_list"] = ev.get("_source", "")
                rows.append(r)
                found += 1
        STATS["matches"] += found
        print(f"  [{tried}/{len(uniq)}] {ev.get('tournament','?')[:46]:46s} "
              f"子页 {len(all_subs):2d}→取 {len(stages):2d}  比赛 {found:3d}")
        if dump_every and out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def load_events(paths):
    out = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                r["_source"] = os.path.basename(p)
                out.append(r)
    return out


# ------------------------------------------------------------------ selftest
# 真实夹具：Shanghai Major 2024 的 31 个子页（实测抓取）。正赛只有 3 个 stage 页。
_REAL_SUBS = [
    "Perfect World/Major/2024/Shanghai",
    "Perfect World/Major/2024/Shanghai/Americas",
    "Perfect World/Major/2024/Shanghai/Americas/North America",
    "Perfect World/Major/2024/Shanghai/Americas/South America",
    "Perfect World/Major/2024/Shanghai/Asia-Pacific",
    "Perfect World/Major/2024/Shanghai/Asia-Pacific/China",
    "Perfect World/Major/2024/Shanghai/Asia-Pacific/China/Open",
    "Perfect World/Major/2024/Shanghai/Asia-Pacific/East Asia",
    "Perfect World/Major/2024/Shanghai/Asia-Pacific/Oceania/Open",
    "Perfect World/Major/2024/Shanghai/Elimination Stage",
    "Perfect World/Major/2024/Shanghai/Europe",
    "Perfect World/Major/2024/Shanghai/Europe/Qualifier A",
    "Perfect World/Major/2024/Shanghai/Europe A/Qualifier B",
    "Perfect World/Major/2024/Shanghai/Opening Stage",
    "Perfect World/Major/2024/Shanghai/Playoff Stage",
]


def _selftest() -> int:
    fails = []

    # 1) 限速选择：parse 必须走 30s，其余走 2s（第一版正是在这里出错）
    if sleep_for({"action": "parse"}) != SLEEP_PARSE:
        fails.append("parse 限速错误")
    if sleep_for({"action": "query"}) != SLEEP_QUERY:
        fails.append("query 限速错误")
    if sleep_for({"action": "parse"}, 1.0) != 1.0:
        fails.append("override 未生效")
    if SLEEP_PARSE < 30.0:
        fails.append("parse 限速低于 ToU 的 30s 底线")

    # 2) 子页过滤：真实夹具——应只保留 3 个 stage 页，预选赛/公开赛全剔除
    picked = select_subpages(_REAL_SUBS, max_subpages=20)
    want = {
        "Perfect World/Major/2024/Shanghai/Elimination Stage",
        "Perfect World/Major/2024/Shanghai/Opening Stage",
        "Perfect World/Major/2024/Shanghai/Playoff Stage",
    }
    if set(picked) != want:
        fails.append(f"子页过滤错误: 取到 {sorted(picked)}")
    if any(("Qualifier" in p) or p.endswith("/Open") or p.endswith("/Closed") for p in picked):
        fails.append("预选赛/公开赛页未被剔除")
    if len(select_subpages(_REAL_SUBS, max_subpages=1)) != 1:
        fails.append("max_subpages 上限未生效")

    # 3) 去重 + 过滤
    evs = [
        {"page": "/counterstrike/A", "tournament": "A"},
        {"page": "/counterstrike/A", "tournament": "A dup"},
        {"page": "", "tournament": "none"},
        {"page": "/counterstrike/B", "tournament": "B"},
    ]
    kept = [t for t, _ in dedup_titles(evs)]
    if kept != ["A", "B"]:
        fails.append(f"去重/过滤错误: {kept}")

    # 4) 解析接线
    sample = ('{{Match|opponent1={{TeamOpponent|a}}|opponent2={{TeamOpponent|b}}'
              '|map1={{Map|map=Mirage|t1t=7|t1ct=6|t2t=3|t2ct=3}}|hltv=1}}')
    recs = LP.parse_wikitext(sample, page="T", event="E")
    if len(recs) != 1 or recs[0]["series"]["winner"] != "a":
        fails.append(f"解析接线错误: {recs}")

    if fails:
        print("BATCH_COLLECT_SELFTEST_FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("BATCH_COLLECT_SELFTEST_OK (限速 + 子页过滤[真实夹具] + 去重 + 解析接线)")
    return 0


def main(argv):
    if "--selftest" in argv:
        return _selftest()
    ev_paths = []
    if "--events" in argv:
        i = argv.index("--events") + 1
        while i < len(argv) and not argv[i].startswith("--"):
            ev_paths.append(argv[i])
            i += 1
    if not ev_paths:
        print(__doc__)
        return 0
    out = argv[argv.index("--out") + 1] if "--out" in argv else "matches_all.jsonl"
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    years = argv[argv.index("--years") + 1].split(",") if "--years" in argv else None
    override = float(argv[argv.index("--sleep") + 1]) if "--sleep" in argv else None
    max_subs = int(argv[argv.index("--max-subpages") + 1]) if "--max-subpages" in argv else MAX_SUBPAGES_DEFAULT

    events = load_events(ev_paths)
    if years:
        events = [e for e in events if e.get("year") in years]
    uniq = dedup_titles(events)
    n = limit if limit is not None else len(uniq)
    print(f"事件 {len(uniq)} 条；query {SLEEP_QUERY}s / parse {SLEEP_PARSE}s"
          f"{'（已覆盖）' if override else ''}；每个赛事最多 {max_subs} 个子页")
    print(f"本次拟处理 {n} 场赛事，按 ToU 保守估计约 {n * (max_subs + 1) * 30 / 60:.0f} 分钟")
    t0 = time.time()
    try:
        rows = collect_events(events, out, limit, override, max_subs)
    except RuntimeError as e:
        print(f"中止：{e}")
        print(f"已采比赛 {STATS['matches']} 场，缓存命中 {STATS['cache']}，429 次数 {STATS['req429']}")
        return 2
    dt = time.time() - t0
    print(f"完成：比赛 {len(rows)} 场，图 {sum(len(r['maps']) for r in rows)} 张，用时 {dt:.0f}s")
    print(f"网络 {STATS['net']} / 缓存 {STATS['cache']} / 429 {STATS['req429']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
