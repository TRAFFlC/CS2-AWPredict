"""EPL S24 比赛日期采集器（DS-19）—— 极致保守版。
设计约束（主理人 2026-10-12 裁决 + DS-16/17/19）：
  · 强制串行，**无任何并发**
  · 基础间隔 6000 ms + uniform(3000,6000) 随机抖动（均值 ~9 s，最快 6 s）
  · 单次运行**硬上限 8 个请求**；连续 2 次非 200 立即中止
  · 每次取页前先 GET / 取 cookie（实测决定因素是 TLS 指纹）
  · 命中已知 id/url 即跳过（去重）
  · **dry-run 默认开**：只打印将要发起的请求，不真正访问
自证：selftest（离线夹具）—— 参数下限、去重、抖动区间、上限、解析器。
"""
import io, json, os, re, subprocess, sys, time, random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "epl_s24_dates.json")
COOKIE = os.path.join(HERE, "hltv_dates_cookies.tmp")
BASE = "https://www.hltv.org"

GAP_BASE_MS = 6000          # 底线：不得低于此值（DS-16）
JITTER_MS = (3000, 6000)    # 叠加随机抖动
MAX_REQUESTS = 8            # 单次运行硬上限
MAX_CONSEC_BAD = 2          # 连续非 200 达此数即中止
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

_last_at = [0.0]
_seen = set()
_bad_streak = [0]
_nreq = [0]


def gap_s():
    """下一次请求前的等待秒数（基础 + 抖动），并记录预计时刻。"""
    return (GAP_BASE_MS + random.uniform(*JITTER_MS)) / 1000.0


def wait_turn():
    dt = gap_s()
    now = time.time()
    delay = _last_at[0] + dt - now
    if delay > 0:
        time.sleep(delay)
    _last_at[0] = time.time()


def fetch(url, dry):
    """串行取一页。返回 (status, text)。"""
    if url in _seen:
        print("  SKIP(已取过) %s" % url)
        return 200, None
    _seen.add(url)
    if _nreq[0] >= MAX_REQUESTS:
        print("  ABORT 达到硬上限 %d 个请求" % MAX_REQUESTS)
        return 0, None
    wait_turn()
    cmd = ["curl", "-sS", "--compressed", "--max-time", "45",
           "-A", UA, "-c", COOKIE, "-b", COOKIE,
           "-w", "\n@@HTTP=%{http_code}", url]
    print("  REQ#%d %s" % (_nreq[0] + 1, url))
    if dry:
        _nreq[0] += 1
        return -1, None
    p = subprocess.run(cmd, capture_output=True, text=True)
    body, _, tail = p.stdout.rpartition("\n@@HTTP=")
    try:
        code = int(tail.strip())
    except ValueError:
        code = 0
    _nreq[0] += 1
    _bad_streak[0] = 0 if code == 200 else _bad_streak[0] + 1
    print("     -> http=%d bytes=%d bad_streak=%d" % (code, len(body), _bad_streak[0]))
    if _bad_streak[0] >= MAX_CONSEC_BAD:
        raise SystemExit("ABORT 连续 %d 次非 200，判定被风控，立即停止"
                         % MAX_CONSEC_BAD)
    return code, body


ROW = re.compile(
    r'<tr[^>]*>.*?</tr>', re.S | re.I)
CELL = re.compile(r'<td[^>]*>(.*?)</td>', re.S | re.I)
DATE = re.compile(r'(\d{1,2}/\d{1,2}/\d{4})')
TAG = re.compile(r'<[^>]+>')
LINK = re.compile(r'href="(/results/\d+/[^"]+)"')
TEAM = re.compile(r'/team/\d+/[^"]*"[^>]*>\s*([^<]{1,40})\s*</a>', re.I)


def _clean(x):
    return re.sub(r"\s+", " ", TAG.sub(" ", x)).strip()


def parse_results(html):
    """从 HLTV 赛事 results 表解析 日期 / 队名 / 比分 / match id。
    HLTV 的比分有两种布局，都要吃：
      A) 单格 "2 - 0"
      B) 两个相邻格 "2" 与 "0"（夹在两个队名之间）
    """
    out = []
    for tr in ROW.findall(html):
        md = DATE.search(_clean(tr))
        if not md:
            continue
        cells = [_clean(c) for c in CELL.findall(tr)]
        lk = LINK.search(tr)
        teams = TEAM.findall(tr)
        score = None
        one = [c for c in cells if re.fullmatch(r"\d{1,2}\s*[:\-]\s*\d{1,2}", c)]
        if one:
            score = one[0]
        else:
            # 布局 B：比分是两个独立的纯数字格，位置可夹在队名之间，也可分居两队名两侧
            nums = [c for c in cells if re.fullmatch(r"\d{1,2}", c)]
            if len(nums) >= 2:
                score = "%s - %s" % (nums[0], nums[1])
        out.append({
            "date": md.group(1),
            "match_id": int(lk.group(1).split("/")[2]) if lk else None,
            "url": (BASE + lk.group(1)) if lk else None,
            "teams": [t.strip() for t in teams],
            "score": score,
            "cells": cells,
        })
    return out


def find_event(html, pattern=re.compile(r"pro\s*league\s*season\s*24", re.I)):
    for href, txt in re.findall(r'href="(/events/\d+/[^"]+)"[^>]*>(.*?)</a>', html, re.S | re.I):
        t = _clean(txt)
        if pattern.search(t) or pattern.search(href):
            return BASE + href, t
    return None, None


def selftest():
    n = 0
    # 1) 频率下限
    assert GAP_BASE_MS >= 6000, GAP_BASE_MS
    assert JITTER_MS[0] >= 3000 and JITTER_MS[1] > JITTER_MS[0]
    mins = [GAP_BASE_MS + random.uniform(*JITTER_MS) for _ in range(500)]
    assert min(mins) >= GAP_BASE_MS, min(mins)
    assert random.random() < 1.0
    n += 1
    # 2) 请求硬上限
    assert MAX_REQUESTS <= 8
    assert MAX_CONSEC_BAD <= 2
    n += 1
    # 3) 去重
    _seen.clear(); _nreq[0] = 0
    c1, _ = fetch("https://x/1", dry=True)
    c2, _ = fetch("https://x/1", dry=True)
    assert c1 == -1 and c2 == 200 and _nreq[0] == 1, (_nreq[0], c1, c2)
    _seen.clear(); _nreq[0] = 0
    n += 1
    # 4) 解析器（离线夹具）—— 布局 B：比分是两个独立格
    html = ('<table><tr class="">'
            '<td class="aFavorite">16/10/2026</td>'
            '<td><a href="/team/9999/falcons">Falcons</a></td><td class="team1">2</td>'
            '<td class="aFavorite"><a href="/team/8888/tyloo">TYLOO</a></td><td>0</td>'
            '<td><a href="/results/2499999/espacemasters">a</a></td></tr></table>')
    rows = parse_results(html)
    assert len(rows) == 1, rows
    assert rows[0]["date"] == "16/10/2026", rows[0]
    assert rows[0]["match_id"] == 2499999, rows[0]
    assert rows[0]["teams"] == ["Falcons", "TYLOO"], rows[0]
    assert rows[0]["score"] == "2 - 0", rows[0]
    n += 1
    # 4b) 布局 A：比分是单格 "1 - 2"
    htmlA = ('<table><tr>'
             '<td>01/11/2026</td>'
             '<td><a href="/team/1111/a">A</a></td>'
             '<td>1 - 2</td>'
             '<td><a href="/team/2222/b">B</a></td>'
             '<td><a href="/results/2500001/x">b</a></td></tr></table>')
    rowsA = parse_results(htmlA)
    assert len(rowsA) == 1 and rowsA[0]["score"] == "1 - 2", rowsA
    assert rowsA[0]["teams"] == ["A", "B"], rowsA
    n += 1
    # 5) 事件页定位
    ev = ('<a href="/events/7500/esl-pro-league-season-24">ESL Pro League Season 24</a>')
    url, txt = find_event(ev)
    assert url and url.endswith("/events/7500/esl-pro-league-season-24"), url
    assert txt == "ESL Pro League Season 24", txt
    n += 1
    print("EPL_DATES_SELFTEST_OK (%d checks)" % n)
    return 0


def main():
    dry = "--live" not in sys.argv
    print("MODE = %s" % ("DRY-RUN（不发请求）" if dry else "LIVE"))
    print("间隔 = %d ms + U(%.0f,%.0f) ms   请求上限 = %d   连续失败中止 = %d"
          % (GAP_BASE_MS, JITTER_MS[0], JITTER_MS[1], MAX_REQUESTS, MAX_CONSEC_BAD))
    if dry:
        print("计划请求序列：")
        print("  1) %s/            （取 cookie，TLS 指纹）" % BASE)
        print("  2) %s/events       （定位 EPL S24 赛事页 id）" % BASE)
        print("  3) %s/results/event/<id>/<slug>   （results 表自带日期列，1 页足够）" % BASE)
        print("预估总耗时 ≥ 27 s；不会访问任何 /results/<matchid>/ 单场页。")
        return 0
    # LIVE
    fetch(BASE + "/", dry)
    code, html = fetch(BASE + "/events", dry)
    if code != 200:
        print("events 页失败，中止")
        return 1
    url, title = find_event(html)
    print("赛事：%s -> %s" % (title, url))
    if not url:
        return 1
    code, html = fetch(url + "/results", dry)
    if code != 200:
        print("results 页失败")
        return 1
    rows = parse_results(html)
    keep = [r for r in rows if r["match_id"] and r["teams"]]
    io.open(OUT, "w", encoding="utf-8").write(
        json.dumps({"event": title, "url": url, "rows": keep}, ensure_ascii=False, indent=2))
    print("写出 %s：%d 行" % (OUT, len(keep)))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        sys.exit(selftest())
    sys.exit(main())
