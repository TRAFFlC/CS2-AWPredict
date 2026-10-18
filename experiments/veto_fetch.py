# -*- coding: utf-8 -*-
r"""DS-11 veto 采集器（curl 作传输，纯离线解析）。

## 为什么要这一版（推翻「只能走浏览器」的结论）

2026-10-08 复查通道，三条实测：

| 通道 | 结果 |
|---|---|
| Python `urllib` 直取比赛页 | **403 挑战页**（即使先 GET `/` 拿到 cookie） |
| Python `urllib` 直取 `/` | 200 / 410 KB（首页正常，说明不是网络问题） |
| **`curl` 先 GET `/` 再取比赛页** | **200 / 380–880 KB**，8/8 全成功 |

即：**决定因素是 TLS 指纹**，curl 能过、Python 的 OpenSSL 栈过不了；浏览器通道不是唯一解。
故本工具用 `curl` 子进程当传输，解析与落盘仍在 Python 里。

页面里 veto 是纯 HTML，无需执行 JS：

    <div class="standard-box veto-box"><div class="padding">
      <div>1. Vitality removed Ancient</div> ... <div>7. Inferno was left over</div>
    </div></div>

## 坑（实测）

1. **空 cookie 直取比赛页 = 403 挑战页（5.7 KB）**；必须先 GET `/` 拿 `__cf_bm`。
2. **挑战是间歇性的**：同一 id 前一轮 200 / 543 KB，后一轮就连着 3 次 403。
   故判定式取三条件任一：`status in (403,429,503)`、页面含 `Just a moment`、**响应体 < 20 KB**。
   （真比赛页 380–880 KB，与挑战页差两个数量级，阈值很安全。）
3. **200 但页面没有 veto** 也会发生（比如该场 HLTV 没展示 Maps 段）——
   记为坏记录、**下轮重回工作清单**，不污染好记录。
4. 非贪婪配 `<div>([\s\S]*?)</div></div>` 会在第 7 步自己的 `</div>` 处收手，
   把决胜图连着标签一起丢掉（只剩 6 步）→ 用「起点标记 + `was left over` + 下一个 `</div>`」切块。

## 产物

与页内采集器**同格式**追加写 `csapi_veto.jsonl`（故 `veto_labels.py` 无需改动）：

    {"id","bo","played","veto":[[n,kind,team,map],...],"bytes","status","attempts",
     "src","picks_match","decider","veto_ok"}

去重：好记录不再重取；坏记录（403/挑战页/无 veto）下轮自动重回工作清单。

## 子命令

    selftest                     解析器自证（离线夹具 + 合成样例），无网络
    pending                      待取清单与按月分布
    run [limit=N] [gap_ms=2500] [order=priority|desc|asc]
        [max_blocks=6] [backoff_s=120] [max_attempts=3]
    probe id1 id2 ...            单场实取（打通道，不落盘）
"""

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）
WORK = os.path.join(HERE, "veto_worklist.json")
OUT = os.path.join(DATA, "csapi_veto.jsonl")
MATCHES = os.path.join(DATA, "csapi_matches_all.jsonl")
JAR = os.path.join(HERE, "hltv_cookies.tmp")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36")
BASE = "https://www.hltv.org"

# 2025-10 起的实际池只用到这 9 张；出现别的名字不算解析失败，但要留痕。
KNOWN_MAPS = {"Ancient", "Anubis", "Cache", "Dust2", "Inferno",
              "Mirage", "Nuke", "Overpass", "Train"}

MIN_REAL_BYTES = 20000          # 挑战页 ~5.7 KB，真比赛页 380–880 KB
KIND_RE = re.compile(r"^\s*(\d+)\.\s+(.*?)\s+(removed|picked)\s+(.+?)\s*$")
LEFT_RE = re.compile(r"^\s*(\d+)\.\s+(.+?)\s+was left over\s*$")
BOX_START = '<div class="standard-box veto-box"><div class="padding">'
DIV_RE = re.compile(r"<div>([\s\S]*?)</div>")
TAG_RE = re.compile(r"<[^>]+>")


def strip_tags(s):
    s = TAG_RE.sub(" ", s)
    for a, b in (("&nbsp;", " "), ("&amp;", "&"), ("&#39;", "'"), ("&quot;", '"')):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def veto_segment(html):
    """切出 veto 那一小块 HTML（到「was left over」所在的 `</div>` 为止，含该标签）。"""
    i = html.find(BOX_START)
    if i >= 0:
        start = i + len(BOX_START)
        k = html.find("was left over", start)
        if k >= 0:
            k2 = html.find("</div>", k)
            return html[start:(k2 + 6) if k2 >= 0 else len(html)]
    m = re.search(r"1\.[\s\S]{0,700}?was left over[^<]*", html)
    return m.group(0) if m else None


def is_block(status, body):
    if status in (403, 429, 503):
        return True
    if len(body) < MIN_REAL_BYTES:
        return True
    return b"Just a moment" in body[:4096]


def parse_veto(html):
    """→ (veto_list, err)。[[1,'re','Vitality','Ancient'], ..., [7,'L',None,'Inferno']]"""
    if "Just a moment" in html:
        return None, "challenge"
    if "was left over" not in html:
        return None, "no-veto-block"
    body = veto_segment(html)
    if not body:
        return None, "no-veto-block"
    lines = [strip_tags(d) for d in DIV_RE.findall(body)]
    if not lines:
        lines = [strip_tags(x) for x in body.split("</div>")]
    lines = [l for l in lines if l]
    steps = []
    for line in lines:
        m = LEFT_RE.match(line)
        if m:
            steps.append([int(m.group(1)), "L", None, m.group(2).strip()])
            continue
        m = KIND_RE.match(line)
        if m:
            kind = "re" if m.group(3) == "removed" else "pi"
            steps.append([int(m.group(1)), kind, m.group(2).strip(), m.group(4).strip()])
    if len(steps) != 7:
        return None, "steps=%d" % len(steps)
    lo = [s for s in steps if s[1] == "L"]
    if len(lo) != 1:
        return None, "leftover=%d" % len(lo)
    return steps, None


# ---------------------------------------------------------------- 传输（curl）

def curl_get(url, referer=None, timeout=30):
    """→ (status, body, secs)。用 curl 是因为 Python TLS 指纹会被 Cloudflare 挡住。"""
    fd, tmp = tempfile.mkstemp(suffix=".html", dir=HERE)
    os.close(fd)
    cmd = ["curl", "-sS", "--max-time", str(timeout), "-A", UA,
           "-b", JAR, "-c", JAR, "-L",
           "-H", "Accept: text/html,application/xhtml+xml,*/*;q=0.8",
           "-H", "Accept-Language: en-US,en;q=0.9",
           "-o", tmp, "-w", "%{http_code}"]
    if referer:
        cmd += ["-e", referer]
    cmd.append(url)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        try:
            status = int((p.stdout or "0").strip().splitlines()[-1])
        except Exception:
            status = 0
        body = b""
        if os.path.exists(tmp):
            body = io.open(tmp, "rb").read()
        return status, body, time.time() - t0
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def bootstrap(quiet=False):
    st, body, secs = curl_get(BASE + "/")
    if not quiet:
        print("bootstrap http=%d bytes=%d %.2fs" % (st, len(body), secs), flush=True)
    return st == 200 and not is_block(st, body)


# ---------------------------------------------------------------- 工作清单

def load_state():
    good, bad = set(), set()
    if os.path.exists(OUT):
        for line in io.open(OUT, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            (good if r.get("veto_ok") else bad).add(r["id"])
    bad -= good
    return good, bad


def load_dates():
    d = {}
    if os.path.exists(MATCHES):
        for line in io.open(MATCHES, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            d[r["id"]] = r.get("date")
    return d


def pending_list(order="priority"):
    good, _bad = load_state()
    work = json.load(io.open(WORK, encoding="utf-8"))
    pend = [w for w in work if w["id"] not in good]
    if order == "priority":
        # 先取 2026-06..09（Overpass→Cache 换图日 + 7 月池归属的关键窗口），其余按 id 降序
        dates = load_dates()

        def key(w):
            d = dates.get(w["id"]) or ""
            hot = 0 if "2026-06" <= d[:7] <= "2026-09" else 1
            return (hot, -w["id"])
        pend.sort(key=key)
    elif order == "desc":
        pend.sort(key=lambda w: -w["id"])
    elif order == "asc":
        pend.sort(key=lambda w: w["id"])
    return pend


# ---------------------------------------------------------------- 采集

def _append(rec):
    with io.open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run(limit=None, gap_ms=2500, order="priority", max_blocks=6, backoff_s=120,
        max_attempts=3, quiet=False):
    if not shutil.which("curl"):
        print("NO_CURL —— 中止", flush=True)
        return 2
    if not bootstrap(quiet=quiet):
        print("BOOTSTRAP_FAILED —— 中止", flush=True)
        return 2
    good, _ = load_state()
    pend = pending_list(order)
    print("VETO_FETCH_START pending=%d good=%d gap_ms=%d order=%s" %
          (len(pend), len(good), gap_ms, order), flush=True)

    done = blocked = failed = 0
    consecutive_blocks = 0
    t_start = time.time()
    for w in pend:
        if limit is not None and done + blocked + failed >= limit:
            break
        mid = w["id"]
        attempts, steps, note, st, body = 0, None, None, None, b""
        while attempts < max_attempts:
            attempts += 1
            st, body, _secs = curl_get(BASE + "/matches/%d/x" % mid, referer=BASE + "/")
            if is_block(st, body):
                blocked += 1
                consecutive_blocks += 1
                note = "challenge" if b"Just a moment" in body[:4096] else "blocked"
                if consecutive_blocks >= max_blocks:
                    print("VETO_FETCH_ABORT consecutive_blocks=%d done=%d blocked=%d failed=%d"
                          % (consecutive_blocks, done, blocked, failed), flush=True)
                    return 3
                if not quiet:
                    print("  block id=%d http=%s bytes=%d → sleep %ds"
                          % (mid, st, len(body), backoff_s), flush=True)
                time.sleep(backoff_s)
                bootstrap(quiet=True)
                continue
            steps, note = parse_veto(body.decode("utf-8", "replace"))
            break
        if steps:
            consecutive_blocks = 0
            picks = sorted(s[3] for s in steps if s[1] == "pi")
            played = sorted(w.get("played") or [])
            rec = {"id": mid, "bo": w.get("bo"), "played": played, "veto": steps,
                   "bytes": len(body), "status": st, "attempts": attempts, "src": "hltv-curl",
                   "picks_match": picks == played,
                   "decider": [s[3] for s in steps if s[1] == "L"][0],
                   "veto_ok": True}
            unknown = [s[3] for s in steps if s[3] not in KNOWN_MAPS]
            if unknown:
                rec["unknown_maps"] = unknown
            _append(rec)
            done += 1
        else:
            failed += 1
            _append({"id": mid, "bo": w.get("bo"), "played": w.get("played"), "veto": None,
                     "bytes": len(body), "status": st, "attempts": attempts,
                     "src": "hltv-curl", "note": note, "picks_match": False,
                     "decider": None, "veto_ok": False})
            if not quiet:
                print("  fail id=%d http=%s bytes=%d note=%s" % (mid, st, len(body), note),
                      flush=True)
        n = done + blocked + failed
        if n % 10 == 0:
            el = time.time() - t_start
            print("PROGRESS done=%d blocked=%d failed=%d elapsed=%.0fs rate=%.2fs/item"
                  % (done, blocked, failed, el, el / max(1, n)), flush=True)
        time.sleep(gap_ms / 1000.0)
    el = time.time() - t_start
    print("VETO_FETCH_DONE done=%d blocked=%d failed=%d elapsed=%.0fs"
          % (done, blocked, failed, el), flush=True)
    return 0


def probe(ids):
    if not bootstrap():
        return 2
    for mid in ids:
        st, body, secs = curl_get(BASE + "/matches/%s/x" % mid, referer=BASE + "/")
        steps, err = parse_veto(body.decode("utf-8", "replace"))
        print("%s http=%s bytes=%d %.2fs parse=%s"
              % (mid, st, len(body), secs, steps[0] if steps else err))
        time.sleep(2.5)
    return 0


# ---------------------------------------------------------------- 自证

FIXTURE = os.path.join(HERE, "fixtures", "veto_2396950.html")


def selftest():
    ok = True
    checks = []

    def chk(name, cond):
        nonlocal ok
        checks.append((name, bool(cond)))
        if not cond:
            ok = False

    if os.path.exists(FIXTURE):
        html = io.open(FIXTURE, encoding="utf-8").read()
        steps, err = parse_veto(html)
        chk("fixture parses", steps is not None and err is None)
        if steps:
            chk("fixture 7 steps", len(steps) == 7)
            chk("fixture step order 1..7", [s[0] for s in steps] == list(range(1, 8)))
            chk("fixture decider Inferno", [s[3] for s in steps if s[1] == "L"] == ["Inferno"])
            chk("fixture picks Dust2/Mirage",
                sorted(s[3] for s in steps if s[1] == "pi") == ["Dust2", "Mirage"])
            chk("fixture bans Ancient/Anubis/Nuke/Cache",
                sorted(s[3] for s in steps if s[1] == "re") == ["Ancient", "Anubis", "Cache", "Nuke"])
            chk("fixture team names keep spaces",
                steps[0][2] == "Vitality" and steps[1][2] == "MOUZ")
    else:
        chk("fixture present", False)

    chk("challenge page rejected", parse_veto("<html>Just a moment...</html>")[1] == "challenge")
    chk("page without veto rejected", parse_veto("<html>no data</html>")[1] == "no-veto-block")

    bo5 = ('<div class="standard-box veto-box"><div class="padding">'
           '<div>1. A removed Cache</div><div>2. B removed Anubis</div>'
           '<div>3. A picked Ancient</div><div>4. B picked Inferno</div>'
           '<div>5. A picked Overpass</div><div>6. B picked Dust2</div>'
           '<div>7. Mirage was left over</div></div></div>')
    st5, e5 = parse_veto(bo5)
    chk("bo5 synthetic parses", st5 is not None and e5 is None)
    if st5:
        chk("bo5 has 4 picks", len([s for s in st5 if s[1] == "pi"]) == 4)
        chk("bo5 decider Mirage", [s[3] for s in st5 if s[1] == "L"] == ["Mirage"])

    chk("6 steps rejected", parse_veto(bo5.replace('<div>6. B picked Dust2</div>', ''))[0] is None)
    chk("two leftovers rejected",
        parse_veto(bo5.replace('<div>6. B picked Dust2</div>', '<div>6. Dust2 was left over</div>'))[0]
        is None)
    chk("nbsp cleaned", strip_tags("<div>3.&nbsp;A picked&nbsp;Cache</div>") == "3. A picked Cache")

    # 拦截判定：挑战页（5.7 KB / 403 / Just a moment）必须被挡，真页面必须放行
    chk("block: 403", is_block(403, b"x" * 300000))
    chk("block: tiny body 200", is_block(200, b"x" * 5688))
    chk("block: just-a-moment body", is_block(200, b"<html>Just a moment...</html>" + b"x" * 300000))
    chk("pass: real page", not is_block(200, b"x" * 400000))

    n = sum(1 for _n, c in checks if c)
    for name, c in checks:
        print(("  ok   " if c else "  FAIL ") + name)
    print("VETO_FETCH_SELFTEST_%s (%d/%d checks)" % ("OK" if ok else "FAIL", n, len(checks)))
    return 0 if ok else 1


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "pending"
    if cmd == "selftest":
        return selftest()
    if cmd == "pending":
        from collections import Counter
        pend = pending_list("priority")
        dates = load_dates()
        c = Counter((dates.get(w["id"]) or "?")[:7] for w in pend)
        print("pending=%d" % len(pend))
        for m in sorted(c):
            print("  %s %d" % (m, c[m]))
        print("first 5:", [w["id"] for w in pend[:5]])
        return 0
    if cmd == "probe":
        return probe(argv[2:])
    if cmd == "run":
        kw = {}
        for a in argv[2:]:
            if "=" in a:
                k, v = a.split("=", 1)
                kw[k] = int(v) if v.lstrip("-").isdigit() else v
        return run(**kw)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
