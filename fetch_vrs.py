#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 HLTV 抓取 Valve 全球排名（VRS），并生成 `data/vrs_rank_<日期>.json`。

为什么用 Playwright
------------------
hltv.org 前面有 Cloudflare 挑战。普通 `requests` / `curl` 会拿到
403 "Just a moment..."，所以这里用真实 Chrome（`channel="chrome"`）无头加载，
等 `div.ranking` 里的 `ranked-team` 出现后，直接从 DOM 读取，不依赖脆弱的
HTML 片段正则。

防风控
------
- **先看本地快照**：`data/vrs_rank_<请求日期>.json` 已存在且可解析时，
  直接跳过，不发起任何网络请求。
- **页面实际日期再查一次**：如果 HLTV 当天还没更新、页面仍是前一天，
  而前一天快照已存在，也不重复写。
- 一天只跑一次；失败不自动重试（交给第二天的计划任务）。
- 成功一次后把 Cloudflare cookie 存到 `data/cache/hltv_storage.json`，
  下次复用；图片 / 字体 / 媒体请求全部拦截，减少流量。
- 并发锁 `data/cache/fetch_vrs.lock`，防止手动和计划任务撞车。

用法
----
    python -X utf8 fetch_vrs.py                 # 抓今天；已有快照则跳过
    python -X utf8 fetch_vrs.py --date 2026-10-26
    python -X utf8 fetch_vrs.py --force         # 已有也重抓并覆盖
    python -X utf8 fetch_vrs.py --dry-run       # 只抓取/校验，不写盘
    python -X utf8 fetch_vrs.py --headed        # Cloudflare 卡住时手动过一次

依赖：playwright（本机已装；Chrome 用系统安装的，不需要 `playwright install`）。

退出码：0 成功或已存在跳过；2 抓取失败；3 解析/校验失败。
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import html
import io
import json
import os
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CACHE = os.path.join(DATA, "cache")
STORAGE = os.path.join(CACHE, "hltv_storage.json")
LOCK = os.path.join(CACHE, "fetch_vrs.lock")
URL_TMPL = "https://www.hltv.org/valve-ranking/teams/{year}/{month}/{day}"
LATEST_URL = "https://www.hltv.org/valve-ranking/teams"
BJ = dt.timezone(dt.timedelta(hours=8))

MONTHS = ["january", "february", "march", "april", "may", "june",
          "july", "august", "september", "october", "november", "december"]

DATE_RE = re.compile(r"Valve global ranking on ([A-Za-z]+) (\d{1,2})(?:st|nd|rd|th)?, (\d{4})", re.I)
POS_RE = re.compile(r"#\s*(\d+)")
POINTS_RE = re.compile(r"\(\s*(\d+)")

sys.path.insert(0, HERE)
import ingest_vrs as IV  # noqa: E402

JS_ROWS = r"""
(root) => {
  const blocks = [...root.querySelectorAll('div.ranked-team.standard-box')];
  return blocks.map(b => {
    const q = (sel) => {
      const el = b.querySelector(sel);
      return el ? el.textContent.trim() : null;
    };
    return {
      pos: q('span.position.wide-position'),
      name: q('span.name'),
      points: q('span.points'),
      region: q('span.region')
    };
  });
}
"""


class FetchError(RuntimeError):
    pass


class ParseError(RuntimeError):
    pass


def url_for(day):
    return URL_TMPL.format(year=day.year, month=MONTHS[day.month - 1], day=day.day)


def parse_cli_date(s):
    if not s:
        return dt.date.today()
    try:
        return dt.date.fromisoformat(s)
    except ValueError:
        raise SystemExit("✗ --date 必须是 YYYY-MM-DD，实际是：%r" % s)


def snapshot_doc(path, date):
    """读取已有快照；不存在返回 None，存在但内容不对则报错（不静默覆盖）。"""
    if not os.path.exists(path):
        return None
    try:
        doc = json.load(io.open(path, encoding="utf-8"))
    except Exception as e:                                  # noqa: BLE001
        raise ParseError("已有快照无法解析：%s（%s；可用 --force 覆盖）" % (path, e))
    if not isinstance(doc, dict) or doc.get("date") != date:
        raise ParseError("已有快照日期不符：%s（内部 date=%r；可用 --force 覆盖）"
                         % (path, (doc or {}).get("date") if isinstance(doc, dict) else None))
    if not isinstance(doc.get("teams"), list) or not doc["teams"]:
        raise ParseError("已有快照没有 teams：%s（可用 --force 覆盖）" % path)
    return doc


def parse_page_date(text):
    """从 `Valve global ranking on October 26th, 2026` 解析日期；失败返回 None。"""
    m = DATE_RE.search(text or "")
    if not m or m.group(1).lower() not in MONTHS:
        return None
    month = MONTHS.index(m.group(1).lower()) + 1
    return dt.date(int(m.group(3)), month, int(m.group(2)))


def parse_rows(raw_rows):
    if not raw_rows:
        raise ParseError("页面里没有找到任何 div.ranked-team.standard-box")
    rows = []
    for i, r in enumerate(raw_rows, 1):
        pos = (r.get("pos") or "").strip()
        name = html.unescape((r.get("name") or "").strip())
        points = (r.get("points") or "").strip()
        region = (r.get("region") or "").strip().upper()
        mp = POS_RE.search(pos)
        mq = POINTS_RE.search(points)
        if not (mp and name and mq and region):
            raise ParseError("第 %d 个 ranked-team 区块字段缺失：%r" % (i, r))
        rows.append({"rank": int(mp.group(1)), "name": name,
                     "points": int(mq.group(1)), "region": region})
    return rows


def atomic_write(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def append_log(entry):
    os.makedirs(CACHE, exist_ok=True)
    entry = dict(entry)
    entry.setdefault("ts", dt.datetime.now().isoformat(timespec="seconds"))
    with io.open(os.path.join(CACHE, "fetch_vrs_runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def prune_html_cache(keep=14):
    """只保留最近 N 份成功抓取的 ranking HTML（约 1.5MB/天，避免无限增长）。"""
    try:
        files = sorted(glob.glob(os.path.join(CACHE, "hltv_valve_*.html")))
        for path in files[:-keep]:
            os.remove(path)
    except OSError:
        pass


def summarize_changes(old_teams, new_teams):
    """比较两版 teams，返回可打印/可解析的变化摘要。"""
    old = {t.get("name"): t for t in old_teams or []}
    new = {t.get("name"): t for t in new_teams or []}
    deltas = []
    for name in old.keys() & new.keys():
        op, np_ = old[name].get("points"), new[name].get("points")
        orank, nrank = old[name].get("rank"), new[name].get("rank")
        if op != np_ or orank != nrank:
            deltas.append((name, op, np_, orank, nrank))
    deltas.sort(key=lambda x: -abs((x[2] or 0) - (x[1] or 0)))
    return {
        "changed": len(deltas),
        "rank_changed": sum(1 for d in deltas if d[3] != d[4]),
        "added": sorted(set(new) - set(old)),
        "dropped": sorted(set(old) - set(new)),
        "top": [[d[0], d[1], d[2]] for d in deltas[:5]],
    }


def archive_snapshot(path, doc, revision):
    """覆盖同日期快照前，把旧版本存到 data/cache/vrs_history/。"""
    hist = os.path.join(CACHE, "vrs_history")
    try:
        os.makedirs(hist, exist_ok=True)
        stamp = str(doc.get("fetched_at") or "")
        if not stamp:
            try:
                stamp = dt.datetime.fromtimestamp(os.path.getmtime(path), BJ).isoformat()
            except OSError:
                stamp = "unknown"
        stamp = re.sub(r"[^0-9A-Za-z]+", "-", stamp).strip("-") or "unknown"
        base = os.path.splitext(os.path.basename(path))[0]
        dst = os.path.join(hist, "%s_r%s_%s.json" % (base, revision, stamp))
        shutil.copy2(path, dst)
        files = sorted(glob.glob(os.path.join(hist, "*.json")))
        for old in files[:-60]:
            os.remove(old)
        return dst
    except OSError as e:
        print("⚠ 旧版本归档失败（不影响新快照写入）：%s" % e)
        return None


def acquire_lock():
    os.makedirs(CACHE, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii"))
            os.close(fd)
            return True
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(LOCK) > 600:
                    os.remove(LOCK)
                    continue
            except OSError:
                continue
            return False
    return False


def release_lock():
    try:
        os.remove(LOCK)
    except OSError:
        pass


def _safe_title(page):
    try:
        return page.title()
    except Exception:                                       # noqa: BLE001
        return "?"


def _save_failure(page, url):
    os.makedirs(CACHE, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(CACHE, "hltv_failed_%s.html" % stamp)
    try:
        io.open(path, "w", encoding="utf-8").write(page.content())
        print("  失败页面已保存：%s" % path)
    except Exception:                                       # noqa: BLE001
        pass
    print("  url=%s title=%r" % (url, _safe_title(page)))


def _route(r):
    if r.request.resource_type in ("image", "media", "font"):
        r.abort()
    else:
        r.continue_()


def fetch_page(url, timeout=60, headed=False, date_hint=None):
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except Exception as e:                                  # noqa: BLE001
        raise FetchError("未安装 Playwright：pip install playwright（%s）" % e)

    with sync_playwright() as p:
        args = ["--disable-blink-features=AutomationControlled", "--no-sandbox"]
        try:
            browser = p.chromium.launch(channel="chrome", headless=not headed, args=args)
        except Exception as e:                              # noqa: BLE001
            raise FetchError("无法启动系统 Chrome；请确认已安装 Chrome，或先安装 "
                             "Playwright Chromium：python -m playwright install chromium（%s）" % e)
        try:
            major = browser.version.split(".")[0]
            ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/%s.0.0.0 Safari/537.36" % major)
            ctx_kw = dict(user_agent=ua, locale="en-US", viewport={"width": 1365, "height": 900},
                          extra_http_headers={"Accept-Language": "en-US,en;q=0.9"})
            if os.path.exists(STORAGE):
                ctx_kw["storage_state"] = STORAGE
            try:
                ctx = browser.new_context(**ctx_kw)
            except Exception:                               # noqa: BLE001
                print("⚠ 旧 storage_state 不可用，已忽略：%s" % STORAGE)
                ctx_kw.pop("storage_state", None)
                ctx = browser.new_context(**ctx_kw)

            page = ctx.new_page()
            page.route("**/*", _route)
            try:
                resp = page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            except PWTimeout as e:
                _save_failure(page, url)
                raise FetchError("页面加载超时（%ss）" % timeout) from e
            status = resp.status if resp else None
            if status and status >= 400 and status != 403:
                _save_failure(page, url)
                raise FetchError("HLTV 返回 HTTP %s" % status)

            try:
                page.wait_for_selector("div.ranking div.ranked-team.standard-box",
                                       timeout=timeout * 1000)
            except PWTimeout as e:
                title = _safe_title(page)
                _save_failure(page, url)
                if "just a moment" in title.lower() or "attention required" in title.lower():
                    raise FetchError("Cloudflare 挑战未通过。可先运行 "
                                     "`python -X utf8 fetch_vrs.py --headed --date %s` "
                                     "手动过盾；成功后 cookie 会缓存到 data/cache/。"
                                     % (date_hint or "YYYY-MM-DD")) from e
                raise FetchError("页面已打开，但没等到 div.ranking（title=%r）" % title) from e

            try:
                date_text = page.eval_on_selector(
                    "div.ranking .regional-ranking-header-text", "el => el.textContent.trim()")
                raw_rows = page.eval_on_selector("div.ranking", JS_ROWS)
                ranking_html = page.eval_on_selector("div.ranking", "el => el.outerHTML")
            except Exception as e:                          # noqa: BLE001
                _save_failure(page, url)
                raise FetchError("页面结构异常，无法读取 div.ranking：%s" % e) from e
            try:
                ctx.storage_state(path=STORAGE)
            except Exception as e:                          # noqa: BLE001
                print("⚠ 无法保存 Cloudflare cookie：%s" % e)
            return {"status": status, "date_text": date_text, "raw_rows": raw_rows,
                    "ranking_html": ranking_html, "title": _safe_title(page)}
        finally:
            browser.close()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="从 HLTV 抓取 Valve 全球排名（VRS）→ data/vrs_rank_<日期>.json")
    ap.add_argument("--date", default=None,
                    help="要抓取的日期 YYYY-MM-DD（默认本机今天）")
    ap.add_argument("--latest", action="store_true",
                    help="抓最新页（不带日期 URL），按页面 header 的实际日期写入")
    ap.add_argument("--out-dir", default=DATA, help="快照输出目录（默认 ./data）")
    ap.add_argument("--text-out", default=os.path.join(HERE, "valve_rank.txt"),
                    help="可读榜单输出路径（默认 ./valve_rank.txt）")
    ap.add_argument("--no-text", action="store_true", help="不写 valve_rank.txt")
    ap.add_argument("--force", action="store_true",
                    help="已有同日快照也重新抓取并覆盖")
    ap.add_argument("--dry-run", action="store_true", help="只抓取/校验，不写盘")
    ap.add_argument("--headed", action="store_true", help="显示 Chrome 窗口（过 Cloudflare 用）")
    ap.add_argument("--timeout", type=int, default=60, help="页面等待秒数（默认 60）")
    ap.add_argument("--keep-last", action="store_true",
                    help="重名队伍保留最后一次出现（默认保留第一次）")
    a = ap.parse_args(argv)

    requested = None if a.latest else parse_cli_date(a.date)
    url = LATEST_URL if a.latest else url_for(requested)
    req_iso = requested.isoformat() if requested else None
    req_path = os.path.join(a.out_dir, "vrs_rank_%s.json" % req_iso) if req_iso else None

    if not a.force and req_path:
        try:
            doc = snapshot_doc(req_path, req_iso)
        except ParseError as e:
            print("✗ %s" % e)
            return 3
        if doc:
            print("✓ 已有 %s（%d 支队），跳过网络请求。" % (req_path, len(doc["teams"])))
            append_log({"status": "skip_existing", "requested": req_iso,
                        "path": req_path, "teams": len(doc["teams"]),
                        "note": "已有同期日期快照，跳过网络请求"})
            return 0

    if not acquire_lock():
        print("⚠ 另一个 fetch_vrs.py 正在运行，本次跳过。")
        return 0
    try:
        print("抓取 %s ..." % url)
        got = fetch_page(url, timeout=a.timeout, headed=a.headed,
                         date_hint=req_iso or "latest")
        header = got.get("date_text") or ""
        if "valve global ranking" not in header.lower():
            raise ParseError("页面 header 不是 Valve global ranking：%r" % header)
        actual = parse_page_date(header)
        if actual is None:
            raise ParseError("无法从页面 header 解析日期：%r" % header)
        actual_iso = actual.isoformat()
        if requested and actual != requested:
            print("⚠ URL 日期 %s，页面实际日期 %s（HLTV 可能还没更新）" % (req_iso, actual_iso))
        rows = parse_rows(got["raw_rows"])
        try:
            IV.check(rows)
        except SystemExit as e:
            raise ParseError(str(e))
        actual_path = os.path.join(a.out_dir, "vrs_rank_%s.json" % actual_iso)
        existing = None
        if os.path.exists(actual_path):
            try:
                existing = snapshot_doc(actual_path, actual_iso)
            except ParseError as e:
                if not a.force:
                    raise ParseError(str(e))
                print("⚠ 旧快照不可用，将覆盖：%s" % e)
            if existing and not a.force:
                print("✓ 页面仍是 %s，且已有快照（%d 支队），跳过写入。"
                      % (actual_iso, len(existing["teams"])))
                append_log({"status": "skip_actual_existing", "requested": req_iso,
                            "actual": actual_iso, "teams": len(existing["teams"]),
                            "url": url,
                            "note": "页面日期未更新（仍是 %s），已有快照存在，跳过写入" % actual_iso})
                return 0

        doc = IV.build(actual_iso, rows, keep_first=not a.keep_last,
                       origin="fetch_vrs.py 从 HLTV Valve 全球排名页面抓取")
        print("日期 %s · 解析 %d 行 · 去重后 %d 支队 · 重名丢弃 %d 条"
              % (actual_iso, len(rows), len(doc["teams"]), len(doc["duplicates"])))
        for d in doc["duplicates"][:10]:
            print("  · %-24s 保留 #%d(%d)　丢弃 #%d(%d)"
                  % (d["name"], d["kept_rank"], d["kept_points"],
                     d["dropped_rank"], d["dropped_points"]))
        if len(doc["duplicates"]) > 10:
            print("  · …其余 %d 条见 JSON 的 duplicates 字段" % (len(doc["duplicates"]) - 10))

        old_teams = (existing or {}).get("teams") or []
        changed = summarize_changes(old_teams, doc["teams"]) if existing else None
        old_revision = int((existing or {}).get("revision") or 1)
        has_change = bool(changed and (changed["changed"]
                                      or changed["added"] or changed["dropped"]))
        revision = old_revision + 1 if has_change else old_revision
        fetched_at = dt.datetime.now(BJ).isoformat(timespec="seconds")
        doc["revision"] = revision
        doc["fetched_at"] = fetched_at

        if existing:
            if has_change:
                print("内容变化（第 %d 版）：积分变动 %d 支 · 新增 %d 支 · 掉榜 %d 支"
                      % (revision, changed["changed"],
                         len(changed["added"]), len(changed["dropped"])))
                for name, old_p, new_p in changed["top"]:
                    print("  · %-24s %s → %s" % (name, old_p, new_p))
            else:
                print("内容未变（第 %d 版），只更新抓取时间" % revision)
        print("VRS_CHANGE " + json.dumps({
            "revision": revision,
            "changed": (changed or {}).get("changed", 0),
            "added": len((changed or {}).get("added") or []),
            "dropped": len((changed or {}).get("dropped") or []),
            "fetched_at": fetched_at,
            "latest": bool(a.latest),
        }, ensure_ascii=False))

        if a.dry_run:
            print("--dry-run：未写盘。")
            append_log({"status": "dry_run", "requested": req_iso, "actual": actual_iso,
                        "teams": len(doc["teams"]), "url": url, "latest": bool(a.latest),
                        "revision": revision, "changed": (changed or {}).get("changed", 0)})
            return 0

        if has_change:
            archive_snapshot(actual_path, existing, old_revision)
        atomic_write(actual_path, json.dumps(doc, indent=1, ensure_ascii=False))
        try:
            if not a.no_text:
                lines = [actual_iso]
                for r in rows:
                    lines.append("%d. %s — %d — %s"
                                 % (r["rank"], r["name"], r["points"], r["region"]))
                atomic_write(a.text_out, "\n".join(lines) + "\n")
        except OSError as e:
            print("⚠ 快照已写入，但可读榜单写入失败：%s" % e)
        try:
            os.makedirs(CACHE, exist_ok=True)
            atomic_write(os.path.join(CACHE, "hltv_valve_%s.html" % actual_iso),
                         got["ranking_html"])
            prune_html_cache()
        except OSError as e:
            print("⚠ 快照已写入，但 HTML 缓存写入失败：%s" % e)
        append_log({"status": "ok", "requested": req_iso, "actual": actual_iso,
                    "teams": len(doc["teams"]), "duplicates": len(doc["duplicates"]),
                    "url": url, "path": actual_path, "latest": bool(a.latest),
                    "revision": revision, "changed": (changed or {}).get("changed", 0),
                    "fetched_at": fetched_at})
        print("✓ 已写入 %s（%d 字节）" % (actual_path, os.path.getsize(actual_path)))
        if not a.no_text:
            print("  可读榜单：%s" % a.text_out)
        print("  提示：常驻 predict_gui 会在下一个请求时自动重建 rank 索引，无需重启。")
        return 0
    except FetchError as e:
        print("✗ %s" % e)
        append_log({"status": "fetch_error", "requested": req_iso, "url": url, "error": str(e)})
        return 2
    except ParseError as e:
        print("✗ %s" % e)
        append_log({"status": "parse_error", "requested": req_iso, "url": url, "error": str(e)})
        return 3
    finally:
        release_lock()


if __name__ == "__main__":
    sys.exit(main() or 0)
