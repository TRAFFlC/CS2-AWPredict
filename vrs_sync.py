#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""页面刷新时自动抓取 HLTV VRS，并同步「分数预测」与「Major 竞猜」的数据。

日期规则（北京时间）
--------------------
* **08:00 之后（含 08:00）**：抓「当天」。
* **08:00 之前**：抓「前一天」。

例：10-29 凌晨 03:30 → 目标 10-28；10-29 上午 09:00 → 目标 10-29。
理由：HLTV 的 Valve 全球排名按欧洲/北美日界更新，北京时间凌晨通常还停在
前一日榜单；直接抓「本机今天」会得到 404 或空页。

行为
----
* `maybe_start()` 由页面加载时调用。快照新鲜且强度库已同步 → 立即 no-op。
* **同一天的榜单会变**：HLTV 在赛后持续重算，28 号凌晨和 28 号晚上看到的
  「2026-10-28」可能不是同一份数据。因此当前快照按 `REVALIDATE_S`（默认 3600 秒）
  限频复核：页面加载时如果距上次抓取已超过 1 小时，就后台重抓最新页。
* 抓取用 `fetch_vrs.py --latest --force`：抓不带日期的最新页，按页面 header 的
  实际日期写入；内容变了才升 `revision` 并把旧版本归档到 `data/cache/vrs_history/`。
* 抓取后跑 `pickem/update_vrs.py --write`，用 `content_sha` 判断强度库是否需要更新
  ——**队数不变、只有积分变化**时也会更新。根页面直接读 `data/vrs_rank_*.json`，
  靠已有的 mtime 指纹热重载，无需重启。
* 失败按 `RETRY_S`（默认 900 秒）重试；成功后按 `REVALIDATE_S` 复核。
* 跨进程文件锁 `data/cache/vrs_sync.lock`；状态写 `data/cache/vrs_sync_state.json`。
* `VRS_SYNC_DISABLE=1` 时只报告状态、不起线程（自检/离线环境用）。

用法
----
    python -X utf8 vrs_sync.py status
    python -X utf8 vrs_sync.py refresh [--force]
    python -X utf8 vrs_sync.py selftest
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CACHE = os.path.join(DATA, "cache")
STATE = os.path.join(CACHE, "vrs_sync_state.json")
LOCK = os.path.join(CACHE, "vrs_sync.lock")
FETCH = os.path.join(HERE, "fetch_vrs.py")
UPDATE = os.path.join(HERE, "pickem", "update_vrs.py")
FIXTURE = os.path.join(HERE, "pickem", "fixtures", "team_strength.json")

BJ = datetime.timezone(datetime.timedelta(hours=8))
CUTOFF_HOUR = 8
REVALIDATE_S = int(os.environ.get("VRS_SYNC_REVALIDATE", "3600"))
RETRY_S = int(os.environ.get("VRS_SYNC_RETRY",
                             os.environ.get("VRS_SYNC_COOLDOWN", "900")))
FETCH_TIMEOUT = int(os.environ.get("VRS_SYNC_FETCH_TIMEOUT", "240"))
UPDATE_TIMEOUT = int(os.environ.get("VRS_SYNC_UPDATE_TIMEOUT", "120"))
LOCK_STALE_S = 1200
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")

_MUTEX = threading.Lock()
_STATE_LOCK = threading.Lock()
_RUNNING = False
_CACHE = {}


# ------------------------------------------------------------------ 时间
def now_bj():
    return datetime.datetime.now(BJ)


def target_date(now=None):
    """按北京时间 08:00 切分：08:00 起取当天，之前取前一天。"""
    if now is None:
        now = now_bj()
    elif now.tzinfo is None:
        now = now.replace(tzinfo=BJ)
    bj = now.astimezone(BJ)
    if bj.hour >= CUTOFF_HOUR:
        return bj.date()
    return bj.date() - datetime.timedelta(days=1)


# ------------------------------------------------------------------ 快照/强度库
def _snapshot_paths():
    out = []
    for path in sorted(glob.glob(os.path.join(DATA, "vrs_rank_*.json"))):
        m = DATE_RE.match(os.path.basename(path)[len("vrs_rank_"):-len(".json")])
        if m:
            out.append((m.group(1), path))
    return out


def latest_snapshot():
    """返回 (date, path)；没有快照返回 (None, None)。只按文件名排序，不读内容。"""
    items = _snapshot_paths()
    return items[-1] if items else (None, None)


def content_sha(rows):
    """只对 rank/name/points 排序后做 sha256（与 pickem/update_vrs.py 同口径）。

    不含 fetched_at / revision：同一份榜单重复抓取时内容哈希不变。
    """
    blob = json.dumps(sorted((int(r["rank"]), str(r["name"]), float(r["points"]))
                             for r in rows),
                      ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _snapshot_info(path, filename_date=None):
    """读快照的 date / 队数 / revision / 抓取时间 / content_sha。

    按 (path, mtime, size) 缓存，状态轮询不重复读盘。
    """
    try:
        st = os.stat(path)
    except OSError as exc:
        return {"path": path, "date": filename_date, "teams": None,
                "revision": None, "fetched_at": None, "mtime": None,
                "content_sha": None, "error": str(exc)}
    key = (path, st.st_mtime_ns, st.st_size)
    hit = _CACHE.get("snapshot")
    if hit and hit[0] == key:
        return hit[1]
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        teams = doc.get("teams") or []
        try:
            sha = content_sha(teams)
        except (KeyError, TypeError, ValueError):
            sha = None
        info = {"path": path,
                "date": str(doc.get("date") or filename_date or ""),
                "teams": len(teams),
                "revision": int(doc.get("revision") or 1),
                "fetched_at": doc.get("fetched_at"),
                "mtime": st.st_mtime,
                "content_sha": sha,
                "error": None}
    except (OSError, ValueError) as exc:
        info = {"path": path, "date": filename_date, "teams": None,
                "revision": None, "fetched_at": None, "mtime": st.st_mtime,
                "content_sha": None, "error": str(exc)}
    _CACHE["snapshot"] = (key, info)
    return info


def _library_info():
    """读强度库的 asof / VRS 队数 / vrs_source；按 (mtime, size) 缓存。"""
    try:
        st = os.stat(FIXTURE)
    except OSError as exc:
        return {"asof": None, "vrs": None, "teams": 0,
                "vrs_source": {}, "error": str(exc)}
    key = (st.st_mtime_ns, st.st_size)
    hit = _CACHE.get("library")
    if hit and hit[0] == key:
        return hit[1]
    try:
        with open(FIXTURE, encoding="utf-8") as fh:
            doc = json.load(fh)
        counts = doc.get("counts") or {}
        info = {"asof": doc.get("asof"), "vrs": counts.get("vrs"),
                "teams": len(doc.get("teams") or []),
                "vrs_source": doc.get("vrs_source") or {},
                "error": None}
    except (OSError, ValueError) as exc:
        info = {"asof": None, "vrs": None, "teams": 0,
                "vrs_source": {}, "error": str(exc)}
    _CACHE["library"] = (key, info)
    return info


# ------------------------------------------------------------------ 状态文件
def _read_state():
    try:
        with open(STATE, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_state(updates):
    with _STATE_LOCK:
        doc = _read_state()
        doc.update(updates)
        doc["updated_at"] = now_bj().isoformat(timespec="seconds")
        try:
            os.makedirs(CACHE, exist_ok=True)
            tmp = STATE + ".tmp"
            with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(doc, fh, ensure_ascii=False, indent=1)
                fh.write("\n")
            os.replace(tmp, STATE)
        except OSError:
            pass


# ------------------------------------------------------------------ 跨进程锁
def _lock_active():
    try:
        age = time.time() - os.path.getmtime(LOCK)
    except OSError:
        return False
    if age > LOCK_STALE_S:
        try:
            os.remove(LOCK)
        except OSError:
            pass
        return False
    return True


def _acquire_lock():
    os.makedirs(CACHE, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, ("%d %s" % (os.getpid(),
                                     now_bj().isoformat(timespec="seconds"))).encode("ascii"))
            os.close(fd)
            return True
        except FileExistsError:
            if _lock_active():
                return False
            continue
        except OSError:
            return False
    return False


def _release_lock():
    try:
        os.remove(LOCK)
    except OSError:
        pass


# ------------------------------------------------------------------ 状态
def _parse_ts(text):
    try:
        ts = datetime.datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=BJ)
    return ts


def _age_seconds(now, fetched_at, mtime):
    """优先用快照自带的 fetched_at，旧快照回退到文件 mtime。"""
    ts = _parse_ts(fetched_at) if fetched_at else None
    if ts is not None:
        return max(0, int((now - ts).total_seconds()))
    if mtime:
        return max(0, int(now.timestamp() - mtime))
    return None


def _cooldown_seconds(now, target_ready, last_ts, last_result):
    """联网抓取的剩余冷却。

    目标日已就绪、只是超过 1 小时未复核 → 60 分钟窗口；
    目标日还没发布（例如 08:00 刚切到新一天）或上次失败 → 15 分钟重试。
    """
    if last_ts is None:
        return 0
    window = (REVALIDATE_S if target_ready
              and last_result in ("ok", "not_published") else RETRY_S)
    return max(0, int(window - (now - last_ts).total_seconds()))


def _library_matches(lib, snap):
    """强度库是否已与快照同步。优先 content_sha，缺失时回退 asof + 队数。"""
    if not (lib and snap and snap.get("date")):
        return False
    src = lib.get("vrs_source") or {}
    sha = snap.get("content_sha")
    if sha and src.get("content_sha"):
        return (src.get("date") == snap.get("date")
                and src.get("content_sha") == sha)
    return (lib.get("asof") == snap.get("date")
            and lib.get("vrs") == snap.get("teams"))


def status():
    """给页面轮询的完整状态。只读盘，不触发抓取。"""
    now = now_bj()
    target = target_date(now)
    latest_date, latest_path = latest_snapshot()
    snap = _snapshot_info(latest_path, latest_date) if latest_path else None
    lib = _library_info()
    state = _read_state()

    target_ready = bool(snap and snap.get("date") and snap["date"] >= target.isoformat())
    library_current = bool(target_ready and _library_matches(lib, snap))
    running = bool(_RUNNING or _lock_active())

    age = _age_seconds(now, snap.get("fetched_at"), snap.get("mtime")) if snap else None
    needs_update = bool(target_ready and not library_current)
    needs_fetch = bool(not target_ready or age is None or age >= REVALIDATE_S)

    # 只有「需要联网抓取」才看冷却；本地同步强度库不联网，不应被冷却挡住。
    cooldown = 0
    if needs_fetch:
        last_ts = _parse_ts(state.get("last_fetch_at") or state.get("last_attempt"))
        last_result = str(state.get("last_fetch_result")
                          or state.get("last_result") or "")
        cooldown = _cooldown_seconds(now, target_ready, last_ts, last_result)

    if running:
        message = "正在检查 VRS %s…" % target.isoformat()
    elif needs_update:
        message = "VRS %s 已就绪，强度库待同步" % snap["date"]
    elif target_ready and needs_fetch:
        message = "VRS %s 已超过 %d 分钟未复核" % (snap["date"], REVALIDATE_S // 60)
    elif target_ready:
        message = "VRS 已是最新（%s · 第 %s 版），强度库已同步" % (
            snap["date"], snap.get("revision") or "?")
    elif cooldown > 0:
        message = "HLTV 尚未发布 %s，%d 秒后重试" % (target.isoformat(), cooldown)
    else:
        message = "待抓取 VRS %s" % target.isoformat()
    if state.get("last_error") and not running:
        message += "；上次错误：" + str(state["last_error"])[:160]

    return {
        "ok": True,
        "now": now.isoformat(timespec="seconds"),
        "target": target.isoformat(),
        "target_rule": "北京时间 08:00 前取前一天，08:00 起取当天",
        "latest": snap.get("date") if snap else None,
        "latest_teams": snap.get("teams") if snap else None,
        "snapshot_revision": snap.get("revision") if snap else None,
        "snapshot_fetched_at": snap.get("fetched_at") if snap else None,
        "snapshot_age_s": age,
        "snapshot_content_sha": snap.get("content_sha") if snap else None,
        "target_ready": target_ready,
        "needs_fetch": needs_fetch,
        "needs_update": needs_update,
        "running": running,
        "library_asof": lib.get("asof"),
        "library_vrs": lib.get("vrs"),
        "library_content_sha": (lib.get("vrs_source") or {}).get("content_sha"),
        "library_revision": (lib.get("vrs_source") or {}).get("revision"),
        "library_current": library_current,
        "last_attempt": state.get("last_attempt"),
        "last_fetch_at": state.get("last_fetch_at"),
        "last_fetch_result": state.get("last_fetch_result"),
        "last_change": state.get("last_change"),
        "last_result": state.get("last_result"),
        "last_error": state.get("last_error"),
        "last_ok": state.get("last_ok"),
        "cooldown_s": cooldown,
        "revalidate_s": REVALIDATE_S,
        "retry_s": RETRY_S,
        "message": message,
    }


# ------------------------------------------------------------------ 抓取
def _run_cmd(cmd, timeout):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout,
                              env=env)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return 124, out + err + "\n[timeout %ss]" % timeout
    except OSError as exc:
        return 127, str(exc)


def _tail_line(text):
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    return lines[-1][:500] if lines else ""


def _parse_change(text):
    """从 fetch_vrs.py 输出里取 `VRS_CHANGE {json}` 行。"""
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line.startswith("VRS_CHANGE "):
            continue
        try:
            doc = json.loads(line[len("VRS_CHANGE "):])
        except ValueError:
            return None
        return doc if isinstance(doc, dict) else None
    return None


def _log(message):
    sys.stderr.write("[vrs_sync %s] %s\n" % (now_bj().strftime("%H:%M:%S"), message))
    sys.stderr.flush()


def _run(force):
    """后台线程主体：按需抓取 → 更新强度库 → 落状态。"""
    global _RUNNING
    target = target_date().isoformat()
    started = now_bj().isoformat(timespec="seconds")
    before = status()
    _write_state({"attempted_target": target, "last_attempt": started,
                  "last_result": "running", "last_error": None,
                  "message": "正在检查 VRS %s" % target})
    if not _acquire_lock():
        _write_state({"last_result": "locked",
                      "message": "另一个进程正在抓取 VRS，本次跳过"})
        _RUNNING = False
        return
    _log("开始：目标 %s（force=%s，needs_fetch=%s，needs_update=%s）"
         % (target, force, before.get("needs_fetch"), before.get("needs_update")))
    fetch_rc, fetch_out, change = None, "", None
    try:
        do_fetch = bool(force or before.get("needs_fetch"))
        if do_fetch:
            _log("调用 fetch_vrs.py --latest --force")
            _write_state({"last_fetch_at": now_bj().isoformat(timespec="seconds"),
                          "last_fetch_result": "running"})
            fetch_rc, fetch_out = _run_cmd(
                [sys.executable, "-X", "utf8", FETCH, "--latest", "--force"],
                FETCH_TIMEOUT)
            change = _parse_change(fetch_out)
            _write_state({
                "fetch_rc": fetch_rc,
                "fetch_tail": fetch_out[-2000:],
                "last_fetch_result": "ok" if fetch_rc == 0 else "fetch_error",
                "last_change": change,
            })
            _log("fetch_vrs.py rc=%s" % fetch_rc)

        latest_date, latest_path = latest_snapshot()
        snap = _snapshot_info(latest_path, latest_date) if latest_path else None
        lib = _library_info()
        stale = bool(snap and not _library_matches(lib, snap))
        if stale:
            _log("更新强度库：%s（%s 队，content_sha %s）"
                 % (snap.get("date"), snap.get("teams"),
                    str(snap.get("content_sha"))[:12]))
            urc, uout = _run_cmd(
                [sys.executable, "-X", "utf8", UPDATE, "--write"], UPDATE_TIMEOUT)
            _write_state({"update_rc": urc, "update_tail": uout[-2000:]})
            _log("update_vrs.py rc=%s" % urc)
            if urc != 0:
                _write_state({
                    "last_result": "update_error",
                    "last_error": _tail_line(uout) or ("update_vrs rc=%s" % urc),
                    "message": "快照已就绪，但强度库更新失败",
                })
                return

        latest_date, _ = latest_snapshot()
        if latest_date and latest_date >= target:
            _write_state({
                "last_result": "ok",
                "last_ok": now_bj().isoformat(timespec="seconds"),
                "last_error": None,
                "message": "VRS 已更新到 %s" % latest_date,
            })
            _log("完成：VRS %s" % latest_date)
        elif fetch_rc == 0:
            _write_state({
                "last_result": "not_published",
                "last_error": None,
                "message": "HLTV 尚未发布 %s（页面仍是 %s）" % (target, latest_date or "—"),
            })
            _log("未发布：目标 %s，最新 %s" % (target, latest_date))
        else:
            err = _tail_line(fetch_out) or ("fetch_vrs rc=%s" % fetch_rc)
            _write_state({
                "last_result": "fetch_error",
                "last_error": err,
                "message": "抓取 %s 失败" % target,
            })
            _log("失败：%s" % err)
    except Exception as exc:                                   # noqa: BLE001
        _write_state({
            "last_result": "error",
            "last_error": "%s: %s" % (type(exc).__name__, exc),
            "message": "VRS 同步异常",
        })
        _log("异常：%s: %s" % (type(exc).__name__, exc))
    finally:
        _release_lock()
        _RUNNING = False


def maybe_start(force=False):
    """页面加载时调用：需要时起后台线程。立即返回，不阻塞请求。"""
    global _RUNNING
    if os.environ.get("VRS_SYNC_DISABLE") == "1":
        return {"ok": True, "started": False, "reason": "disabled", "status": status()}
    with _MUTEX:
        if _RUNNING:
            return {"ok": True, "started": False, "reason": "running", "status": status()}
        st = status()
        if st["running"]:
            return {"ok": True, "started": False, "reason": "locked", "status": st}
        if not force:
            if st["needs_fetch"] and st["cooldown_s"] > 0:
                return {"ok": True, "started": False, "reason": "cooldown", "status": st}
            if not st["needs_fetch"] and not st["needs_update"]:
                return {"ok": True, "started": False, "reason": "up_to_date", "status": st}
        _RUNNING = True
    threading.Thread(target=_run, args=(bool(force),), daemon=True,
                     name="vrs-sync").start()
    return {"ok": True, "started": True, "reason": "started", "status": status()}


# ------------------------------------------------------------------ CLI
def _selftest():
    """不联网的自检：日期切分、状态结构、禁用开关。"""
    ok = 0

    def chk(cond, msg):
        nonlocal ok
        if not cond:
            raise AssertionError("FAIL: " + msg)
        ok += 1

    cases = [
        (datetime.datetime(2026, 10, 29, 3, 30, tzinfo=BJ), "2026-10-28"),
        (datetime.datetime(2026, 10, 29, 7, 59, 59, tzinfo=BJ), "2026-10-28"),
        (datetime.datetime(2026, 10, 29, 8, 0, 0, tzinfo=BJ), "2026-10-29"),
        (datetime.datetime(2026, 10, 29, 23, 59, 59, tzinfo=BJ), "2026-10-29"),
    ]
    for now, want in cases:
        got = target_date(now).isoformat()
        chk(got == want, "%s -> %s（实得 %s）" % (now.isoformat(), want, got))

    st = status()
    chk(st.get("ok") is True and st.get("target"), "status 有目标日期")
    chk("08:00" in (st.get("target_rule") or ""), "status 说明 08:00 日期规则")
    chk(bool(st.get("latest")) and bool(st.get("library_asof")),
        "status 能读到最新快照与强度库日期")
    chk(isinstance(st.get("snapshot_revision"), int)
        and st["snapshot_revision"] >= 1, "status 带快照 revision")
    chk(isinstance(st.get("snapshot_age_s"), int)
        and st["snapshot_age_s"] >= 0, "status 带快照年龄（秒）")
    chk(isinstance(st.get("needs_fetch"), bool)
        and isinstance(st.get("needs_update"), bool), "status 带抓取/同步判断")
    chk(st.get("revalidate_s") == REVALIDATE_S
        and st.get("retry_s") == RETRY_S, "status 带复核/重试间隔")

    rows_a = [{"rank": 1, "name": "A", "points": 100},
              {"rank": 2, "name": "B", "points": 90}]
    rows_b = list(reversed(rows_a))
    chk(content_sha(rows_a) == content_sha(rows_b), "content_sha 与顺序无关")
    chk(content_sha(rows_a) != content_sha(
        [{"rank": 1, "name": "A", "points": 101},
         {"rank": 2, "name": "B", "points": 90}]), "content_sha 对积分敏感")
    parsed = _parse_change('noise\nVRS_CHANGE {"revision": 2, "changed": 3}\n')
    chk(parsed and parsed.get("revision") == 2 and parsed.get("changed") == 3,
        "能解析 fetch_vrs 的 VRS_CHANGE 行")

    fixed_now = datetime.datetime(2026, 10, 29, 12, 0, tzinfo=BJ)
    five_min = fixed_now - datetime.timedelta(minutes=5)
    chk(_cooldown_seconds(fixed_now, True, five_min, "ok") > 3000,
        "目标日已就绪 → 60 分钟复核窗口")
    chk(_cooldown_seconds(fixed_now, False, five_min, "ok") <= RETRY_S,
        "目标日未发布 → 15 分钟重试窗口")
    chk(_cooldown_seconds(fixed_now, True,
                          fixed_now - datetime.timedelta(minutes=61), "ok") == 0,
        "超过 60 分钟 → 冷却归零")
    chk(_cooldown_seconds(fixed_now, True, five_min, "fetch_error") <= RETRY_S,
        "上次失败 → 15 分钟重试窗口")

    old = os.environ.get("VRS_SYNC_DISABLE")
    os.environ["VRS_SYNC_DISABLE"] = "1"
    try:
        result = maybe_start()
        chk(result.get("started") is False and result.get("reason") == "disabled",
            "VRS_SYNC_DISABLE=1 时不起后台线程")
    finally:
        if old is None:
            os.environ.pop("VRS_SYNC_DISABLE", None)
        else:
            os.environ["VRS_SYNC_DISABLE"] = old

    print("VRS_SYNC_SELFTEST_OK (%d checks)" % ok)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="VRS 自动抓取与强度库同步")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("status", help="打印当前状态（JSON）")
    p_ref = sub.add_parser("refresh", help="触发一次刷新（需要时抓网络）")
    p_ref.add_argument("--force", action="store_true", help="忽略冷却与已是最新判断")
    sub.add_parser("selftest", help="不联网自检日期规则与状态")
    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        return _selftest()

    if args.cmd == "refresh":
        result = maybe_start(force=args.force)
        print(json.dumps(result, ensure_ascii=False, indent=1))
        while result.get("started") and status()["running"]:
            time.sleep(1)
        if result.get("started"):
            print(json.dumps(status(), ensure_ascii=False, indent=1))
        return 0

    print(json.dumps(status(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
