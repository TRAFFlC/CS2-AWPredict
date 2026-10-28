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
* `maybe_start()` 由页面加载时调用。目标快照已存在且强度库已同步 → 立即 no-op。
* 需要抓取时起**后台线程**跑 `fetch_vrs.py --date <目标>`，不阻塞页面。
* 抓取成功后跑 `pickem/update_vrs.py --write`，把 rank/points 写进强度库；
  根页面直接读 `data/vrs_rank_*.json`，靠已有的 mtime 指纹热重载，无需重启。
* 失败或 HLTV 尚未发布时进入冷却（默认 900 秒），避免每次刷新都打网络。
* 跨进程文件锁 `data/cache/vrs_sync.lock`；状态写 `data/cache/vrs_sync_state.json`。
* `VRS_SYNC_DISABLE=1` 时只报告状态、不起线程（自检/离线环境用）。

用法
----
    python -X utf8 vrs_sync.py status
    python -X utf8 vrs_sync.py refresh [--force]
"""
from __future__ import annotations

import argparse
import datetime
import glob
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
COOLDOWN_S = int(os.environ.get("VRS_SYNC_COOLDOWN", "900"))
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


def _snapshot_info(path, filename_date=None):
    """读快照的 date / 队数；按 (path, mtime, size) 缓存，状态轮询不重复读盘。"""
    try:
        st = os.stat(path)
    except OSError as exc:
        return {"path": path, "date": filename_date, "teams": None, "error": str(exc)}
    key = (path, st.st_mtime_ns, st.st_size)
    hit = _CACHE.get("snapshot")
    if hit and hit[0] == key:
        return hit[1]
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        info = {"path": path,
                "date": str(doc.get("date") or filename_date or ""),
                "teams": len(doc.get("teams") or []),
                "error": None}
    except (OSError, ValueError) as exc:
        info = {"path": path, "date": filename_date, "teams": None, "error": str(exc)}
    _CACHE["snapshot"] = (key, info)
    return info


def _library_info():
    """读强度库的 asof / VRS 队数；按 (mtime, size) 缓存。"""
    try:
        st = os.stat(FIXTURE)
    except OSError as exc:
        return {"asof": None, "vrs": None, "teams": 0, "error": str(exc)}
    key = (st.st_mtime_ns, st.st_size)
    hit = _CACHE.get("library")
    if hit and hit[0] == key:
        return hit[1]
    try:
        with open(FIXTURE, encoding="utf-8") as fh:
            doc = json.load(fh)
        counts = doc.get("counts") or {}
        info = {"asof": doc.get("asof"), "vrs": counts.get("vrs"),
                "teams": len(doc.get("teams") or []), "error": None}
    except (OSError, ValueError) as exc:
        info = {"asof": None, "vrs": None, "teams": 0, "error": str(exc)}
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
def status():
    """给页面轮询的完整状态。只读盘，不触发抓取。"""
    now = now_bj()
    target = target_date(now)
    latest_date, latest_path = latest_snapshot()
    snap = _snapshot_info(latest_path, latest_date) if latest_path else None
    lib = _library_info()
    state = _read_state()

    target_ready = bool(snap and snap.get("date") and snap["date"] >= target.isoformat())
    library_current = bool(
        target_ready and snap and snap.get("teams") is not None
        and lib.get("asof") == snap["date"] and lib.get("vrs") == snap["teams"]
    )
    running = bool(_RUNNING or _lock_active())

    cooldown = 0
    if not target_ready and state.get("attempted_target") == target.isoformat():
        try:
            last = datetime.datetime.fromisoformat(str(state.get("last_attempt") or ""))
            if last.tzinfo is None:
                last = last.replace(tzinfo=BJ)
            cooldown = max(0, int(COOLDOWN_S - (now - last).total_seconds()))
        except ValueError:
            cooldown = 0

    if running:
        message = "正在检查 VRS %s…" % target.isoformat()
    elif target_ready and library_current:
        message = "VRS 已是最新（%s），强度库已同步" % snap["date"]
    elif target_ready:
        message = "VRS %s 已就绪，强度库待同步" % snap["date"]
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
        "target_ready": target_ready,
        "running": running,
        "library_asof": lib.get("asof"),
        "library_vrs": lib.get("vrs"),
        "library_current": library_current,
        "last_attempt": state.get("last_attempt"),
        "last_result": state.get("last_result"),
        "last_error": state.get("last_error"),
        "last_ok": state.get("last_ok"),
        "cooldown_s": cooldown,
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


def _log(message):
    sys.stderr.write("[vrs_sync %s] %s\n" % (now_bj().strftime("%H:%M:%S"), message))
    sys.stderr.flush()


def _run(force):
    """后台线程主体：抓取（如需要）→ 更新强度库 → 落状态。"""
    global _RUNNING
    target = target_date().isoformat()
    started = now_bj().isoformat(timespec="seconds")
    _write_state({"attempted_target": target, "last_attempt": started,
                  "last_result": "running", "last_error": None,
                  "message": "正在检查 VRS %s" % target})
    if not _acquire_lock():
        _write_state({"last_result": "locked",
                      "message": "另一个进程正在抓取 VRS，本次跳过"})
        _RUNNING = False
        return
    _log("开始：目标 %s（force=%s）" % (target, force))
    try:
        latest_date, latest_path = latest_snapshot()
        fetch_rc, fetch_out = None, ""
        if not latest_date or latest_date < target:
            _log("调用 fetch_vrs.py --date %s" % target)
            fetch_rc, fetch_out = _run_cmd(
                [sys.executable, "-X", "utf8", FETCH, "--date", target], FETCH_TIMEOUT)
            _write_state({"fetch_rc": fetch_rc, "fetch_tail": fetch_out[-2000:]})
            _log("fetch_vrs.py rc=%s" % fetch_rc)

        latest_date, latest_path = latest_snapshot()
        if latest_path:
            snap = _snapshot_info(latest_path, latest_date)
            lib = _library_info()
            stale = (lib.get("asof") != snap.get("date")
                     or lib.get("vrs") != snap.get("teams"))
            if stale:
                _log("更新强度库：%s（%s 队）" % (snap.get("date"), snap.get("teams")))
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
            if st["target_ready"] and st["library_current"]:
                return {"ok": True, "started": False, "reason": "up_to_date", "status": st}
            if not st["target_ready"] and st["cooldown_s"] > 0:
                return {"ok": True, "started": False, "reason": "cooldown", "status": st}
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
