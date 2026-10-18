#!/usr/bin/env python
"""预测器 Web 界面 —— 输入/选择两个队名，直接出预测结果。

为什么是常驻服务：`Predictor` 初始化 **9.5 秒**（要建账本、拟合选图模型），
但单次预测只要 **0.01 秒**。每次对话里手填参数跑脚本，等于每次都付那 9.5 秒。
本工具在后台线程里初始化一次，之后常驻复用。

零依赖：只用标准库 `http.server`（flask/fastapi 虽然装了，但不想给使用添加装依赖）。

启动：
    python -X utf8 predict_gui.py            # 默认 http://127.0.0.1:8765
    python -X utf8 predict_gui.py --port 9000 --open

界面要点（都是这几轮实测逼出来的，务必保留）：
  * **胜方取 rank 热门**（不是模型 argmax）：2026-10-13 复测——112 场时间前进式
    无泄漏（**模型已摄入 09-08~10-11 赛果**）：rank 67.0% vs 模型 63.4%（p=0.481），
    9 档置信门控无一超过 rank ⇒ **遵循 rank、不做门控**；rank 缺失才回退模型。
    分歧时显示 ⚠ 并给出模型的意见，由人决定（112 场分歧 18：rank 11 对 vs 模型 7 对）。
  * **⚠ 富度提示**：MVP 项按 lineup 选手历史出场场次分档
    （<18 场 ≈ 随机 20%，≥18 场 ≈ 40%）。冷门队伍必须看到这条。
  * **期望得分**是标定后的（MVP 项不再虚高 3 倍）。
  * **比分朝向**：换胜方时比分线自动翻转（scorer 要求二者一致）。
  * bo1 **不计比分项**（满分 25）；bo5 满分 55 且比分有 3:0/3:1/3:2 三种可能。
"""
from __future__ import annotations

import argparse
import datetime
import glob
import io
import json
import os
import re
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）
sys.path.insert(0, HERE)

# ---- rank 快照陈旧度阈值（实测标定，见下方注释）----
# 2026-10-13 实测：csapi 09-05 快照 与 HLTV 10-05 排名相比，
# 129 支有交集的队伍里**只有 3 支排名未变（2.3%）**，126 支都动了，
# 最大变动 159 位（Sangal #190→#48）。
# ⇒ 30 天不是保守估计，是**排名普遍翻动的量级**。
RANK_STALE_D = 30          # 超过即显著陈旧
RANK_DEAD_D = 60           # 超过即基本失效

# ⚠ 不要以为「抓到新的 rank 就更好」。同口径无泄漏对照（69 场真实对局）：
#     csapi 09-05   50/69 = 72.5%   ← 现用
#     HLTV  09-07   43/69 = 62.3%   ← 更新的 HLTV 体系，反而低 10.2pp
#   McNemar p=0.065（csapi 赢 9 场、输 2 场）。
#   ⇒ 差距来自**排名体系不同**，不是「数据旧」。
#   所以 csapi 恢复后重抓的价值是「消除陈旧」，**不是**「换算法」，
#   重抓后必须重新验证净值，不能默认变好。

ALIAS = {
    "mouz": "MOUZ", "navi": "Natus Vincere", "faze": "FaZe",
    "betboom": "BetBoom", "big": "BIG", "mibr": "MIBR", "nip": "Ninjas in Pyjamas",
    "vitality": "Vitality", "fnatic": "fnatic", "100thieves": "100 Thieves",
    "jijiehao": "JiJieHao", "eternalfire": "Eternal Fire",
    "gamelegion": "GamerLegion", "gamerlegion": "GamerLegion",
    "mongolz": "The MongolZ", "parivision": "PARIVISION", "pvision": "PARIVISION",
    "innercircle": "Inner Circle", "infinite": "INFINITE", "3dmax": "3DMAX",
    "eyeballers": "EYEBALLERS", "luminosity": "Luminosity",
    "metizport": "Metizport", "m80": "M80", "b8": "B8", "astr": "Astralis",
    "sashi": "Sashi", "fokus": "FOKUS", "betclic": "betclic", "bbl": "BBL",
    "johnnyspeeds": "Johnny Speeds", "johnny speeds": "Johnny Speeds",
    "heroic": "HEROIC", "virtus.pro": "Virtus.pro", "sangal": "Sangal",
    "phantom": "Phantom", "astral": "ASTRAL", "eac": "EAC", "5star": "5star",
    "magic": "magic", "tyloo": "TYLOO", "legacy": "Legacy", "alliance": "Alliance",
    "g2": "G2", "furia": "FURIA", "aurora": "Aurora", "nrg": "NRG",
    "9z": "9z", "magic_": "magic",
}

_STATE = {"ready": False, "error": None, "pred": None, "teams": [],
          "t0": time.time(), "reloads": 0, "last_reload": None,
          "rank_syncs": 0, "last_rank_sync": None, "_rank_fp": None}

# 2026-10-14：Predictor 在**启动时**把 state_2026-10.pkl / rank 索引一次性读进内存。
# 数据重建（ingest_new.py --rebuild）后不重启就永远用旧账本 —— 实测表现为选图
# 概率明显偏低（Legacy vs PARIVISION 的 Ancient 0.60 vs 重载后 0.75）。
# 故提供 /api/reload 重新构造 Predictor，并显式重建 rank 索引。
_RELOAD_LOCK = threading.Lock()

# ⚠ 2026-10-19：rank 索引**不再**只能靠手动刷新。
#   起因——把新的 `data/vrs_rank_<日期>.json` 丢进目录后，页面排名纹丝不动，
#   而日志里没有任何报错：`Predictor._build_rank()` 只在构造时跑一次，
#   常驻进程永远不会重新看磁盘。旧快照是全服务共用的单例状态，
#   修法不能是「让调用方记得刷新」。
#   这里改成**每个请求前比对 rank 数据的 (mtime, size) 指纹**，变了就地重建：
#   只读一份几十 KB 的 JSON + 1.2MB 兜底文件，毫秒级完成，
#   不触发重建 Predictor 那 10~20 秒。指纹相同则完全不做事，零常态开销。
#   独立于 _RELOAD_LOCK：完整重载要十几秒，不能让查状态的请求跟它排队。
_SYNC_LOCK = threading.Lock()


def rank_fingerprint():
    """rank 数据的指纹：所有 `vrs_rank_*.json` + csapi 兜底文件的 (名, mtime_ns, size)。

    只认这三个属性而不认内容哈希：这些文件是几 MB~几十 KB 级的小 JSON，
    请求路径上一次 stat 就够；真去读内容哈希反而把「毫秒级」变成「百毫秒级」。
    """
    fps = []
    for p in sorted(glob.glob(os.path.join(DATA, "vrs_rank_*.json"))):
        try:
            stt = os.stat(p)
        except OSError:                                # 并发删除/替换中
            continue
        fps.append((os.path.basename(p), stt.st_mtime_ns, stt.st_size))
    p = os.path.join(DATA, "csapi_matches_all.jsonl")
    try:
        stt = os.stat(p)
        fps.append((os.path.basename(p), stt.st_mtime_ns, stt.st_size))
    except OSError:
        pass
    return tuple(fps)


def sync_rank():
    """磁盘上的 rank 数据变了就自动重建索引。返回变化摘要，没变返回 None。

    与 `reload_model()` 的分工：那个重建整个 Predictor（含 state 账本与选图模型，
    十几秒），这个只换 rank 索引（毫秒级）。换排名快照是**高频**操作，
    账本重建是低频操作，不该绑在同一条路径上。
    """
    pr = _STATE["pred"]
    if pr is None or not hasattr(pr, "_build_rank"):
        return None
    fp = rank_fingerprint()
    if fp == _STATE.get("_rank_fp"):
        return None
    with _SYNC_LOCK:
        pr = _STATE["pred"]                          # 等锁期间可能已被 reload 换掉
        if pr is None:
            return None
        fp = rank_fingerprint()
        if fp == _STATE.get("_rank_fp"):
            return None
        before = getattr(pr, "_rank_snap", "")
        try:
            pr._build_rank()
            teams = build_teams(pr)
        except Exception as e:                       # noqa: BLE001
            sys.stderr.write("[predict_gui] sync_rank: %s: %s\n"
                             % (type(e).__name__, e))
            sys.stderr.flush()
            return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
        if _STATE["pred"] is not pr:                  # 期间发生了完整重载，交给它
            return None
        _STATE["teams"] = teams
        _STATE["_rank_fp"] = fp
        _STATE["rank_syncs"] += 1
        _STATE["last_rank_sync"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return {"ok": True, "from": before, "to": getattr(pr, "_rank_snap", ""),
                "teams": len(teams),
                "rank_teams": len(getattr(pr, "_rank_idx", None) or {})}


def reload_model():
    """重新加载模型 + rank 索引 + 队伍清单。返回给前端的状态摘要。"""
    import predict as PR
    with _RELOAD_LOCK:
        old = _STATE["pred"]
        pr = PR.Predictor(quiet=True)
        # ⚠ _build_rank() 只在首次调用时建表；已有 Predictor 不会自动感知
        #   磁盘上新的 vrs_rank_*.json，故显式重建。
        if hasattr(pr, "_build_rank"):
            try:
                pr._build_rank()
            except Exception as e:                       # noqa: BLE001
                return {"ok": False, "error": "_build_rank: %s" % e,
                        "reloads": _STATE["reloads"]}
        _STATE["pred"] = pr
        _STATE["teams"] = build_teams(pr)
        _STATE["_rank_fp"] = rank_fingerprint()      # 刚建过，别让 sync_rank 立刻再跑一次
        _STATE["ready"] = True
        _STATE["error"] = None
        _STATE["reloads"] += 1
        _STATE["last_reload"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if old is not None:
            del old
        rk = getattr(pr, "_rank_idx", None) or {}
        return {"ok": True, "reloads": _STATE["reloads"],
                "last_reload": _STATE["last_reload"],
                "teams": len(_STATE["teams"]),
                "rank_teams": len(rk),
                "rank_snapshot": getattr(pr, "_rank_snap", ""),
                "last_date": getattr(getattr(pr, "state", None), "last_date", None)}


# ------------------------------------------------------------------ 队伍清单
def build_teams(pr):
    """取全部队伍 + rank，并附上当前账本的 games 与选手丰度。

    ⚠ 2026-10-19 修：rank 原先**只**读 `csapi_matches_all.jsonl` 内嵌的名次，
      那是 2026-09-05 的**冻结**快照 ⇒ 页面下拉里的 `#N` 永远是 09-05 的旧名次，
      哪怕磁盘上已经换了更新的 `vrs_rank_*.json`。实测表现就是「更新了数据、
      页面排名纹丝不动」，很容易误判成快照没生效。
      现改为与 `predict.py:rank_pick` 同一套两级口径：
      **VRS 索引优先，csapi 只兜底**，并在 `rank_src` 里标出每支队名的来源，
      前端据此给兜底项加 `*` 标记。
    """
    info = {}
    p = os.path.join(DATA, "csapi_matches_all.jsonl")
    if os.path.exists(p):
        for line in io.open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            for t in (r["team1"], r["team2"]):
                nm = t["name"]
                d = info.setdefault(nm, {"rank": None})
                if t.get("rank"):
                    d["rank"] = int(t["rank"])
    # ---- VRS 覆盖 csapi（同名归一化后比对，与 predict.py:_norm 完全一致）----
    ridx = getattr(pr, "_rank_idx", None) or {}
    rsrc = getattr(pr, "_rank_src", None) or {}
    nrm = getattr(pr, "_norm", None) or (
        lambda t: "".join(ch for ch in str(t).lower() if ch.isalnum()))
    for nm, d in info.items():
        k = nrm(nm)
        if k in ridx:
            d["rank"], d["rank_src"] = int(ridx[k]), rsrc.get(k)
        elif d["rank"] is not None:
            d["rank_src"] = "csapi"
        else:
            d["rank_src"] = None
    st = pr.state
    date = getattr(st, "last_date", None)
    out = []
    for nm, d in info.items():
        lu = st.lineup(nm, date) if date else []
        gs = sorted(float(st.p_w.get(x) or 0) for x in lu) if lu else []
        rich = gs[len(gs) // 2] if gs else 0.0
        out.append({"name": nm, "rank": d["rank"], "rank_src": d["rank_src"],
                    "games": int(st.games.get(nm, 0)),
                    "richness": round(rich, 1),
                    "has_lineup": bool(lu)})
    out.sort(key=lambda z: (z["rank"] is None, z["rank"] or 999, z["name"]))
    return out


def resolve_team(q):
    q = (q or "").strip()
    if not q:
        return None
    k = re.sub(r"[^a-z0-9]", "", q.lower())
    names = _STATE["teams"]
    exact = [t for t in names if t["name"].lower() == q.lower()]
    if exact:
        return exact[0]["name"]
    al = ALIAS.get(k)
    if al:
        for t in names:
            if t["name"].lower() == al.lower():
                return t["name"]
    norm = {re.sub(r"[^a-z0-9]", "", t["name"].lower()): t["name"] for t in names}
    if k in norm:
        return norm[k]
    hit = [v for kk, v in norm.items() if k and (k in kk or kk in k)]
    return hit[0] if len(hit) == 1 else None


# ------------------------------------------------------------------ 预测
def predict(pr, a, b, bo):
    A, B = resolve_team(a), resolve_team(b)
    if not A:
        raise ValueError("找不到队伍：%r" % a)
    if not B:
        raise ValueError("找不到队伍：%r" % b)
    if A == B:
        raise ValueError("两队不能相同")
    bo = int(bo)
    if bo not in (1, 3, 5):
        raise ValueError("bo 只能是 1/3/5")
    p = pr.predict(A, B, bo)
    maxp = {1: 25, 3: 45, 5: 55}[bo]

    warn = []
    src = p.get("rank_src") or {}
    stale = None
    if src.get("ok"):
        # 来源标签形如 "VRS 2026-10-13" / "csapi 2026-09-05"，
        # ⚠ **不能**直接 [:10] 切片取日期——那会切出 "VRS 2026-1" 而解析失败。
        def _d(tag):
            m = re.search(r"(\d{4}-\d{2}-\d{2})", tag or "")
            if not m:
                return None
            try:
                return datetime.date(*map(int, m.group(1).split("-")))
            except Exception:                           # noqa: BLE001
                return None
        tags = [src.get("srcA"), src.get("srcB")]
        dA, dB = _d(src.get("srcA")), _d(src.get("srcB"))
        ds = [x for x in (dA, dB) if x]
        if ds:
            stale = (datetime.date.today() - min(ds)).days   # 用**较旧**那一侧算
        oldest_date = min(ds).isoformat() if ds else (src.get("snapshot") or "?")
        if src.get("mixed"):
            pairs = [(t, d) for t, d in zip(tags, (dA, dB)) if d is not None]
            lo = min(ds)
            older = "、".join("%s（%s）" % (t, d) for t, d in pairs if d == lo)
            warn.append("🟠 本场两队 rank **来源不同**：较旧的一侧是 %s，只能沿用旧榜兜底。"
                        "判定混了两个时间点（相差 %d 天），请谨慎"
                        % (older or "旧榜", stale or 0))
    if not src.get("ok"):
        warn.append("🔴 rank 缺失（%s）⇒ 胜方退回模型判断，而模型在双方≤16 上"
                    "不优于热门基线，请谨慎" % src.get("why", "?"))
    elif stale is not None and stale >= RANK_DEAD_D:
        warn.append("🔴 本场 rank 名次来自 %s 的旧榜（已 %d 天）。实测 30 天内 96.8%% "
                    "的队伍排名已变、最大 159 位 ⇒ 胜方几乎只是「一个月前的共识」。"
                    "有把握请自己覆盖" % (oldest_date, stale))
    elif stale is not None and stale >= RANK_STALE_D:
        warn.append("🟡 本场 rank 名次来自 %s 的旧榜（已 %d 天），排名普遍翻动，"
                    "胜方可信度下降；模型在此时分歧更值得听" % (oldest_date, stale))
    tmap = {t["name"]: t for t in _STATE["teams"]}
    for nm in (A, B):
        t = tmap.get(nm)
        if not t:
            continue
        if t["rank"] is None:
            warn.append("🔴 %s 无 rank 数据" % nm)
        elif t["rank"] > 60:
            warn.append("🟡 %s rank#%d（冷门）" % (nm, t["rank"]))
        if not t["has_lineup"]:
            warn.append("🔴 %s 无阵容数据 ⇒ MVP 项只能取随机" % nm)
        elif t["richness"] < 18:
            warn.append("🟡 %s 选手历史中位 %.0f 场（<18）⇒ MVP 项≈随机 20%%"
                        % (nm, t["richness"]))
    if p.get("disagreement"):
        warn.append("⚠ 模型与 rank 分歧：我方 rank 选 %s，模型倾向 %s（%.1f%%）。"
                    "实测分歧仅占 11%%，其中 rank 对 6/8——**默认跟 rank**，"
                    "但 rank 差距大（>6）或快照陈旧时更该听模型"
                    % (p["winner"], p["model_winner"],
                       100 * max(p["win_prob"].values())))
    if bo == 1:
        warn.append("ℹ bo1 不计比分项（满分 25），选图是 7 选 1、MVP 占 40%")
    if bo == 5:
        warn.append("ℹ bo5 比分有 3:0/3:1/3:2 三种，语料横扫率仅 36.7%"
                    "（bo3 的 2:0 是 55.2%）")
    return {
        "A": A, "B": B, "bo": bo, "max_points": maxp,
        "winner": p["winner"], "loser": B if p["winner"] == A else A,
        "p_win": round(100 * p["win_prob"][p["winner"]], 1),
        "model_winner": p.get("model_winner"),
        "disagreement": bool(p.get("disagreement")),
        "rank_src": src, "rank_stale_days": stale,
        "score": "%d:%d" % (p["score"][0], p["score"][1]),
        "maps": p["maps"], "map_probs": p.get("map_probs") or {},
        # ⚠ 概率降序专供显示。`maps` 是字母规范序（= 计划身份，计分用，
        # scorer 默认按集合判分不看顺序）。前端若直接 join(maps)，
        # 会把 Mirage 74% 显示在 Dust2 57% 后面 —— 2026-10-19 主理人当场抓到过。
        "maps_ranked": list(p.get("maps_ranked") or p["maps"]),
        "mvp": p.get("mvp"), "mvp_probs": p.get("mvp_probs") or {},
        "exp_points": round(p["exp_points"], 1),
        "sd": round(p.get("sd", 0), 1),
        "warnings": warn,
    }


# ------------------------------------------------------------------ HTTP
HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CS2 预测器</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--line:#262b36;--fg:#e6e9ef;--dim:#9aa3b2;
--acc:#4da3ff;--ok:#3ecf8e;--warn:#ffb020;--bad:#ff5c5c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.6 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:20px 16px 60px}
h1{font-size:19px;margin:0 0 4px}
.sub{color:var(--dim);font-size:12px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:16px;margin-bottom:14px}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end}
.f{flex:1 1 200px;min-width:180px;position:relative}
label{display:block;font-size:12px;color:var(--dim);margin-bottom:5px}
select,input,button{width:100%;padding:9px 11px;border-radius:7px;
border:1px solid var(--line);background:#0e1117;color:var(--fg);font-size:14px}
button{cursor:pointer;background:var(--acc);color:#07121f;font-weight:600;
border:none;padding:10px 22px}
button:disabled{opacity:.5;cursor:not-allowed}
/* ⚠ 下拉必须**绝对定位**浮在内容之上。
   原先它在文档流里：展开时把卡片撑高并溢出，后续卡片（.card 无 z-index）
   会后绘制把它盖住 ⇒ 队名列表被压在「最近」框下面，点不到。
   .f 需 position:relative 作定位锚点，z-index 要高于任何普通卡片。 */
.lst{position:absolute;top:100%;left:0;right:0;z-index:60;
max-height:260px;overflow:auto;border:1px solid var(--line);
border-radius:8px;margin-top:6px;display:none;
background:#141821;box-shadow:0 10px 28px rgba(0,0,0,.55)}
.lst.on{display:block}
.it{padding:7px 11px;cursor:pointer;border-bottom:1px solid #1c212b;
display:flex;justify-content:space-between;gap:10px}
.it:hover{background:#1d2330}
.it b{font-weight:600}
.tag{font-size:11px;color:var(--dim);white-space:nowrap}
.win{font-size:26px;font-weight:700;margin:2px 0 4px}
.bar{height:9px;background:#0e1117;border-radius:5px;overflow:hidden;
border:1px solid var(--line)}
.bar>i{display:block;height:100%;background:var(--acc)}
.kv{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
.kv div{flex:1 1 150px;background:#0e1117;border:1px solid var(--line);
border-radius:8px;padding:10px 12px}
.kv b{display:block;font-size:11px;color:var(--dim);font-weight:400;margin-bottom:3px}
.kv span{font-size:15px}
ul.w{margin:10px 0 0;padding-left:18px;color:var(--warn);font-size:13px}
ul.w li{margin:3px 0}
.pill{display:inline-block;font-size:11px;padding:2px 8px;border-radius:20px;
background:#0e1117;border:1px solid var(--line);color:var(--dim);margin-right:6px}
.pillOk{border-color:#1f5c3d;color:var(--ok)}
.pillWarn{border-color:#6b4a10;color:var(--warn);background:#1d1608}
.pillBad{border-color:#6b2020;color:var(--bad);background:#1d0e0e;font-weight:600}
#hist{font-size:12px;color:var(--dim)}
#hist a{color:var(--acc);text-decoration:none;margin-right:10px;
cursor:pointer}
#hist a:hover{text-decoration:underline}
/* 错误提示卡：只着色，不覆盖 .card 的 padding（旧 padding:10px 0 会把内边距压成 0） */
.err{color:var(--bad)}
.hint{color:var(--dim);font-size:12px;margin-top:6px}
</style></head><body><div class="wrap">
<h1>CS2 预测器</h1>
<div class="sub">胜方取 rank 热门（模型 argmax 实测不优于它）· MVP 按数据丰度标定 · bo1 不计比分</div>

<div class="card">
  <div class="row">
    <div class="f"><label>队伍 A</label>
      <input id="a" placeholder="输入或点选，如 vitality / MOUZ" autocomplete="off">
      <div class="lst" id="la"></div></div>
    <div class="f"><label>队伍 B</label>
      <input id="b" placeholder="输入或点选，如 G2 / FURIA" autocomplete="off">
      <div class="lst" id="lb"></div></div>
    <div class="f" style="flex:0 0 110px"><label>赛制</label>
      <select id="bo"><option value="3" selected>BO3 · 45</option>
      <option value="1">BO1 · 25</option>
      <option value="5">BO5 · 55</option></select></div>
    <div class="f" style="flex:0 0 auto"><label>&nbsp;</label>
      <button id="go" disabled>预测</button></div>
  </div>
  <div class="hint" id="st">正在加载模型（首次约 10 秒）…</div>
  <div class="hint">支持简称：mouz→MOUZ、navi→Natus Vincere、gamelegion→GamerLegion、
    nip→Ninjas in Pyjamas 等</div>
</div>

<div id="out"></div>
<div class="card"><label>最近</label><div id="hist"></div></div>
</div>
<script>
let TEAMS=[],HIST=[],LAST_SNAP=null,LAST_SYNC=null;
const $=s=>document.getElementById(s), esc=s=>String(s??"").replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
// ⚠ 名次后带 `*` = 该队名次来自 csapi 冻结兜底（2026-09-05），不是最新 VRS 快照。
function rankTxt(t){return t.rank?('#'+t.rank+(/csapi/.test(t.rank_src||'')?'*':'')):'无rank';}
function gamesTxt(t){return t.games+'场';}
function richTxt(t){return t.has_lineup?('丰度'+t.richness):'无阵容';}

function bind(inputId,listId){
  const inp=$(inputId),lst=$(listId);
  function closeAll(except){ ['la','lb'].forEach(id=>{ if(id!==except) $(id).classList.remove('on'); }); }
  function draw(){
    closeAll(listId);              // 同时只开一个，否则两个浮层互相盖住
    const q=inp.value.trim().toLowerCase();
    let f=TEAMS;
    if(q) f=TEAMS.filter(t=>t.name.toLowerCase().includes(q));
    f=f.slice(0,80);
    lst.innerHTML=f.map(t=>`<div class="it" data-n="${esc(t.name)}"><b>${esc(t.name)}</b>
      <span class="tag">${rankTxt(t)} · ${gamesTxt(t)} · ${richTxt(t)}</span></div>`).join('')
      ||'<div class="it">无匹配</div>';
    lst.classList.add('on');
    lst.querySelectorAll('.it[data-n]').forEach(el=>el.onclick=()=>{
      inp.value=el.dataset.n;lst.classList.remove('on');inp.blur();go();});
  }
  inp.onfocus=draw;
  inp.oninput=draw;
  inp.onblur=()=>setTimeout(()=>lst.classList.remove('on'),180);
  // ⚠ 必须阻止列表自身的 mousedown 默认行为：否则按下鼠标的瞬间 input 先 blur，
  //   180ms 后列表被关掉——拖动滚动条会闪没，快速点击也可能落空。
  lst.onmousedown=e=>e.preventDefault();
  inp.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();lst.classList.remove('on');go();}
    else if(e.key==='Escape'){lst.classList.remove('on');inp.blur();}};
}
document.addEventListener('mousedown',e=>{
  if(!e.target.closest('.f')) ['la','lb'].forEach(id=>$(id).classList.remove('on'));
});
function bar(p){return `<div class="bar"><i style="width:${p}%"></i></div>`;}

async function go(){
  const a=$('a').value.trim(),b=$('b').value.trim(),bo=$('bo').value;
  if(!a||!b){$('out').innerHTML='<div class="card err">请填两支队伍</div>';return;}
  $('go').disabled=true;$('st').textContent='预测中…';
  try{
    const r=await fetch(`/api/predict?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}&bo=${bo}`);
    const j=await r.json();
    if(!r.ok) throw new Error(j.error||'失败');
    render(j);
    HIST.unshift(`${j.A} vs ${j.B} (bo${j.bo})`);
    HIST=HIST.slice(0,12);
    $('hist').innerHTML=HIST.map(h=>`<a>${esc(h)}</a>`).join('');
  }catch(e){$('out').innerHTML=`<div class="card err">${esc(e.message)}</div>`;}
  finally{$('go').disabled=false;$('st').textContent='就绪';}
}

function render(j){
  const mp=Object.entries(j.map_probs||{}).sort((a,b)=>b[1]-a[1]);
  const pm=Object.fromEntries(mp);
  // 选图按**概率降序**，每张图自带概率；池内未选的单列一行。
  // ⚠ 绝不能用 j.maps —— 它是字母规范序（计划身份/计分用），
  // 直接 join 会把 Mirage 74% 排在 Dust2 57% 后面，看着像首选小镇。
  const picked=(j.maps_ranked&&j.maps_ranked.length?j.maps_ranked:j.maps).slice();
  picked.sort((a,b)=>(pm[b]||0)-(pm[a]||0));
  const pickedTxt=picked.map(m=>`${esc(m)} ${pm[m]!=null?Math.round(pm[m]*100)+'%':''}`).join('  ·  ');
  const restTxt=mp.filter(([m])=>picked.indexOf(m)<0)
                 .map(([m,v])=>`${esc(m)} ${Math.round(v*100)}%`).join('  ');
  const vp=Object.entries(j.mvp_probs||{}).sort((a,b)=>b[1]-a[1]).slice(0,4);
  const wins=j.warnings.length?`<ul class="w">${j.warnings.map(w=>`<li>${esc(w)}</li>`).join('')}</ul>`:'';
  const st=j.rank_stale_days;
  const staleCls = st==null?'':(st>=60?'pillBad':(st>=30?'pillWarn':'pillOk'));
  const staleTxt = st==null?'':`<span class="pill ${staleCls}">rank快照 ${esc(j.rank_src.snapshot)} · ${st}天前</span>`;
  $('out').innerHTML=`
  <div class="card">
    <div><span class="pill">BO${j.bo} · 满分 ${j.max_points}</span>
      ${j.rank_src&&j.rank_src.ok?`<span class="pill">${rankTxt({rank:j.rank_src.rankA})} vs ${rankTxt({rank:j.rank_src.rankB})}</span>${staleTxt}`:''}</div>
    <div class="win" style="color:var(--acc)">${esc(j.winner)}</div>
    <div style="color:var(--dim);font-size:12px">对 ${esc(j.loser)} · 预测比分 ${esc(j.score)}</div>
    ${bar(j.p_win)}
    <div style="margin-top:5px;font-size:12px;color:var(--dim)">胜方概率 ${j.p_win}%</div>
    <div class="kv">
      <div><b>期望得分</b><span>${j.exp_points} <small style="color:var(--dim)">±${j.sd}</small> / ${j.max_points}</span></div>
      <div><b>预测 MVP</b><span>${esc(j.mvp||'—')}</span></div>
      <div><b>模型倾向</b><span>${esc(j.model_winner||'—')}${j.disagreement?' ⚠':''}</span></div>
    </div>
    <div style="margin-top:14px"><b style="font-size:12px;color:var(--dim)">选图（按概率降序） ${pickedTxt}</b></div>
    ${restTxt?`<div style="font-size:11px;color:var(--dim);margin-top:3px">池内未选 ${esc(restTxt)}</div>`:''}
    ${vp.length?`<div style="margin-top:12px"><b style="font-size:12px;color:var(--dim)">MVP 概率（未标定）</b></div>
    <div style="margin-top:6px">${vp.map(([m,p])=>`<span class="pill">${esc(m)} ${(p*100).toFixed(0)}%</span>`).join('')}</div>`:''}
    ${wins}
  </div>`;
}
bind('a','la');bind('b','lb');
fetch('/api/status').then(r=>r.json()).then(j=>{
  if(j.error){$('st').textContent='加载失败：'+j.error;return;}
  TEAMS=j.teams;$('go').disabled=false;
  LAST_SNAP=j.rank_snapshot;LAST_SYNC=j.rank_syncs;
  $('st').textContent=`就绪 · ${TEAMS.length} 支队伍 · 模型加载 ${j.load_s}s`
    +(j.rank_snapshot?` · rank 快照 ${j.rank_snapshot}`:'');
}).catch(()=>{$('st').textContent='无法连接服务';});
// 兜底模板同 predict_page.html：30s 拉一次状态，换了 rank 快照自动跟上。
setInterval(function(){
  if(!TEAMS.length) return;
  fetch('/api/status').then(r=>r.json()).then(j=>{
    if(!j.ready||!j.teams) return;
    const changed=j.rank_snapshot!==LAST_SNAP||j.rank_syncs!==LAST_SYNC;
    TEAMS=j.teams;$('go').disabled=false;
    LAST_SNAP=j.rank_snapshot;LAST_SYNC=j.rank_syncs;
    $('st').textContent=`就绪 · ${TEAMS.length} 支队伍`
      +(j.rank_snapshot?` · rank 快照 ${j.rank_snapshot}`:'')
      +(j.rank_syncs?` · 自动同步 ${j.rank_syncs} 次`:'')
      +(changed?' · rank 已更新':'');
  }).catch(()=>{});
},30000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    # ⚠ 用 HTTP/1.0（单请求即断），**不用** HTTP/1.1 keep-alive：
    #   1.1 下部分客户端（urllib / 某些浏览器预连接）在 Connection 头缺失时
    #   会读到 `IncompleteRead(0 bytes read)`。1.0 + 显式 Connection: close
    #   在浏览器与脚本两侧都稳定。
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        # ⚠ 必须处理 dict/list：第一版直接 `body.encode()`，而调用点多处传的是
        #   Python 对象 ⇒ `TypeError: a bytes-like object is required, not 'dict'`，
        #   连接被挂断、客户端侧表现为 IncompleteRead / Timeout（现象与原因相隔很远）。
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            self._route()
        except Exception as e:                           # noqa: BLE001
            # ⚠ 关键：异常必须变成**完整**的 JSON 错误响应。
            #   若异常发生在 send_response 之后（响应头已冲出去、body 还没写），
            #   客户端拿到的是「有状态行、无 Content-Length、0 字节 body」的残缺响应，
            #   urllib 报 IncompleteRead、浏览器报 `HTTP/1.1 400 Bad Request`——
            #   现象与「500」相隔十万八千里，极难定位。
            #   这里用 send_error 走统一的错误路径，保证状态行+body 完整。
            sys.stderr.write("[predict_gui] %s: %s\n" % (type(e).__name__, e))
            sys.stderr.flush()
            if not getattr(self, "_headers_buffer", None):
                return
            try:
                self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})
            except Exception:                            # noqa: BLE001
                pass

    def _route(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            # 2026-10-14 视觉重设计：同目录 predict_page.html 存在时**优先服务它**，
            # 文件缺失则回退到下方内嵌 HTML。⇒ 换皮肤 = 换一个 HTML 文件，
            # 不动内嵌模板、更不动 JS 逻辑。
            _page = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "predict_page.html")
            if os.path.exists(_page):
                return self._send(200, io.open(_page, encoding="utf-8").read(),
                                  "text/html; charset=utf-8")
            return self._send(200, HTML, "text/html; charset=utf-8")
        if u.path == "/favicon.ico":
            # 浏览器每次都会自动要它；给个空 200，省掉控制台 404 噪音
            return self._send(200, b"", "image/x-icon")
        if u.path == "/api/status":
            if _STATE["error"]:
                return self._send(200, {"error": _STATE["error"]})
            if not _STATE["ready"]:
                return self._send(200, {"ready": False})
            try:                       # rank 数据变了就自动跟上（毫秒级）
                sync_rank()
            except Exception:          # noqa: BLE001  同步失败不该让状态接口整个挂掉
                pass
            rk = getattr(_STATE["pred"], "_rank_idx", None) or {}
            return self._send(200, {"ready": True,
                                    "teams": _STATE["teams"],
                                    "load_s": round(time.time() - _STATE["t0"], 1),
                                    "rank_teams": len(rk),
                                    "rank_snapshot": getattr(
                                        _STATE["pred"], "_rank_snap", ""),
                                    "rank_syncs": _STATE["rank_syncs"],
                                    "last_rank_sync": _STATE["last_rank_sync"],
                                    "reloads": _STATE["reloads"],
                                    "last_reload": _STATE["last_reload"],
                                    "last_date": getattr(
                                        getattr(_STATE["pred"], "state", None),
                                        "last_date", None)})
        if u.path == "/api/reload":
            if q.get("sync", ["0"])[0] == "1":
                try:
                    r = reload_model()
                except Exception as e:                   # noqa: BLE001
                    r = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
                return self._send(200 if r.get("ok") else 500, r)
            # 默认异步：重建 Predictor 要十几秒，同步会撞浏览器/urllib 默认超时
            if _STATE.get("_reloading"):
                return self._send(200, {"ok": True, "busy": True})
            _STATE["_reloading"] = True

            def _bg():
                try:
                    reload_model()
                finally:
                    _STATE["_reloading"] = False
            threading.Thread(target=_bg, daemon=True).start()
            return self._send(200, {"ok": True, "started": True})
        if u.path == "/api/predict":
            if not _STATE["ready"]:
                return self._send(503, {"error": "模型仍在加载"})
            sync_rank()      # 预测前同步：换快照后第一次预测就该用新名次
            try:
                r = predict(_STATE["pred"], q.get("a", [""])[0],
                            q.get("b", [""])[0], q.get("bo", ["3"])[0])
                return self._send(200, r)
            except Exception as e:                       # noqa: BLE001
                return self._send(400, {"error": str(e)})
        self._send(404, {"error": "not found"})


class Server(ThreadingHTTPServer):
    """独占端口的 HTTP 服务（Windows 下用 SO_EXCLUSIVEADDRUSE）。"""

    allow_reuse_address = False
    # SO_REUSEADDR 在 Windows 上语义与 Unix **相反**：允许多个 socket 绑定同一端口。
    # 于是第二个实例不报 EADDRINUSE，而是抢占绑定后 serve_forever() 永久阻塞，
    # 两个新旧服务并存、请求随机落到其中一个 ⇒ 时好时坏的 400 / IncompleteRead。
    # 下面在 socket() 时直接带上 SO_EXCLUSIVEADDRUSE，绑定后独占，别人一律失败。
    _EXCL = {"socketOptions": [(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)]} \
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE") else {}

    def __init__(self, addr, handler):
        super().__init__(addr, handler, bind_and_activate=False)
        self.socket = socket.socket(self.address_family, self.socket_type)
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET,
                                   socket.SO_EXCLUSIVEADDRUSE, 1)
        self.server_address = addr
        self.server_bind()          # bind + activate（listen）
        self.server_activate()


def _port_busy(host, port):
    """先探端口再绑定。

    ⚠ Windows 的 SO_EXCLUSIVEADDRUSE 只在**本进程**的 bind 时生效，
    不能阻止别的进程抢占，所以仍需显式探测，失败要给出可执行的提示。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(2)
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true", help="自动打开浏览器")
    a = ap.parse_args()

    # ⚠ 陈旧进程守卫——本项目已经因此出了一次线上事故：
    #   改完 predict_gui.py 后忘了重启，浏览器连到**内存里的旧代码**，
    #   于是报 `HTTP/1.1 400 Bad Request`；而旧进程还占着端口，
    #   人就一直对着坏服务调试。现在占端口就直接拒并给出处理命令。
    if _port_busy(a.host, a.port):
        print("✗ %s:%d 已有服务在监听，很可能是一个**旧版** predict_gui。"
              % (a.host, a.port))
        print("  本项目模型初始化要 9.5 秒，旧进程会一直占着端口；")
        print("  浏览器拿到的是内存里的旧代码，报错与当前文件无关。")
        print("  处理（一行）：")
        print("    Get-NetTCPConnection -LocalPort %d | "
              "Select-Object -Expand OwningProcess -Unique | "
              "Where-Object { $_ -gt 0 } | "
              "ForEach-Object { Stop-Process -Id $_ -Force }" % a.port)
        print("  确认杀掉后重新运行。")
        return 1

    try:
        srv = Server((a.host, a.port), H)
    except OSError as e:
        print("✗ 端口 %d 绑定失败：%s" % (a.port, e))
        return 1

    def boot():
        reload_model()

    def run():
        try:
            boot()
        except Exception as e:                           # noqa: BLE001
            _STATE["error"] = "%s: %s" % (type(e).__name__, e)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    url = "http://%s:%d/" % (a.host, a.port)
    print("预测器已启动：%s" % url)
    print("模型在后台加载（约 10 秒），页面会自行变为可用。Ctrl-C 退出。")
    if a.open:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == "__main__":
    sys.exit(main() or 0)
