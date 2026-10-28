#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用最新 VRS 快照刷新 `fixtures/team_strength.json` 的 rank / points。

为什么需要这个脚本
-----------------
根页面 `/` 直接读 `data/vrs_rank_<日期>.json`（`Predictor._build_rank()`），
换快照后靠 mtime 指纹热重载即可。但 `/major` 用的是公开强度库
`pickem/fixtures/team_strength.json`：它的 `rank` / `points` 是**生成时**抄进去的，
不会自己跟着快照走。没有这一步，就会出现「根页面名次已更新、Major 页面还是旧积分」
——两个页面各说各话。

口径
----
* 输入：`data/vrs_rank_<日期>.json`（默认取文件名日期最大的那份）。
* 匹配：队名精确（不区分大小写）→ 别名 → 规范化（小写 + 去非字母数字）
  → 规范化别名。**不做模糊匹配**；同一支库内队伍只能被一行占用。
* 快照里有、库里没有 → 追加新记录，`rating=null`（不伪造 Elo），`aliases=[]`。
* 库里有、快照里没有 → `rank=null` / `points=null`（掉榜），Elo 原样保留。
* `asof` 改成快照日期，`counts` 重算，`_provenance` 追加一段可追溯说明。
* 写入 `vrs_source`（date / teams / content_sha / fetched_at / revision）：
  `vrs_sync` 用它判断「同日期、同队数、但积分变了」的情况。
* 默认只预览；`--write` 才原子替换强度库。content_sha 相同则直接跳过，
  不制造无意义的 mtime / git 变化；`--force` 可强制重写。

用法
----
    python pickem/update_vrs.py                 # 预览
    python pickem/update_vrs.py --write         # 写回 fixtures/team_strength.json
    python pickem/update_vrs.py --snapshot data/vrs_rank_2026-10-28.json --write
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXTURE = os.path.join(HERE, "fixtures", "team_strength.json")
DATA = os.path.join(ROOT, "data")
MARK = "VRS rank/points 刷新自 "
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def norm(text):
    """与 predict.py:_norm 同口径：小写 + 只保留字母数字。"""
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def latest_snapshot(data_dir=DATA):
    paths = sorted(glob.glob(os.path.join(data_dir, "vrs_rank_*.json")))
    if not paths:
        raise SystemExit("没有找到任何 data/vrs_rank_*.json 快照")
    return paths[-1]


def load_snapshot(path):
    """读并校验快照；返回 (date, rows, meta)。非法即退出，不静默修补。"""
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit("快照读不到或不是合法 JSON：%s（%s）" % (path, exc))
    date = str(doc.get("date") or "")
    if not DATE_RE.match(date):
        raise SystemExit("快照 %s 的 date 非法：%r" % (path, doc.get("date")))
    teams = doc.get("teams")
    if not isinstance(teams, list) or not teams:
        raise SystemExit("快照 %s 没有 teams" % path)
    rows, seen_rank, seen_name = [], set(), set()
    for i, rec in enumerate(teams, 1):
        if not isinstance(rec, dict):
            raise SystemExit("快照 %s 第 %d 条不是对象" % (path, i))
        name = str(rec.get("name") or "").strip()
        rank, points = rec.get("rank"), rec.get("points")
        if not name or not isinstance(rank, int) or not isinstance(points, int):
            raise SystemExit("快照 %s 第 %d 条字段缺失：%r" % (path, i, rec))
        if rank in seen_rank:
            raise SystemExit("快照 %s 名次重复：%d" % (path, rank))
        if name.lower() in seen_name:
            raise SystemExit("快照 %s 队名重复：%r" % (path, name))
        seen_rank.add(rank)
        seen_name.add(name.lower())
        rows.append({"name": name, "rank": rank, "points": points})
    rows.sort(key=lambda r: r["rank"])
    meta = {"fetched_at": doc.get("fetched_at"),
            "revision": doc.get("revision")}
    return date, rows, meta


def content_sha(rows):
    """只对 rank/name/points 排序后做 sha256（与 vrs_sync.content_sha 同口径）。

    不含 fetched_at / revision：同一份榜单重复抓取时内容哈希不变，
    强度库就不会因为「只是又抓了一次」而被重写。
    """
    blob = json.dumps(sorted((int(r["rank"]), str(r["name"]), float(r["points"]))
                             for r in rows),
                      ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_lookup(teams):
    """建四张查找表；后出现的别名不覆盖先出现的。"""
    exact, alias, nexact, nalias = {}, {}, {}, {}
    for t in teams:
        name = t["name"]
        exact.setdefault(name.lower(), name)
        nexact.setdefault(norm(name), name)
        for a in t.get("aliases") or []:
            alias.setdefault(str(a).lower(), name)
            nalias.setdefault(norm(a), name)
    return exact, alias, nexact, nalias


def match_team(name, lookups, used):
    """返回 (规范队名, 匹配方式)；没有匹配返回 (None, None)。

    若某行命中了一个**已被别的行占用**的规范队名，返回 duplicate 原因，
    由调用方报错——静默丢弃会让积分凭空少一队，且没人看得见。
    """
    exact, alias, nexact, nalias = lookups
    key = name.lower()
    candidates = (
        (exact.get(key), "exact"),
        (alias.get(key), "alias"),
        (nexact.get(norm(name)), "normalized"),
        (nalias.get(norm(name)), "alias-normalized"),
    )
    for cand, how in candidates:
        if cand is None:
            continue
        if cand in used:
            return None, "duplicate:%s" % cand
        return cand, how
    return None, None


def strip_previous(text):
    """去掉本脚本上一次写入的段落（可能在开头，也可能在 update_elo 之后）。"""
    while True:
        i = text.find(MARK)
        if i < 0:
            return text
        j = text.find("原 provenance: ", i)
        if j < 0:
            return text[:i].rstrip()
        text = text[:i] + text[j + len("原 provenance: "):]


def build_doc(doc, date, rows, matched, renamed, added, dropped, sha, meta):
    old_prov = strip_previous(doc.get("_provenance") or "")
    rev = meta.get("revision")
    when = meta.get("fetched_at")
    extra = ""
    if rev or when:
        extra = "（第 %s 版%s）" % (
            rev if rev is not None else "?",
            "，抓取于 %s" % when if when else "")
    segment = (
        "%s%s 的 Valve 全球排名快照%s（%d 队；新增 %d 支 rating=null；"
        "掉榜 %d 支 rank/points=null；按别名/规范化匹配 %d 支；content_sha %s）。"
        "默认强度源仍是 VRS 积分；Elo 列来自公开语料重放，不随 VRS 刷新。"
        % (MARK, date, extra, len(rows), added, dropped, renamed, sha[:12])
    )
    new_prov = segment + "原 provenance: " + old_prov
    teams = doc["teams"]
    counts = dict(doc.get("counts") or {})
    counts["teams"] = len(teams)
    counts["elo"] = sum(1 for t in teams if t.get("rating") is not None)
    counts["vrs"] = sum(1 for t in teams if t.get("points") is not None)
    counts["both"] = sum(1 for t in teams
                         if t.get("rating") is not None and t.get("points") is not None)
    elo_asof = doc.get("elo_asof")
    role = ("公开强度库：队名查找与两种强度来源。rating = 公开语料重算的 Elo"
            "（截至 %s）；rank/points = VRS 快照（%s）。公开侧只查表，不读 data/。"
            % (elo_asof or "—", date))
    return {
        "asof": date,
        "elo_asof": elo_asof,
        "elo_matches": doc.get("elo_matches"),
        "source": doc.get("source", "rank_points"),
        "_role": role,
        "_provenance": new_prov,
        "vrs_source": {
            "date": date,
            "teams": len(rows),
            "content_sha": sha,
            "fetched_at": meta.get("fetched_at"),
            "revision": meta.get("revision"),
        },
        "counts": counts,
        "teams": teams,
    }


def apply_snapshot(doc, date, rows):
    """就地改 `doc["teams"]`，返回统计摘要。"""
    teams = doc["teams"]
    lookups = build_lookup(teams)
    by_name = {t["name"]: t for t in teams}
    old_vrs = {t["name"] for t in teams if t.get("points") is not None}

    # 先清空旧 VRS 列：掉榜队伍不能留着上一次的积分
    for t in teams:
        t["rank"] = None
        t["points"] = None

    used, new_records = set(), []
    matched = renamed = added = 0
    for row in rows:
        canon, how = match_team(row["name"], lookups, used)
        if how and how.startswith("duplicate:"):
            raise SystemExit("快照队名 %r 与已匹配的队伍冲突：%s"
                             % (row["name"], how.split(":", 1)[1]))
        if canon is None:
            rec = {"name": row["name"], "rating": None,
                   "rank": row["rank"], "points": float(row["points"]),
                   "aliases": []}
            new_records.append(rec)
            by_name[row["name"]] = rec
            used.add(row["name"])
            added += 1
            continue
        rec = by_name[canon]
        rec["rank"] = row["rank"]
        rec["points"] = float(row["points"])
        used.add(canon)
        matched += 1
        if canon.lower() != row["name"].lower():
            renamed += 1

    dropped = sum(1 for t in teams
                  if t["name"] in old_vrs and t.get("points") is None)
    off_list = sum(1 for t in teams if t.get("points") is None)
    teams.extend(new_records)
    return {"matched": matched, "renamed": renamed,
            "added": added, "dropped": dropped, "off_list": off_list,
            "new_records": new_records}


def main(argv=None):
    ap = argparse.ArgumentParser(description="用 VRS 快照刷新 pickem 强度库的 rank/points")
    ap.add_argument("--snapshot", default=None,
                    help="快照路径（默认 data/vrs_rank_*.json 中日期最大的一份）")
    ap.add_argument("--data-dir", default=DATA, help="快照目录（默认 ./data）")
    ap.add_argument("--fixture", default=FIXTURE, help=argparse.SUPPRESS)
    ap.add_argument("--write", action="store_true", help="写回 fixture（默认只预览）")
    ap.add_argument("--force", action="store_true",
                    help="即使 content_sha 相同也重写（手工修复用）")
    args = ap.parse_args(argv)

    snap_path = args.snapshot or latest_snapshot(args.data_dir)
    date, rows, meta = load_snapshot(snap_path)
    sha = content_sha(rows)
    try:
        with open(args.fixture, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit("强度库读不到或不是合法 JSON：%s（%s）" % (args.fixture, exc))
    if not isinstance(doc.get("teams"), list) or not doc["teams"]:
        raise SystemExit("强度库没有 teams：%s" % args.fixture)

    old_src = doc.get("vrs_source") or {}
    if (not args.force and old_src.get("date") == date
            and old_src.get("content_sha") == sha):
        print("强度库已与快照一致（content_sha %s），无需更新。" % sha[:12])
        return 0

    stats = apply_snapshot(doc, date, rows)
    out = build_doc(doc, date, rows, stats["matched"], stats["renamed"],
                    stats["added"], stats["dropped"], sha, meta)

    print("快照：%s（%s，%d 队，第 %s 版%s）"
          % (snap_path, date, len(rows), meta.get("revision") or "?",
             "，抓取于 %s" % meta["fetched_at"] if meta.get("fetched_at") else ""))
    print("content_sha %s" % sha[:12])
    print("匹配 %d 支（其中别名/规范化 %d 支）；新增 %d 支；掉榜 %d 支（另有 %d 支本来就不在榜上）"
          % (stats["matched"], stats["renamed"], stats["added"],
             stats["dropped"], stats["off_list"] - stats["dropped"]))
    print("覆盖：%d 队 / 有 Elo %d / 有 VRS %d / 两者都有 %d"
          % (out["counts"]["teams"], out["counts"]["elo"],
             out["counts"]["vrs"], out["counts"]["both"]))
    for rec in stats["new_records"][:20]:
        print("  + %-24s #%d %d" % (rec["name"], rec["rank"], rec["points"]))
    if len(stats["new_records"]) > 20:
        print("  + …其余 %d 支见强度库" % (len(stats["new_records"]) - 20))

    if not args.write:
        print("\n（预览模式；加 --write 才会写回 %s）" % args.fixture)
        return 0

    tmp = args.fixture + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    os.replace(tmp, args.fixture)
    print("\n已写回：%s" % args.fixture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
