#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 Valve 排名快照转成 `data/vrs_rank_<日期>.json`。

为什么要有这个脚本（2026-10-19）：
  排名更新是**每周一次的高频操作**，而 `Predictor._build_rank()` 只在进程构造时
  读一次盘。原来从抓到的排名文本到可用快照之间没有任何可复现的转换步骤——
  每次都靠手工拼 JSON，拼错一个字段没人会发现，只表现为「页面名次悄悄不对」。
  转换必须是**一条命令 + 明确校验**，否则它就是下一次事故的入口。

输入（默认 `valve_rank.txt`，也可 `--src` 指定）：
    2026-10-18
    1. Vitality — 1987 — EU
    2. MOUZ — 1940 — EU
    ...
    第 1 行是日期，其余每行 `序号. 队名 — 积分 — 大区`。

输出：`data/vrs_rank_<日期>.json`，字段与既有快照完全一致
    （date / source / type / teams[{rank,name,points}]），额外带 region 与
    下面的 notes / duplicates。

⚠ **重名队伍按「保留第一次出现」处理**（2026-10-19 与使用者约定）。
  抓下来的列表里同一队名会出现两次、名次积分都不同（如 Betclic 49/1245 与
  206/700）。`predict.py:_build_rank()` 是 `idx[name] = rank`，**后出现覆盖
  先出现** ⇒ 直接照抄会让这支队静默取到更差的那次名次。保留第一次（名次更好）
  并把被丢弃的行写进 `duplicates` 留档，可审计可回溯。

用法：
    python -X utf8 ingest_vrs.py                       # 读 ./valve_rank.txt
    python -X utf8 ingest_vrs.py --src rank.txt
    python -X utf8 ingest_vrs.py --dry-run             # 只校验不落盘
    python -X utf8 ingest_vrs.py --keep-first|--keep-last   # 覆盖去重口径
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# 破折号可能是 U+2014 EM DASH / U+2013 EN DASH / 连字符，统一收进来
ROW_RE = re.compile(r"^(\d+)\.\s+(.+?)\s*[—–-]\s*(\d+)\s*[—–-]\s*([A-Za-z]+)\s*$")


def norm(name):
    """与 `predict.py:Predictor._norm` 完全一致的归一化（小写 + 去非字母数字）。"""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def parse(path):
    """解析排名文本 → (date, rows, problems)。rows 为 [{rank,name,points,region}]。"""
    raw = io.open(path, encoding="utf-8").read().splitlines()
    if not raw or not re.match(r"^\d{4}-\d{2}-\d{2}\s*$", raw[0].strip()):
        raise SystemExit("✗ 第 1 行必须是 YYYY-MM-DD 日期，实际是：%r" % (raw[:1],))
    date = raw[0].strip()
    rows, problems = [], []
    for ln, line in enumerate(raw[1:], start=2):
        if not line.strip():
            continue
        m = ROW_RE.match(line.strip())
        if not m:
            problems.append((ln, line))
            continue
        rows.append({"rank": int(m.group(1)), "name": m.group(2).strip(),
                     "points": int(m.group(3)), "region": m.group(4).upper()})
    return date, rows, problems


def check(rows):
    """结构性校验。返回警告列表；致命问题直接抛。"""
    if not rows:
        raise SystemExit("✗ 一行队伍都没解析出来")
    ranks = [r["rank"] for r in rows]
    if ranks != list(range(1, len(ranks) + 1)):
        bad = [(i, r) for i, r in enumerate(ranks, 1) if r != i][:5]
        raise SystemExit("✗ 名次不连续（应为 1..%d），前几处异常：%s" % (len(ranks), bad))
    for a, b in zip(rows, rows[1:]):
        if b["points"] > a["points"]:
            raise SystemExit("✗ 积分未随名次单调下降：#%d %s(%d) → #%d %s(%d)"
                             % (a["rank"], a["name"], a["points"],
                                b["rank"], b["name"], b["points"]))
    return []


def build(date, rows, keep_first=True, origin="ingest_vrs.py 从排名文本转换"):
    teams, first, dupes = [], {}, []
    for r in rows:
        k = norm(r["name"])
        if k in first:
            kept = first[k]
            dupes.append({"name": r["name"],
                          "kept_rank": kept["rank"], "kept_points": kept["points"],
                          "dropped_rank": r["rank"], "dropped_points": r["points"]})
            if keep_first:
                continue
        first[k] = r
        teams.append(r)
    doc = {"date": date, "source": "Valve Regional Standings (VRS)", "type": "world",
           "notes": "由 %s（%s）。%s"
                    % (origin, date, ("源文件中 %d 支队伍重名，按约定保留第一次出现"
                                      "（名次/积分更高），被丢弃行见 duplicates。" % len(dupes))
                       if dupes else "无重名队伍。"),
           "teams": teams, "duplicates": dupes}
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description="Valve 排名文本 → data/vrs_rank_<日期>.json")
    ap.add_argument("--src", default=os.path.join(HERE, "valve_rank.txt"),
                    help="排名文本路径（默认 ./valve_rank.txt）")
    ap.add_argument("--out-dir", default=DATA, help="输出目录（默认 ./data）")
    ap.add_argument("--dry-run", action="store_true", help="只解析与校验，不写盘")
    ap.add_argument("--keep-last", action="store_true",
                    help="重名队伍保留最后一次出现（默认保留第一次）")
    a = ap.parse_args(argv)

    if not os.path.exists(a.src):
        raise SystemExit("✗ 找不到输入文件：%s" % a.src)
    date, rows, problems = parse(a.src)
    check(rows)
    if problems:
        for ln, line in problems[:10]:
            sys.stderr.write("⚠ 第 %d 行无法解析：%s\n" % (ln, line))
        raise SystemExit("✗ %d 行无法解析，已中止（不产出半份快照）" % len(problems))

    doc = build(date, rows, keep_first=not a.keep_last)
    out = os.path.join(a.out_dir, "vrs_rank_%s.json" % date)

    print("日期 %s · 解析 %d 行 · 去重后 %d 支队 · 重名丢弃 %d 条"
          % (date, len(rows), len(doc["teams"]), len(doc["duplicates"])))
    for d in doc["duplicates"][:10]:
        print("  · %-24s 保留 #%d(%d)　丢弃 #%d(%d)"
              % (d["name"], d["kept_rank"], d["kept_points"],
                 d["dropped_rank"], d["dropped_points"]))
    if len(doc["duplicates"]) > 10:
        print("  · …其余 %d 条见 JSON 的 duplicates 字段" % (len(doc["duplicates"]) - 10))

    if a.dry_run:
        print("--dry-run：未写盘。")
        return 0
    os.makedirs(a.out_dir, exist_ok=True)
    # 与既有快照逐字节同格式：indent=1 / ensure_ascii=False / 无尾换行
    io.open(out, "w", encoding="utf-8", newline="").write(
        json.dumps(doc, indent=1, ensure_ascii=False))
    print("✓ 已写入 %s（%d 字节）" % (out, os.path.getsize(out)))
    print("  提示：常驻 predict_gui 会在下一个请求时自动重建 rank 索引，"
          "无需重启；页面 30s 内自行刷新。")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)