"""把 111 场真实盲预测的赛果导出成 match_record，并做入库前的干跑校验。

**为什么现在能做**：主理人 2026-10-12 确认「我列图顺序 = pick 顺序」，
而 veto 顺序正是 Plackett–Luce 选图模型的**唯一**入口。
⇒ 84 场（EPL 41 + 裂变天地 29 + StarLadder 14，日期齐全）可立即入库选图模型。

**本脚本不碰爬虫**，只做导出 + 校验（干跑），入库由 verify 另行确认。

逐图比分（map_results）仍缺：csapi SSL 阻断、HLTV 403、Liquipedia Cargo 不可用
（cargotables 返回 0 张表）⇒ 只能逐页 action=parse（ToU 30s/次）。
但它只影响 elo_m（地图净胜 Elo），而胜方项现已改用 rank ⇒ **优先级低**。
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.0）
sys.path.insert(0, HERE)
import record_schema as RS

ZH2EN = {"遗迹": "Dust2", "迷城": "Mirage", "沙二": "Anubis", "阿努比斯": "Anubis",
         "死城": "Cache", "核子": "Nuke", "小镇": "Inferno", "远古": "Ancient"}

EPL = (r"E:\world\another\data\Mimo\MiMo-V2.6-Flash\2026-10-12"
       r"\live_tournament_results.jsonl")
# (日期, 结果文件)
DATED = [
    ("2026-09-08", "fissure_2026-09-08_results.json"),
    ("2026-09-09", "fissure_2026-09-09_results.json"),
    ("2026-09-10", "fissure_2026-09-10_results.json"),
    ("2026-09-11", "fissure_2026-09-11_results.json"),
    ("2026-09-12", "fissure_2026-09-12_results.json"),
    ("2026-09-13", "fissure_2026-09-13_results.json"),
    ("2026-09-17", "fissure_2026-09-17_results.json"),
    ("2026-09-18", "fissure_2026-09-18_results.json"),
    ("2026-09-19", "fissure_2026-09-19_results.json"),
    ("2026-09-20", "fissure_2026-09-20_results.json"),
    ("2026-09-20", "fissure_2026-09-20-bo5_results.json"),
]

# 冷门小赛事（2026-09-21 ~ 09-27）逐日日期，主理人 2026-10-12 给出：
#   cold1 的 1-4 → 21 日、5-8 → 22 日；cold2 的 1 → 22、2-6 → 23、7-11 → 24、
#   12-15 → 25、16-18 → 26、19 → 27。
# **顺序按 TRUTHS / 结果文件里的原始次序**（与主理人列出的先后一致）。
COLD_DATES = {
    "fissure_cold1_results.json": ["2026-09-21"] * 4 + ["2026-09-22"] * 4,
    "fissure_cold2_results.json": ["2026-09-22"] + ["2026-09-23"] * 5
    + ["2026-09-24"] * 5 + ["2026-09-25"] * 4 + ["2026-09-26"] * 3
    + ["2026-09-27"],
}


def epl_records():
    out = []
    for l in io.open(EPL, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        out.append({
            "id": "live-" + r["id"],
            "date": r["date"],
            "best_of": r["best_of"],
            "team1": {"name": r["team1"]["name"], "score": r["team1"]["score"]},
            "team2": {"name": r["team2"]["name"], "score": r["team2"]["score"]},
            "winner": r["winner"],
            # 主理人确认列图顺序 = pick 顺序 ⇒ 同时作为 veto_order
            "maps_full_B": list(r["maps_full_B"]),
            "veto_order": list(r["maps_full_B"]),
            "mvp": r.get("mvp", ""),
            "src": "EPL S24 主理人赛果",
        })
    return out


def dated_records():
    out = []
    for d, f in DATED:
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            continue
        for x in json.load(io.open(p, encoding="utf-8")):
            bo = 5 if len(x["maps"]) == 5 else 3
            out.append({
                "id": "%s|%s|%s" % (d, x["A"], x["B"]),
                "date": d,
                "best_of": bo,
                "team1": {"name": x["A"], "score": x["score"][0]},
                "team2": {"name": x["B"], "score": x["score"][1]},
                "winner": x["winner"],
                "maps_full_B": list(x["maps"]),
                "veto_order": list(x["maps"]),
                "mvp": x.get("mvp", ""),
                "src": "主理人赛果",
            })
    for f, dates in COLD_DATES.items():
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            print("  ⚠ 缺 %s，跳过" % f)
            continue
        rows = json.load(io.open(p, encoding="utf-8"))
        if len(rows) != len(dates):
            print("  ⚠ %s 场次数(%d)与日期数(%d)不符，**跳过**（不猜）"
                  % (f, len(rows), len(dates)))
            continue
        for x, d in zip(rows, dates):
            bo = 5 if len(x["maps"]) == 5 else 3
            out.append({
                "id": "%s|%s|%s" % (d, x["A"], x["B"]),
                "date": d, "best_of": bo,
                "team1": {"name": x["A"], "score": x["score"][0]},
                "team2": {"name": x["B"], "score": x["score"][1]},
                "winner": x["winner"],
                "maps_full_B": list(x["maps"]),
                "veto_order": list(x["maps"]),
                "mvp": x.get("mvp", ""),
                "src": "冷门小赛事 主理人赛果",
            })
    return out


def main():
    recs = epl_records() + dated_records()
    print("待入库 %d 场（EPL %d + 裂变/StarLadder %d）"
          % (len(recs), len(epl_records()), len(dated_records())))
    ok = bad = 0
    fuel_count = {"map_results": 0, "rosters": 0, "by_map": 0, "veto_order": 0}
    problems = []
    for r in recs:
        good, probs, fuel = RS.validate(r, strict=False)
        if good:
            ok += 1
        else:
            bad += 1
            problems.append((r["id"], probs))
        for k, v in fuel.items():
            fuel_count[k] += bool(v)
    print("  校验通过 %d 场，不通过 %d 场" % (ok, bad))
    for pid, pr in problems[:10]:
        print("    ✗ %s → %s" % (pid, pr))
    print("\n  四组燃料到位情况：")
    for k, v in fuel_count.items():
        print("    %-13s %3d/%d 场" % (k, v, len(recs)))
    dates = sorted({r["date"] for r in recs})
    print("\n  日期覆盖：%s .. %s（%d 个不同日期）" % (dates[0], dates[-1], len(dates)))
    teams = sorted({t["name"] for r in recs for t in (r["team1"], r["team2"])})
    print("  涉及队伍 %d 支" % len(teams))
    out = os.path.join(DATA, "groundtruth", "new_records_2026-09.jsonl")
    with io.open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n已写 %s" % os.path.basename(out))
    print("\n⚠ 干跑结束，**未入库**。缺 map_results/rosters/by_map 三组燃料，")
    print("  故本次入库只会更新 team Elo 与选图（veto）模型，不会动评分/阵容特征。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
