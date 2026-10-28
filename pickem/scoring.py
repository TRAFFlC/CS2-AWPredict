#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pick'Em **真实命中**计分器 —— issue-015 §4 验收 3/4 的那把尺子。

`predict_stage.py` 回答「十格该怎么填、P(≥5) 是多少」；本模块回答**另一半**：
把**真实战绩**与任意一套十格对拍，给出**真实命中格数**与**哪几格中、哪几格错**。
没有它，「模型到底有没有用」只能停在「不劣于热门基线」这种弱断言上。

**为什么单独一个模块**：`predict_stage.py` 的报表是已公布数字（P(≥5)=0.2106）的
复现入口，其输出文本被逐字节盯着；计分是「事后」环节，混进去会动摇复现契约。

真值表格式（**真值留在私有区**，公共区只放本计分器与用法）::

    {
      "stage": "StarLadder Budapest Major 2025 Stage 1",
      "records": { "FaZe": [3, 2], "FlyQuest": [3, 0], "Lynn Vision": [0, 3], ... }
    }

`records` 必须是**完整 16 队瑞士阶段**：胜场=3 恰 8 支（其中 3-0 恰 2 支）、
负场=3 恰 8 支（其中 0-3 恰 2 支）、晋级格恰 6 支。**不符即 rc=1 拒绝**——
错的真值表会**静默**算出错的命中数，这正是本项目最忌讳的静默失败；
十格一侧同样硬校验（2/6/2、互不重复、队名必须在真值表内 ⇒ 顺手挡住「阶段错配」）。

用法::

    python scoring.py --truth truth.json --card card.json          # 真值 × 十格
    python scoring.py --truth truth.json --card card.json --k 6    # 换过关线
    python scoring.py --truth truth.json --card card.json --json out.json

`--card` 既接受 `predict_stage.py --json` 的完整报表（同时评优化器与三路基线），
也接受最简的 `{"p30": [...], "padv": [...], "p03": [...]}` 或 `{"picks": {...}}`。

退出码：0 计分成功；1 真值表 / 十格非法或读文件失败。
"""
from __future__ import annotations

import argparse
import json
import sys

CATS = ("p30", "padv", "p03")
LABEL = {"p30": "3:0", "padv": "晋级", "p03": "0:3"}
SIZES = {"p30": 2, "padv": 6, "p03": 2}
TEAMS = 16
WINNERS = 8          # 16 队瑞士阶段：恰 8 支拿到 3 胜
N30 = 2              # 其中恰 2 支 3-0
N03 = 2              # 恰 2 支 0-3


# ------------------------------------------------------------------ 真值表
def classify(records):
    """真值战绩 → 三个集合（3:0 / 晋级 / 0:3）。"""
    out = {"p30": set(), "padv": set(), "p03": set()}
    for t, (w, l) in records.items():
        if w == 3 and l == 0:
            out["p30"].add(t)
        elif w == 3 and l in (1, 2):
            out["padv"].add(t)
        elif w == 0 and l == 3:
            out["p03"].add(t)
    return out


def actual_label(wl):
    """一队真实战绩 → 人类可读类别（用于「未中格」的归因）。"""
    w, l = wl
    if w == 3 and l == 0:
        return "3:0"
    if w == 3:
        return "晋级"
    if w == 0 and l == 3:
        return "0:3"
    return "淘汰"


def check_records(records, expect=TEAMS):
    """结构校验；返回 problems 列表（空 = 合法）。"""
    probs = []
    if not isinstance(records, dict) or not records:
        return ["真值表缺少可用的 records（应为 {队名: [胜, 负]}）"]
    for t, v in records.items():
        if not isinstance(t, str) or not t:
            probs.append("队名非法：%r" % (t,))
            break
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            probs.append("队伍 %s 的战绩应为 [胜, 负]，实得 %r" % (t, v))
            break
        if any(isinstance(x, bool) or not isinstance(x, int) for x in v):
            probs.append("队伍 %s 的战绩必须是整数，实得 %r" % (t, list(v)))
            break
        if not (0 <= v[0] <= 3 and 0 <= v[1] <= 3):
            probs.append("队伍 %s 的战绩越界（0..3），实得 %r" % (t, list(v)))
            break
        if v[0] == 3 and v[1] == 3:
            probs.append("队伍 %s 不可能同时 3 胜 3 负" % t)
            break
    if probs:
        return probs
    n = len(records)
    w3 = [t for t, v in records.items() if v[0] == 3]
    l3 = [t for t, v in records.items() if v[1] == 3]
    s30 = [t for t, v in records.items() if v[0] == 3 and v[1] == 0]
    s03 = [t for t, v in records.items() if v[0] == 0 and v[1] == 3]
    if n != expect:
        probs.append("真值表必须恰好 %d 队（完整瑞士阶段），实得 %d" % (expect, n))
    if len(w3) != WINNERS:
        probs.append("拿满 3 胜的应恰 %d 支，实得 %d" % (WINNERS, len(w3)))
    if len(l3) != WINNERS:
        probs.append("吃满 3 败的应恰 %d 支，实得 %d" % (WINNERS, len(l3)))
    if len(s30) != N30:
        probs.append("3:0 的应恰 %d 支，实得 %d" % (N30, len(s30)))
    if len(s03) != N03:
        probs.append("0:3 的应恰 %d 支，实得 %d" % (N03, len(s03)))
    return probs


def truth_from_doc(doc, expect=TEAMS):
    """从**已在内存里**的真值表 dict 构造真值（CLI / Web / 测试共用）。非法即 ValueError。"""
    if not isinstance(doc, dict):
        raise ValueError("真值表应为 JSON 对象（{stage, records}）")
    records = doc.get("records")
    probs = check_records(records, expect)
    if probs:
        raise ValueError("; ".join(probs))
    records = {t: [int(v[0]), int(v[1])] for t, v in records.items()}
    return {"stage": doc.get("stage") or "(未命名阶段)",
            "source": doc.get("source"), "records": records,
            "sets": classify(records)}


def load_truth(path, expect=TEAMS):
    """读真值表并硬校验。非法即抛 ValueError。"""
    with open(path, encoding="utf-8") as fh:
        return truth_from_doc(json.load(fh), expect)


# ------------------------------------------------------------------ 十格
def norm_picks(obj, records):
    """从多种输入形状里取出十格并硬校验。非法即抛 ValueError。

    接受：{p30,padv,p03} / {picks: {...}} / `predict_stage.py` 完整报表 / {baselines:{name:{picks}}}
    """
    if not isinstance(obj, dict):
        raise ValueError("十格应为 JSON 对象")
    if all(c in obj for c in CATS):
        picks = {c: obj[c] for c in CATS}
    elif isinstance(obj.get("picks"), dict):
        picks = obj["picks"]
    else:
        raise ValueError("十格缺少 %s" % "/".join(CATS))
    probs = []
    for c in CATS:
        v = picks.get(c)
        if not isinstance(v, (list, tuple)) or len(v) != SIZES[c]:
            probs.append("%s 组应恰 %d 项，实得 %r" % (LABEL[c], SIZES[c], v))
            continue
        for t in v:
            if not isinstance(t, str):
                probs.append("%s 组含非字符串项：%r" % (LABEL[c], t))
    if probs:
        raise ValueError("; ".join(probs))
    allp = [t for c in CATS for t in picks[c]]
    if len(set(allp)) != len(allp):
        probs.append("十格三组之间不得重复（同一队只能占一格）")
    unknown = sorted(set(allp) - set(records))
    if unknown:
        probs.append("十格含真值表里没有的队伍（阶段错配？）：%s" % ", ".join(unknown))
    if probs:
        raise ValueError("; ".join(probs))
    return {c: list(picks[c]) for c in CATS}


def _looks_like_picks(v):
    return isinstance(v, dict) and all(c in v for c in CATS)


def card_strategies(card):
    """从 card JSON 里取出「策略名 → 十格」。

    接受四种形状：`predict_stage.py` 完整报表（`baselines` 下多路，其值可带 `picks`
    也可直接是十格）/ 最简 `{p30,padv,p03}` / `{picks: {...}}`。
    **认不出就报错**——静默返回空会让 rc=0 而什么也没评（本项目最忌讳的静默失败）。
    """
    if isinstance(card, dict) and isinstance(card.get("baselines"), dict):
        out = {}
        for name, v in card["baselines"].items():
            pk = v.get("picks") if isinstance(v, dict) else None
            if not _looks_like_picks(pk) and _looks_like_picks(v):
                pk = v                      # 未包一层 picks 的写法也接受
            if _looks_like_picks(pk):
                out[name] = pk
        if _looks_like_picks(card.get("picks")):
            out.setdefault("optimizer", card["picks"])
        if out:
            return out
    if _looks_like_picks(card):
        return {"card": card}
    if _looks_like_picks((card or {}).get("picks")):
        return {"card": card["picks"]}
    raise ValueError("card JSON 里找不到十格（应为 p30/padv/p03，或 baselines 下的多路十格）")


# ------------------------------------------------------------------ 计分
def score_card(truth, picks, k=5):
    """真值 × 十格 → 命中明细。picks 需已由 `norm_picks` 规范化。"""
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 10:
        raise ValueError("过关线 k 应为 1..10 的整数，实得 %r" % (k,))
    sets = truth["sets"]
    records = truth["records"]
    hit, miss, miss_detail = {}, {}, {}
    for c in CATS:
        hit[c] = sorted(t for t in picks[c] if t in sets[c])
        miss[c] = sorted(t for t in picks[c] if t not in sets[c])
        for t in miss[c]:
            miss_detail[t] = {"picked_as": LABEL[c], "actual": actual_label(records[t])}
    by_cat = {c: len(hit[c]) for c in CATS}
    hits = sum(by_cat.values())
    return {"stage": truth["stage"], "k": k, "cells": sum(SIZES.values()),
            "hits": hits, "passed": hits >= k, "by_cat": by_cat,
            "hit": hit, "miss": miss, "miss_detail": miss_detail,
            "truth_sets": {c: sorted(sets[c]) for c in CATS}}


def score_document(truth, card, k=5, only=None):
    """报表 / 单卡 → {策略名: 计分结果}。only 可限一个策略名。"""
    out = {}
    for name, raw in card_strategies(card).items():
        if only and name != only:
            continue
        out[name] = score_card(truth, norm_picks(raw, truth["records"]), k)
    if only and only not in out:
        raise ValueError("card 里没有策略 %r（含：%s）"
                         % (only, ", ".join(sorted(card_strategies(card)))))
    if not out:
        raise ValueError("card JSON 里没有可评的策略")
    return out


def format_report(truth, results, k=5):
    lines = ["== Pick'Em 真实命中：%s ==" % truth["stage"]]
    sets = truth["sets"]
    lines.append("   过关线 k=%d ｜ 十格 %d 项 ｜ 真值规模：3:0 %d / 晋级 %d / 0:3 %d"
                 % (k, sum(SIZES.values()), len(sets["p30"]), len(sets["padv"]),
                    len(sets["p03"])))
    lines.append("   真实 3:0：%s" % ", ".join(sorted(sets["p30"])))
    lines.append("   真实晋级：%s" % ", ".join(sorted(sets["padv"])))
    lines.append("   真实 0:3：%s" % ", ".join(sorted(sets["p03"])))
    for name in sorted(results):
        r = results[name]
        lines.append("   %-10s 命中 %d/%d（%s）  3:0 %d/%d ｜ 晋级 %d/%d ｜ 0:3 %d/%d"
                     % (name, r["hits"], r["cells"], "过" if r["passed"] else "不过",
                        r["by_cat"]["p30"], SIZES["p30"], r["by_cat"]["padv"],
                        SIZES["padv"], r["by_cat"]["p03"], SIZES["p03"]))
    for name in sorted(results):
        r = results[name]
        for c in CATS:
            if r["hit"][c]:
                lines.append("   命中格 %s（%s）：%s"
                             % (name, LABEL[c], ", ".join(r["hit"][c])))
        for t, d in sorted(r["miss_detail"].items()):
            lines.append("   未中格 %s：%s 填 %s，实为 %s"
                         % (name, t, d["picked_as"], d["actual"]))
    lines.append("   注：真实命中是**赛后**的量，与报表里的 P(≥k)（赛前模型概率）是两回事；")
    lines.append("       单阶段命中是小样本，只能作量级示意，不能当准确率。")
    return "\n".join(lines)


# ------------------------------------------------------------------ 自证
def _selftest():
    ok = 0

    def chk(cond, msg):
        nonlocal ok
        if not cond:
            raise AssertionError("FAIL: " + msg)
        ok += 1

    truth_doc = {
        "stage": "SYNTHETIC",
        "records": {"T01": [3, 0], "T02": [3, 1], "T03": [3, 1], "T04": [3, 2],
                    "T05": [3, 0], "T06": [3, 2], "T07": [3, 1], "T08": [3, 2],
                    "T09": [1, 3], "T10": [1, 3], "T11": [2, 3], "T12": [1, 3],
                    "T13": [2, 3], "T14": [0, 3], "T15": [2, 3], "T16": [0, 3]},
    }
    chk(check_records(truth_doc["records"]) == [], "合成真值表应合法")
    truth = {"stage": "SYNTHETIC", "records": truth_doc["records"],
             "sets": classify(truth_doc["records"])}
    chk(len(truth["sets"]["p30"]) == 2 and len(truth["sets"]["padv"]) == 6
        and len(truth["sets"]["p03"]) == 2, "三集合规模应为 2/6/2")
    chk(actual_label([2, 3]) == "淘汰" and actual_label([0, 3]) == "0:3"
        and actual_label([3, 2]) == "晋级" and actual_label([3, 0]) == "3:0",
        "actual_label 四态")

    # 真值表结构负样本
    def bad(mutate, msg):
        d = json.loads(json.dumps(truth_doc))
        mutate(d["records"])
        chk(check_records(d["records"]) != [], msg)

    bad(lambda r: r.pop("T16"), "15 队必须被拒")
    bad(lambda r: r.update({"T16": [3, 0]}), "3 支 3:0 必须被拒")
    bad(lambda r: r.update({"T16": [0, 0]}), "0 胜 0 负必须被拒（结构不符）")
    bad(lambda r: r.update({"T16": [3, 3]}), "3 胜 3 负必须被拒")
    bad(lambda r: r.update({"T16": [2.5, 3]}), "非整数战绩必须被拒")
    bad(lambda r: r.update({"T16": [True, 3]}), "布尔不得当作整数")
    bad(lambda r: r.update({"T16": [1, 4]}), "越界战绩必须被拒")
    bad(lambda r: r.update({"T16": [3, 2]}), "晋级 7 支必须被拒")

    perfect = {"p30": ["T01", "T05"], "padv": ["T02", "T03", "T04", "T06", "T07", "T08"],
               "p03": ["T14", "T16"]}
    zero = {"p30": ["T09", "T10"], "padv": ["T11", "T12", "T13", "T14", "T15", "T16"],
            "p03": ["T07", "T08"]}
    seven = {"p30": ["T01", "T09"], "padv": ["T02", "T03", "T04", "T06", "T07", "T14"],
             "p03": ["T16", "T08"]}

    r = score_card(truth, norm_picks(perfect, truth["records"]), 5)
    chk(r["hits"] == 10 and r["passed"] and not r["miss_detail"], "完美卡 10/10 且过关")
    chk(r["by_cat"] == {"p30": 2, "padv": 6, "p03": 2}, "完美卡分类命中 2/6/2")

    r0 = score_card(truth, norm_picks(zero, truth["records"]), 5)
    chk(r0["hits"] == 0 and not r0["passed"], "全错卡 0/10 不过")
    chk(len(r0["miss_detail"]) == 10, "全错卡 10 格都要有归因")

    r7 = score_card(truth, norm_picks(seven, truth["records"]), 5)
    chk(r7["hits"] == 7 and r7["passed"], "7/10 在 k=5 下过关")
    chk(not score_card(truth, norm_picks(seven, truth["records"]), 8)["passed"],
        "7/10 在 k=8 下不过（边界）")
    chk(score_card(truth, norm_picks(seven, truth["records"]), 7)["passed"],
        "7/10 在 k=7 下过关（恰好等于门槛）")
    chk(r7["miss_detail"]["T14"]["actual"] == "0:3"
        and r7["miss_detail"]["T14"]["picked_as"] == "晋级"
        and r7["miss_detail"]["T08"]["actual"] == "晋级"
        and r7["miss_detail"]["T09"]["actual"] == "淘汰", "未中格归因三态")
    chk(len(r7["hit"]["p30"]) + len(r7["hit"]["padv"]) + len(r7["hit"]["p03"]) == r7["hits"],
        "命中明细合计应等于命中数")

    # 十格负样本
    def badpicks(p, msg):
        try:
            norm_picks(p, truth["records"])
        except ValueError:
            ok_ = True
        else:
            ok_ = False
        chk(ok_, msg)

    badpicks({"p30": ["T01"], "padv": perfect["padv"], "p03": perfect["p03"]}, "3:0 只填 1 格必须被拒")
    badpicks({"p30": ["T01", "T05"], "padv": perfect["padv"][:5], "p03": perfect["p03"]},
             "晋级只填 5 格必须被拒")
    badpicks({"p30": ["T01", "T01"], "padv": perfect["padv"], "p03": perfect["p03"]},
             "同组重复必须被拒")
    badpicks({"p30": ["T01", "T02"], "padv": ["T02", "T03", "T04", "T06", "T07", "T08"],
              "p03": perfect["p03"]}, "跨组重复必须被拒")
    badpicks({"p30": ["T01", "ZZZ"], "padv": perfect["padv"], "p03": perfect["p03"]},
             "真值表外的队名（阶段错配）必须被拒")
    badpicks({"p30": ["T01", "T05"]}, "缺组必须被拒")
    chk(norm_picks(perfect, truth["records"]) == perfect, "合法十格应原样返回")

    # 过关线负样本
    try:
        score_card(truth, norm_picks(perfect, truth["records"]), 0)
    except ValueError:
        ok += 1
    else:
        raise AssertionError("FAIL: k=0 必须被拒")

    # 形状兼容 + 幂等
    doc = {"baselines": {"optimizer": {"picks": perfect}, "hot": {"picks": seven}},
           "picks": perfect}
    res = score_document(truth, doc, 5)
    chk(set(res) == {"optimizer", "hot"} and res["optimizer"]["hits"] == 10
        and res["hot"]["hits"] == 7, "报表形状应评出两条策略")
    bare = {"baselines": {"a": perfect, "b": seven}}     # baselines 值直接是十格
    res_bare = score_document(truth, bare, 5)
    chk(set(res_bare) == {"a", "b"} and res_bare["a"]["hits"] == 10, "未包 picks 的写法也应接受")
    chk(score_document(truth, {"picks": perfect}, 5)["card"]["hits"] == 10,
        "{picks: {...}} 形状应可评")
    for bad_card, why in (({"foo": 1}, "认不出的 card 必须报错（不许 rc=0 空跑）"),
                          ({"baselines": {"x": {"p30": ["T01"]}}},
                           "baselines 里没有合法十格必须报错")):
        try:
            score_document(truth, bad_card, 5)
        except ValueError:
            ok += 1
        else:
            raise AssertionError("FAIL: " + why)
    chk(score_document(truth, doc, 5, only="hot")["hot"]["hits"] == 7, "--only 应只评一条")
    chk(score_document(truth, perfect, 5)["card"]["hits"] == 10, "最简形状应可评")
    chk(score_document(truth, doc, 5) == score_document(truth, doc, 5), "同输入结果应一致")
    try:
        score_document(truth, doc, 5, only="nope")
    except ValueError:
        ok += 1
    else:
        raise AssertionError("FAIL: 不存在的策略名必须被拒")

    rep = format_report(truth, res, 5)
    chk("命中 10/10（过）" in rep and "未中格" in rep and "真实 0:3" in rep, "报表关键行齐")

    print("SCORING_SELFTEST_OK (%d checks)" % ok)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pick'Em 真实命中计分器")
    ap.add_argument("--truth", required=True, help="真值表 JSON（16 队最终战绩）")
    ap.add_argument("--card", required=True, help="十格 JSON（报表或 {p30,padv,p03}）")
    ap.add_argument("--k", type=int, default=5, help="过关线（默认 5，Pick'Em 官方口径）")
    ap.add_argument("--json", dest="json_out", help="把计分结果写入该路径")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    try:
        truth = load_truth(a.truth)
        with open(a.card, encoding="utf-8") as fh:
            card = json.load(fh)
        results = score_document(truth, card, k=a.k)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("[ERR] 计分输入无效：%s" % exc, file=sys.stderr)
        return 1
    if not a.quiet:
        print(format_report(truth, results, a.k))
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as fh:
            json.dump({"stage": truth["stage"], "k": a.k, "results": results},
                      fh, ensure_ascii=False, indent=1)
        print("\n[pickem] 计分结果写入 %s" % a.json_out)
    print("SCORING_DONE")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(_selftest())
    sys.exit(main())
