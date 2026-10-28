#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐队边际概率的**校准检验** —— issue-015 §4 验收第 4 条。

`predict_stage.py` 给出逐队 P(3:0) / P(晋级) / P(0:3)（"模型说"），`scoring.py` 给出
十格的真实命中（"押对几格"）；本模块回答第三个问题：**那些概率本身可信吗**——
按队算 Brier / LogLoss / 可靠性分箱，并与**常数基线**（全押该类的平均发生率）对照。

依据（issue-015 §5.1）：判一个概率源好坏看它**有没有比"永远输出平均发生率"更好**。
实测教训：赛前排名源的 3:0 Brier（0.1220）**比常数基线（0.1094）还差** ⇒ 该源在 3:0 上无效。
不查这一项，会把"看起来很专业的小数"当成信息。

输入两个文件（都可由本工具链产生）：
  * 真值表：与 `scoring.py` 同格式（16 队最终战绩，**留私有区**）；
  * 报表：任何含 `marginals` 的 JSON —— `predict_stage.py --json` 的产物、
    或私有回测 `backtest_budapest_*.json`（同名字段，故私有数字可直接复核）。
重复给 `--report` 即可**跨阶段汇总**（比较强度源时这是唯一有统计意义的做法）。

用法::

    python calibration.py --truth truth.json --report rep_elo.json
    python calibration.py --truth truth.json --report point.json --report sigma.json --json out.json

退出码：0 成功；1 输入非法（真值表结构不符 / 报表缺 `marginals` / 队名对不上 / 读文件失败）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys

CATS = ("p30", "padv", "p03")
LABEL = {"p30": "3:0", "padv": "晋级", "p03": "0:3"}
BINS = 5


# ---------------------------------------------------------------- 输入
def truth_labels(records):
    """真值战绩 → {队名: {p30: 0/1, padv: 0/1, p03: 0/1}}。"""
    out = {}
    for t, (w, l) in records.items():
        out[t] = {"p30": 1 if (w == 3 and l == 0) else 0,
                  "padv": 1 if (w == 3 and l in (1, 2)) else 0,
                  "p03": 1 if (w == 0 and l == 3) else 0}
    return out


def report_marginals(doc, name=None):
    """从报表 JSON 里取 `marginals`（队名 → 三概率），并做形状校验。"""
    if not isinstance(doc, dict):
        raise ValueError("报表应为 JSON 对象")
    marg = doc.get("marginals")
    if not isinstance(marg, dict) or not marg:
        raise ValueError("报表缺少 marginals（本模块只吃 predict_stage/backtest 的报表）")
    out = {}
    for t, v in marg.items():
        if not isinstance(v, dict) or any(c not in v for c in CATS):
            raise ValueError("marginals[%s] 缺 p30/padv/p03" % t)
        row = {}
        for c in CATS:
            x = v[c]
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                raise ValueError("marginals[%s][%s] 不是数字（%r）" % (t, c, x))
            if not (0.0 <= float(x) <= 1.0):
                raise ValueError("marginals[%s][%s] = %r 越界（0..1）" % (t, c, x))
            row[c] = float(x)
        out[str(t)] = row
    # 标签尽量能区分臂：`source`（+ `arm`，如 uncertain）——否则同一源的两个臂会重名
    label = name or doc.get("source") or "(未命名报表)"
    arm = doc.get("arm")
    if not name and arm:
        label = "%s/%s" % (label, arm)
    return {"label": label,
            "stage": doc.get("stage") or doc.get("event") or "(未命名阶段)",
            "marginals": out}


def check_against_truth(doc, labels):
    """报表与真值表必须是**同一阶段**；队名对不上就是阶段错配（静默算错校准）。"""
    missing = sorted(set(doc["marginals"]) - set(labels))
    extra = sorted(set(labels) - set(doc["marginals"]))
    probs = []
    if missing:
        probs.append("报表含真值表里没有的队伍（阶段错配？）：%s" % ", ".join(missing))
    if extra:
        probs.append("真值表有队伍不在报表里：%s" % ", ".join(extra))
    if probs:
        raise ValueError("; ".join(probs))
    return True


# ---------------------------------------------------------------- 指标
def brier(ps, ys):
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps)


def logloss(ps, ys, eps=1e-6):
    tot = 0.0
    for p, y in zip(ps, ys):
        q = min(max(p, eps), 1.0 - eps)
        tot += -(y * math.log(q) + (1 - y) * math.log(1 - q))
    return tot / len(ps)


def reliability(ps, ys, bins=BINS):
    """可靠性分箱：每箱给 队数 / 预测均值 / 实测发生率。"""
    out = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        idx = [j for j, p in enumerate(ps)
               if (lo <= p < hi) or (i == bins - 1 and p == 1.0)]
        if not idx:
            continue
        out.append({"lo": lo, "hi": hi, "n": len(idx),
                    "pred": sum(ps[j] for j in idx) / len(idx),
                    "obs": sum(ys[j] for j in idx) / len(idx)})
    return out


def calibrate_category(doc, labels, cat):
    teams = sorted(doc["marginals"])
    ps = [doc["marginals"][t][cat] for t in teams]
    ys = [labels[t][cat] for t in teams]
    base = sum(ys) / len(ys)
    b_model, b_base = brier(ps, ys), brier([base] * len(ys), ys)
    return {"n": len(teams), "brier": b_model, "brier_base": b_base,
            "delta": b_model - b_base, "better_than_base": b_model < b_base,
            "logloss": logloss(ps, ys),
            "logloss_base": logloss([base] * len(ys), ys),
            "base_rate": base, "reliability": reliability(ps, ys)}


def calibrate(truth, docs):
    """真值表 × 若干报表 → 逐报表、逐类别的校准指标。"""
    labels = truth["labels"]
    out = {"stage": truth["stage"], "reports": []}
    for doc in docs:
        check_against_truth(doc, labels)
        entry = {"label": doc["label"], "stage": doc["stage"],
                 "categories": {c: calibrate_category(doc, labels, c) for c in CATS}}
        out["reports"].append(entry)
    return out


def format_report(res):
    lines = ["== 校准检验：%s ==" % res["stage"]]
    lines.append("   （逐队边际概率 vs 真实战绩；判据是「能否击败常数基线」——"
                 "常数基线 = 永远输出该类的平均发生率）")
    for r in res["reports"]:
        lines.append("   [%s]" % r["label"])
        lines.append("     类别    Brier    常数基线    差        LogLoss   实测率")
        for c in CATS:
            v = r["categories"][c]
            lines.append("     %-4s   %.4f   %.4f    %+.4f   %.4f    %.3f%s"
                         % (LABEL[c], v["brier"], v["brier_base"], v["delta"],
                            v["logloss"], v["base_rate"],
                            "" if v["better_than_base"] else "   ⚠ 劣于常数基线"))
        for c in CATS:
            rb = r["categories"][c]["reliability"]
            if not rb:
                continue
            lines.append("     可靠性分箱 %s：%s" % (LABEL[c], " ｜ ".join(
                "[%.1f-%.1f) n=%d 预测%.3f 实测%.3f" % (b["lo"], b["hi"], b["n"],
                                                     b["pred"], b["obs"]) for b in rb)))
    lines.append("   注：① 过关概率 P(≥k) 的校准需要**多阶段**（每个阶段只有 1 个 0/1 结果），"
                 "此处只校逐队边际；")
    lines.append("       ② 单阶段 16 队的 Brier 噪声很大，看「差」的符号与量级，不要看小数点后三位。")
    return "\n".join(lines)


# ---------------------------------------------------------------- 自证
def _selftest():
    ok = 0

    def chk(cond, msg):
        nonlocal ok
        if not cond:
            raise AssertionError("FAIL: " + msg)
        ok += 1

    teams = ["T%02d" % i for i in range(1, 17)]
    records = {t: [1, 3] for t in teams}
    for t, wl in (("T01", [3, 0]), ("T05", [3, 0]), ("T14", [0, 3]), ("T16", [0, 3])):
        records[t] = wl
    for t in ("T02", "T03", "T07"):
        records[t] = [3, 1]
    for t in ("T04", "T06", "T08"):
        records[t] = [3, 2]
    labels = truth_labels(records)
    chk(sum(labels[t]["p30"] for t in teams) == 2, "真值应恰 2 支 3:0")
    chk(sum(labels[t]["padv"] for t in teams) == 6, "真值应恰 6 支晋级")
    chk(sum(labels[t]["p03"] for t in teams) == 2, "真值应恰 2 支 0:3")

    chk(abs(brier([1.0, 0.0], [1, 0])) < 1e-12, "完美预测 Brier=0")
    chk(abs(brier([0.5, 0.5], [1, 0]) - 0.25) < 1e-12, "0.5 预测 Brier=0.25")
    # 概率被裁剪到 [eps, 1-eps] ⇒ 完美预测的 LogLoss 是 ~eps 量级而不是 0
    # （故意不写 0：真值那个 0 只有靠不裁剪才拿得到，而裁剪是防 log(0) 的必要手段）
    chk(logloss([1.0, 0.0], [1, 0]) < 1e-5, "完美预测 LogLoss 应 ≈ 0（裁剪后 ≤1e-5）")
    chk(logloss([0.5, 0.5], [1, 0]) > 0.69, "0.5 的 LogLoss 应 ≈ ln2")
    rb = reliability([0.1, 0.9], [0, 1])
    chk(len(rb) == 2 and abs(rb[0]["pred"] - 0.1) < 1e-12 and rb[1]["obs"] == 1.0,
        "可靠性分箱应分到两箱")

    # 常数基线：等于发生率时 delta 应恰为 0
    base = 2 / 16.0
    chk(abs(brier([base] * 16, [labels[t]["p30"] for t in teams])
            - brier([base] * 16, [labels[t]["p30"] for t in teams])) < 1e-12, "自反")

    def doc(marg, label="src"):
        return {"label": label, "stage": "SYN", "marginals": marg}

    truth = {"stage": "SYN", "labels": labels}
    # 完美预测源：Brier=0 且击败常数基线
    perfect = {t: {c: float(labels[t][c]) for c in CATS} for t in teams}
    res = calibrate(truth, [doc(perfect, "perfect")])
    v = res["reports"][0]["categories"]["p30"]
    chk(v["brier"] == 0.0 and v["better_than_base"], "完美源 Brier=0 且优于常数基线")
    # 常数源：与基线同值 ⇒ 不算更好
    const = {t: {c: labels[t][c] * 0 + (2 / 16.0 if c == "p30" else 0.3) for c in CATS}
             for t in teams}
    v = calibrate(truth, [doc(const, "const")])["reports"][0]["categories"]["p30"]
    chk(not v["better_than_base"] and abs(v["delta"]) < 1e-12, "常数源应恰好打平基线")
    # 反向源：应明显劣于基线
    anti = {t: {c: 1.0 - float(labels[t][c]) for c in CATS} for t in teams}
    v = calibrate(truth, [doc(anti, "anti")])["reports"][0]["categories"]["p30"]
    chk(v["delta"] > 0 and not v["better_than_base"], "反向源应劣于基线")

    # 多报表汇总 + 标签
    res2 = calibrate(truth, [doc(perfect, "a"), doc(const, "b")])
    chk([r["label"] for r in res2["reports"]] == ["a", "b"], "多报表应各留一条")
    chk("perfect" in format_report(res) and "可靠性分箱" in format_report(res),
        "报表应含标签与分箱行")
    chk("劣于常数基线" in format_report(calibrate(truth, [doc(anti, "anti")])),
        "劣于基线应被标出")

    # 负样本
    def must_fail(fn, why):
        try:
            fn()
        except ValueError:
            ok_local = True
        else:
            ok_local = False
        chk(ok_local, why)

    must_fail(lambda: report_marginals({"stage": "x"}), "缺 marginals 应报错")
    must_fail(lambda: report_marginals({"marginals": {"A": {"p30": 0.5}}}), "缺类别应报错")
    must_fail(lambda: report_marginals({"marginals": {"A": {"p30": 1.5, "padv": 0.5, "p03": 0.5}}}),
              "越界概率应报错")
    must_fail(lambda: report_marginals({"marginals": {"A": {"p30": "0.5", "padv": 0.5, "p03": 0.5}}}),
              "非数字应报错")
    bad = {t: {c: 0.5 for c in CATS} for t in teams[:15]}
    must_fail(lambda: calibrate(truth, [doc(bad, "bad")]), "少一队（阶段错配）应报错")
    extra = {t: {c: 0.5 for c in CATS} for t in teams + ["ZZZ"]}
    must_fail(lambda: calibrate(truth, [doc(extra, "extra")]), "多出未知队应报错")

    # 从报表里读 marginals（与 predict_stage --json 同形）
    d = report_marginals({"stage": "S", "source": "elo", "marginals": perfect})
    chk(d["label"] == "elo" and d["stage"] == "S" and len(d["marginals"]) == 16,
        "应能从报表里读出 marginals 与标签")
    d = report_marginals({"stage": "S", "source": "elo", "arm": "uncertain",
                          "marginals": perfect})
    chk(d["label"] == "elo/uncertain", "有 arm 时应拼进标签（否则两臂重名）")
    chk(report_marginals({"stage": "S", "source": "elo", "marginals": perfect},
                         name="自定义")["label"] == "自定义", "显式 name 应优先")

    print("CALIBRATION_SELFTEST_OK (%d checks)" % ok)
    return 0


# ---------------------------------------------------------------- CLI
def truth_from_doc(doc):
    """从**已在内存里**的真值表 dict 构造标签（CLI / Web / 测试共用）。非法即 ValueError。"""
    records = (doc or {}).get("records")
    if not isinstance(records, dict) or not records:
        raise ValueError("真值表缺少 records（{队名: [胜, 负]}）")
    norm = {}
    for t, v in records.items():
        if not isinstance(v, (list, tuple)) or len(v) != 2 \
                or any(isinstance(x, bool) or not isinstance(x, int) for x in v):
            raise ValueError("队伍 %s 的战绩应为两个整数 [胜, 负]，实得 %r" % (t, v))
        if not (0 <= v[0] <= 3 and 0 <= v[1] <= 3) or (v[0] == 3 and v[1] == 3):
            raise ValueError("队伍 %s 的战绩越界或矛盾：%r" % (t, list(v)))
        norm[t] = [int(v[0]), int(v[1])]
    return {"stage": doc.get("stage") or "(未命名阶段)", "records": norm,
            "labels": truth_labels(norm)}


def load_truth(path):
    """读真值表（与 scoring.py 同格式）。非法即 ValueError。"""
    with open(path, encoding="utf-8") as fh:
        return truth_from_doc(json.load(fh))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pick'Em 逐队边际概率校准检验")
    ap.add_argument("--truth", required=True, help="真值表 JSON（16 队最终战绩）")
    ap.add_argument("--report", action="append", required=True,
                    help="含 marginals 的报表 JSON（可重复给多个）")
    ap.add_argument("--json", dest="json_out", help="把结果写入该路径")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    try:
        truth = load_truth(a.truth)
        docs = []
        for p in a.report:
            with open(p, encoding="utf-8") as fh:
                docs.append(report_marginals(json.load(fh)))
        res = calibrate(truth, docs)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("[ERR] 校准输入无效：%s" % exc, file=sys.stderr)
        return 1
    if not a.quiet:
        print(format_report(res))
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)
        print("\n[pickem] 校准结果写入 %s" % a.json_out)
    print("CALIBRATION_DONE")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(_selftest())
    sys.exit(main())
