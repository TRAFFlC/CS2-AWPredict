"""按赛事层级评估胜方模型：T1（真大赛）vs T4（纯弱队局）。

为什么必须做：语料 67.7% 的场次（T4）连一支前16队都没有，22.2%（T1）才是有前8队的真大赛。
模型在这种切分下的表现差异，直接决定它在"你要预测的大赛"上还有没有价值。

分层只用 csapi 的 `rank` 字段（HLTV 式，0 缺失），**不按赛事名猜**：
  T1 前8队≥3支 或 名字含 major 且有前8队
  T2 有前8队但 <3 支
  T3 无前8队但前16队≥4支
  T4 连前16队都没有
"""
import collections, io, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.0）
sys.path.insert(0, HERE)
import model_v2 as M


def event_tiers():
    ev = collections.OrderedDict()
    for l in io.open(os.path.join(DATA, "csapi_matches_all.jsonl"), encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        d = ev.setdefault(r.get("event") or "?", {"r": set()})
        for t in (r["team1"], r["team2"]):
            if t.get("rank"):
                d["r"].add(int(t["rank"]))
    out = {}
    for e, d in ev.items():
        n8 = sum(1 for x in d["r"] if x <= 8)
        n16 = sum(1 for x in d["r"] if x <= 16)
        s = e.lower()
        out[e] = ("T1" if (n8 >= 3 or ("major" in s and n8 >= 1))
                  else "T2" if n8 >= 1
                  else "T3" if n16 >= 4 else "T4")
    return out


def main():
    tier = event_tiers()
    # split 记录**没有 event 字段**（只有 id / teamA / teamB），必须按 id 关联，
    # 按 event 字符串关联会全部落进 T4（第一版就这么错的）。
    ev_by_id = {}
    for l in io.open(os.path.join(DATA, "csapi_matches_all.jsonl"), encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            ev_by_id[str(r["id"])] = r.get("event")
    prep = M.prepare(verbose=False)
    data = prep["data"]
    ws, wm = prep["wsel"], prep["wmodel"]

    print("== 1) 各层语料分布（train/valid/test）==")
    print("  %-4s %8s %8s %8s %8s" % ("层", "train", "valid", "test", "合计"))
    cnt = {s: collections.Counter() for s in ("train", "valid", "test")}
    for s in ("train", "valid", "test"):
        for t in data[s]:
            cnt[s][tier.get(ev_by_id.get(str(t.get("id"))), "T4")] += 1
    for k in ("T1", "T2", "T3", "T4"):
        a, b, c = cnt["train"][k], cnt["valid"][k], cnt["test"][k]
        print("  %-4s %8d %8d %8d %8d" % (k, a, b, c, a + b + c))
    tot = sum(sum(v.values()) for v in cnt.values())
    print("  %-4s %8d %8d %8d %8d" % ("合计",
          sum(cnt['train'].values()), sum(cnt['valid'].values()),
          sum(cnt['test'].values()), tot))

    print("\n== 2) 胜方准确率（train 用 wsel，valid/test 用对应模型）==")
    print("  %-4s %-8s %6s %8s %8s %8s" % ("层", "集合", "n", "准确率", "平均信心", "校准"))
    for s, model in (("valid", ws), ("test", wm)):
        buckets = collections.defaultdict(list)
        for t in data[s]:
            p = M.predict_proba_win(model, t)
            y = float(t["y"])
            conf = max(p, 1 - p)
            ok = 1.0 if ((p >= 0.5) == (y >= 0.5)) else 0.0
            buckets[tier.get(ev_by_id.get(str(t.get("id"))), "T4")].append((ok, conf, y))
        for k in ("T1", "T2", "T3", "T4"):
            v = buckets.get(k)
            if not v:
                continue
            acc = np.mean([x[0] for x in v])
            mc = np.mean([x[1] for x in v])
            base = max(np.mean([x[2] for x in v]), 1 - np.mean([x[2] for x in v]))
            print("  %-4s %-8s %6d %7.1f%% %8.3f %+7.1fpp"
                  % (k, s, len(v), 100 * acc, mc, 100 * (acc - base)))

    print("\n== 3) 一线队（两队都在前16）子集 vs 其余 ==")
    rank_by_event = {}
    for l in io.open(os.path.join(DATA, "csapi_matches_all.jsonl"), encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        rank_by_event.setdefault(r.get("event") or "?", set()).update(
            int(t["rank"]) for t in (r["team1"], r["team2"]) if t.get("rank"))
    for s, model in (("valid", ws), ("test", wm)):
        top, rest = [], []
        for t in data[s]:
            p = M.predict_proba_win(model, t)
            y = float(t["y"])
            ok = 1.0 if ((p >= 0.5) == (y >= 0.5)) else 0.0
            rk = rank_by_event.get(ev_by_id.get(str(t.get("id"))), set())
            (top if len(rk) and min(rk) <= 16 else rest).append(ok)
        if top:
            print("  %-6s 赛事含前16队 n=%4d 准确率 %.1f%%  |  纯弱队赛事 n=%4d 准确率 %.1f%%"
                  % (s, len(top), 100 * np.mean(top), len(rest), 100 * np.mean(rest)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
