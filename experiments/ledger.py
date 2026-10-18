"""排行榜台账：把「每轮我的冻结预测 / 我的得分 / 榜首得分 / 校准偏移」记成可复算的账。

为什么需要它（三条理由，都不是"记个流水"）：
  1. **δ 自愈**：每轮算 `δ = 我最近 K 场的实际准确率 − 平均信心`，
     把 δ 平移进 E[rank] 的概率。实测证明**全局校准器不管用**
     （test 欠自信 −4.9pp、EPL S24 过自信 +7.3pp，方向会翻转），
     而这个 δ 用**我自己的实时表现**纠偏，自动吸收时间漂移。
  2. **零风险的 A/B 台**：v3 结论「两目标在本语料 D/(2σ)<0 大体重合」是**模拟**结论。
     台账让我们能在**真实赛场**上按轮次切分——某几轮用 score 目标、某几轮用 rank 目标——
     然后看哪个真的多拿分。这是 41 场之外唯一能证伪那个结论的办法。
  3. **可审计**：冻结件哈希 + 得分 + 榜首分数三者对齐，日后能复算每一轮的决策是否最优。

用法：
  python ledger.py record --round 1 --pred predictions_2026-10-12.json \
      --my-score 230 --leader-score 230 --leader-total 100 --field-n 20
  python ledger.py status
"""
import collections, hashlib, io, json, os, sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）
LEDGER = os.path.join(DATA, "groundtruth", "leaderboard_ledger.jsonl")
LOOKBACK = 20          # δ 用最近 K 场


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load():
    if not os.path.exists(LEDGER):
        return []
    return [json.loads(l) for l in io.open(LEDGER, encoding="utf-8") if l.strip()]


def save(rows):
    io.open(LEDGER, "w", encoding="utf-8", newline="\n").write(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


def record(args):
    rows = load()
    rounds = [r["round"] for r in rows]
    if args.rnd in rounds:
        print("第 %d 轮已在台账里，拒绝重复记录（append-only）" % args.rnd)
        return 1
    blob = sha256_of(args.pred)
    doc = json.load(io.open(args.pred, encoding="utf-8"))
    ms = doc["matches"]
    frozen = [{
        "A": m["A"], "B": m["B"], "bo": m["bo"],
        "winner": m["winner"], "win_prob": m["win_prob"],
        "score": m["score"], "maps": m["maps"], "mvp": m["mvp"],
        "exp_points": m["exp_points"],
    } for m in ms]
    rows.append({
        "round": args.rnd,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pred_file": os.path.basename(args.pred),
        "pred_sha256": blob[:16],
        "n_matches": len(ms),
        "max_points": sum(25 if m["bo"] == 1 else 45 if m["bo"] == 3 else 55
                          for m in ms),
        "self_est_total": round(sum(m["exp_points"] for m in ms), 2),
        "my_score": args.my_score,
        "leader_score": args.leader_score,
        "leader_total": args.leader_total,
        "field_n": args.field_n,
        "objective": args.obj,
        "frozen_predictions": frozen,
    })
    save(rows)
    print("已记录第 %d 轮：%d 场（满分 %d）  自估 %.1f  我 %d / 榜首 %d"
          % (args.rnd, len(ms), rows[-1]["max_points"],
             rows[-1]["self_est_total"], args.my_score, args.leader_score))
    return 0


def cmd_status(args):
    rows = load()
    if not rows:
        print("台账为空。先 record。")
        return 0
    # **按赛事分账**：不同赛事的分数不可相加（2026-10-12 主理人纠正过一次）。
    # 累计分按**轮内得分**重算，不能信任 my_score 字段——R10 曾因 res 累积了
    # 两天的文件而把 09-08 重复计入（135+295=430，真值 295）。
    groups = collections.OrderedDict()
    for r in rows:
        groups.setdefault(r.get("competition") or "EPL S24", []).append(r)

    for comp, rs in groups.items():
        run = 0
        run_max = 0
        run_est = 0
        print("\n=== %s ===" % comp)
        print("轮次 | 场次 | 本轮 | 我自估 | 累计 | 累计满分 | 累计%% | 榜首 | 差")
        print("-" * 82)
        for r in rs:
            rp = r.get("round_points")
            if rp is None:                      # 早期回填行只有 my_score
                run = r["my_score"]
            else:
                run += rp
            run_max += r["max_points"]
            run_est += r["self_est_total"]
            ls = r.get("leader_score")
            r["my_score"] = run                 # 回写，保证后续轮次接着算
            print("%4d | %4d | %4s | %6.1f | %5d | %8d | %5.1f%% | %5s | %4s"
                  % (r["round"], r["n_matches"],
                     rp if rp is not None else "-", r["self_est_total"], run,
                     run_max, 100 * run / max(1, run_max),
                     "?" if ls is None else str(ls),
                     "?" if ls is None else "%+d" % (run - ls)))
        save(rows)
        line = ("  累计 %d / %d = %.2f%%    自估 %.1f (%.2f%%)  偏差 %+.1f 分 (%+.1fpp)"
                % (run, run_max, 100 * run / max(1, run_max), run_est,
                   100 * run_est / max(1, run_max), run_est - run,
                   100 * (run_est - run) / max(1, run_max)))
        ls = rs[-1].get("leader_score")
        if ls is not None:
            line += "    榜首 %d    差 %+d" % (ls, run - ls)
        print(line)

    # 校准自检也必须**按赛事分**，混算没有意义
    for comp, rs in groups.items():
        flat = [m for r in rs for m in r["frozen_predictions"]]
        conf_ok, conf_bad = [], []
        for m in flat[-LOOKBACK:]:
            # 必须比**真实胜方**（truth_winner），不能比 teamA——比 teamA 等于在问
            # 「我有没有选 A 队」，那是选边倾向不是准确率（v1 的 bug：δ 全错）。
            tw = m.get("truth_winner")
            if not tw:
                continue
            pa = float((m.get("win_prob") or {}).get(m["A"], 0.5))
            conf = max(pa, 1 - pa)
            (conf_ok if m["winner"] == tw else conf_bad).append(conf)
        n = len(conf_ok) + len(conf_bad)
        if not n:
            continue
        mo = sum(conf_ok) / len(conf_ok) if conf_ok else float("nan")
        mb = sum(conf_bad) / len(conf_bad) if conf_bad else float("nan")
        acc = len(conf_ok) / n
        mc = (sum(conf_ok) + sum(conf_bad)) / n
        se = (acc * (1 - acc) / n) ** 0.5
        print("\n[%s] 胜方校准：猜对 %d 平均信心 %.3f / 猜错 %d 平均信心 %.3f"
              % (comp, len(conf_ok), mo, len(conf_bad), mb))
        print("  ⇒ δ = 实际 %.3f − 平均信心 %.3f = %+.3f (标准误 %.3f, %.1fσ)  %s"
              % (acc, mc, acc - mc, se, (acc - mc) / se,
                 "欠自信" if (acc - mc) > se else
                 ("过自信" if (acc - mc) < -se else "与 0 无异")))
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r1 = sub.add_parser("record")
    r1.add_argument("--round", dest="rnd", type=int, required=True)
    r1.add_argument("--pred", required=True)
    r1.add_argument("--my-score", type=int, required=True)
    r1.add_argument("--leader-score", type=int, default=0)
    r1.add_argument("--leader-total", type=int, default=0)
    r1.add_argument("--field-n", type=int, default=20)
    r1.add_argument("--objective", default="score", choices=["score", "rank"])
    r1.set_defaults(func=record)
    r2 = sub.add_parser("status")
    r2.set_defaults(func=cmd_status)
    a = ap.parse_args()
    sys.exit(a.func(a))
