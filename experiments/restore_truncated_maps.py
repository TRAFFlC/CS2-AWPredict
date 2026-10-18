"""恢复被 `_fix_provenance.py` 等中间脚本截断的 `maps_full_B`（干跑脚本）。

## 问题
`record_schema` 定义 `maps_full_B` = **读法 B 全图集**（bo 张，含 veto 已定但
从未上场的决胜图），`validate()` 也硬性要求 bo3 恰 3 张。

但 `_fix_provenance.py:81-86` 对 2:0 记录做了：
    if (s1,s2) in ((2,0),(0,2)) and len(m)==3:
        r["maps_full_B"] = m[:2];  r["veto_order"] = m[:2]

动机是「决胜图从未上场，不该算进 app/tapp 的『该图出现过』」。这个顾虑本身
**成立**，但手段错在三处：

1. 它改的是 `maps_full_B` —— 该字段的语义就是含 decider，一改就与自身文档矛盾。
2. 它让 63 场过不了 `validate()`，于是被 `ingest_new.as_rows` **整条丢弃**。
   而 `record_schema.to_ingest_row` 里本来有一条专门的降级通道
   `_skip_veto = maps_verified is False`（图序不可信但比赛是真的 ⇒
   照常更新胜负/图池统计，只是不进选图训练）—— **这条通道被顺带废掉了**。
   「别进选图模型」被实现成了「整场别进」。
3. 它与 2,800 场基座语料口径相反。`model_v2.py:690` 喂的是
   `truths[id]["maps"]`，而 `truth_maps_B.jsonl` 的 `maps` 是读法 B、
   `maps_unplayed` 另存 —— **基座本来就把决胜图算进 app**。

⇒ 正确做法是恢复第 3 张图，而不是砍掉。decider 会不会被误算成「出现过」
是 `to_ingest_row` 该用 `maps_played` 决定的事，不该改数据。

## 恢复来源（按优先级，都必须是主理人原始数据，不许用中间产物猜）
1. 主理人原始赛果文件：Mimo 的 EPL 明细 + fissure_*_results.json
2. `new_records_2026-09.jsonl.bak_before_provenance`（截断前的快照）

用法：
    python -X utf8 restore_truncated_maps.py           # 干跑，只报告
    python -X utf8 restore_truncated_maps.py --apply   # 写盘（自动备份）
"""
import collections
import glob
import io
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）
sys.path.insert(0, HERE)
import record_schema as RS

SRC = os.path.join(DATA, "groundtruth", "new_records_2026-09.jsonl")
BAK_TRUNC = SRC + ".bak_before_provenance"
OUT_SRC = os.path.join(HERE, "new_records_2026-09.jsonl.bak_before_restore")
MIMO = r"E:\world\another\data\Mimo\MiMo-V2.6 Flash\2026-10-12" \
       r"\live_tournament_results.jsonl"


def key(d, a, b):
    return (str(d)[:10], frozenset([str(a).strip(), str(b).strip()]))


def load_original():
    """主理人原始赛果 → {(日期, 队伍对): 图序}。只收 bo 张数的列表。"""
    out, prov = {}, collections.defaultdict(set)

    def put(d, a, b, maps, src):
        if not maps or len(maps) not in (3, 5):
            return
        out.setdefault(key(d, a, b), list(maps))
        prov[key(d, a, b)].add(src)

    if os.path.exists(MIMO):
        for l in io.open(MIMO, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            put(r["date"], r["team1"]["name"], r["team2"]["name"],
                r.get("maps_full_B"), "EPL S24 明细")
    for f in sorted(glob.glob(os.path.join(HERE, "fissure_*_results.json"))):
        # `my_maps` 是模型预测，只有 `maps` 是主理人原话 —— 只取后者
        for x in json.load(io.open(f, encoding="utf-8")):
            put(f[10:20], x["A"], x["B"], x.get("maps"),
                os.path.basename(f))
    return out, prov


def main():
    apply_ = "--apply" in sys.argv
    orig, prov = load_original()
    print("主理人原始赛果索引：%d 场（EPL 明细 + fissure_*_results.json）\n" % len(orig))

    recs = [json.loads(l) for l in io.open(SRC, encoding="utf-8") if l.strip()]
    backup = {}
    if os.path.exists(BAK_TRUNC):
        backup = {r["id"]: r for r in
                  (json.loads(l) for l in io.open(BAK_TRUNC, encoding="utf-8")
                   if l.strip())}

    fixed, still2, bad_prefix, conflict = [], [], [], []
    for r in recs:
        m = list(r.get("maps_full_B") or [])
        if r["best_of"] != 3 or len(m) != 2:
            continue
        k = key(r.get("date"), r["team1"]["name"], r["team2"]["name"])
        # ⚠ 优先级：**截断前快照优先**。它是 10-14 的状态，反映了主理人在
        #   Mimo 明细（10-12）之后对图序做过的更正（如 _apply_eplfix/_apply_mapfix）。
        #   拿 10-12 的旧档去覆盖 10-14 的更正 = 用旧数据盖新数据。
        #   只有快照缺失/不匹配时才退回原始明细。两者都要**前缀一致**才敢追加。
        cands = []
        if r["id"] in backup and len(backup[r["id"]].get("maps_full_B") or []) == 3:
            cands.append((backup[r["id"]]["maps_full_B"], "截断前快照(10-14)"))
        if k in orig:
            cands.append((orig[k], "原始明细:" + "+".join(sorted(prov[k]))))
        if not cands:
            still2.append((r, k))
            continue
        hit = [(sm, w) for sm, w in cands if sm[:2] == m]
        if not hit:
            bad_prefix.append((r, m, [c[0] for c in cands], [c[1] for c in cands]))
            continue
        src_map, where = hit[0]
        if r.get("veto_order") == m and r.get("veto_order") != src_map:
            # veto_order 是纯图名列表形式 → 同步补第 3 张（顺序 = 主理人列图顺序）
            r["veto_order"] = src_map
        r["maps_full_B"] = src_map
        r["maps_played"] = m                      # 读法 B 全集 vs 实际打过
        r["restored_from"] = where
        fixed.append((r, where))

    print("可恢复（现 2 图、原始数据有 3 图且前 2 张一致）：%d 场" % len(fixed))
    print("  来源分布：%s" % dict(collections.Counter(w for _r, w in fixed)))
    print("恢复后 maps_verified 分布：%s"
          % dict(collections.Counter(bool(r.get("maps_verified")) for r, _w in fixed)))
    if bad_prefix:
        print("\n⚠ 前 2 张与任何权威来源都对不上、**未动**的 %d 场"
              "（图序本身有分歧，需主理人裁决）：" % len(bad_prefix))
        for r, m, sms, whs in bad_prefix[:8]:
            print("    %-28s 现 %s" % (r["id"], m))
            for sm, w in zip(sms, whs):
                print("        候选 %s（%s）%s"
                      % (sm, w, "← 前缀一致" if sm[:2] == m else ""))
    if still2:
        print("\n⚠ 主理人原始数据里也只有 2 张、**无法恢复**的 %d 场：" % len(still2))
        for r, k in still2:
            print("    %-30s %s %d-%d %s  （缺决胜图，需主理人补）"
                  % (r["id"], r["team1"]["name"], r["team1"]["score"],
                     r["team2"]["score"], r["team2"]["name"]))

    ok = bad = 0
    probs = []
    for r in recs:
        good, p, _f = RS.validate(r, strict=False)
        if good:
            ok += 1
        else:
            bad += 1
            probs.append((r["id"], p))
    print("\n恢复后 schema 校验：通过 %d / 不通过 %d" % (ok, bad))
    for i, p in probs[:5]:
        print("    ✗ %s → %s" % (i, p))

    if not apply_:
        print("\n⚠ 干跑结束，**未写盘**。加 --apply 才写（会自动备份）。")
        return 0
    shutil.copy2(SRC, OUT_SRC)
    print("\n已备份 → %s" % os.path.basename(OUT_SRC))
    with io.open(SRC, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("已写 %s（%d 条，恢复 %d 场）" % (os.path.basename(SRC), len(recs), len(fixed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())