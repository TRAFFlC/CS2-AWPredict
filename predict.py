#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""predict.py —— issue-014 的 **模拟预测工具**（2026-10-10）。

主理人要求：手动输入对战双方 → 模型给出 **选图 / 胜方 / 比分 / MVP**；
输入真实结果后 **自动计分**；并能按「最高分」与「最高排名」两个目标出两版预测。

命令
----
  selftest                     离线自证（不读大数据）
  match --A 队A --B 队B [--bo 3] [--date 2026-10-12] [--objective score|rank]
                                出单场四项预测（含概率与理由）
  score --file results.json    批量：读「对战+真实结果」，自动计分并汇总
  interact [--session out.json]
                                交互式：逐场输入双方→出预测；再输入真实结果→自动计分
  backtest [--n 60]            把本工具走一遍 test，核对是否复现已离线回测的四项分
  field [--opponents 20]       打印「场（对手）模型」的每场矩，用于 --objective rank

`results.json` / session 文件格式
--------------------------------
{
  "matches": [
    {"A": "G2", "B": "NAVI", "bo": 3,
     "winner": "G2", "score": [2, 1], "maps": ["Mirage","Nuke","Ancient"], "mvp": "m0NESY"}
  ]
}
`winner/score/maps/mvp` 可留空表示「尚未有结果」（只出预测、不计分）。

计分口径完全沿用 `014-scorer/scorer.py`（bo1 25 分、bo3 45 分、bo5 55 分；
选图读法 B：真值 = veto 后全部图，含未打的决胜图）。
"""
from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")  # # 仓库布局：数据与代码分目录（Alpha 1.0）
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)  # scorer.py 与本模块同目录（仓库布局）

import veto_model as VM  # noqa: E402
import model_v2 as M  # noqa: E402
import decide as D  # noqa: E402
import coldstart as C  # noqa: E402
import split as SPLIT  # noqa: E402
import pool_experiment as PE  # noqa: E402
from scorer import Prediction, Truth, score_match, max_total  # noqa: E402


# ------------------------------------------------------------------ 预测器
class Predictor:
    """把 v2 的四项模型 + 冷门映射器打包成一个可直接调用的对象。"""

    def __init__(self, quiet=False):
        # **单一建模路径**：与 `model_v2.run` / `decide.py` 同一套选型与最终模型
        self.prep = M.prepare(verbose=False)
        self.rows = self.prep["rows"]
        self.meta = self.prep["meta"]
        self.data = self.prep["data"]
        self.state = self.prep["state"]
        self.mh = self.prep["mh"]
        self.ranks = self.prep["ranks"]
        self.wmodel = self.prep["wmodel"]
        self._rank_idx = None          # 惰性构建：csapi rank 快照索引
        self._rank_snap = ""
        # ⚠ 扩展名 .pkl 不是笔误：这个文件**曾是 state_2026-10.json**，
        #   但内容其实是 pickle 协议 4（头 \x80\x04\x95），任何 json.load 都会炸。
        #   已改名，与 predict_snapshot.pkl / state_snapshot_live.pkl 口径一致。
        # 增量快照（state_snapshot.ingest_new.py 于 2026-10-12 摄入 110 场，
        # 推进到 2026-10-11）。没有它，每次预测都只用 09-05 的 prepare() 状态，
        # 入库等于白做。常数漂移由 from_blob() 当场报错（硬坑 B），不回退。
        self.snapshot = None
        self.snapshot_note = None
        snap = os.path.join(DATA, "state_2026-10.pkl")
        if os.path.exists(snap) and os.environ.get("ISSUE014_NO_SNAPSHOT") != "1":
            import pickle
            import state_snapshot as SS
            with open(snap, "rb") as f:
                blob = pickle.load(f)
            s_new, mh_new, cur, sha = SS.from_blob(blob)
            self.snapshot = {"cursor": cur, "sha256": sha, "blob": blob,
                             "state": s_new, "mh": mh_new}
            # 账本换新：**特征读取口必须一并换**，否则 state/mh 更新不生效
            self.state = s_new
            self.mh = mh_new
            self.last_date = s_new.last_date
            self.snapshot_note = ("已载入增量快照 %s（游标 %s/%s，state.last_date=%s）"
                                  % (os.path.basename(snap), cur.get("last_date"),
                                     cur.get("last_id"), s_new.last_date))
        else:
            self.snapshot_note = "未载入增量快照（ISSUE014_NO_SNAPSHOT=%s 或文件缺失）" % (
                os.environ.get("ISSUE014_NO_SNAPSHOT", "0"),)
        self.mmodel = self.prep["mmodel"]
        self.winfeats = self.prep["winfeats"]
        self.score_name = self.prep["score_name"]
        wsel = self.prep["wsel"]
        for sp in self.data["train"]:
            sp["pA"] = M.predict_proba_win(wsel, sp)
        for k in ("valid", "test"):
            for sp in self.data[k]:
                sp["pA"] = M.predict_proba_win(self.wmodel, sp)
        self.smodel = D.ScoreModel.fit(self.data["train"])
        self.pool = C.current_pool(self.mh)
        self.profiles = C.all_profiles(self.state, self.mh, self.pool)
        self._slope, self._intercept = C.rating_to_elo_slope(self.profiles)
        # ⚠ 快照账本日期优先：本行原来无条件覆盖 line 89，把 last_date 打回
        # 语料末日 09-05 ⇒ 实时预测的 date 依赖特征（阵容时点/休整天数/先手方）
        # 全部停在 09-05，增量更新等于对这些特征失效（2026-10-13 实测）。
        _dates = [max(r["date"] for r in self.rows)]
        if self.snapshot:
            _dates.append(self.snapshot["state"].last_date)
        self.last_date = max(_dates)
        self._injected = set()
        if not quiet:
            print("[predict] 模型就绪：已知队伍 %d 支，当前池 %s" % (len(self.profiles), ",".join(self.pool)))

    # ---- 冷门队伍 ----
    def resolve_team(self, name, profile=None, quiet=True):
        """返回 (真实队名, 是否为冷门注入)。已注入过的队不重复注入。"""
        known = self.state.games.get(name, 0) >= 3 or name in self._injected
        if known:
            return name, name in self._injected
        if not profile:
            if not quiet:
                print("  [冷门] 「%s」历史不足 3 场且未提供档案 ⇒ 用默认 Elo 1500" % name)
            return name, False
        mp = C.map_team(dict(profile, name=name), self.profiles, self.pool, self._slope, self._intercept)
        C.apply_mapping(self.state, self.mh, mp)
        self._injected.add(name)
        if not quiet:
            print("  [冷门] 「%s」→ 最相似已知队伍「%s」（相似度 %.3f），注入 Elo %.0f" % (
                name, mp["donor"], mp["top"][0][1] if mp["top"] else 0.0, mp["elo"]))
        return name, True

    # ---- 构造单场样本 ----
    def live_sample(self, A, B, bo=3, date=None):
        date = date or self.last_date
        pool = self.pool
        ea, _ = self.state.elo_of(A)
        eb, _ = self.state.elo_of(B)
        ma, _ = self.state.elo_m_of(A)
        mb, _ = self.state.elo_m_of(B)
        ra, pa = self.ranks.rank_points(A, date)
        rb, pb = self.ranks.rank_points(B, date)
        la, lb = self.state.lineup_strength(A, date), self.state.lineup_strength(B, date)
        ca, cb = self.state.churn_rate(A, date), self.state.churn_rate(B, date)
        fa, fb = self.state.form(A), self.state.form(B)
        rest_a, rest_b = self.state.rest_days(A, date), self.state.rest_days(B, date)
        feats = {"d_elo": (ea - eb) / 400.0, "d_elom": (ma - mb) / 400.0,
                 "d_rank": float(rb - ra), "d_logp": math.log1p(pa) - math.log1p(pb),
                 "d_wr": self.state.winrate(A) - self.state.winrate(B),
                 "d_h2h": self.state.h2h_rate(A, B) - self.state.h2h_rate(B, A),
                 "d_poolwr": self.state.pool_wr(A, pool) - self.state.pool_wr(B, pool),
                 "d_lineup": (la - lb) if (la is not None and lb is not None) else 0.0,
                 "a_churn": ca, "b_churn": cb, "d_churn": ca - cb, "sum_churn": ca + cb,
                 "d_form": fa - fb,
                 "d_rest": math.log1p(min(rest_a, 180.0)) - math.log1p(min(rest_b, 180.0)),
                 "d_games": self.state.exp_log(A) - self.state.exp_log(B)}
        map_rows = self.mh.features_for(A, B, pool, bo, date=date)
        player_rows = {}
        for team, opp in ((A, B), (B, A)):
            lineup = self.state.lineup(team, date)
            opp_elo = self.state.elo_of(opp)[0]
            player_rows[team] = [dict(self.state.player_row(team, p, date, opp_elo, lineup), label=0)
                                 for p in lineup]
        pA = M.predict_proba_win(self.wmodel, {"feats": feats})
        return {"teamA": A, "teamB": B, "bo": bo, "pA": pA, "map_rows": map_rows,
                "player_rows": player_rows, "truth": None, "mvp_truth": "",
                "feats": feats, "elo": {A: ea, B: eb}, "elom": {A: ma, B: mb}}

    # ---- 出预测 ----
    def predict_from_sample(self, sample, objective="score", rank_field=None,
                            opponents=20, noise=0.25):
        """对「已备好的单场样本」出预测（live 与 backtest 共用同一条代码路径）。

        `opponents` / `noise`：**逐场**对手场的人数与噪声（主理人 2026-10-11：参赛人数按比赛不同
        ⇒ 这是参数，默认 20；对外一律同时报「期望分 + 该人数下的期望名次」）。
        """
        md = D.MatchDist(sample, self.mmodel, self.smodel)
        plan = D.choose_score(md)
        er = None
        if objective == "rank":
            # **逐场**对手场：对手也只为这一场提交 ⇒ 名次是 1..N+1（而非「整批赛场」的名次）。
            field = rank_field or D.Field(opponents, noise).fit([md])
            plan = D.choose_rank(md, 0, [md.moments(plan)], field)
            er = field.expected_rank(*md.moments(plan))

        # ---- 胜方项：**遵循 rank**（2026-10-13 复测定案，不做置信度门控）----
        # 2026-10-13 时间前进式复测（exp_walkforward_winner.py / walkforward_winner_results.json）：
        # 112 场 09-08~10-11，按日推进——每场只用**更早日期**的状态预测后再摄入当日，
        # 全部 out-of-sample、无同日泄漏；且**在增量数据已摄入的模型上重跑**，
        # 补上了 2026-10-12 结论「用的是停在 09-05 的模型」那个漏洞：
        #   rank 热门        75/112 = 67.0%      ← 胜方
        #   模型·更新        71/112 = 63.4%      McNemar vs rank：仅模型对 7 / 仅 rank 对 11，p=0.481
        #   模型·冻结 09-05  71/112 = 63.4%      与更新臂只差 2 场（净 0）⇒ 新数据**已摄入**
        #                                        （账本 09-05→10-11）但对胜方 argmax 几乎无影响：
        #                                        胜方模型权重与 d_rank 特征都是冻结的
        #   置信度门控 t=0.50~0.90（9 档）：最高 67.0% = rank，**没有任何一档超过 rank**
        #   分歧 18 场：rank 11 对 vs 模型 7 对（p=0.481）
        #   分层：9 月 71 场 rank 71.8% vs 模型 64.8%；10 月 41 场（快照已陈旧 25~36 天）
        #         rank 58.5% vs 模型 61.0%（仅差 1 场）⇒ 陈旧度确实在吃掉 rank 的优势，
        #         但样本 41 场、差 1 场，**尚不足以翻案**
        # ⇒ 裁决：**遵循 rank**；rank 缺失才回退模型（rank_pick 返回 None 时自动回退）。
        #    模型输出仍保留在 disagreement 字段，供人工复核与后续检验；
        #    rank 快照 >60 天（RANK_DEAD_D）或 csapi 恢复重抓后，必须重跑本实验再定。
        # 旧证据（2026-10-12，43 场，rank 69.8% vs 模型 62.8%）方向一致，保留备查。
        #
        # ★ 换排名**体系**并不会更好（2026-10-14 补，无泄漏口径 69 场）：
        #   csapi/VRS 09-05   50/69 = 72.5%   ← 现用
        #   HLTV       09-07  43/69 = 62.3%   ← 本地有一份更新的 HLTV 快照，反而低 10.2pp
        #   McNemar p=0.065（csapi 赢 9 场、输 2 场）
        #   （HLTV 10-05 能刷出 73.9%，但那是**前视偏差**：该快照已吃进被评比赛的
        #    赛果，不可用于评估——这正是"拿最新排名回头评旧比赛"最容易自欺的地方。）
        # ⇒ 差距来自**体系不同**而非「数据旧」。所以陈旧度确实在衰减 rank（见上），
        #   但**换源不是解法**；csapi 恢复后重抓的价值是消除陈旧，不是换算法，
        #   且重抓后必须重跑本实验再定，不能默认变好。
        model_winner = plan["winner"]
        rank_winner, rank_info = self.rank_pick(sample, model_winner)
        if rank_winner and rank_winner != model_winner:
            # 换胜方必须**连带重算比分与 MVP**：scorer 要求 ①比分隐含的胜方 ==
            # pred.winner（否则第 2 项整项 0，scorer.py:144）②MVP 出自预测胜方
            # （否则第 4 项 0）。第一版只翻了比分没动 MVP ⇒ 分歧场次 MVP 必 0。
            plan = D.retarget(md, plan, rank_winner)

        mu, var = md.moments(plan)
        A, B, bo = sample["teamA"], sample["teamB"], sample["bo"]
        out = {
            "A": A, "B": B, "bo": bo, "objective": objective,
            "pA": sample["pA"], "win_prob": {A: sample["pA"], B: 1 - sample["pA"]},
            "winner": plan["winner"], "score": list(plan["score"]),
            "maps": list(plan["maps"]), "mvp": plan["mvp"],
            "model_winner": model_winner,
            "disagreement": bool(rank_winner and rank_winner != model_winner),
            "rank_src": rank_info,
            "map_probs": {r["map"]: None for r in sample["map_rows"]},
            "exp_points": mu, "sd": math.sqrt(var), "exp_rank": er,
            "rank_field_n": (len(field.moments) + 1) if er is not None else None,
            "cold": {},
        }
        _picked, probs = VM.predict_set(self.mmodel, {"bo": bo, "rows": sample["map_rows"]}, max(1, min(bo, 5)))
        out["map_probs"] = probs
        out["mvp_probs"] = D.mvp_probs(
            sample["player_rows"].get(plan["winner"]) or [], bo=bo)
        out["score_probs"] = md.score_true.get(plan["winner"], {})
        return out

    def rank_pick(self, sample, model_winner):
        """返回 (rank 更靠前的队, 说明 dict)。查不到任一方则返回 (None, ...)。

        **rank = Valve Regional Standings (VRS)**，两级来源：

        1. `vrs_rank_<日期>.json`（**首选**，用户提供的 VRS 榜单快照）
           - 体系与 csapi 相同（csapi 的 `/rankings/` 端点自述即 VRS），
             故「VRS 胜过模型」这个 72.5% 的验证结论**适用于它**
           - 唯一区别是**新鲜度**：实测 09-05→10-05 期间 129 支队只有 3 支名次
             未变、126 支都动（最大 159 位）⇒ 同体系的新快照严格更优
        2. `csapi_matches_all.jsonl` 内嵌 rank（**兜底**，09-05 冻结快照）
           - 仅填补新榜未收录的队伍，覆盖率 120/179 → 179/179
           - 触发兜底时 `mixed=True`，调用方可据此下调对本次判定的信任

        ⚠ **只做规范化精确匹配（小写 + 去非字母数字），绝不模糊匹配**。
          模糊匹配实测会产出 `Jam→Ninjas in Pyjamas`、`ASTRAL→Astralis`、
          `MAGICOS→magic` 这类**把强队名次安给无关队伍**的错误，代价远大于漏匹配。
        """
        if self._rank_idx is None:
            self._build_rank()
        ka, kb = self._norm(sample["teamA"]), self._norm(sample["teamB"])
        ra, rb = self._rank_idx.get(ka), self._rank_idx.get(kb)
        if ra is None or rb is None:
            missing = [n for n, k, v in ((sample["teamA"], ka, ra),
                                         (sample["teamB"], kb, rb)) if v is None]
            return None, {"ok": False, "why": "rank 缺失: %s" % ",".join(missing)}
        win = sample["teamA"] if ra < rb else sample["teamB"]
        sa, sb = self._rank_src.get(ka), self._rank_src.get(kb)
        return win, {"ok": True, "rankA": ra, "rankB": rb,
                     "snapshot": self._rank_snap, "srcA": sa, "srcB": sb,
                     "mixed": bool(sa != sb)}

    @staticmethod
    def _norm(t):
        return "".join(ch for ch in str(t).lower() if ch.isalnum())

    def _build_rank(self):
        """加载 VRS 排名：新快照优先，csapi 快照仅作兜底填补。"""
        here = DATA  # 仓库布局：VRS 排名与 csapi 兜底档都在 data/
        idx, src = {}, {}
        # ---- 1) VRS 用户快照（越新越优先）----
        best_date = ""
        for vf in sorted(glob.glob(os.path.join(here, "vrs_rank_*.json"))):
            try:
                d = json.load(open(vf, encoding="utf-8"))
            except Exception:                              # noqa: BLE001
                continue
            date = str(d.get("date") or "")
            if date < best_date:                           # 只用最新那份
                continue
            if date > best_date:
                best_date = date
                idx, src = {}, {}
            for t in d.get("teams") or []:
                nm, rk = t.get("name"), t.get("rank")
                if nm and rk:
                    idx[self._norm(nm)] = int(rk)
                    src[self._norm(nm)] = "VRS " + date
        n_vrs = len(idx)
        # ---- 2) csapi 兜底（只填空位，绝不覆盖新榜）----
        p = os.path.join(here, "csapi_matches_all.jsonl")
        if os.path.exists(p):
            rows = []
            mx = ""
            for line in io.open(p, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                d = str(r.get("date") or "")
                mx = max(mx, d)          # ⚠ 先扫完全文取全局最大日期
                rows.append(r)            #    不能用「读到该行时的累计最大值」
            # 实测 csapi_matches_all.jsonl 内179 支队 rank 零变化 ⇒ 是一次性抓取，
            # 所以兜底项**统一标注同一个日期**。若按行内累计日期标注，会给
            # ASTRAL 标成 2025-11-20、SPARTA 标成 2025-10-19，纯属读取顺序假象。
            for r in rows:
                for t in (r["team1"], r["team2"]):
                    if not t.get("rank"):
                        continue
                    k = self._norm(t["name"])
                    if k not in idx:
                        idx[k] = int(t["rank"])
                        src[k] = "csapi " + mx
            self._csapi_snap = mx
        self._rank_idx, self._rank_src = idx, src
        self._rank_snap = max(best_date, getattr(self, "_csapi_snap", ""))

    def predict(self, A, B, bo=3, date=None, objective="score", rank_field=None,
                profileA=None, profileB=None, quiet=True, opponents=20, noise=0.25):
        A, coldA = self.resolve_team(A, profileA, quiet)
        B, coldB = self.resolve_team(B, profileB, quiet)
        sample = self.live_sample(A, B, bo, date)
        out = self.predict_from_sample(sample, objective, rank_field,
                                       opponents=opponents, noise=noise)
        out["cold"] = {A: coldA, B: coldB}
        return out

    # ---- 场（对手）模型 ----
    def field(self, opponents=20, noise=0.25):
        """**赛场级**对手场：把整个 test 当成一场竞赛（对手总分 = 317 场之和）。

        用于 `field` 命令与「竞赛级」诊断；**单场预测请用 `match_field`**。
        """
        mds = [D.MatchDist(s, self.mmodel, self.smodel) for s in self.data["test"]]
        return D.Field(opponents, noise).fit(mds)

    def match_field(self, md, opponents=20, noise=0.25):
        """**逐场**对手场：对手只为这一场提交 ⇒ 名次 1..N+1。"""
        return D.Field(opponents, noise).fit([md])


# ------------------------------------------------------------------ 打印
def fmt_prediction(p):
    L = []
    L.append("  胜方: %-24s  P=%.1f%%   (对家 %.1f%%)" % (
        p["winner"], 100 * max(p["pA"], 1 - p["pA"]), 100 * min(p["pA"], 1 - p["pA"])))
    L.append("  比分: %d:%d   (bo%d)" % (p["score"][0], p["score"][1], p["bo"]))
    L.append("  选图: %s" % " / ".join(p["maps"]))
    L.append("  MVP : %s" % (p["mvp"] or "（无选手数据）"))
    if p.get("map_probs"):
        top = sorted(p["map_probs"].items(), key=lambda kv: -kv[1])[:5]
        L.append("       图集概率: " + "  ".join("%s %.2f" % (m, pr) for m, pr in top))
    if p.get("mvp_probs"):
        top = sorted(p["mvp_probs"].items(), key=lambda kv: -kv[1])[:3]
        # ⚠ 这行概率**不可当作把握程度**：候选池是胜方 5 名选手（随机基线 0.20）。
        # 实测命中率**按选手历史丰度分档**，不是单一数字：
        #   丰度 >= decide.MVP_RICH_MIN(18)：真实命中 0.39~0.41
        #   丰度 < 18              ：真实命中 0.12~0.15 ≈ 随机
        # 冷门队几乎总是落在低丰度档 ⇒ 这一行对它们基本没有信息量。
        # `decide.MVP_CAL_*` 已按此分档标定，故 E[分] 不受此影响；
        # 但**不要**把这些数当成「我有多确信」。
        L.append("       MVP 概率(未标定,仅供参考): "
                 + "  ".join("%s %.2f" % (m, pr) for m, pr in top)
                 + "   ⚠ 实际命中率按丰度分档 0.31/0.20，见 decide.MVP_CAL_RICH/POOR")
    L.append("  期望得分 %.1f（±%.1f）%s" % (
        p["exp_points"], p["sd"],
        "" if p["exp_rank"] is None else "，本场提交名次 %.2f（1 = 最好；共 %d 人）"
        % (p["exp_rank"], p.get("rank_field_n", 21))))
    if any(p["cold"].values()):
        L.append("  冷门注入: " + ", ".join(t for t, v in p["cold"].items() if v))
    return "\n".join(L)


def score_one(entry):
    """entered = {"A","B","bo","winner","score","maps","mvp"} → (Prediction, Truth)"""
    truth = Truth(teamA=entry["A"], teamB=entry["B"], winner=entry["winner"],
                  score=tuple(entry["score"]), maps=list(entry.get("maps") or []),
                  mvp=entry.get("mvp") or "", best_of=entry["bo"])
    return truth


# ------------------------------------------------------------------ 命令
def cmd_match(args):
    pred = Predictor()
    profA = profB = None
    if args.profileA:
        profA = json.load(io.open(args.profileA, encoding="utf-8"))
    if args.profileB:
        profB = json.load(io.open(args.profileB, encoding="utf-8"))
    p = pred.predict(args.A, args.B, args.bo, args.date, args.objective,
                     profileA=profA, profileB=profB, quiet=False,
                     opponents=args.opponents, noise=args.noise)
    print("")
    print("== %s vs %s （bo%d，目标=%s）==" % (p["A"], p["B"], p["bo"], p["objective"]))
    print(fmt_prediction(p))


def cmd_score(args):
    doc = json.load(io.open(args.file, encoding="utf-8"))
    pred = Predictor()
    total = mx = 0
    rows = []
    for m in doc["matches"]:
        p = pred.predict(m["A"], m["B"], m.get("bo", 3), m.get("date"), args.objective, quiet=True)
        if not m.get("winner"):
            rows.append((m, p, None))
            continue
        truth = score_one(m)
        b = score_match(Prediction(winner=p["winner"], score=tuple(p["score"]),
                                   maps=list(p["maps"]), mvp=p["mvp"]), truth)
        total += b.total
        mx += max_total(truth)
        rows.append((m, p, b))
    for m, p, b in rows:
        print("")
        print("-- %s vs %s (bo%s)" % (m["A"], m["B"], m.get("bo", 3)))
        print(fmt_prediction(p))
        if b is None:
            print("  （无真实结果，未计分）")
        else:
            ok = "✓" if b.winner else "✗"
            print("  计分: 胜方 %s %2d ｜ 比分 %2d ｜ 选图 %2d %s ｜ MVP %2d ｜ 合计 %2d / %d" % (
                ok, b.winner, b.score, b.maps, b.maps_hit, b.mvp, b.total, max_total(score_one(m))))
    if mx:
        print("")
        print("== 合计 %d / %d = %.1f%% （%d 场）==" % (total, mx, 100.0 * total / mx, len(rows)))


def cmd_interact(args):
    pred = Predictor()
    session = {"matches": []}
    total = mx = 0
    print("\n交互模式：依次输入 队A / 队B / bo（直接回车=3）。输入 q 结束。")
    while True:
        try:
            A = input("\n队A> ").strip()
        except EOFError:
            break
        if not A or A.lower() in ("q", "quit", "exit"):
            break
        B = input("队B> ").strip()
        if not B:
            continue
        bo_s = input("bo (1/3/5，回车=3)> ").strip()
        bo = int(bo_s) if bo_s in ("1", "3", "5") else 3
        p = pred.predict(A, B, bo, None, args.objective, quiet=False)
        print("")
        print("== %s vs %s （bo%d，目标=%s）==" % (p["A"], p["B"], bo, args.objective))
        print(fmt_prediction(p))
        print("\n输入真实结果（可直接回车跳过计分）。")
        w = input("胜方 (回车=%s)> " % p["winner"]).strip()
        if not w:
            session["matches"].append({"A": A, "B": B, "bo": bo, **{k: p[k] for k in ("winner", "maps", "mvp")},
                                       "score": p["score"], "predicted_only": True})
            continue
        sc = input("比分 (如 2:1，bo1 可回车)> ").strip()
        scoreline = p["score"]
        if ":" in sc or "：" in sc:
            a, b2 = sc.replace("：", ":").split(":")
            scoreline = [int(a), int(b2)]
        else:
            scoreline = ([1, 0] if w == A else [0, 1]) if bo == 1 else \
                        ([2, 1] if w == A else [1, 2]) if bo == 3 else ([3, 1] if w == A else [1, 3])
        mp = input("选图 (逗号分隔，回车=不填)> ").strip()
        maps = [x.strip() for x in mp.replace("，", ",").split(",") if x.strip()]
        mvp = input("MVP (回车=不填)> ").strip()
        entry = {"A": A, "B": B, "bo": bo, "winner": w, "score": scoreline,
                 "maps": maps, "mvp": mvp}
        truth = score_one(entry)
        b = score_match(Prediction(winner=p["winner"], score=tuple(p["score"]),
                                   maps=list(p["maps"]), mvp=p["mvp"]), truth)
        total += b.total
        mx += max_total(truth)
        print("  本场计分: 胜方 %2d ｜ 比分 %2d ｜ 选图 %2d ｜ MVP %2d ｜ 合计 %2d / %d" % (
            b.winner, b.score, b.maps, b.mvp, b.total, max_total(truth)))
        print("  累计: %d / %d = %.1f%%" % (total, mx, 100.0 * total / mx))
        session["matches"].append(dict(entry, predicted={"winner": p["winner"], "score": p["score"],
                                                         "maps": p["maps"], "mvp": p["mvp"]},
                                       points=b.total))
    if args.session:
        with io.open(args.session, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(session, ensure_ascii=False, indent=1) + "\n")
        print("\n已写入 %s（%d 场）" % (args.session, len(session["matches"])))
    if mx:
        print("合计 %d / %d = %.1f%%" % (total, mx, 100.0 * total / mx))


def cmd_backtest(args):
    """把本工具走一遍 test，核对是否复现离线回测的四项分。"""
    pred = Predictor(quiet=True)
    agg = collections.Counter()
    res = {k: None for k in ("winner", "score", "maps", "mvp", "total")}
    n = 0
    for s in pred.data["test"]:
        t = s["truth"]
        if not t:
            continue
        p = pred.predict_from_sample(s, "score")
        truth = M.to_truth(t, s["mvp_truth"])
        b = score_match(Prediction(winner=p["winner"], score=tuple(p["score"]),
                                   maps=list(p["maps"]), mvp=p["mvp"]), truth)
        agg["winner"] += b.winner
        agg["score"] += b.score
        agg["maps"] += b.maps
        agg["mvp"] += b.mvp
        agg["total"] += b.total
        sc_max = 10 if s["bo"] in (3, 5) else 0
        agg["max"] += max_total(truth)
        agg["winner_max"] += 10
        agg["score_max"] += sc_max
        agg["maps_max"] += max_total(truth) - 20 - sc_max
        agg["mvp_max"] += 10
        n += 1
        if args.n and n >= args.n:
            break
    print("== predict.py backtest（test 前 %d 场）==" % n)
    for k in ("winner", "score", "maps", "mvp"):
        print("  %-7s %6d / %6d = %5.1f%%" % (k, agg[k], agg[k + "_max"],
                                               100.0 * agg[k] / agg[k + "_max"] if agg[k + "_max"] else 0))
    print("  TOTAL   %6d / %6d = %5.1f%%" % (agg["total"], agg["max"],
                                             100.0 * agg["total"] / agg["max"] if agg["max"] else 0))


def cmd_field(args):
    pred = Predictor(quiet=True)
    f = pred.field(args.opponents)
    print("== 场（对手）模型：%d 人，策略=%s ==" % (args.opponents, f.assign[:5]))
    for i, (mu, var) in enumerate(f.moments[:8]):
        print("  对手#%-2d 期望总分 %6.1f  sd %5.1f" % (i, mu, math.sqrt(var)))


# ------------------------------------------------------------------ selftest
def _selftest():
    fails = []
    n_chk = [0]                      # 2026-10-11：此前打印的是**硬编码 5**

    def chk(name, got, want):
        n_chk[0] += 1
        if got != want:
            fails.append("%s: got=%r want=%r" % (name, got, want))

    # 计分口径（与 014-scorer 一致）
    t = Truth("A", "B", "A", (2, 1), ["Mirage", "Nuke", "Inferno"], "star", best_of=3)
    b = score_match(Prediction("A", (2, 1), ["Mirage", "Nuke", "Inferno"], "star"), t)
    chk("perfect bo3", b.total, 45)
    chk("max_total bo3", max_total(t), 45)
    chk("max_total bo1", max_total(Truth("A", "B", "A", (1, 0), ["Dust2"], "x", best_of=1)), 25)

    # 预测格式化不炸
    fake = {"A": "X", "B": "Y", "bo": 3, "pA": 0.7, "winner": "X", "score": [2, 1],
            "maps": ["Mirage", "Nuke", "Inferno"], "mvp": "p1",
            "map_probs": {"Mirage": 0.7}, "mvp_probs": {"p1": 0.6},
            "exp_points": 20.0, "sd": 8.0, "exp_rank": None, "cold": {"X": False, "Y": False}}
    chk("fmt has winner", "X" in fmt_prediction(fake), True)
    chk("fmt has maps", "Inferno" in fmt_prediction(fake), True)

    # 换胜方必须**连带重算比分与 MVP**（rank-first 覆盖模型胜方时走 D.retarget），
    # 否则分歧场次的 MVP 出自对手队伍 ⇒ 第 4 项必然 0 分（2026-10-13 实测到的真 bug）。
    class _MDStub(object):
        bo, A = 3, "X"
        score_true = {"Y": {(0, 2): 0.6, (1, 2): 0.4}}
        mvp_true = {"Y": {"py1": 0.7, "py2": 0.3}}
    _rp = D.retarget(_MDStub, {"winner": "X", "score": (2, 1),
                               "maps": ("Mirage", "Nuke", "Inferno"), "mvp": "px"}, "Y")
    chk("retarget: MVP 改到新胜方", _rp["mvp"], "py1")
    chk("retarget: 比分取新胜方众数（A/B 朝向）", _rp["score"], (0, 2))
    chk("retarget: 选图不动", _rp["maps"], ("Mirage", "Nuke", "Inferno"))
    chk("retarget: 胜方一致时原样返回",
        D.retarget(_MDStub, {"winner": "Y", "score": (0, 2), "mvp": "py1"}, "Y")["mvp"], "py1")

    # 接线守卫（同 model_v2）：map 特征必须走 features_for（带日期）
    _src = io.open(os.path.abspath(__file__), encoding="utf-8").read()
    _needle = "." + "features("       # 拼接以免守卫命中它自己的检查名
    chk("predict 不裸调 features()（须走 features_for）",
        _needle not in _src, True)

    if fails:
        print("PREDICT_SELFTEST_FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("PREDICT_SELFTEST_OK (%d checks)" % n_chk[0])
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(_selftest())
    ap = argparse.ArgumentParser(prog="predict.py")
    sub = ap.add_subparsers(dest="cmd")
    m = sub.add_parser("match")
    m.add_argument("--A", required=True)
    m.add_argument("--B", required=True)
    m.add_argument("--bo", type=int, default=3)
    m.add_argument("--date", default=None)
    m.add_argument("--objective", choices=["score", "rank"], default="score")
    m.add_argument("--profileA", default=None)
    m.add_argument("--profileB", default=None)
    m.add_argument("--opponents", type=int, default=20,
                   help="逐场对手场人数（rank 目标用；主理人 2026-10-11：参赛人数按比赛不同）")
    m.add_argument("--noise", type=float, default=0.25, help="对手场噪声（rank 目标用）")
    m.set_defaults(func=cmd_match)

    s = sub.add_parser("score")
    s.add_argument("--file", required=True)
    s.add_argument("--objective", choices=["score", "rank"], default="score")
    s.set_defaults(func=cmd_score)

    i = sub.add_parser("interact")
    i.add_argument("--objective", choices=["score", "rank"], default="score")
    i.add_argument("--session", default=None)
    i.set_defaults(func=cmd_interact)

    bt = sub.add_parser("backtest")
    bt.add_argument("--n", type=int, default=60)
    bt.set_defaults(func=cmd_backtest)

    fl = sub.add_parser("field")
    fl.add_argument("--opponents", type=int, default=20)
    fl.set_defaults(func=cmd_field)

    if len(sys.argv) == 1:
        print(__doc__)
        raise SystemExit(2)
    a = ap.parse_args()
    if not getattr(a, "func", None):
        print(__doc__)
        raise SystemExit(2)
    a.func(a)
