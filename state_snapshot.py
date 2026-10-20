"""State / MapHistory 快照 + 增量更新（[P0]-1 第 1 步）

设计依据（来自两份精确清点）：
1. `State._decay`（model_v2.py:292-308）与 `MapHistory.decay`（veto_model.py:159-174）
   **按族统一乘一个标量因子**，且 `days_apart` 可加 ⇒
   `f(t1-t2)·f(t2-t3) == f(t1-t3)`，衰减是标量半群。
   ⇒ 落盘的是「截至 last_date 的真账本」，`advance(D)` 一次即可精确推进到任意未来日期。
2. **硬坑 A**：`State.__init__` 有 6 个 lambda 做 defaultdict 工厂 → `pickle` 直接失败。
   本模块**不改 model_v2.py / veto_model.py**：落盘前把 lambda-defaultdict 摊平成普通 dict，
   读回时先造一个全新 `State()`（自带正确工厂）再 bulk update。
3. **硬坑 B**：`veto_model` 的模块级常数（`K_TEAM` / `EPS_GLOBAL` / `FIRST_BAN_TEAM2` /
   `SHAPE_MIX` / `HALF_LIFE`）**不在实例上**，pickle 抓不到 ⇒ 快照必须显式钉住并校验，
   否则「旧账本 + 新常数」会被静默重新解释。
4. 查询路径会用 `defaultdict.__getitem__` **插入 0 值键**（无害，但污染快照与 diff）
   ⇒ 序列化时统一剔除 0 值项。
5. 游标：`last_date` 不够（同日不排他）。本模块另记 `ingested` 的最大 `(date, id)`。

自证 `selftest`（离线夹具）+ 等价性验证 `verify`（全语料）。
"""
import collections, hashlib, io, json, os, pickle, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_v2 as M
import veto_model as VM
import pool_experiment as PE

BLOB_VERSION = "state-snapshot/1"

# State 上需要持久化的全部实例属性（CONST 12 + 水位 1 + RAW 9 + DERIVED 13 + 注入 2）
STATE_KEYS = [
    # CONST
    "hl_hist", "hl_player", "hl_form", "k_new", "k_old", "k_sat", "elo0",
    "opp_adj", "rho", "rho_g", "churn_window", "churn_half",
    # 水位
    "last_date",
    # RAW
    "elo", "elo_m", "games", "last_match", "roster_sig", "churn",
    "p_team", "p_date", "team_players",
    # DERIVED（按族预衰减）
    "mw", "mf", "mapw", "h2h_hist",
    "p_sum", "p_w", "p_adj", "p_play", "p_top",
    "pm_sum", "pm_w", "t_level_sum", "t_level_w",
    # 注入配置
    "aliases", "injected",
    # 明确离队的选手（2026-10-19 加）：{队: {选手: 离队日期}}
    # 不持久化的话，重建快照会把「cmtry 已下放」这件事**忘掉** ⇒
    # 他又会从 240 天窗口里回到 MVP 候选池。
    "roster_departed",
]

MH_KEYS = ["uniform", "half_life", "app", "pooln", "tapp", "tpooln",
           "tban", "tpick", "gban", "gpick", "nban", "npick",
           "tw", "tp", "seen_pool", "last_date"]

# 这 6 个是 lambda 工厂的 defaultdict，必须摊平后才可 pickle
LAMBDA_DD = ("elo", "elo_m", "mw", "mf", "mapw", "h2h_hist")

# veto_model 的模块级常数：不在实例上，快照必须钉住（硬坑 B）
VM_CONSTS = ["K_TEAM", "EPS_GLOBAL", "UNIFORM_RATE", "FIRST_BAN_TEAM2",
             "FIRST_BAN_LOWER", "SHAPE_MIX", "HALF_LIFE"]
M_CONSTS = ["HALF_LIFE_HIST", "HALF_LIFE_FORM", "HALF_LIFE_PLAYER",
            "K_NEW", "K_OLD", "K_SAT", "ELO0", "OPP_ADJ", "RHO", "RHO_G",
            "CHURN_WINDOW", "CHURN_HALF", "LINEUP_WINDOW", "K_MAP"]


def _nz(v):
    """0 值剔除：列表全 0 / 数值 == 0 / 空集合"""
    if isinstance(v, list):
        return not (len(v) == 2 and v[0] == 0 and v[1] == 0)
    if isinstance(v, (int, float)):
        return v != 0
    return True


def _flat(obj, keys):
    out = {}
    for k in keys:
        v = getattr(obj, k)
        if k in LAMBDA_DD:
            out[k] = {kk: vv for kk, vv in v.items() if _nz(vv)}
        elif isinstance(v, collections.defaultdict) and v.default_factory in (int, float):
            out[k] = {kk: vv for kk, vv in v.items() if vv}
        elif isinstance(v, collections.defaultdict) and v.default_factory is list:
            out[k] = {kk: vv for kk, vv in v.items() if vv}
        elif isinstance(v, collections.defaultdict) and v.default_factory is dict:
            out[k] = {kk: vv for kk, vv in v.items() if vv}
        elif isinstance(v, collections.defaultdict) and v.default_factory is collections.Counter:
            out[k] = {kk: collections.Counter(vv) for kk, vv in v.items()
                      if any(vv.values())}
        elif isinstance(v, collections.Counter):
            out[k] = collections.Counter({kk: vv for kk, vv in v.items() if vv})
        else:
            out[k] = v
    return out


def to_blob(state, mh, cursor=None, source_sha256="", note=""):
    vm_consts = {n: getattr(VM, n, None) for n in VM_CONSTS}
    m_consts = {n: getattr(M, n, None) for n in M_CONSTS}
    return {
        "version": BLOB_VERSION,
        "cursor": cursor or {"last_date": state.last_date, "last_id": None},
        "source_sha256": source_sha256,
        "note": note,
        "state": _flat(state, STATE_KEYS),
        "mh": _flat(mh, MH_KEYS),
        "consts": {"veto_model": vm_consts, "model_v2": m_consts},
    }


def from_blob(blob):
    assert blob["version"] == BLOB_VERSION, blob["version"]
    # 硬坑 B：常数漂移必须当场报错，不许静默
    for n, v in blob["consts"]["veto_model"].items():
        cur = getattr(VM, n, None)
        if cur != v:
            raise RuntimeError("veto_model.%s 已变：快照 %r ≠ 现值 %r ⇒ 旧快照作废"
                               % (n, v, cur))
    for n, v in blob["consts"]["model_v2"].items():
        cur = getattr(M, n, None)
        if cur != v:
            raise RuntimeError("model_v2.%s 已变：快照 %r ≠ 现值 %r ⇒ 旧快照作废"
                               % (n, v, cur))
    s = M.State(half_life_hist=blob["state"]["hl_hist"],
                half_life_player=blob["state"]["hl_player"],
                half_life_form=blob["state"]["hl_form"],
                k_new=blob["state"]["k_new"], k_old=blob["state"]["k_old"],
                k_sat=blob["state"]["k_sat"], elo0=blob["state"]["elo0"],
                opp_adj=blob["state"]["opp_adj"], rho=blob["state"]["rho"],
                rho_g=blob["state"]["rho_g"],
                churn_window=blob["state"]["churn_window"],
                churn_half=blob["state"]["churn_half"])
    for k in STATE_KEYS:
        v = blob["state"][k]
        if isinstance(v, dict):
            getattr(s, k).update(v)
        else:
            setattr(s, k, v)
    h = VM.MapHistory(uniform=blob["mh"]["uniform"], half_life=blob["mh"]["half_life"])
    for k in MH_KEYS:
        cur = getattr(h, k)
        val = blob["mh"][k]
        if isinstance(cur, collections.defaultdict) and cur.default_factory is collections.Counter:
            for t, c in val.items():
                cur[t] = collections.Counter(c)
        elif isinstance(cur, set):
            cur.update(val)
        elif isinstance(cur, collections.Counter):
            cur.update(val)
        elif isinstance(cur, dict):
            cur.update(val)
        else:
            setattr(h, k, val)
    return s, h, blob["cursor"], blob.get("source_sha256", "")


# ------------------------------------------------------------------ 增量摄入

def make_ctx(truths, veto, stats, map_stats, eras, by_pool, tl):
    def pool_of(r):
        t = truths.get(r["id"]) or {}
        if t.get("pool"):
            return list(t["pool"])
        return list(PE.era_and_pool(r["date"], [], eras, by_pool, tl)[1])
    return {"truths": truths, "veto": veto, "stats": stats,
            "map_stats": map_stats or {}, "pool_of": pool_of}


def ingest(state, mh, rows, ctx, cursor=None):
    """镜像 model_v2.build() 的驱动循环（只维护账本，不抽特征）。
    逐日：先 advance(d) 一次，再 observe 当日全部比赛 —— 与 build() 的
    「同日两趟」时序安全约束一致（同日比赛互不可见）。"""
    usable = [r for r in rows
              if isinstance(r.get("best_of"), int) and r["best_of"] in (1, 3, 5)]
    order = sorted(usable, key=lambda r: (r["date"], r["id"]))
    # 游标：跳过已摄入的（date, id）
    if cursor and cursor.get("last_date") is not None:
        ld, li = cursor["last_date"], cursor.get("last_id")
        order = [r for r in order
                 if (r["date"], r["id"]) > (ld, li if li is not None else -1)]
    n = 0
    skipped_veto = 0
    i = 0
    while i < len(order):
        d = order[i]["date"]
        state.advance(d)
        mh.advance(d)
        group = []
        while i < len(order) and order[i]["date"] == d:
            group.append(order[i])
            i += 1
        for r in group:
            tA, tB = r["team1"]["name"], r["team2"]["name"]
            mres = [(m["name"], m["team1_score"], m["team2_score"])
                    for m in (r.get("maps") or [])]
            t = ctx["truths"].get(r["id"]) or {}
            state.observe(d, tA, tB, r["winner"]["name"], mres,
                          ctx["stats"].get(r["id"], {}),
                          ctx["map_stats"].get(r["id"]))
            # ⚠ 新记录（主理人手工赛果）**不在 ctx["truths"] 里** ⇒ 旧式取图会
            # 拿到空集，导致它们的选图信息整批丢失（2026-10-12 实测：112 场全丢）。
            # 故优先用记录自带的 `_maps` / `_veto`。
            ms = r.get("_maps")
            if ms is None:
                ms = t.get("maps") or []
            vr = r.get("_veto")
            if vr is None:
                vr = ctx["veto"].get(r["id"])
            # 2026-10-14：`maps_verified=false` 的记录图序是脚本猜的
            # （Ancient/Anubis/Dust2 曾反复记错），**不进选图训练**，
            # 避免污染 veto 模型。其余维度（胜负/比分/图池统计）照常更新。
            if r.get("_skip_veto"):
                skipped_veto += 1
                ms, vr = [], None
            mh.observe_match(tA, tB, ctx["pool_of"](r), ms, vr, mres)
            n += 1
    new_cursor = {"last_date": order[-1]["date"], "last_id": order[-1]["id"]} if order else cursor
    if skipped_veto:
        print("  [veto 闸门] %d 场因 maps_verified=false 未进选图训练" % skipped_veto)
    return n, new_cursor


# ------------------------------------------------------------------ 比对

def _canon(obj, keys):
    """把账本摊平成可比对的 {字段: 值} 字典。
    ⚠ 必须剔除 **0 值项**：查询路径（`pool_wr` / `winrate` / `elo_of` …）会用
    `defaultdict.__getitem__` 插入零值键（清点报告预警过）。它们数值无害但会污染
    快照与 diff，且带查询副作用的参考态必然比纯账本态多出一批零值键。"""
    out = {}
    for k in keys:
        v = getattr(obj, k)
        if k in ("mw", "mf", "mapw", "h2h_hist"):
            # lambda 工厂的 defaultdict，值是 [W, L]；双 0 视为「未观测」
            out[k] = {kk: (float(vv[0]), float(vv[1])) for kk, vv in sorted(v.items())
                      if not (vv[0] == 0 and vv[1] == 0)}
            continue
        if isinstance(v, (int, float)) or v is None:
            out[k] = v
        elif isinstance(v, (set, list, tuple)):
            out[k] = sorted(map(str, v))
        elif isinstance(v, collections.defaultdict) and v.default_factory in (int, float):
            out[k] = {kk: float(vv) for kk, vv in sorted(v.items()) if vv}
        elif isinstance(v, collections.defaultdict) and v.default_factory is list:
            out[k] = {kk: [tuple(x) for x in vv] for kk, vv in sorted(v.items()) if vv}
        elif isinstance(v, collections.defaultdict) and v.default_factory is dict:
            out[k] = {kk: {a: b for a, b in sorted(vv.items())} for kk, vv in sorted(v.items()) if vv}
        elif isinstance(v, collections.defaultdict) and v.default_factory is collections.Counter:
            out[k] = {kk: dict(sorted(vv.items())) for kk, vv in sorted(v.items())
                      if any(vv.values())}
        elif isinstance(v, collections.Counter):
            out[k] = dict(sorted(v.items()))
        elif isinstance(v, dict):
            out[k] = {kk: (sorted(map(str, vv)) if isinstance(vv, (set, list, tuple))
                           else vv) for kk, vv in sorted(v.items(), key=lambda x: str(x[0]))
                      if not (isinstance(vv, (int, float)) and vv == 0)}
        else:
            out[k] = repr(v)
    return out


def diff_ledgers(a, b, keys, tol=1e-9):
    ca, cb = _canon(a, keys), _canon(b, keys)
    worst, worst_k, nnum, ncmp = 0.0, None, 0, 0
    for k in keys:
        va, vb = ca[k], cb[k]
        if isinstance(va, float) or isinstance(vb, float):
            d = abs(float(va) - float(vb))
            nnum += 1
            if d > worst:
                worst, worst_k = d, k
        elif isinstance(va, dict) and isinstance(vb, dict):
            ncmp += len(set(va) | set(vb))
            if set(va) != set(vb):
                only_a = sorted(set(va) - set(vb))[:3]
                only_b = sorted(set(vb) - set(va))[:3]
                return float("inf"), "%s 键集不同（ref 多 %r / new 多 %r）" % (k, only_a, only_b)
            for kk in va:
                x, y = va[kk], vb[kk]
                if isinstance(x, float) or isinstance(y, float):
                    d = abs(float(x) - float(y))
                    nnum += 1
                    if d > worst:
                        worst, worst_k = d, "%s[%r]" % (k, kk)
                elif x != y:
                    return float("inf"), "%s[%r] 值不同（%r vs %r）" % (k, kk, str(x)[:50], str(y)[:50])
        elif va != vb:
            return float("inf"), "%s 结构不同（%r vs %r）" % (k, str(va)[:60], str(vb)[:60])
    return worst, ("%s 绝对差 %.3e（比对 %d 个标量字段 / %d 个 dict 项）"
                   % (worst_k or "(无差异)", worst, nnum, ncmp) if (nnum or ncmp)
                   else "无数值字段")


def selftest():
    n = 0
    s = M.State()
    h = VM.MapHistory()
    ctx = {"truths": {}, "veto": {}, "stats": {}, "map_stats": {},
           "pool_of": lambda r: ["Ancient", "Dust2", "Mirage", "Nuke"]}
    rows = [{"id": 1, "date": "2026-01-01", "best_of": 3,
             "team1": {"name": "A"}, "team2": {"name": "B"},
             "winner": {"name": "A"},
             "maps": [{"name": "Dust2", "team1_score": 13, "team2_score": 9},
                      {"name": "Mirage", "team1_score": 16, "team2_score": 12}]},
            {"id": 2, "date": "2026-01-05", "best_of": 3,
             "team1": {"name": "B"}, "team2": {"name": "C"},
             "winner": {"name": "C"},
             "maps": [{"name": "Dust2", "team1_score": 8, "team2_score": 16}]}]
    cnt, cur = ingest(s, h, rows, ctx)
    assert cnt == 2, cnt
    assert cur == {"last_date": "2026-01-05", "last_id": 2}, cur
    n += 1
    # 快照 → 还原 → 与原对象逐字段一致
    blob = to_blob(s, h, cur, source_sha256="deadbeef")
    raw = pickle.dumps(blob)          # 硬坑 A：这里过去会抛 AttributeError
    assert len(raw) > 0
    s2, h2, cur2, sha = from_blob(pickle.loads(raw))
    d1, w1 = diff_ledgers(s, s2, STATE_KEYS)
    d2, w2 = diff_ledgers(h, h2, MH_KEYS)
    assert d1 == 0.0 and d2 == 0.0, (w1, w2)
    assert cur2 == cur and sha == "deadbeef"
    n += 1
    # 增量：还原体继续吃新比赛 == 原对象吃同样新比赛
    new = [{"id": 3, "date": "2026-02-01", "best_of": 3,
            "team1": {"name": "A"}, "team2": {"name": "C"},
            "winner": {"name": "A"},
            "maps": [{"name": "Nuke", "team1_score": 13, "team2_score": 10}]}]
    ingest(s, h, new, ctx)
    ingest(s2, h2, new, ctx, cursor=cur2)
    d1, w1 = diff_ledgers(s, s2, STATE_KEYS)
    d2, w2 = diff_ledgers(h, h2, MH_KEYS)
    assert d1 < 1e-12 and d2 < 1e-12, (w1, w2)
    n += 1
    # 游标幂等：重复摄入同一批必须不改变账本
    before = _canon(s, STATE_KEYS)
    cnt2, _ = ingest(s, h, new, ctx, cursor={"last_date": "2026-02-01", "last_id": 3})
    assert cnt2 == 0, cnt2
    assert _canon(s, STATE_KEYS) == before
    n += 1
    # 硬坑 B：常数漂移必须当场报错
    bad = json.loads(json.dumps({"version": BLOB_VERSION}))
    bad = pickle.loads(pickle.dumps(blob))
    bad["consts"]["veto_model"]["FIRST_BAN_TEAM2"] = 0.999
    try:
        from_blob(bad)
        raise AssertionError("常数漂移未被拦截")
    except RuntimeError:
        pass
    n += 1
    print("SNAPSHOT_SELFTEST_OK (%d checks)" % n)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        sys.exit(selftest())
    sys.exit(2)
