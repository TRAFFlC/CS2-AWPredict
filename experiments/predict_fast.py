"""predict 快速路径：快照优先，避免每次全量重放（[P0]-1 第 B 步）

**不修改 `predict.py` / `model_v2.py` / `decide.py` / `veto_model.py`**：
用 `Predictor.__new__` 绕过 `__init__`，把快照里的账本 + 已拟合模型灌进 `__dict__`。

快照内容：
  · `state_snapshot` 的账本 blob（State 30 字段 + MapHistory 16 字段 + 游标）
  · 已拟合模型：`wsel` / `wmodel` / `mmodel` / `gmodel` / `smodel`（sklearn + decide.ScoreModel）
  · 选型元数据：`winfeats` / `C` / `score_name`
  · 指纹：`source_sha256`（语料）+ `records_sha256`（match_record）
  · 常数快照（`veto_model` / `model_v2` 模块级），读回时逐个校验，不符即作废
不缓存：`data`（backtest 用的全部样本，体积大）⇒ 快照态下 `backtest` 会明确报错，
要求走全量路径。这是刻意的取舍：日常预测是高频、backtest 是低频。

用法：
    import predict_fast as PF
    P_ = PF.build()                    # 有有效快照就用快照，否则全量建并落盘
    P_.predict("G2", "Natus Vincere", 3, None, "score")
    P_.ingest_records(path)            # 把新的 match_record 增量吃进去并更新快照
"""
import hashlib, io, json, os, pickle, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）
sys.path.insert(0, HERE)
import model_v2 as M
import split as SPLIT
import pool_experiment as PE
import state_snapshot as SS
import record_schema as RS
import decide as D
import coldstart as C
import predict as P

SNAP = os.path.join(HERE, "predict_snapshot.pkl")
SIDECAR = os.path.join(DATA, "predict_snapshot.meta.json")
VERSION = "predict-snapshot/1"
# 只缓存**预测路径真正用到的**：`wsel` / `gmodel` 在 predict.py 里只是 `__init__` 的局部量
# （`predict.py:72` 的 wsel 仅用于给 data["train"] 算 pA，而 data 不缓存）；
# `C` 只是 prepare() 的选型超参、预测期不用。三者故都不入快照。
MODEL_ATTRS = ["wmodel", "mmodel", "smodel"]
META_ATTRS = ["winfeats", "score_name"]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save(obj, path=SNAP, records=None):
    blob = SS.to_blob(obj.state, obj.mh, {"last_date": obj.state.last_date,
                                          "last_id": None},
                      source_sha256=sha256_of(M.MATCHES),
                      note="predict snapshot")
    side = {
        "version": VERSION,
        "corpus_sha256": blob["source_sha256"],
        "records_sha256": sha256_of(records) if records and os.path.exists(records) else None,
        "ledger_last_date": obj.state.last_date,
        "models": {k: type(getattr(obj, k)).__name__ for k in MODEL_ATTRS},
        "sel": {k: getattr(obj, k) for k in META_ATTRS},
    }
    payload = {"blob": blob, "models": {k: getattr(obj, k) for k in MODEL_ATTRS},
               "meta": {k: getattr(obj, k) for k in META_ATTRS}}
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    io.open(SIDECAR, "w", encoding="utf-8").write(json.dumps(side, ensure_ascii=False, indent=2))
    return os.path.getsize(path)


def load(path=SNAP, strict_sha=True):
    """返回还原好的 Predictor；任何不一致返回 None（调用方退回全量路径）。"""
    if not (os.path.exists(path) and os.path.exists(SIDECAR)):
        return None, "无快照"
    side = json.load(io.open(SIDECAR, encoding="utf-8"))
    if side.get("version") != VERSION:
        return None, "快照版本不符"
    if strict_sha and side.get("corpus_sha256") != sha256_of(M.MATCHES):
        return None, "语料已变（sha256 不符）⇒ 快照作废"
    if side.get("records_sha256"):
        rp = side.get("_records_path")
        if rp and os.path.exists(rp) and sha256_of(rp) != side["records_sha256"]:
            return None, "match_record 已变 ⇒ 快照作废"
    try:
        payload = pickle.load(open(path, "rb"))
        s, h, cur, _sha = SS.from_blob(payload["blob"])     # 内含常数漂移校验
    except Exception as e:                                  # 常数漂移 / pickle 损坏
        return None, "快照不可用：%s" % e

    obj = P.Predictor.__new__(P.Predictor)
    obj.prep = None
    obj.rows = None
    obj.meta = {}
    obj.data = None                     # backtest 需要，见模块 docstring
    obj.state, obj.mh = s, h
    obj.ranks = PE.Ranks()
    for k in MODEL_ATTRS:
        setattr(obj, k, payload["models"][k])
    for k in META_ATTRS:
        setattr(obj, k, payload["meta"][k])
    obj.pool = C.current_pool(obj.mh)
    obj.profiles = C.all_profiles(obj.state, obj.mh, obj.pool)
    obj._slope, obj._intercept = C.rating_to_elo_slope(obj.profiles)
    obj.last_date = obj.state.last_date or side.get("ledger_last_date")
    obj._injected = set()
    return obj, "OK（账本末日 %s，已注入 0 支）" % obj.last_date


def build(path=SNAP, records=None, allow_fresh=True, verbose=True):
    t0 = time.time()
    obj, why = load(path)
    if obj is not None:
        if verbose:
            print("[predict_fast] 快照命中：%s  (%.2fs，原路径约 9-10s)"
                  % (why, time.time() - t0))
        return obj
    if not allow_fresh:
        raise RuntimeError("快照不可用：%s" % why)
    if verbose:
        print("[predict_fast] 快照不可用（%s）⇒ 走全量路径" % why)
    obj = P.Predictor()
    t1 = time.time()
    n = save(obj, path, records)
    if verbose:
        print("[predict_fast] 全量建模 %.1fs，落盘快照 %d B ⇒ 下次走快速路径"
              % (t1 - t0, n))
    return obj


def ingest_records(obj, records_path, save_after=True, verbose=True):
    """把一批 match_record 增量喂进已加载的 Predictor，并（可选）更新快照。"""
    recs = [json.loads(l) for l in io.open(records_path, encoding="utf-8").read().split("\n")
            if l.strip()]
    truths, veto, stats, map_stats, rows = {}, {}, {}, {}, []
    for rec in recs:
        if not rec.get("date"):
            continue
        if rec.get("date") <= (obj.state.last_date or ""):
            continue                                    # 游标之前的不重复摄入
        row = RS.to_ingest_row(rec)
        rows.append(row)
        truths[row["id"]] = {"maps": rec["maps_full_B"], "pool": rec.get("pool") or []}
        if rec.get("veto_order"):
            veto[row["id"]] = {"bo": rec["best_of"],
                               "order": [tuple(s) for s in rec["veto_order"]]}
        if rec.get("rosters"):
            stats[row["id"]] = row["_rosters"]
        if rec.get("by_map"):
            map_stats[row["id"]] = row["_by_map"]
    ctx = SS.make_ctx(truths, veto, stats, map_stats, *PE.load_pool_eras(),
                      json.load(io.open(PE.TIMELINE, encoding="utf-8")))
    rows.sort(key=lambda r: (r["date"], r["id"]))
    cur = {"last_date": obj.state.last_date, "last_id": None}
    n, newcur = SS.ingest(obj.state, obj.mh, rows, ctx, cursor=cur)
    obj.pool = C.current_pool(obj.mh)
    obj.profiles = C.all_profiles(obj.state, obj.mh, obj.pool)
    obj._slope, obj._intercept = C.rating_to_elo_slope(obj.profiles)
    obj.last_date = obj.state.last_date
    if verbose:
        print("[predict_fast] 增量摄入 %d 场（跳过 %d 条日期不新于水位）⇒ 账本末日 %s"
              % (n, len(recs) - len(rows), obj.state.last_date))
    if save_after:
        save(obj, SNAP, records_path)
    return n


def _guard_backtest(obj):
    if obj.data is None:
        raise RuntimeError(
            "快照态不缓存 data（体积大）⇒ 无法 backtest。请用 "
            "`predict_fast.build(allow_fresh=True)` 走全量路径，或直接跑 "
            "`predict.py backtest --n 317`。")
