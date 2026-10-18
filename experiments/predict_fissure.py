"""裂变天地 S3（2026-09-08）8 场盲预测。

队伍名对齐：csapi 用 `The MongolZ` / `GamerLegion` / `FaZe` / `BetBoom` / `BIG` /
`PARIVISION` / `TYLOO` 等写法，主理人给的简称需映射（pvision→PARIVISION 等）。

冷门提示规则（主理人 2026-10-12 明确要求）：
  🔴 rank>50 或 语料<10 场   —— 该队 Elo 估计不可靠
  🟡 语料<40 场              —— 数据偏少
  并标出 both≤16 子集：**上一轮实测该子集模型相对热门基线 −5.3pp，是无增值区**。
"""
import collections, io, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）

ALIAS = {"pvision": "PARIVISION", "parivision": "PARIVISION",
         "mongolz": "The MongolZ", "gamelegion": "GamerLegion",
         # 主理人有时拼作 gamerlegion（多 r），两个都收
         "gamerlegion": "GamerLegion",
         "faze": "FaZe", "betboom": "BetBoom", "big": "BIG", "mibr": "MIBR",
         "legacy": "Legacy", "alliance": "Alliance", "astralis": "Astralis",
         "g2": "G2", "furia": "FURIA", "tyloo": "TYLOO", "9z": "9z",
         "magic": "magic", "5star": "5star",
         # StarLadder 秋季群星系列赛
         "mouz": "MOUZ", "nrg": "NRG", "vitality": "Vitality",
         "navi": "Natus Vincere", "aurora": "Aurora",
         # 冷门小赛事
         "johnnyspeeds": "Johnny Speeds", "johnny speeds": "Johnny Speeds",
         "heroic": "HEROIC", "virtus.pro": "Virtus.pro", "astral": "ASTRAL",
         "sangal": "Sangal", "betclic": "betclic", "bbl": "BBL",
         "fokus": "FOKUS", "eac": "EAC", "sashi": "Sashi",
         "phantom": "Phantom", "astralis": "Astralis",
         "fnatic": "fnatic", "100thieves": "100 Thieves",
         "jijiehao": "JiJieHao", "eternalfire": "Eternal Fire",
         # Logitech G Connect 2026
         # ⚠ 「nip」= Ninjas in Pyjamas（2026-09-17 该赛事揭幕战即 NIP vs M80，
         #   阵容 stavn/Krimbo/xKacpersky/sjuush/adamb）。**但 csapi 语料里
         #   完全不存在 NiP** ⇒ 该队进不了模型，胜方项也无 rank 可依。
         #   仍写进别名表是为了让失败信息明确，而不是抛 KeyError。
         "nip": "Ninjas in Pyjamas",
         "innercircle": "Inner Circle", "infinite": "INFINITE",
         "3dmax": "3DMAX", "eyeballers": "EYEBALLERS",
         "luminosity": "Luminosity", "metizport": "Metizport",
         "m80": "M80", "b8": "B8", "liquid": "Liquid"}

PAIRS = [("legacy", "magic"), ("9z", "5star"), ("mongolz", "mibr"), ("faze", "alliance"),
         ("betboom", "big"), ("pvision", "tyloo"), ("furia", "gamelegion"),
         ("g2", "astralis")]


def parse_pairs(spec):
    """'magic:faze,5star:mongolz' -> [('magic','faze'), ('5star','mongolz')]"""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok or ":" not in tok:
            continue
        a, b = tok.split(":", 1)
        out.append((a.strip().lower(), b.strip().lower()))
    return out


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def coverage():
    seen = collections.defaultdict(lambda: {"n": 0, "rank": None, "last": "", "nm": ""})
    for l in io.open(os.path.join(DATA, "csapi_matches_all.jsonl"), encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        dt = str(r.get("date") or "")
        for t in (r["team1"], r["team2"]):
            d = seen[t["name"]]
            d["n"] += 1
            d["nm"] = t["name"]
            if t.get("rank"):
                d["rank"] = int(t["rank"])
            d["last"] = max(d["last"], dt)
    idx = collections.defaultdict(list)
    for n in seen:
        idx[norm(n)].append(n)
    return idx, seen


def resolve(q, idx, seen):
    k = norm(q)
    cands = idx.get(k)
    if not cands:
        if k == "pvision":
            cands = idx.get("parivision")
        if not cands:
            cands = [n for kk, v in idx.items() if k in kk for n in v]
    return seen[cands[0]] if cands else None


def main():
    idx, seen = coverage()
    bo = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    pairs, tag, comp = PAIRS, "2026-09-08", "裂变天地 S3"
    if len(sys.argv) > 2 and sys.argv[2].startswith("--"):
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--date", default="2026-09-08")
        ap.add_argument("--pairs", required=True)
        ap.add_argument("--bo", type=int, default=bo)
        ap.add_argument("--comp", default="裂变天地 S3", help="赛事名（仅用于标题）")
        a = ap.parse_args(sys.argv[2:])
        pairs, tag, bo, comp = parse_pairs(a.pairs), a.date, a.bo, a.comp
    print("== %s · %s · %d 场 bo%d 盲预测 ==\n" % (comp, tag, len(pairs), bo))
    out = []
    for a, b in pairs:
        A, B = ALIAS[a], ALIAS[b]
        # resolve 必须用**别名后的规范名**（"The MongolZ"/"GamerLegion"），否则简称匹配不上
        da, db = resolve(A, idx, seen), resolve(B, idx, seen)
        r = subprocess.run([sys.executable, "-X", "utf8", "predict.py", "match",
                            "--A", A, "--B", B, "--bo", str(bo)],
                           capture_output=True, cwd=HERE)
        txt = r.stdout.decode("utf-8", "replace")
        # 队名可能带空格（The MongolZ），必须用非贪婪 + 惰性匹配到 " P=" 前
        w = re.search(r"胜方:\s*(.+?)\s+P=([\d.]+)%", txt)
        win, p = (w.group(1), float(w.group(2))) if w else ("?", 50.0)
        sc = re.search(r"比分:\s*(\d):(\d+)", txt)
        maps = re.search(r"选图:\s*(.+)", txt)
        mvp = re.search(r"MVP\s*:\s*(\S+)", txt)
        exp = re.search(r"期望得分\s*([\d.]+)", txt)
        rk = lambda d: (d["rank"] if d and d["rank"] else 999)
        ga, gb = rk(da), rk(db)
        elite = max(ga, gb) <= 16
        flags = []
        for q, d, g in ((a, da, ga), (b, db, gb)):
            if d is None or d["n"] < 10 or g > 50:
                flags.append("🔴%s(rank%s, %d场)" % (ALIAS[q], ("#%d" % g) if d and d["rank"] else "?", d["n"] if d else 0))
            elif d["n"] < 40:
                flags.append("🟡%s(%d场)" % (ALIAS[q], d["n"]))
        print("--- %s vs %s (bo%d) ---" % (A, B, bo))
        print("  胜方 %s  P=%.1f%%   比分 %s   MVP %s   期望 %.1f 分"
              % (win, p, (sc.group(0).split()[1] if sc else "?"),
                 (mvp.group(1) if mvp else "?"), float(exp.group(1)) if exp else 0))
        print("  选图 %s" % (maps.group(1).strip() if maps else "?"))
        print("  rank %s#%d vs %s#%d   %s" % (A, ga, B, gb,
              "⚠ 双方≤16 = 上一轮实测的**无增值子集**" if elite else ""))
        print("  冷门提示：%s" % ("；".join(flags) if flags else "无"))
        out.append({"A": A, "B": B, "bo": bo, "winner": win, "p_win": p,
                    "score": (sc.group(0).split()[1] if sc else "?"),
                    "maps": maps.group(1).strip() if maps else "",
                    "mvp": mvp.group(1) if mvp else "",
                    "exp": float(exp.group(1)) if exp else 0.0,
                    "rankA": ga, "rankB": gb, "elite": elite, "flags": flags})
    json.dump(out, open(os.path.join(HERE, "fissure_%s_predictions.json" % tag), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    # 满分随 bo 变：bo1=25 / bo3=45 / bo5=55（第一版硬编码 45，bo5 会算错）
    mx = {1: 25, 3: 45, 5: 55}.get(bo, 45)
    print("\n合计期望分 %.1f / 满分 %d（%d 场 bo%d）"
          % (sum(o["exp"] for o in out), mx * len(out), len(out), bo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
