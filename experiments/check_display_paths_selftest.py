"""check_display_paths_selftest.py —— 验证 check_display_paths 的检测器本身有效。

一个抓不到 bug 的守卫等于没有守卫。这里喂它**主理人实际抓到的三处旧代码**，
必须全部抓到；同时喂计分用法，必须全部放行。
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "cdp", os.path.join(HERE, "check_display_paths.py"))
CDP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CDP)

OLD = [
    ("predict_gui.py",
     """<div style="margin-top:14px"><b>选图 ${esc(j.maps.join(' / '))}</b></div>"""),
    ("blind_batch.py",
     """print("  选图 %s" % " / ".join(p["maps"]))"""),
    ("decide_rank.py",
     """pl["score"][1], ",".join(pl["maps"]), pl["mvp"], mu, sd))"""),
]
SCORING = [
    """out.append({"A": a.A[i], "maps": list(pl["maps"]), "mvp": pl["mvp"],""",
    """maps=list(plan["maps"]), mvp=plan["mvp"])""",
    """maps=list(p["maps"]), mvp=p["mvp"]), truth)""",
    """maps=list(p["maps"]), mvp=p["mvp"])""",
]

fails = []
print("=== 检测器必须抓到的（主理人抓到的三处旧代码）===")
for fn, line in OLD:
    hit = bool(CDP.DISPLAY_PAT.search(line))
    print(("  抓到   " if hit else "  ★漏掉 ") + "%-16s %s" % (fn, line.strip()[:66]))
    if not hit:
        fails.append("漏掉 " + fn)

print()
print("=== 检测器必须放行的（计分用法，不许误伤）===")
for line in SCORING:
    hit = bool(CDP.DISPLAY_PAT.search(line))
    print(("  ★误伤 " if hit else "  放行   ") + line.strip()[:70])
    if hit:
        fails.append("误伤 " + line.strip()[:40])

print()
if fails:
    print("DETECTOR_SELFTEST_FAIL")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("DETECTOR_SELFTEST_OK（3 抓到 / %d 放行）" % len(SCORING))
