"""check_rank_sync.py —— 守卫「rank 快照没有掉队 / 没有分裂」。

## 起因（2026-10-19，主理人当场抓到）

主理人已把 VRS 榜更新到 10-19，web 上却仍显示
「rank快照 2026-10-13 · 7天前」，而且**每次重启都是旧数据**。

根因链（逐环查证）：

1. `_build_repo.py` 的 DATA 白名单**钉死**了 `vrs_rank_2026-10-13.json`。
   ⇒ 新快照（10-18 / 10-19）**永远不会随构建发布**。
2. 10-18 / 10-19 两份快照只存在于**发布仓** `data/`（git 跟踪，构建不删），
   **scratch 这边没有**。
3. 而从 scratch 起的 GUI（`glob(HERE, "vrs_rank_*.json")`）只看得见 10-13。
4. `_build_rank()` 挑的是 JSON **内部 `date` 字段**最大的那份，不是文件名
   —— 所以只要文件在，它就会正确选最新的；**问题不在选文件，在文件没到位**。

## 本守卫查两件事

A. **两侧同步**：scratch 与发布仓里最新的 VRS 快照（按内部 date）必须同日期、
   同内容。差一天就说明「有一份更新只落在一侧」。
B. **选文件正确性**：把 GUI 的选文件逻辑（按内部 date 取最大）单独跑一遍，
   确认它选中的就是最新那份，而不是被文件名骗了。

用法：python -X utf8 check_rank_sync.py
"""
import glob
import hashlib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）
REL = r"E:\world\CS2-AWPredict"
REL_DATA = os.path.join(REL, "data")

# ⚠ 输出里的 ✓/✗ 是非 ASCII，Windows 默认 GBK 编不了 ⇒ 不带 `-X utf8` 跑就
#   UnicodeEncodeError **崩掉**，于是「守卫」在有人不用 -X utf8 时等于不存在。
#   显式把 stdout 重配成 UTF-8（Python 3.7+），让脚本不依赖调用方式。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                   # noqa: BLE001
        pass


def snap_files(base):
    """该目录下的 VRS 快照列表。

    ⚠ 必须经**参数**拼路径，不能写 `os.path.join(DATA, "vrs_rank_*.json")`：
      构建脚本会把那个字面量重写成 `os.path.join(DATA, "vrs_rank_*.json")`
      （因为它认得「HERE + 已知数据文件」这个模式）⇒ 本脚本想判断的
      「**脚本所在目录**有没有快照」被偷换成了「**数据目录**有没有快照」，
      在发布仓里逻辑整个反过来（实测：发布仓里报「选不到快照」）。
      走参数就绕开了重写。
    """
    return sorted(glob.glob(os.path.join(base, "vrs_rank_*.json")))


def load_all(base):
    """{内部 date: (路径, 队数, 内容 sha)}"""
    out = {}
    for f in snap_files(base):
        try:
            d = json.load(io.open(f, encoding="utf-8"))
        except Exception:                                # noqa: BLE001
            continue
        date = str(d.get("date") or "")
        b = io.open(f, "rb").read()
        out[date] = (f, len(d.get("teams") or []),
                     hashlib.sha256(b).hexdigest()[:12])
    return out


def pick_newest(snap):
    """复刻 predict.py::_build_rank() 的选文件口径：按**内部 date**取最大。"""
    if not snap:
        return None
    return max(snap.keys())


def main():
    fails = []
    # 两侧怎么定：本脚本在 scratch 与发布仓里各有一份，HERE 不同。
    #   在 scratch 跑 ⇒ HERE 就是「开发侧」，另一侧是发布仓 → 比对两侧
    #   在发布仓跑 ⇒ HERE 是「发布侧」，没有开发侧可比 → 只做 §B 自检
    # ⚠ 不能硬编码 scratch 的绝对路径（含用户名），故按「HERE 下有没有快照」判别。
    here_has = bool(snap_files(HERE))
    rel_has = bool(snap_files(REL_DATA))
    if here_has:
        dev_dir, dev_tag, rel_dir, rel_tag = HERE, "开发侧(scratch)", REL_DATA, "发布仓"
    else:
        dev_dir, dev_tag = REL_DATA, "发布仓"
        rel_dir, rel_tag = REL_DATA, "发布仓(同一份)"
    s1 = load_all(dev_dir)
    s2 = load_all(rel_dir)
    print("=" * 72)
    print("A. 两侧 VRS 快照")
    print("=" * 72)
    for tag, s in ((dev_tag, s1), (rel_tag, s2)):
        print("  %-16s %s" % (tag, {d: (v[1], v[2]) for d, v in sorted(s.items())}))
    if not s1 or not s2:
        print("  ★ 有一侧完全没有快照")
        fails.append("一侧无快照")
    else:
        d1, d2 = pick_newest(s1), pick_newest(s2)
        print("  各自最新：%s=%s  %s=%s" % (dev_tag, d1, rel_tag, d2))
        if dev_dir == rel_dir:
            print("  （本脚本不在开发侧运行 ⇒ 跳过跨侧比对）")
        elif d1 != d2:
            print("  ★ **两侧最新快照不同** ⇒ 更新只落在一侧")
            print("    缺的那侧：%s" % ("开发侧" if d1 < d2 else "发布仓"))
            fails.append("两侧不同步")
        elif s1[d1][2] != s2[d2][2]:
            print("  ★ 同日期但内容不同 ⇒ 被改过且没同步")
            fails.append("同日期内容不同")
        else:
            print("  ✓ 两侧一致")

    print()
    print("=" * 72)
    print("B. 选文件逻辑（按内部 date 取最大，不是按文件名）")
    print("=" * 72)
    for tag, s, base in ((dev_tag, s1, dev_dir), (rel_tag, s2, rel_dir)):
        p = pick_newest(s)
        if p:
            print("  %-16s 选中 %s（%d 队）" % (tag, p, s[p][1]))
        else:
            print("  %-16s ★ 选不到" % tag)
            fails.append(tag + " 选不到快照")
        if len(s) >= 2:
            by_name = max(glob.glob(os.path.join(base, "vrs_rank_*.json")),
                          key=os.path.basename)
            nm_date = str(json.load(io.open(by_name, encoding="utf-8")).get("date") or "")
            print("  %-16s 文件名最新=%s（内部 %s）  应选=%s  %s"
                  % ("", os.path.basename(by_name), nm_date, p,
                     "✓" if nm_date == p else "★不一致"))
            if nm_date != p:
                fails.append(tag + " 文件名与内部 date 不一致")

    print()
    if fails:
        print("RANK_SYNC_FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("RANK_SYNC_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
