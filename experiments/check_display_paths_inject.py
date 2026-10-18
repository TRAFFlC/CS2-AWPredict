"""check_display_paths_inject.py —— 把主理人抓到的旧 bug 注回 predict_gui.py，验证守卫真的会 FAIL。

一个抓不到 bug 的守卫等于没有守卫（第一版 check_display_paths 就是这样：
正则从未匹配任何东西，却一直「通过」）。本脚本做端到端验证：

    1. 备份
    2. 注入旧渲染 `j.maps.join(' / ')`
    3. 跑 check_display_paths ⇒ 必须 rc=1
    4. 还原 ⇒ 必须 rc=0

用法：python -X utf8 check_display_paths_inject.py
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHK = os.path.join(HERE, "check_display_paths.py")
BAD = """    <div>选图 ${esc(j.maps.join(' / '))}</div>"""


def find_gui():
    """定位 predict_gui.py —— 仓库里它与本脚本**不在同一目录**。

    scratch：两者都在 014-scratch/
    发布仓：本脚本在 experiments/，GUI 在**根目录**（平铺布局）
    ⇒ 只按 HERE 找会在发布仓里 FileNotFoundError（2026-10-19 实测）。
    """
    for c in (os.path.join(HERE, "predict_gui.py"),
              os.path.join(HERE, "..", "predict_gui.py")):
        if os.path.exists(c):
            return os.path.abspath(c)
    raise SystemExit("找不到 predict_gui.py（试了 %s 与 %s/..）"
                     % (HERE, HERE))


def run_check(script, cwd):
    """⚠ 两个必须做对的地方，缺一个就得到**假 OK**：

    1) 跑**传入的** script，不是模块级 CHK。检查器用
       `HERE = dirname(abspath(__file__))` 决定扫哪个目录 ⇒ 拿 scratch 里的
       真检查器配 cwd=tmp 是**无效注入**（它仍扫 scratch，永远返回 0）。
    2) `encoding="utf-8"` 必须显式给。`text=True` 用的是**系统默认编码**
       （这台机器是 GBK），而子进程是 `-X utf8` 跑的中文输出 ⇒
       UnicodeDecodeError 发生在 reader 线程里，**stdout 变成空串**，
       于是 `hit = "predict_gui.py" in out` 恒为 False，而 rc 已经是对的
       —— 症状是「rc=1 但命中 False」，非常容易误判成「守卫没生效」。
    """
    r = subprocess.run([sys.executable, "-X", "utf8", script],
                       capture_output=True, text=True, cwd=cwd,
                       encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout or "")[-800:]


def main():
    gui = find_gui()
    # ⚠ **绝不就地改写 predict_gui.py** —— 它是 git 跟踪的源文件，
    #   注入测试失败一次就可能把反模式留在提交里。整套操作在临时目录里做。
    tmp = tempfile.mkdtemp(prefix="cdp_inject_")
    try:
        local_gui = os.path.join(tmp, "predict_gui.py")
        shutil.copyfile(gui, local_gui)
        local_chk = os.path.join(tmp, "check_display_paths.py")
        shutil.copyfile(CHK, local_chk)
        src = io.open(local_gui, encoding="utf-8").read()
        if BAD in src:
            print("!! 旧 bug 已在 predict_gui.py 里，先手工还原")
            return 1
        io.open(local_gui, "w", encoding="utf-8", newline="\n").write(
            src + "\n" + BAD + "\n")
        rc, out = run_check(local_chk, tmp)
        print("=== 注入旧 bug 后（临时目录，不碰真文件） ===")
        print("  rc = %d  （必须非 0）" % rc)
        hit = "predict_gui.py" in out
        print("  命中 predict_gui.py：%s  （必须 True）" % hit)
        if rc == 0:
            print("  注入无效！检查 stdout 尾部：")
            print("  " + out.strip().replace("\n", "\n  "))
        ok = (rc != 0 and hit)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    rc2, out2 = run_check(CHK, HERE)
    print("=== 真目录复查（未被污染） ===")
    print("  rc = %d  （必须 0）" % rc2)
    if rc2 != 0:
        print("  " + out2.strip().replace("\n", "\n  "))
    if ok and rc2 == 0:
        print("INJECT_SELFTEST_OK（守卫真的会响，且没碰真文件）")
        return 0
    print("INJECT_SELFTEST_FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
