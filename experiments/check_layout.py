"""predict_page.html / predict_gui.py 的浮层布局与交互回归检查。

背景：本项目有两个「前端」来源，容易改错文件——
  · `predict_page.html`     —— 2026-10-14 的视觉重设计版，**实际生效**的页面
  · `predict_gui.py` 内联 HTML —— 兜底，仅当上面那个文件不存在时才用
所以校验必须对着**线上真正下发的内容**做，否则会「改了半天没生效」。

本脚本只做静态断言：真层叠要在浏览器里目视，这里防的是改回文档流 / 改错文件。
"""
import io
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
Q = chr(39)


def main():
    bad = 0
    print("=== 1. 线上真正下发的页面（权威）===")
    try:
        live = urllib.request.urlopen("http://127.0.0.1:8765/", timeout=10).read().decode()
    except Exception as e:                                # noqa: BLE001
        print("  服务不可达：%s" % e)
        return 1
    disk = io.open(os.path.join(HERE, "predict_page.html"), encoding="utf-8").read()
    same = live == disk
    print("  线上 %d 字符 / 磁盘 predict_page.html %d 字符 / 一致=%s"
          % (len(live), len(disk), same))
    if not same:
        bad += 1
        print("  ⚠ 线上与磁盘不同 ⇒ 要么服务未重启，要么服务的是别的文件")

    checks = [
        ("下拉定位锚点", r"\.f\{[^}]*position:relative"),
        ("下拉绝对定位", r"\.lst\{[^}]*position:absolute"),
        ("下拉有层级", r"\.lst\{[^}]*z-index:\s*\d+"),
        ("下拉不透明底", r"\.lst\{[^}]*background:"),
        ("浮层投影", r"\.lst\{[^}]*box-shadow"),
        ("下拉互斥", r"function closeAll"),
        ("阻止失焦", r"lst\.onmousedown"),
        ("Esc 关闭", r"e\.key===" + Q + "Escape"),
        ("点外部关闭", r"addEventListener\(" + Q + "mousedown"),
    ]
    print("\n=== 2. 关键行为标记（对着线上内容查）===")
    for name, pat in checks:
        ok = bool(re.search(pat, live))
        if not ok:
            bad += 1
        print("  %-14s %s" % (name, "OK" if ok else "缺失"))

    print("\n=== 3. 反向检查（不该出现的写法）===")
    anti = [
        ("下拉仍在文档流", r"\.lst\{(?![^}]*position:absolute)"),
        ("下拉无层级", r"\.lst\{(?![^}]*z-index)"),
    ]
    for name, pat in anti:
        hit = bool(re.search(pat, live))
        if hit:
            bad += 1
        print("  %-14s %s" % (name, "✗ 出现!" if hit else "OK"))

    print("\n=== 4. JS 语法自洽（括号/引号配平粗检）===")
    js = ""
    m = re.search(r"<script>(.*?)</script>", live, re.S)
    if m:
        js = m.group(1)
    for ch_open, ch_close, nm in (("{", "}", "花括号"), ("(", ")", "圆括号"),
                                  ("[", "]", "方括号")):
        d = js.count(ch_open) - js.count(ch_close)
        flag = "OK" if d == 0 else "不平衡 %+d" % d
        if d != 0:
            bad += 1
        print("  %-8s %s (开%d 闭%d)" % (nm, flag, js.count(ch_open), js.count(ch_close)))
    # 之前真出过：PowerShell 把 JS 里的 $(id) 当变量展开，吃掉后留下 `};` 与 `});;`
    for junk in ("});;", "});\n};", "forEach(id=>.classList"):
        if junk in live:
            bad += 1
            print("  ⚠ 发现上次被 shell 破坏的残留：%r" % junk)

    print("\n=== 5. 后端接口仍正常 ===")
    for path in ("/api/status", "/api/predict?a=vitality&b=furia&bo=3"):
        try:
            j = json.loads(urllib.request.urlopen(
                "http://127.0.0.1:8765" + path, timeout=20).read().decode())
            if "status" in path:
                print("  /api/status        ready=%s 队伍 %d" % (j.get("ready"), len(j.get("teams", []))))
            else:
                print("  /api/predict       %s 胜 %s E=%.1f/%d" %
                      (j["A"], j["winner"], j["exp_points"], j["max_points"]))
        except Exception as e:                            # noqa: BLE001
            bad += 1
            print("  %s ✗ %s" % (path, e))

    print("\n%s" % ("LAYOUT_OK" if bad == 0 else "LAYOUT_FAIL(%d)" % bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())