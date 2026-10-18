"""用真实浏览器验证队名下拉浮层与「最近」组件。

为什么必须真浏览器：上一轮我只做了**静态标记检查**（z-index 存在、position 存在…），
全部通过，但实际仍然错。原因是我没查**层叠上下文**：
`.card` 带 `backdrop-filter`，它会创建层叠上下文，于是子元素 `.lst` 的
`z-index` 只在卡片内部比较，对后面的兄弟卡片无效。
静态 grep 永远发现不了这件事，只有真实布局（getBoundingClientRect +
elementFromPoint 的命中测试）才能测出来。

检查项：
  1. 下拉可见时，其矩形是否**压在**「最近」卡片之上
  2. 下拉区域里命中测试拿到的是下拉本身（而不是「最近」卡片）
  3. 点击下拉里的队伍能回填并出结果
  4. 「最近」条目可点击，点击后能回填两队与赛制并重跑
"""
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8765/"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def wait_ready(page):
    page.wait_for_function(
        "() => !document.getElementById('go').disabled", timeout=120000)


def main():
    fails = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=CHROME)
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        pg.goto(URL, wait_until="domcontentloaded")
        wait_ready(pg)
        print("页面就绪")

        # ---------- 1) 下拉浮层 vs 「最近」卡片 ----------
        pg.click("#a")
        pg.wait_for_selector("#la.on", timeout=5000)
        pg.wait_for_timeout(300)
        dd = pg.locator("#la")
        hist_card = pg.locator("div.card").last
        db = dd.bounding_box()
        hb = hist_card.bounding_box()
        print("\n=== 1. 几何位置 ===")
        print("  下拉 #la   :", {k: round(v) for k, v in db.items()})
        print("  「最近」卡片:", {k: round(v) for k, v in hb.items()})
        overlap_v = min(db["y"] + db["height"], hb["y"] + hb["height"]) - max(db["y"], hb["y"])
        overlap_h = min(db["x"] + db["width"], hb["x"] + hb["width"]) - max(db["x"], hb["x"])
        overlap = overlap_v > 0 and overlap_h > 0
        print("  垂直重叠 %.0f px / 水平重叠 %.0f px ⇒ %s"
              % (max(0, overlap_v), max(0, overlap_h), "确实重叠" if overlap else "不重叠"))

        # ---------- 2) 命中测试：重叠点上到底点得到谁 ----------
        print("\n=== 2. 命中测试（取下拉与「最近」重叠区中心）===")
        hit_ok = False
        if overlap:
            cx = int(max(db["x"], hb["x"]) + min(db["x"] + db["width"], hb["x"] + hb["width"]) / 2
                     - db["x"]) + int(db["x"])
            cy = int(max(db["y"], hb["y"]) + min(db["y"] + db["height"],
                                                 hb["y"] + hb["height"]) / 2)
            el = pg.evaluate(
                """([x,y])=>{const e=document.elementFromPoint(x,y);
                   return e? (e.id||e.className||e.tagName):null;}""", [cx, cy])
            print("  点 (%d,%d) 命中 -> %s" % (cx, cy, el))
            hit_ok = el in ("la", "lb") or (el and "it" in str(el))
            if not hit_ok:
                fails.append("重叠区命中的是 %s，不是下拉项" % el)
        else:
            hit_ok = True
            print("  不重叠，跳过（不重叠本身即正确）")

        # ---------- 3) 用下拉点选队伍 ----------
        # ⚠ 必须先填好 B 队：只选 A 队时 go() 会走 `if(!a||!b)` 分支提示
        #   「请填两支队伍」，本来就**不该**出结果——早先误把它当成了 bug。
        print("\n=== 3. 点选队伍 ===")
        pg.fill("#b", "FURIA")
        pg.click("#a")
        pg.fill("#a", "vita")
        pg.wait_for_timeout(400)
        diag = pg.evaluate(
            """()=>{const l=document.getElementById('la');
                 const it=l.querySelector('.it[data-n]');
                 const cs=it?getComputedStyle(it):null;
                 const ls=getComputedStyle(l);
                 return {laClass:l.className, laDisplay:ls.display,
                         laVis:ls.visibility, itemH:it?it.offsetHeight:null,
                         itemDisp:cs?cs.display:null, n:l.querySelectorAll('.it').length};}""")
        print("  诊断: %s" % diag)
        items = pg.locator("#la .it[data-n]")
        n = items.count()
        print("  过滤 'vita' 命中 %d 项" % n)
        if n == 0:
            fails.append("下拉过滤不出队伍")
        else:
            name = items.first.get_attribute("data-n")
            items.first.click()
            pg.wait_for_timeout(3000)
            print("  已点选 %s → 输入框值 %r" % (name, pg.input_value("#a")))
            out = pg.locator("#out .win")
            if out.count() == 0:
                fails.append("点选队伍后没出结果")
            else:
                print("  结果区胜方: %s" % out.first.inner_text().strip())

        # 3b) 「预测」按钮必须真的绑了事件——曾经它是个死按钮
        print("\n=== 3b. 「预测」按钮 ===")
        pg.fill("#a", "")
        pg.fill("#b", "")
        pg.wait_for_timeout(200)
        empty_before = pg.locator("#out .win").count()
        pg.fill("#a", "Legacy")
        pg.fill("#b", "magic")
        pg.click("#go")
        pg.wait_for_timeout(3000)
        after = pg.locator("#out .win").count()
        bound = pg.evaluate("() => typeof document.getElementById('go').onclick")
        print("  onclick 类型=%s   点击前结果区=%d  点击后结果区=%d"
              % (bound, empty_before, after))
        if bound == "undefined":
            fails.append("「预测」按钮未绑定 onclick（死按钮）")
        if after == 0:
            fails.append("点「预测」按钮后没出结果")

        # ---------- 4) 「最近」条目可点且能回放 ----------
        print("\n=== 4. 「最近」组件 ===")
        # 先造第二条记录，确保有可点的条目
        pg.fill("#a", "furia")
        pg.fill("#b", "G2")
        pg.click("#go")
        pg.wait_for_timeout(2500)
        hist = pg.locator("#hist a.hitem")
        hn = hist.count()
        print("  条目数 %d" % hn)
        if hn == 0:
            fails.append("「最近」没有渲染出可点条目")
        else:
            # 先清空输入框，点历史条目应当把值填回来
            pg.fill("#a", "")
            pg.fill("#b", "")
            pg.select_option("#bo", "1")
            txt = hist.nth(0).inner_text().strip()
            hist.nth(0).click()
            pg.wait_for_timeout(2500)
            a2, b2, bo2 = pg.input_value("#a"), pg.input_value("#b"), pg.input_value("#bo")
            print("  点第一条 %r" % txt)
            print("  回填 → A=%r  B=%r  bo=%s" % (a2, b2, bo2))
            if not a2 or not b2:
                fails.append("点「最近」没回填队伍")
            if bo2 == "1":
                fails.append("点「最近」没回填赛制（仍是 bo1）")
            out2 = pg.locator("#out .win")
            print("  重跑结果: %s" % (out2.first.inner_text().strip()
                                    if out2.count() else "（无）"))
            if out2.count() == 0:
                fails.append("点「最近」后没重跑")

        pg.screenshot(path="_ui_check.png", full_page=True)
        print("\n截图: _ui_check.png")
        b.close()

    print("\n%s" % ("UI_OK" if not fails else "UI_FAIL"))
    for f in fails:
        print("  ✗ %s" % f)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())