"""check_display_paths.py —— 全仓扫描「显示层丢弃模型排名」的反模式。

## 为什么有这个脚本

2026-10-19 主理人两次踩同一个坑：

1. 第一次：CLI 输出 `选图: Dust2 / Mirage / Nuke`，而概率是
   `Mirage 74% · Nuke 65% · Dust2 57%` —— **第一位是概率第三的图**。
2. 第二次：修完 CLI，**web 还是没变** —— 因为 `predict_gui.py` 有自己的
   JS 渲染（`j.maps.join(' / ')`），根本没走 CLI 那份 `fmt_prediction`。

根因是同一个：**同一份逻辑有两份实现**。这正是本仓库反复强调的
「不另写一套规则」（CONVENTIONS）—— 我自己刚犯了一次。

## 规则

`plan["maps"]` 是**字母规范序**，用途是「计划身份」：
- `_plans()` 去重要比 tuple
- `scorer.py` 默认 `maps_order_matters=False`，按**集合**判分，完全不看顺序

所以对计分而言顺序毫无意义，字母序纯属显示层的坏习惯。
真正的排名在 `plan["maps_ranked"]`（`VM.predict_set` 的原始概率降序）。

**任何把 `maps` 直接 join/拼给人看的地方都是 bug。**
传��� `scorer.Prediction(maps=...)` / `dict(maps=...)` 是**正确**用法，不许改。

用法：python -X utf8 check_display_paths.py
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# 任何「把 maps 拼成给人看的字符串」的形态都是 bug。两种形态：
#   Python:  ", ".join(pl["maps"])      /  x.join(p["maps"])
#   JS:      j.maps.join(' / ')         /  j.maps_ranked.join(...)
# 计分传参（Prediction(maps=...) / "maps": list(...) / maps=list(...)）不含 join，不匹配。
DISPLAY_PAT = re.compile(
    r"""join\(\s*[^)]*\bmaps\b[^)]*\)"""       # Python：join(... maps ...)
    r"""|\bmaps\b\s*\.\s*join\s*\(""",        # JS：xxx.maps.join(
    re.X)

# 允许的例外：这些文件里 maps 只进 scorer，不进显示
SCORING_ONLY = {
    "score_batch.py", "seed_ledger.py", "replay_mvp_rule.py",
    "replay_rank_winner.py", "exp_decide_ab.py", "exp_item_ceiling.py",
    "exp_maps_ceiling.py", "exp_maps_confound.py", "exp_maps_entropy.py",
    "exp_maps_richness.py", "exp_mvp_scoring.py", "exp_score_diag.py",
    "exp_score_threshold.py", "verify_exp_calibration.py", "veto_labels.py",
    "export_new_records.py", "liquipedia_matches.py", "probe_major.py",
    "record_schema.py", "state_snapshot.py", "map_pools.py", "split.py",
    "veto_model.py", "model_v2.py", "pool_experiment.py", "decide.py",
}


# 计分传参（同一行里出现 Prediction( / "maps": list( / maps=list(）——传参不是显示
SCORING_CTX = re.compile(r"""Prediction\s*\(|\bmaps\s*=\s*list\(|"maps"\s*:\s*list\(""")


def main():
    bad = []
    scanned = 0
    gated = []
    EXTS = (".py", ".html", ".js")
    for fn in sorted(os.listdir(HERE)):
        if not fn.endswith(EXTS) or fn == "check_display_paths.py":
            continue                      # 脚本自身含反模式样例，别自杀
        scanned += 1
        src = io.open(os.path.join(HERE, fn), encoding="utf-8").read()
        # 文件级闸门：**只有拿得到概率数据时**，按字母序显示才算缺陷。
        # `_dump_*.py` 之类打印历史记录图集，压根没有 map_probs —— 字母序无害。
        if "map_probs" not in src and "maps_ranked" not in src:
            continue
        gated.append(fn)
        for i, line in enumerate(src.splitlines(), 1):
            if not DISPLAY_PAT.search(line):
                continue
            st = line.lstrip()
            if st.startswith("#") or st.startswith("//") or st.startswith("'") \
                    or st.startswith('"'):
                continue                  # 注释/文档里的反模式样例
            if SCORING_CTX.search(line):
                continue
            bad.append((fn, i, line.strip()[:100]))

    print("=" * 70)
    print("扫描 %d 个 .py/.html/.js；其中 %d 个拿得到概率数据（闸门内），逐行查反模式"
          % (scanned, len(gated)))
    print("=" * 70)
    if not bad:
        print("OK：闸门内没有任何显示层直接拼字母序 maps")
        print("     （新增显示路径请用 plan['maps_ranked'] 或按 map_probs 排序）")
        return 0
    print("FAIL：以下位置有概率数据却按字母序显示：\n")
    for fn, i, line in bad:
        print("  %s:%d\n    %s\n" % (fn, i, line))
    print("修法：用 plan['maps_ranked']（概率降序），或按 map_probs 排序。")
    print("计分用的 list(p['maps']) **不要动** —— scorer 按集合判分。")

    dual = check_dual_template()
    if dual:
        print("FAIL：两份页面模板不同步（外部 predict_page.html 优先，改内联不生效）\n")
        for d in dual:
            print("  - " + d)
        return 1
    print("双模板同步检查：OK（内联 PAGE 与 predict_page.html 都已按概率降序渲染）")
    return 1


def check_dual_template():
    """★ predict_gui.py 有**两份**页面模板，且外部文件优先 —— 这是第三次踩坑的根源。

    `predict_gui.py:506-509`：若同目录存在 `predict_page.html` 就用它，
    否则退回内联 PAGE 常量。改内联模板**完全不生效**（2026-10-19 实测：
    磁盘上内联已修好，web 仍返回旧的 `j.maps.join(' / ')`）。

    ⇒ 两份模板必须一起改，或者干脆只留一份。这里强制它们同步。
    """
    gui = os.path.join(HERE, "predict_gui.py")
    page = os.path.join(HERE, "predict_page.html")
    if not (os.path.exists(gui) and os.path.exists(page)):
        return None
    a = io.open(gui, encoding="utf-8").read()
    b = io.open(page, encoding="utf-8").read()
    if "predict_page.html" not in a:
        return None                        # 已改为单一来源，无需比对
    out = []
    if ("maps_ranked" in a) != ("maps_ranked" in b):
        out.append("两份模板对 maps_ranked 的支持**不一致**（内联=%s / 外部=%s）"
                   % ("maps_ranked" in a, "maps_ranked" in b))
    for tag, txt in (("内联 PAGE (predict_gui.py)", a), ("外部模板 (predict_page.html)", b)):
        if "按概率降序" not in txt:
            out.append("%s 缺少「按概率降序」渲染 —— 改它不生效的话 web 不会变" % tag)
    return out


if __name__ == "__main__":
    sys.exit(main())
