"""诊断：已训练的选图模型到底在不在用「对手弱图」？

H1 实测：pick 的「对手在该图胜率百分位」= 0.4299（z = −11.5），自身频率无效应（z = −2.0）。
若模型已捕捉 ⇒ 系数里 b_wr/a_wr 应显著；若无权重 ⇒ 有明确的头寸。
本脚本只读，不训练、不改任何文件。
"""
import io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_v2 as M
import veto_model as VM


def main():
    prep = M.prepare(verbose=False)
    mm = prep["mmodel"]                      # dict{scaler, clf, features}
    names = list(mm["features"])
    clf, scaler = mm["clf"], mm["scaler"]
    print("选图特征组 = %d 维" % len(names))
    print("模型 = LogisticRegression(C=%s)" % clf.C)
    coef = list(clf.coef_[0])
    pairs = sorted(zip(coef, names), key=lambda x: -abs(x[0]))
    print("\n按 |标准化系数| 排序（正号 = 倾向该图进入图集）：")
    print("%-22s %10s  %s" % ("特征", "coef", "解读"))
    print("-" * 76)
    NOTE = {
        "a_wr": "己方在该图胜率", "b_wr": "**对手**在该图胜率",
        "a_pick": "己方 pick 频率", "b_pick": "对手 pick 频率",
        "a_ban": "己方 ban 频率", "b_ban": "对手 ban 频率",
        "a_app": "己方在该图出场", "b_app": "对手在该图出场",
        "g_app": "该图全局出场", "new": "该图是否新图",
        "own_pick_free": "自己不会先 pick 的程度", "opp_pick_free": "**对手**不会先 pick 的程度",
        "both_keep": "双方都不 ban 的程度", "a_keep": "己方不 ban", "b_keep": "对手不 ban",
        "sim_in_set": "veto 模拟（自由顺序）", "sim2_in_set": "veto 模拟（分层）",
        "sim3_in_set": "veto 模拟（带先手规则）",
    }
    for c, n in pairs:
        print("%-22s %10.4f  %s" % (n, c, NOTE.get(n, "")))

    d = dict(zip(names, coef))
    print("\n== 与 H1 实测对照 ==")
    print("  a_wr（己方胜率） = %+.4f      b_wr（对手胜率） = %+.4f" % (d["a_wr"], d["b_wr"]))
    print("  a_pick（己方 pick）= %+.4f      b_pick（对手 pick）= %+.4f" % (d["a_pick"], d["b_pick"]))
    print("  a_ban （己方 ban） = %+.4f      b_ban （对手 ban） = %+.4f" % (d["a_ban"], d["b_ban"]))
    net = d["a_wr"] + d["b_wr"]
    print("\n  → 己方胜率与对手胜率的**净权重** = %+.4f" % net)
    print("     %s" % ("负值 ⇒ 模型确实在「倾向选对手不擅长的图」（与 H1 的 z=−11.5 同向）"
                      if net < 0 else
                      "正值 ⇒ 模型偏好「双方都擅长」的图，与 H1 实测**相反**"))
    print("\n  单特征量级对比：|a_wr|=%.4f  |b_wr|=%.4f  |sim2_in_set|=%.4f  |sim_in_set|=%.4f"
          % (abs(d["a_wr"]), abs(d["b_wr"]), abs(d["sim2_in_set"]), abs(d["sim_in_set"])))
    print("  ⇒ 对手弱图信号在模型里的相对地位：%s"
          % ("**主导**" if abs(net) > 0.5 * abs(d["sim2_in_set"]) else "**次要**（被 veto 模拟盖过）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
