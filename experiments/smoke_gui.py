"""predict_gui 的冒烟测试：起服务 → 等模型加载 → 打几个真实对局 → 关掉。

用子进程跑（Windows 下 -NoNewWindow 起 python 会被沙箱拦），主进程用 HTTP 探活。
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8799
BASE = "http://127.0.0.1:%d" % PORT


def get(path, tries=60):
    """轮询直到拿到**有意义**的响应。

    ⚠ 第一版只等「HTTP 能通」，而 `/api/status` 在模型未就绪时会正常返回
    `{"ready": false}`（没有 teams 字段）⇒ 测试脚本立刻 KeyError。
    状态端点的语义是「通了但还没好」，必须等 ready=True。
    """
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(BASE + path, timeout=8) as r:
                last = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return json.loads(e.read().decode("utf-8"))
        except Exception:                                # noqa: BLE001
            time.sleep(2)
            continue
        if path == "/api/status":
            if last.get("ready") or last.get("error"):
                return last
            time.sleep(2)
            continue
        return last
    return last


def main():
    p = subprocess.Popen([sys.executable, "-u", "-X", "utf8", "predict_gui.py",
                          "--port", str(PORT)], cwd=HERE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        st = get("/api/status")
        if st is None:
            print("FAIL 服务未起来")
            return 1
        if st.get("error"):
            print("FAIL 模型加载出错：%s" % st["error"])
            return 1
        print("== 1. /api/status ==")
        print("  ready=%s  队伍 %d 支  加载 %.1fs"
              % (st.get("ready"), len(st["teams"]), st["load_s"]))

        print("\n== 2. /（HTML 页面）==")
        with urllib.request.urlopen(BASE + "/", timeout=8) as r:
            html = r.read().decode("utf-8")
        need = ["队伍 A", "BO3", "predict", "期望得分", "丰度"]
        miss = [k for k in need if k not in html]
        print("  HTTP %s，长度 %d，关键元素缺失：%s"
              % (r.status, len(html), miss or "无"))

        print("\n== 3. /api/predict（真实对局，含简称与错输入）==")
        cases = [
            ("vitality", "furia", 3),      # 简称
            ("navi", "G2", 3),             # navi → Natus Vincere
            ("gamelegion", "bbl", 1),      # bo1 + gamerlegion 拼写
            ("legacy", "magic", 5),        # bo5
            ("mo", "tyloo", 3),            # 模糊/短输入
            ("不存在队", "G2", 3),          # 错误输入
            ("G2", "G2", 3),              # 同队
            ("G2", "FURIA", 4),           # 非法 bo
        ]
        ok = True
        for a, b, bo in cases:
            u = "/api/predict?a=%s&b=%s&bo=%d" % (urllib.parse.quote(a),
                                                  urllib.parse.quote(b), bo)
            r = get(u, tries=1)
            if "error" in r:
                print("  %-12s vs %-12s bo%d → 拒绝：%s" % (a, b, bo, r["error"]))
                continue
            print("  %-12s vs %-12s bo%d → %-14s %s  P=%.1f%%  E=%.1f/%d  警告%d条"
                  % (a, b, bo, r["A"] + " vs " + r["B"], r["winner"],
                     r["p_win"], r["exp_points"], r["max_points"],
                     len(r["warnings"])))
            if r["bo"] == 1 and r["max_points"] != 25:
                print("     ✗ bo1 满分应为 25")
                ok = False
            if r["bo"] == 5 and r["max_points"] != 55:
                print("     ✗ bo5 满分应为 55")
                ok = False
            if not r["winner"] or r["winner"] not in (r["A"], r["B"]):
                print("     ✗ winner 不在两队之中")
                ok = False

        print("\n== 4. 重复请求应即时（模型常驻）==")
        t = time.time()
        for _ in range(10):
            get("/api/predict?a=G2&b=FURIA&bo=3", tries=1)
        print("  10 次预测共 %.2fs（单次 %.3fs）"
              % (time.time() - t, (time.time() - t) / 10))
        print("\n%s" % ("SMOKE_OK" if ok else "SMOKE_FAIL"))
        return 0 if ok else 1
    finally:
        p.terminate()
        try:
            p.wait(timeout=8)
        except Exception:                                # noqa: BLE001
            p.kill()


if __name__ == "__main__":
    import urllib.parse
    sys.exit(main())
