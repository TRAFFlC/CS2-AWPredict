# -*- coding: utf-8 -*-
"""DS-11 veto 采集用的本地回环小服务（纯标准库）。

为什么需要：HLTV 是 HTTPS 页面，页内后台任务不能直接写本机文件。
本服务负责 ① 发工作清单 GET /worklist.json ② 收解析结果 POST /veto 并即时落盘。

去重语义：
  - 「好记录」（veto_ok=True）不再重取；
  - 「坏记录」（veto_ok=False，例如撞上限流页）**会被重新发回工作清单**，
    以便重跑时自动补取（新一轮 append 一行，标签构建时按 id 取最后一条/优先好记录）。

安全：只绑 127.0.0.1；CORS 放开只为跨源回传；除追加写本目录的 jsonl 外无副作用。
"""
import json, io, os, sys, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）
WORK = os.path.join(HERE, "veto_worklist.json")
OUT = os.path.join(DATA, "csapi_veto.jsonl")

_lock = threading.Lock()
_good, _bad = set(), set()
if os.path.exists(OUT):
    for line in io.open(OUT, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        (_good if r.get("veto_ok") else _bad).add(r["id"])
_bad -= _good   # 有任一好记录即算好


def _cors(h):
    h.send_header("Access-Control-Allow-Origin", "*")
    h.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    h.send_header("Access-Control-Allow-Headers", "Content-Type")
    h.send_header("Access-Control-Max-Age", "600")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_OPTIONS(self):
        self.send_response(204)
        _cors(self)
        self.end_headers()

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        _cors(self)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path, _, qs = self.path.partition("?")
        if path == "/progress":
            return self._json({"good": len(_good), "bad": len(_bad)})
        if path != "/worklist.json":
            return self._json({"err": "not found"}, 404)
        q = {}
        for kv in qs.split("&"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                q[k] = v
        off = int(q.get("offset", 0))
        lim = int(q.get("limit", 10 ** 9))
        work = json.load(io.open(WORK, encoding="utf-8"))
        if q.get("pending", "1") == "1":
            work = [w for w in work if (w["id"] not in _good) or (w["id"] in _bad)]
        return self._json({"total": len(work), "good": len(_good), "bad": len(_bad),
                           "items": work[off:off + lim]})

    def do_POST(self):
        if self.path != "/veto":
            return self._json({"err": "not found"}, 404)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            rec = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:
            return self._json({"ok": False, "err": "bad json: %s" % e})
        veto = rec.get("veto") or []
        picks = sorted([s[3] for s in veto if s[1] == "pi"])
        lo = [s[3] for s in veto if s[1] == "L"]
        played = sorted(rec.get("played") or [])
        rec["picks_match"] = (picks == played)
        rec["decider"] = lo[0] if lo else None
        rec["veto_ok"] = (len(veto) == 7 and len(lo) == 1)
        mid = rec["id"]
        with _lock:
            with io.open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec["veto_ok"]:
                _good.add(mid); _bad.discard(mid)
            elif mid not in _good:
                _bad.add(mid)
        return self._json({"ok": True, "veto_ok": rec["veto_ok"],
                           "picks_match": rec["picks_match"], "decider": rec["decider"],
                           "good": len(_good), "bad": len(_bad)})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8791
    print("VETO_SERVER_UP port=%d good=%d bad=%d" % (port, len(_good), len(_bad)), flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
