#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pick'Em 本地 Web 界面 —— issue-015「可交付使用」的那一面。

对标 issue-014 的 `predict_gui.py`：**本地 HTTP 服务 + 一个页面**，不用装依赖、不联网、
不读你的输入以外的东西。主界面只干一件事：

  **填 16 支参赛队伍（按种子顺序）→ 一键出十格**（2×3:0 ｜ 6×3:1·3:2 ｜ 2×0:3）。

每队强度**已经预先算好**，放在 `fixtures/team_strength.json`（公开强度库，provenance
写在库内；由私有侧 `015-scratch/build_strength.py` 一次性生成）。输入框支持按队名/简称
补全（navi → Natus Vincere、nip → Ninjas in Pyjamas）；查不到的队会**点名告警**并按
默认强度兜底，不静默。

其余能力全部收在「高级」折叠区里，默认看不见：

  * 模拟参数（n / 过关线 k / 随机种子 / 强度源 rank_points（默认）|elo / 平局键 / σ 臂）
  * 手填 roster JSON（含 **矩阵 源**——外部模型接入的桥）
  * 赛后**计分**（真值表 → 真实命中格数，`scoring.py`）
  * **校准**（真值表 + 报表 → Brier / 常数基线 / 可靠性分箱，`calibration.py`）

接口（够脚本化，不依赖页面）：

    GET  /                 页面
    GET  /api/status       {"ready": true, "version": ..., "fixtures": [...]}
    GET  /api/teams        强度库（队长表 + 别名 + rating/rank/points + asof/provenance）
    GET  /api/demo         默认示例阶段（16 个队名，页面开机预填）
    GET  /api/fixtures     内置 roster 清单（名字 → 阶段/源/队数）
    GET  /api/fixture?name=内置 roster 原文
    POST /api/pickem       {"teams": ["16 个队名"], "n":.., "k":.., "seed":.., "source":..,
                            "uncertain": bool, "random_ties": bool} → 十格 + 报表 + 告警
    POST /api/predict      {"roster": {...} | "fixture": "名字", ...} → 报表（字段同 /api/pickem）
    POST /api/score        {"truth": {...}, "card": {...}, "k": 5} → 真实命中
    POST /api/calibrate    {"truth": {...}, "report": {...}} → 校准指标

安全边界（本地工具也要守）：**只允许**读 `fixtures/*.json` 里的白名单文件（不做路径拼接），
请求体上限 1 MiB；默认只绑 `127.0.0.1`；任何异常都返回 JSON 错误而不是堆栈。

用法::

    python predict_gui.py                       # http://127.0.0.1:8790
    python predict_gui.py --port 9000 --host 0.0.0.0
    python predict_gui.py selftest              # 起服务 + 真实 HTTP 回环自证

**端口**：014 的 `predict_gui.py` 实测常驻 **8765**，本工具默认 **8790**，两者可并存。
端口被占一律 **rc=1** 报错、不静默回退——但**光靠 `OSError` 拦不住**：Windows 的
`SO_REUSEADDR`（`HTTPServer.allow_reuse_address=1`）语义是「允许抢绑」，
第二个进程会静默绑上已有人在 listen 的端口，于是同一端口有两个服务。
故 `serve()` 在绑定前先 `port_busy()` **主动探一次连接**（实测：8790 有人在 listen 时，
不加这一步照样 rc=0 打印就绪）。

退出码：0 正常；1 端口占用 / 参数非法 /（selftest）有断言失败。
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import calibration as CAL      # noqa: E402
import predict_stage as PS     # noqa: E402
import scoring as SC           # noqa: E402

VERSION = "pickem-gui 0.2"
POST_ENDPOINTS = ("/api/predict", "/api/pickem", "/api/score", "/api/calibrate",
                  "/api/vrs/refresh")
DEFAULT_PORT = 8790
MAX_BODY = 1 << 20                     # 1 MiB
FIXDIR = os.path.join(HERE, "fixtures")

PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Major · Pick'Em 十格预测</title>
<style>
:root{
  --bg:#06070a; --ink:#0a0c10; --card:rgba(13,16,21,.9); --card2:#12161d; --line:#1b202a;
  --line-2:#28303d; --fg:#e9edf4; --fg-2:#aeb7c5; --dim:#7b8598;
  --acc:#00e58a; --acc-2:#7dffc4; --acc-line:rgba(0,229,138,.42);
  --ok:#3ecf8e; --warn:#ffb020; --bad:#ff5c5c;
  --mono:ui-monospace,"Cascadia Mono",Consolas,"SF Mono",Menlo,monospace;
}
*{box-sizing:border-box}
body{margin:0;color:var(--fg);
  font:14px/1.6 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
  background:
    radial-gradient(1100px 560px at 80% -10%,rgba(0,229,138,.09),transparent 60%),
    radial-gradient(900px 700px at 8% 110%,rgba(0,229,138,.04),transparent 65%),
    var(--bg)}
.wrap{max-width:900px;margin:0 auto;padding:26px 18px 60px}
/* —— 场景切换：同一服务的两个入口 —— */
.modes{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:0 0 18px}
.modes a{display:grid;grid-template-columns:auto 1fr;grid-template-rows:auto auto;
  column-gap:11px;row-gap:1px;align-items:center;padding:10px 13px;
  border:1px solid var(--line);border-radius:8px;background:rgba(10,12,16,.66);
  color:var(--fg-2);text-decoration:none;
  transition:border-color .14s,background .14s,box-shadow .14s}
.modes a .k{grid-row:1 / 3;font:700 17px/1 var(--mono);color:var(--line-2);
  letter-spacing:.04em}
.modes a b{font-size:13px;color:var(--fg);font-weight:600;letter-spacing:.02em}
.modes a i{font-style:normal;font-size:11px;color:var(--dim)}
.modes a:hover{border-color:var(--acc-line);background:rgba(0,229,138,.06)}
.modes a.on{border-color:var(--acc-line);background:rgba(0,229,138,.08);
  box-shadow:inset 3px 0 0 var(--acc)}
.modes a.on .k{color:var(--acc)}
@media(max-width:640px){.modes{grid-template-columns:1fr}}
h1{font-size:21px;margin:0 0 4px;letter-spacing:.04em;color:var(--fg)}
h1::after{content:"";display:inline-block;width:8px;height:15px;margin-left:8px;
  background:var(--acc);vertical-align:-2px;box-shadow:0 0 10px var(--acc-line)}
h2{font-size:13px;color:var(--dim);margin:0 0 10px;text-transform:uppercase;letter-spacing:.6px}
.card h2{margin-bottom:12px}
.card>details:first-child summary{padding-top:2px}
.card>details:first-child summary::-webkit-details-marker{display:none}
.card>details:first-child summary::before{content:"\25B6 ";color:var(--dim)}
.sub{color:var(--dim);font-size:12px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:16px;margin-bottom:14px}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
@media(max-width:880px){.two-col{grid-template-columns:1fr}}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
.spread{justify-content:space-between}
label{display:block;font-size:12px;color:var(--dim);margin-bottom:5px}
select,input,textarea,button{width:100%;padding:9px 11px;border-radius:7px;
  border:1px solid var(--line);background:var(--ink);color:var(--fg);font-size:14px}
textarea{font:12px/1.5 ui-monospace,Consolas,monospace;min-height:118px;resize:vertical}
button{cursor:pointer;background:var(--acc);color:#04140d;font-weight:600;border:none;
  padding:10px 22px;line-height:1.4}
button:hover:not(:disabled){background:var(--acc-2)}

button.ghost{background:var(--ink);color:var(--dim);border:1px solid var(--line);font-weight:400;
  width:auto}
button.ghost:hover{color:var(--fg);border-color:#3a4250}
button:disabled{opacity:.5;cursor:not-allowed}
.hint{color:var(--dim);font-size:12px;margin-top:8px}
.teams{display:grid;grid-template-columns:1fr 1fr;gap:7px 16px;margin-top:4px}
@media(max-width:700px){.teams{grid-template-columns:1fr}}
.t{display:flex;align-items:center;gap:8px}
.t .n{flex:0 0 24px;text-align:right;color:var(--dim);font-size:12px;
  font-variant-numeric:tabular-nums}
.t .f{position:relative;flex:1 1 auto;min-width:0}
.t input{padding:8px 10px}
.t .tag{flex:0 0 96px;font-size:11px;color:var(--dim);text-align:right;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.t .tag.bad{color:var(--warn)}
.lst{position:absolute;top:100%;left:0;right:0;z-index:60;margin-top:4px;display:none;
  max-height:250px;overflow:auto;background:#141821;border:1px solid var(--line);
  border-radius:8px;box-shadow:0 10px 28px rgba(0,0,0,.55)}
.lst.on{display:block}
.it{padding:6px 10px;cursor:pointer;border-bottom:1px solid #1c212b;display:flex;
  justify-content:space-between;gap:10px}
.it:last-child{border-bottom:none}
.it:hover{background:#1d2330}
.it b{font-weight:600}
.it .tag{font-size:11px;color:var(--dim);white-space:nowrap;flex:0 0 auto}
.ten{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:14px}
@media(max-width:700px){.ten{grid-template-columns:1fr}}
.gh{font-size:12px;color:var(--dim);margin-bottom:6px}
.gh b{color:var(--fg);font-weight:600}
.pk{background:var(--card2);border:1px solid var(--line);border-radius:8px;
  padding:8px 10px;margin-bottom:6px}
.pk .top{display:flex;justify-content:space-between;gap:8px;align-items:baseline}
.pk .nm{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pk .no{color:var(--dim);font-size:11px;font-weight:400;margin-right:6px}
.pk .pp{color:var(--dim);font-size:12px;font-variant-numeric:tabular-nums;flex:0 0 auto}
.bar{height:6px;background:#0e1117;border-radius:4px;overflow:hidden;margin-top:6px;
  border:1px solid var(--line)}
.bar>i{display:block;height:100%;background:var(--acc)}
.pk.hit{border-color:#1f5c3d;background:#12241a}
.pk.miss{border-color:#6b2020;background:#241211}
.kv{display:flex;gap:8px;margin-top:14px;flex-wrap:wrap}
.kv div{flex:1 1 128px;background:#0e1117;border:1px solid var(--line);border-radius:8px;
  padding:9px 11px}
.kv b{display:block;font-size:11px;color:var(--dim);font-weight:400;margin-bottom:2px}
.kv span{font-size:16px;font-variant-numeric:tabular-nums}
.pill{display:inline-block;font-size:11px;padding:2px 8px;border-radius:20px;background:#0e1117;
  border:1px solid var(--line);color:var(--dim);margin:0 6px 6px 0}
.pill b{color:var(--fg);font-weight:600}
.pill.ok{border-color:#1f5c3d;color:var(--ok)}
.pill.warn{border-color:#6b4a10;color:var(--warn);background:#1d1608}
.pill.bad{border-color:#6b2020;color:var(--bad);background:#1d0e0e}
ul.w{margin:10px 0 0;padding-left:18px;color:var(--warn);font-size:13px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}
th,td{text-align:right;padding:5px 7px;border-bottom:1px solid #222a36;white-space:nowrap}
th:first-child,td:first-child,th.l,td.l{text-align:left}
th{color:var(--dim);font-weight:500;font-size:12px}
td.good{color:var(--ok);font-weight:600}
td.bad{color:var(--bad);font-weight:600}
tbody tr:last-child td{border-bottom:none}
details{margin-top:12px}
summary{cursor:pointer;color:var(--dim);font-size:12.5px}
summary:hover{color:var(--fg)}
.card>details summary{padding:6px 0 2px;align-items:center}
.card>details summary::-webkit-details-marker{display:none}
.card>details summary::before{content:"\25B6 ";color:var(--dim);margin-right:6px}
.card>details summary:hover::before{color:var(--fg)}
.err{color:var(--bad);padding:8px 0;font-size:13px;white-space:pre-wrap}
code{background:#0b0f14;border:1px solid var(--line);border-radius:5px;padding:1px 5px;
  font:12px ui-monospace,Consolas,monospace}
pre.cmd{background:var(--ink);border:1px solid var(--line);border-radius:8px;padding:9px 11px;
  overflow-x:auto;font:12px ui-monospace,Consolas,monospace;color:#9ff5cd;margin:8px 0 0}
.f4{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
@media(max-width:700px){.f4{grid-template-columns:1fr 1fr}}
.spin{display:inline-block;width:11px;height:11px;border:2px solid #1f5c3d;
  border-top-color:#04140d;border-radius:50%;animation:s .8s linear infinite;
  vertical-align:-1px;margin-right:6px}
@keyframes s{to{transform:rotate(360deg)}}

</style>
</head>
<body>
<div class="wrap">
__MODE_NAV__
<h1>Major · 竞猜十格</h1>
<div class="sub">16 队瑞士轮 Pick&rsquo;Em · 2&times;3:0 ｜ 6&times;3:1&middot;3:2 ｜ 2&times;0:3 · 本地运行，不联网 &middot; <span id="ver">__VERSION__</span></div>

<div class="card">
  <div class="row spread">
    <span class="pill" id="libp">强度库加载中…</span>
    <span class="pill" id="vrsp">VRS 检查中…</span>
    <span class="row" style="gap:8px">
      <button class="ghost" id="autoseed" style="padding:7px 12px"
              title="识别 1v2、3v4… 的首轮对阵顺序并转成种子序；否则按 VRS 排名排序">自动排种子</button>
      <button class="ghost" id="demo" style="padding:7px 12px">载入示例阶段</button>
      <button class="ghost" id="clear" style="padding:7px 12px">清空</button>
    </span>
  </div>
  <label style="margin-top:12px">按 16 支参赛队伍的官方种子顺序填入（1 号位 ↔ 16 号位）；不确定就点「自动排种子」</label>
  <div class="teams" id="teams"></div>
  <div class="row" style="margin-top:14px;gap:14px">
    <span style="flex:0 0 auto"><button id="go" disabled>预测十格</button></span>
    <span class="hint" id="st" style="margin:0">正在加载强度库…</span>
  </div>
  <div class="hint">支持简称/别名：<code>navi</code> &rarr; Natus Vincere、<code>nip</code> &rarr;
    Ninjas in Pyjamas、<code>vp</code> &rarr; Virtus.pro；点候选即选中。
    查不到的队会被点名告警、并按默认强度兜底（该队结果不可信）。</div>
</div>

<div id="out"></div>

<div class="card">
  <h2>高级</h2>

  <details>
    <summary>模拟参数 / 手填 roster（含 矩阵 源）</summary>
    <div class="f4" style="margin-top:10px">
      <div><label>模拟场景数 n</label>
        <select id="n">
          <option value="2000">2000（快）</option>
          <option value="8000">8000</option>
          <option value="20000" selected>20000（默认）</option>
          <option value="50000">50000（慢）</option>
        </select></div>
      <div><label>过关线 k</label>
        <select id="k"><option>5</option><option>6</option><option>7</option></select></div>
      <div><label>随机种子</label><input type="number" id="seed" value="20261012"></div>
      <div><label>强度源</label>
        <select id="source"><option value="rank_points" selected>rank_points（VRS 积分，默认）</option>
          <option value="elo">elo（Elo 参考）</option></select></div>
      <div><label>阶段名（可留空）</label>
        <input id="stage" placeholder="如 StarLadder Budapest Major 2025 Stage 1"></div>
    </div>
    <div class="row" style="margin-top:10px;gap:16px">
      <label style="display:flex;align-items:center;gap:7px;margin:0;color:var(--fg);font-size:13px">
        <input type="checkbox" id="rt" style="width:auto"> 同 Buchholz 并列改均匀抽签</label>
      <label style="display:flex;align-items:center;gap:7px;margin:0;color:var(--fg);font-size:13px">
        <input type="checkbox" id="unc" style="width:auto"> σ 不确定性传播臂
        <span class="hint" style="margin:0">（需每队 exposure/games，强度库不含，故只对手填 roster 有效）</span></label>
    </div>
    <label style="margin-top:14px">手填 roster JSON（source=rank_points / elo / 矩阵）</label>
    <textarea id="roster" spellcheck="false" placeholder='{"stage":"…","source":"rank_points","teams":[{"name":"…","seed":1,"points":1500},…]}'></textarea>
    <div class="row" style="margin-top:8px;gap:8px">
      <span style="flex:0 0 auto"><button class="ghost" id="runraw" style="padding:8px 14px">用手填的 roster 预测</button></span>
      <span style="flex:0 0 auto"><button class="ghost" id="fillraw" style="padding:8px 14px">用上方 16 队填入</button></span>
      <span style="flex:0 0 auto"><button class="ghost" id="dlroster" style="padding:8px 14px" disabled>下载上次预测的 roster</button></span>
    </div>
    <div id="rerr"></div>
  </details>

  <h2 style="margin-top:20px">赛后校验</h2>
  <div class="two-col">
    <div>
      <details style="margin-top:0">
        <summary>赛后计分（真实命中；真值表留私有）</summary>
        <label style="margin-top:10px">真值表 JSON（16 队最终战绩）
          <span class="hint" style="display:inline">必须是完整瑞士阶段：3 胜恰 8（其中 3-0 恰 2）、
            3 败恰 8（其中 0-3 恰 2），否则拒绝</span></label>
        <textarea id="truth" spellcheck="false" style="min-height:150px;line-height:1.4"
          placeholder='{"stage":"…","records":{"FaZe":[3,2],"FlyQuest":[3,0],"…":[2,3]}}'></textarea>
        <div class="row" style="margin-top:8px">
          <span style="flex:0 0 auto"><button class="ghost" id="score" style="padding:8px 14px">用上次预测的十格计分</button></span>
        </div>
        <div id="serr" style="margin-top:8px"></div>
        <div id="sout" style="margin-top:10px"></div>
      </details>
    </div>
    <div>
      <details style="margin-top:0">
        <summary>校准检验（逐队边际 vs 真值）</summary>
        <label style="margin-top:10px">报表 JSON（含 marginals；跑一次预测后自动填入）</label>
        <textarea id="report" spellcheck="false" style="min-height:150px;line-height:1.4" placeholder="运行一次预测后自动填入"></textarea>
        <div class="row" style="margin-top:8px">
          <span style="flex:0 0 auto"><button class="ghost" id="cal" style="padding:8px 14px">跑校准</button></span>
        </div>
        <div id="cerr" style="margin-top:8px"></div>
        <div id="cout" style="margin-top:10px"></div>
      </details>
    </div>
  </div>
</div>
<script>
const $ = (s) => document.getElementById(s);
const esc = (s) => String(s === null || s === undefined ? "" : s)
  .replace(/[&<>]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c]));
const pct = (x) => (x === null || x === undefined) ? "—" : (100 * Number(x)).toFixed(1) + "%";
const fx = (x, d) => (x === null || x === undefined) ? "—" : Number(x).toFixed(d);
const srcLabel = (s) => s === "rank_points" ? "VRS 积分"
  : (s === "matrix" || s === "矩阵") ? "矩阵" : "Elo";

let LIB = null, INDEX = {}, LAST = {}, RUNNING = false;

async function jget(u) {
  const r = await fetch(u);
  const t = await r.text();
  let j = null; try { j = JSON.parse(t); } catch (e) {}
  if (!r.ok) throw new Error((j && j.error) || t.slice(0, 300) || ("HTTP " + r.status));
  return j;
}
async function jpost(u, b) {
  const r = await fetch(u, {method: "POST", headers: {"Content-Type": "application/json"},
                            body: JSON.stringify(b)});
  const t = await r.text();
  let j = null; try { j = JSON.parse(t); } catch (e) {}
  if (!r.ok) throw new Error((j && j.error) || t.slice(0, 300) || ("HTTP " + r.status));
  return j;
}

function suggest(q) {
  const s = String(q || "").trim().toLowerCase();
  let list = LIB ? LIB.teams : [];
  if (s) list = list.filter((t) => t.name.toLowerCase().indexOf(s) >= 0 ||
    (t.aliases || []).some((a) => a.toLowerCase().indexOf(s) >= 0));
  return list.slice(0, 40);
}
function recByQuery(q) {
  const s = String(q || "").trim().toLowerCase();
  if (!s) return null;
  if (INDEX[s]) return INDEX[s];
  const hits = suggest(q);
  return hits.length === 1 ? hits[0] : null;
}
function strengthTag(t) {
  if (!t) return "⚠ 未收录";
  if ($("source").value === "rank_points") {
    if (t.points === null || t.points === undefined) {
      return t.rank !== null && t.rank !== undefined ? "⚠ 无 VRS 分 · #" + t.rank : "⚠ 无 VRS 分";
    }
    return "VRS " + (t.rank ? "#" + t.rank + " · " : "") + Math.round(t.points);
  }
  if (t.rating === null || t.rating === undefined) {
    return t.rank !== null && t.rank !== undefined ? "⚠ 无 Elo · VRS #" + t.rank : "⚠ 无 Elo";
  }
  return "Elo " + Math.round(t.rating) + (t.rank ? " · VRS #" + t.rank : "");
}

function buildSlots() {
  let html = "";
  for (let i = 0; i < 16; i++) {
    html += '<div class="t"><span class="n">' + (i + 1) + '</span>' +
      '<span class="f"><input id="t' + i + '" autocomplete="off" spellcheck="false" ' +
      'placeholder="队伍名"><div class="lst" id="l' + i + '"></div></span>' +
      '<span class="tag" id="g' + i + '"></span></div>';
  }
  $("teams").innerHTML = html;
  for (let i = 0; i < 16; i++) bindSlot(i);
}
function bindSlot(i) {
  const inp = $("t" + i), lst = $("l" + i);
  function closeAll() { for (let k = 0; k < 16; k++) $("l" + k).classList.remove("on"); }
  function draw() {
    closeAll();
    const list = suggest(inp.value);
    lst.innerHTML = list.map((t) => '<div class="it" data-n="' + esc(t.name) + '"><b>' +
      esc(t.name) + '</b><span class="tag">' + esc(strengthTag(t)) + '</span></div>').join("") ||
      '<div class="it">无匹配</div>';
    lst.classList.add("on");
    lst.querySelectorAll(".it[data-n]").forEach((el) => {
      el.onclick = () => {
        inp.value = el.getAttribute("data-n");
        lst.classList.remove("on"); inp.blur(); refresh();
      };
    });
  }
  inp.onfocus = draw;
  inp.oninput = () => { refresh(); draw(); };
  inp.onblur = () => setTimeout(() => lst.classList.remove("on"), 180);
  lst.onmousedown = (e) => e.preventDefault();
  inp.onkeydown = (e) => {
    if (e.key === "Enter") { e.preventDefault(); lst.classList.remove("on"); go(); }
    else if (e.key === "Escape") { lst.classList.remove("on"); inp.blur(); }
  };
}
document.addEventListener("mousedown", (e) => {
  if (!e.target.closest(".t")) {
    for (let i = 0; i < 16; i++) { const l = $("l" + i); if (l) l.classList.remove("on"); }
  }
});

function refresh() {
  let filled = 0, unknown = 0;
  for (let i = 0; i < 16; i++) {
    const v = $("t" + i).value.trim(), tag = $("g" + i), rec = recByQuery(v);
    if (!v) { tag.textContent = ""; tag.className = "tag"; continue; }
    filled++;
    if (!rec) { unknown++; tag.textContent = "⚠ 未收录"; tag.className = "tag bad"; }
    else { tag.textContent = strengthTag(rec); tag.className = "tag"; }
  }
  $("go").disabled = RUNNING || filled < 16;
  if (LIB) {
    $("st").textContent = "强度库 " + LIB.teams.length + " 支（VRS " + LIB.asof +
      (LIB.elo_asof ? " · Elo " + LIB.elo_asof : "") + "）· 已填 " +
      filled + "/16" + (unknown ? " · " + unknown + " 支未收录（会被点名告警）" : "");
  }
}
function slotNames() {
  const out = [];
  for (let i = 0; i < 16; i++) out.push($("t" + i).value.trim());
  return out;
}

function sameRankSet(a, b) {
  if (a.length !== b.length) return false;
  const sa = a.slice().sort((x, y) => x - y), sb = b.slice().sort((x, y) => x - y);
  return sa.every((v, i) => v === sb[i]);
}

function autoSeed() {
  const names = slotNames();
  if (names.length !== 16 || names.some((n) => !n)) {
    $("st").textContent = "先把 16 支队填满，再自动排种子。";
    return;
  }
  const recs = names.map((n) => recByQuery(n));
  if (recs.some((r) => !r || r.rank == null)) {
    $("st").textContent = "有队伍查不到 VRS 排名，无法自动排种子；请手动调整。";
    return;
  }
  const ranks = recs.map((r) => r.rank);
  const sorted = ranks.slice().sort((a, b) => a - b);
  const byRank = names.slice().sort((a, b) => recByQuery(a).rank - recByQuery(b).rank);
  let order, msg;

  // 首轮对阵表顺序：1v2、3v4…，每对第一位是高位种子，且偶数位正好是 VRS 前 8。
  let bracket = ranks.every((r, i) => i % 2 === 0 || ranks[i - 1] < r) &&
    sameRankSet(ranks.filter((_, i) => i % 2 === 0), sorted.slice(0, 8));
  if (bracket) {
    const top = [], bottom = [];
    for (let i = 0; i < 16; i += 2) { top.push(names[i]); bottom.push(names[i + 1]); }
    order = top.concat(bottom);
    msg = "已识别首轮对阵顺序，重排为种子序（前 8 + 后 8）。";
  } else if (sameRankSet(ranks.slice(0, 8), sorted.slice(0, 8))) {
    // 官方种子序可能因区域避让微调 9–16 位（如 MIBR #12 排到 14 号位），不要按 VRS 重排。
    order = names.slice();
    msg = "看起来已是种子序（前 8 正好是 VRS 前 8），保留原顺序。";
  } else {
    order = byRank;
    msg = "已按 VRS 排名排序（如官方抽签不同，请手动调整）。";
  }
  order.forEach((n, i) => { $("t" + i).value = n; });
  refresh();
  $("st").textContent = msg;
}

/* ---------------- 结果渲染 ---------------- */
function slotsHtml(rep, hit) {
  const marg = rep.marginals;
  const groups = [["p30", "3:0", "2 格"], ["padv", "3:1 / 3:2", "6 格"], ["p03", "0:3", "2 格"]];
  let n = 0;
  const cols = groups.map((g) => {
    const c = g[0];
    const list = rep.picks[c].slice().sort((a, b) => marg[b][c] - marg[a][c]);
    const cells = list.map((t) => {
      n++;
      const p = marg[t][c];
      let cls = "pk";
      if (hit && hit[c]) cls += hit[c].indexOf(t) >= 0 ? " hit" : " miss";
      return '<div class="' + cls + '"><div class="top"><span class="nm">' +
        '<span class="no">格 ' + n + '</span>' + esc(t) + '</span><span class="pp">' +
        pct(p) + '</span></div><div class="bar"><i style="width:' +
        (100 * p).toFixed(1) + '%"></i></div></div>';
    }).join("");
    return '<div><div class="gh"><b>' + g[1] + '</b> · ' + g[2] + '</div>' + cells + '</div>';
  }).join("");
  return '<div class="ten" id="pickbox">' + cols + '</div>';
}
function marGHtml(rep) {
  const marg = rep.marginals, seeds = rep.seeds, st = rep.strength;
  const isM = rep.source === "矩阵";
  const rows = Object.keys(marg).sort((a, b) => seeds[a] - seeds[b]).map((t) =>
    '<tr><td class="l">' + seeds[t] + '</td><td class="l">' + esc(t) + '</td><td>' +
    (isM ? Number(st[t]).toFixed(3) : Number(st[t]).toFixed(1)) + '</td><td>' +
    marg[t].p30.toFixed(3) + '</td><td>' + marg[t].padv.toFixed(3) + '</td><td>' +
    marg[t].p03.toFixed(3) + '</td></tr>').join("");
  return '<table><thead><tr><th class="l">种子</th><th class="l">队伍</th><th>' +
    (isM ? "对场胜率" : "强度") + '</th><th>P(3:0)</th><th>P(3:1·3:2)</th>' +
    '<th>P(0:3)</th></tr></thead><tbody>' + rows + '</tbody></table>';
}
function baseHtml(rep) {
  const b = rep.baselines;
  if (!b) return "";
  const rows = ["optimizer", "naive_rank", "hot"].map((k) => {
    const v = b[k];
    const diff = (c) => b.optimizer.expected_by_cat[c] - v.expected_by_cat[c];
    const name = k === "optimizer" ? "优化器" : (k === "naive_rank" ? "naive 种子序" : "热门基线");
    return '<tr><td class="l">' + name + '</td><td>' + v.p_ge_k.toFixed(4) + '</td><td>' +
      v.expected_hits.toFixed(2) + '</td><td>' + diff("p30").toFixed(2) + '</td><td>' +
      diff("padv").toFixed(2) + '</td><td>' + diff("p03").toFixed(2) + '</td></tr>';
  }).join("");
  const same = ["p30", "padv", "p03"].every((c) =>
    JSON.stringify(b.optimizer.picks[c].slice().sort()) ===
    JSON.stringify(b.hot.picks[c].slice().sort()));
  return '<table><thead><tr><th class="l">策略</th><th>P(≥' + rep.k + ')</th><th>期望命中</th>' +
    '<th>Δ3:0</th><th>Δ晋级</th><th>Δ0:3</th></tr></thead><tbody>' + rows + '</tbody></table>' +
    (same ? '<div class="hint">本例热门基线与优化器选出<b>同一批格子</b>：搜索未带来增益。</div>' : "");
}
function render(j) {
  const rep = j.report;
  LAST = {report: rep, card: {picks: rep.picks, baselines: rep.baselines}, command: j.command,
          freeze: j.freeze, roster: j.roster, roster_file: j.roster_file,
          warnings: j.warnings || []};
  const sched = rep.schedule, okSched = sched.bo1 === 20 && sched.bo3 === 13;
  const ind = rep.independent_p_ge_k, d = rep.p_ge_k - ind;
  const warn = LAST.warnings.length
    ? '<ul class="w">' + LAST.warnings.map((w) => '<li>' + esc(w) + '</li>').join("") + '</ul>' : "";
  $("out").innerHTML =
    '<div class="card">' +
    '<div>' +
      '<span class="pill"><b>' + esc(rep.stage) + '</b></span>' +
      '<span class="pill">强度源 ' + esc(srcLabel(rep.source)) + '</span>' +
      '<span class="pill">模拟 ' + rep.n + ' 次 · seed ' + rep.seed + '</span>' +
      '<span class="pill">耗时 ' + fx(j.elapsed_s, 2) + ' s</span>' +
      '<span class="pill">臂 ' + (rep.arm === "uncertain" ? "σ 不确定性" : "点估计") + '</span>' +
      '<span class="pill">平局键 ' + (rep.tie_break === "random" ? "随机抽签" : "种子升序") + '</span>' +
      '<span class="pill ' + (okSched ? "ok" : "warn") + '">赛程校验 bo1=' + sched.bo1 +
        ' / bo3=' + sched.bo3 + (okSched ? "" : "（应 20 / 13）") + '</span>' +
    '</div>' +
    slotsHtml(rep, null) +
    '<div class="kv">' +
      '<div><b>P(命中 ≥' + rep.k + ')</b><span>' + fx(rep.p_ge_k, 4) + '</span></div>' +
      '<div><b>P(≥6)</b><span>' + fx(rep.grid_p["6"], 4) + '</span></div>' +
      '<div><b>P(≥7)</b><span>' + fx(rep.grid_p["7"], 4) + '</span></div>' +
      '<div><b>独立假设 P(≥' + rep.k + ')</b><span>' + fx(ind, 4) + '</span></div>' +
      '<div><b>相关性修正</b><span>' + (d >= 0 ? "+" : "") + fx(d, 4) + '</span></div>' +
    '</div>' +
    '<div class="hint">P(≥k) 是<b>这一组十格</b>在模拟分布下的过关概率，不是模型准确率；' +
      '十格强相关，故与「独立假设」列有差。优化器评估 ' + rep.evals + ' 次。</div>' +
    warn +
    '<details><summary>逐队最终战绩分布（模拟）</summary>' + marGHtml(rep) + '</details>' +
    '<details><summary>三路基线对照（同一批场景，不是准确率检验）</summary>' + baseHtml(rep) + '</details>' +
    '<details><summary>等价命令行 / 赛前冻结件</summary>' +
      '<pre class="cmd">' + esc(j.command) + '</pre>' +
      '<div class="row" style="margin-top:8px;gap:8px">' +
        '<span style="flex:0 0 auto"><button class="ghost" id="dlr2" style="padding:7px 12px">下载 roster（' +
          esc(j.roster_file) + '）</button></span>' +
        '<span style="flex:0 0 auto"><button class="ghost" id="frz" style="padding:7px 12px">下载赛前冻结件</button></span>' +
      '</div>' +
      '<div class="hint">下载 roster 后，把上面的命令行整行粘进终端即可复现同一批数字；' +
        '冻结件另含输入 SHA-256 与参数。</div>' +
    '</details>' +
    '</div>';
  $("dlr2").onclick = () => dl(JSON.stringify(LAST.roster, null, 1), LAST.roster_file);
  $("frz").onclick = () => {
    const f = LAST.freeze || {}, stage = String(f.stage || "stage").replace(/[^\w\u4e00-\u9fa5]+/g, "_");
    dl(JSON.stringify({freeze: f, report: LAST.report}, null, 1),
       "freeze_" + stage + "_" + String(f.utc || "").replace(/[:]/g, "") + ".json");
  };
  $("dlroster").disabled = false;
}
function dl(text, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], {type: "application/json"}));
  a.download = name; a.click(); URL.revokeObjectURL(a.href);
}

async function go() {
  const teams = slotNames();
  const miss = teams.filter((t) => !t).length;
  if (miss) {
    $("out").innerHTML = '<div class="card"><div class="err">16 个位置都要填（还差 ' + miss +
      ' 个）</div></div>';
    return;
  }
  RUNNING = true; $("go").disabled = true;
  $("go").innerHTML = '<span class="spin"></span>模拟中…';
  $("out").innerHTML = "";
  try {
    const j = await jpost("/api/pickem", {teams: teams, n: +$("n").value, k: +$("k").value,
      seed: +$("seed").value, source: $("source").value,
      stage: $("stage").value.trim() || null,
      uncertain: $("unc").checked, random_ties: $("rt").checked});
    render(j);
    $("report").value = JSON.stringify(j.report);
    $("roster").value = JSON.stringify(j.roster, null, 1);
  } catch (e) {
    $("out").innerHTML = '<div class="card"><div class="err">' + esc(e.message) + '</div></div>';
  } finally {
    RUNNING = false; $("go").innerHTML = "预测十格"; refresh();
  }
}

async function runRaw() {
  let roster;
  try { roster = JSON.parse($("roster").value); }
  catch (e) { return; }
  $("rerr").innerHTML = "";
  try {
    const j = await jpost("/api/predict", {roster: roster, n: +$("n").value,
      k: +$("k").value, seed: +$("seed").value, uncertain: $("unc").checked,
      random_ties: $("rt").checked});
    render(j);
    $("report").value = JSON.stringify(j.report);
  } catch (e) { $("rerr").innerHTML = '<div class="err">' + esc(e.message) + '</div>'; }
}

function fillRaw() {
  const src = $("source").value;
  const teams = slotNames().map((v, i) => {
    const rec = recByQuery(v);
    const row = {name: rec ? rec.name : v, seed: i + 1};
    if (src === "rank_points") row.points = (rec && rec.points != null) ? rec.points : 0.0;
    else row.rating = (rec && rec.rating != null) ? rec.rating : 1500.0;
    if (rec && rec.rank != null) row.rank = rec.rank;
    return row;
  });
  $("roster").value = JSON.stringify({stage: $("stage").value.trim() || "手填阶段",
                                      source: src, teams: teams}, null, 1);
  $("rerr").innerHTML = "";
}

async function runScore() {
  let truth;
  try { truth = JSON.parse($("truth").value); }
  catch (e) { $("serr").innerHTML = '<div class="err">真值表 JSON 解析失败：' + esc(e.message) + '</div>'; return; }
  if (!LAST.card) { $("serr").innerHTML = '<div class="err">先跑一次预测（计分用的是上一次的十格）</div>'; return; }
  $("serr").innerHTML = "";
  try {
    const r = await jpost("/api/score", {truth: truth, card: LAST.card, k: +$("k").value});
    const rows = Object.entries(r.results).map((kv) => {
      const name = kv[0], v = kv[1];
      const label = name === "optimizer" ? "优化器" : (name === "naive_rank" ? "naive 种子序" : name);
      return '<tr><td class="l">' + esc(label) + '</td><td>' + v.hits + '/10</td><td class="' +
        (v.passed ? "good" : "bad") + '">' + (v.passed ? "过" : "不过") + '</td><td>' +
        v.by_cat.p30 + '/2</td><td>' + v.by_cat.padv + '/6</td><td>' + v.by_cat.p03 +
        '/2</td></tr>';
    }).join("");
    const det = Object.entries(r.results).map((kv) => {
      const v = kv[1];
      const miss = Object.entries(v.miss_detail).map((mv) =>
        '<tr><td class="l">' + esc(mv[0]) + '</td><td class="l">填 ' + esc(mv[1].picked_as) +
        '</td><td class="l">实为 ' + esc(mv[1].actual) + '</td></tr>').join("");
      return '<div style="margin-top:10px"><div class="gh">' + esc(kv[0]) + ' · 命中 ' + v.hits +
        '/10 ' + (v.passed ? "（过）" : "（不过）") + '</div><table><thead><tr><th class="l">未中格</th>' +
        '<th class="l">填</th><th class="l">实为</th></tr></thead><tbody>' +
        (miss || '<tr><td class="l" colspan="3">无（全中）</td></tr>') + '</tbody></table></div>';
    }).join("");
    const hit = (r.results.optimizer || {}).hit;
    if (hit && LAST.report) {
      const box = $("pickbox");
      if (box) box.outerHTML = slotsHtml(LAST.report, hit);
    }
    $("sout").innerHTML = '<div style="margin-top:12px"><div class="gh">真实命中（赛后量，与赛前的 ' +
      'P(≥k) 是两回事）· 上面十格卡片：绿=命中、红=未中</div><table><thead><tr><th class="l">策略</th>' +
      '<th>命中</th><th>过关</th><th>3:0</th><th>3:1·3:2</th><th>0:3</th></tr></thead><tbody>' +
      rows + '</tbody></table></div>' + det;
  } catch (e) { $("serr").innerHTML = '<div class="err">' + esc(e.message) + '</div>'; }
}

async function runCal() {
  let truth, rep;
  try { truth = JSON.parse($("truth").value); }
  catch (e) { $("cerr").innerHTML = '<div class="err">真值表 JSON 解析失败：' + esc(e.message) + '</div>'; return; }
  try { rep = JSON.parse($("report").value); }
  catch (e) { $("cerr").innerHTML = '<div class="err">报表 JSON 解析失败：' + esc(e.message) + '</div>'; return; }
  $("cerr").innerHTML = "";
  try {
    const r = await jpost("/api/calibrate", {truth: truth, report: rep});
    const d = r.result.reports[0], c = d.categories;
    const lab = (k) => k === "p30" ? "3:0" : (k === "padv" ? "3:1·3:2" : "0:3");
    const rows = ["p30", "padv", "p03"].map((k) => {
      const v = c[k];
      return '<tr><td class="l">' + lab(k) + '</td><td>' + v.brier.toFixed(4) + '</td><td>' +
        v.brier_base.toFixed(4) + '</td><td class="' + (v.delta <= 0 ? "good" : "bad") + '">' +
        (v.delta >= 0 ? "+" : "") + v.delta.toFixed(4) + '</td><td>' + v.logloss.toFixed(4) +
        '</td><td>' + v.base_rate.toFixed(3) + '</td><td class="l">' +
        (v.better_than_base ? "优于常数基线" : "⚠ 劣于常数基线") + '</td></tr>';
    }).join("");
    const bins = ["p30", "padv", "p03"].map((k) => {
      const rb = c[k].reliability.map((b) => "[" + b.lo.toFixed(1) + "-" + b.hi.toFixed(1) +
        ") n=" + b.n + " 预测" + b.pred.toFixed(3) + " 实测" + b.obs.toFixed(3)).join(" ｜ ");
      return '<div class="hint">' + lab(k) + ' 可靠性分箱：' + (rb || "—") + '</div>';
    }).join("");
    $("cout").innerHTML = '<div style="margin-top:12px"><div class="gh">校准（' + esc(d.label) +
      ' · ' + esc(d.stage) + '）</div><table><thead><tr><th class="l">类别</th><th>Brier</th>' +
      '<th>常数基线</th><th>差</th><th>LogLoss</th><th>实测率</th><th class="l">判定</th>' +
      '</tr></thead><tbody>' + rows + '</tbody></table>' + bins +
      '<div class="hint">常数基线 = 永远输出该类平均发生率；判据是「能否击败它」。' +
      '单阶段 16 队噪声大，看差的符号与量级。</div></div>';
  } catch (e) { $("cerr").innerHTML = '<div class="err">' + esc(e.message) + '</div>'; }
}

$("go").onclick = go;
$("autoseed").onclick = autoSeed;
$("runraw").onclick = runRaw;
$("fillraw").onclick = fillRaw;
$("score").onclick = runScore;
$("cal").onclick = runCal;
$("dlroster").onclick = () => dl(JSON.stringify(LAST.roster, null, 1), LAST.roster_file || "roster.json");
$("clear").onclick = () => {
  for (let i = 0; i < 16; i++) $("t" + i).value = "";
  refresh();
};
$("demo").onclick = () => loadDemo();
$("source").onchange = refresh;

async function loadDemo() {
  try {
    const d = await jget("/api/demo");
    for (let i = 0; i < 16; i++) $("t" + i).value = d.teams[i] || "";
    if (d.stage) $("stage").value = d.stage;
    refresh();
  } catch (e) { $("st").textContent = "示例加载失败：" + e.message; }
}

async function loadLib() {
  const t = await jget("/api/teams");
  LIB = t; INDEX = {};
  t.teams.forEach((r) => { INDEX[r.name.toLowerCase()] = r; });
  t.teams.forEach((r) => (r.aliases || []).forEach((a) => {
    if (!INDEX[a.toLowerCase()]) INDEX[a.toLowerCase()] = r;
  }));
  const eloAsOf = t.elo_asof ? " · Elo 参考 " + esc(t.elo_asof) : "";
  $("libp").innerHTML = "强度库 <b>" + t.teams.length + "</b> 支 · 默认 VRS 积分 · VRS " + esc(t.asof) + eloAsOf;
  $("libp").title = (t._provenance || "") + " ｜ 覆盖：" + JSON.stringify(t.counts || {});
  return t;
}

// ---- VRS 自动抓取（2026-10-29）--------------------------------
// 页面每次加载都问服务端「北京时间现在该抓哪一天」；需要时后台抓 HLTV，
// 不阻塞页面。同一天的榜单会被 HLTV 赛后重算，所以服务端每小时复核一次；
// 这里用「日期 + revision」识别版本变化，变了就重拉强度库与预测。
let VRS_KEY = null, VRS_TIMER = null;

function vrsKey(s) {
  return (s && s.latest) ? s.latest + "#" + (s.snapshot_revision || 1) : null;
}

function vrsAge(s) {
  const sec = s && s.snapshot_age_s;
  if (sec === null || sec === undefined) return "";
  if (sec < 90) return "刚刚抓取";
  if (sec < 3600) return Math.round(sec / 60) + " 分钟前抓取";
  return Math.round(sec / 3600) + " 小时前抓取";
}

function vrsText(s) {
  if (!s || s.ok === false) {
    return "VRS 同步不可用" + (s && s.error ? "：" + s.error : "");
  }
  const t = s.target || "—", l = s.latest || "—";
  const meta = [
    s.snapshot_revision ? "第 " + s.snapshot_revision + " 版" : "",
    vrsAge(s)
  ].filter(Boolean).join(" · ");
  if (s.running) return "VRS " + l + " · 正在检查 " + t + "…";
  if (s.needs_update) return "VRS " + l + (meta ? " · " + meta : "") + " · 强度库待同步";
  if (s.target_ready && s.needs_fetch) {
    return "VRS " + l + (meta ? " · " + meta : "") +
      " · 超过 " + Math.round((s.revalidate_s || 3600) / 60) + " 分钟未复核";
  }
  if (s.target_ready) return "VRS " + l + (meta ? " · " + meta : "") + " · 已是最新";
  if (s.cooldown_s > 0) {
    return "VRS " + l + " · 目标 " + t + " 尚未发布，" +
      Math.ceil(s.cooldown_s / 60) + " 分钟后重试";
  }
  return "VRS " + l + " · 目标 " + t + " 待抓取";
}

function vrsShow(s) {
  const el = $("vrsp"); if (!el) return;
  el.textContent = vrsText(s);
  el.title = (s && s.message) || "";
  if (s && s.last_error && !s.running) el.title += " ｜ 上次错误：" + s.last_error;
}

async function vrsPoll() {
  clearTimeout(VRS_TIMER);
  try {
    const s = await jget("/api/vrs/status");
    vrsShow(s);
    if (s.running) { VRS_TIMER = setTimeout(vrsPoll, 1500); return; }
    const key = vrsKey(s);
    if (key && VRS_KEY && key !== VRS_KEY) {
      await loadLib();
      refresh();
      $("st").textContent += " · VRS 已更新到 " + s.latest +
        (s.snapshot_revision ? "（第 " + s.snapshot_revision + " 版）" : "");
    }
    VRS_KEY = key || VRS_KEY;
  } catch (e) { /* 老服务没有该端点时静默 */ }
}

async function vrsKick() {
  try {
    const s = await jget("/api/vrs/status");
    vrsShow(s); VRS_KEY = vrsKey(s);
    const want = (s.needs_fetch === undefined)
      ? !(s.target_ready && s.library_current)
      : (s.needs_update || (s.needs_fetch && !(s.cooldown_s > 0)));
    if (!s.running && want) {
      const j = await jpost("/api/vrs/refresh", {});
      if (j && j.status) vrsShow(j.status); else vrsShow(j);
    }
    vrsPoll();
  } catch (e) { /* 独立部署或老版本没有 VRS 端点 */ }
}

async function boot() {
  try {
    await loadLib();
    buildSlots();
    const st = await jget("/api/status");
    $("ver").textContent = st.version + " · ready";
    await loadDemo();
    $("st").textContent += " · 点「预测十格」出结果（默认 n=20000，约 3～4 秒）";
    vrsKick();
  } catch (e) { $("st").textContent = "加载失败：" + e.message; }
}

boot();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------- 输入辅助
def _fixtures():
    """内置 fixture 白名单（只读 fixtures/*.json；不做任何路径拼接）。"""
    out = {}
    if not os.path.isdir(FIXDIR):
        return out
    for name in sorted(os.listdir(FIXDIR)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(FIXDIR, name), encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        teams = doc.get("teams") or []
        # 只收**真 roster**：恰好 16 条、且每条是带 name 的对象。
        # 这一条把三种非 roster 文件挡在外面：真值表（records，无 teams）、
        # 队名清单（teams 是字符串数组）、强度库（teams 是 255 条对象）。
        if len(teams) != 16 or not all(isinstance(t, dict) and t.get("name") for t in teams):
            continue
        out[name[:-5]] = {"stage": doc.get("stage") or "(未命名)",
                          "source": doc.get("source", "rank_points"),
                          "teams": len(teams), "truth": "records" in doc}
    return out


def _fixture_text(name):
    """按白名单读 fixture 原文；非法名字一律拒绝（防路径穿越）。"""
    if name not in _fixtures():
        raise ValueError("未知 fixture：%r" % name)
    path = os.path.join(FIXDIR, name + ".json")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _doc_from(obj, what):
    """请求体里的对象可以给 dict 也可以给 JSON 字符串（页面上是文本框）。"""
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except ValueError as exc:
            raise ValueError("%s JSON 解析失败：%s" % (what, exc))
    if isinstance(obj, dict):
        return obj
    raise ValueError("%s 应为 JSON 对象或字符串" % what)


def _slug(text, fallback="stage"):
    """阶段名 → 文件名安全片段（只留中英文数字下划线）。"""
    keep = [c for c in str(text or "") if c.isalnum() or c == "_" or "\u4e00" <= c <= "\u9fff"]
    return ("".join(keep) or fallback)[:48]


# ---------------------------------------------------------------- 强度库
_LIB_CACHE = {}          # path -> (mtime, doc)：改库不重启也能生效


def strength_library(path=PS.STRENGTH_LIB):
    """读公开强度库（带 mtime 缓存）。读不到/不合法一律 ValueError。"""
    try:
        mtime = os.path.getmtime(path)
    except OSError as exc:
        raise ValueError("强度库读不到（%s）：%s" % (path, exc))
    hit = _LIB_CACHE.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        doc = PS.load_strength_library(path)
    except OSError as exc:
        raise ValueError("强度库读不到（%s）：%s" % (path, exc))
    _LIB_CACHE[path] = (mtime, doc)
    return doc


def _teams_payload():
    """GET /api/teams：给前端的强度库（含 asof/provenance/别名）。"""
    lib = strength_library()
    return {"asof": lib.get("asof"), "elo_asof": lib.get("elo_asof"),
            "elo_matches": lib.get("elo_matches"), "source": lib.get("source"),
            "_role": lib.get("_role"), "_provenance": lib.get("_provenance"),
            "counts": lib.get("counts"),
            "teams": [{"name": t["name"], "rating": t["rating"], "rank": t["rank"],
                       "points": t["points"], "aliases": t["aliases"]}
                      for t in lib["teams"]]}


def _vrs_mod():
    """懒加载仓库根的 vrs_sync（独立部署只有 pickem/ 时会 ImportError）。"""
    root = os.path.dirname(HERE)
    if root not in sys.path:
        sys.path.append(root)          # 追加而非插到最前，避免遮蔽 pickem 自己的模块
    import vrs_sync
    return vrs_sync


def _vrs_status():
    try:
        return _vrs_mod().status()
    except Exception as exc:                       # noqa: BLE001
        return {"ok": False, "running": False,
                "error": "VRS 同步不可用：%s: %s" % (type(exc).__name__, exc)}


def _vrs_refresh(payload=None):
    try:
        return _vrs_mod().maybe_start(force=bool((payload or {}).get("force")))
    except Exception as exc:                       # noqa: BLE001
        return {"ok": False, "running": False,
                "error": "VRS 同步不可用：%s: %s" % (type(exc).__name__, exc)}


def _demo_stage():
    """GET /api/demo：默认示例阶段（16 个队名，页面开机预填）。"""
    path = os.path.join(FIXDIR, "demo_stage_s1.json")
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ValueError("示例阶段读不到（%s）：%s" % (path, exc))
    teams = [str(t) for t in (doc.get("teams") or [])]
    if len(teams) != 16:
        raise ValueError("示例阶段应有 16 个队名，实得 %d" % len(teams))
    return {"stage": doc.get("stage"), "teams": teams}


# ---------------------------------------------------------------- 跑一次
DEFAULT_N, DEFAULT_K, DEFAULT_SEED = 20000, 5, 20261012


def _params(payload):
    """n / k / seed / σ 臂 / 平局键 参数（照旧的全部合法性检查）。"""
    n = int(payload.get("n", DEFAULT_N))
    k = int(payload.get("k", DEFAULT_K))
    seed = int(payload.get("seed", DEFAULT_SEED))
    if not (100 <= n <= 200000):
        raise ValueError("n 应在 100..200000（实得 %d）" % n)
    if not (1 <= k <= 10):
        raise ValueError("k 应在 1..10（实得 %d）" % k)
    return n, k, seed, bool(payload.get("uncertain")), bool(payload.get("random_ties"))


def _run_roster(roster, n, k, seed, unc, rt, roster_file="my_stage.json"):
    """跑完整链路 → 页面与脚本共用的响应体（/api/predict 与 /api/pickem 同字段）。"""
    try:
        roster, _names, _seeds = PS.validate_roster(roster)
    except ValueError as exc:                    # 与 CLI 同口径：非法一律拒绝，不出结果
        raise ValueError("roster 无效：%s" % exc)
    t0 = time.time()
    rep = PS.predict(roster, n=n, k=k, seed=seed, uncertain=unc,
                     tie_break="random" if rt else "seed")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        PS.report(rep)
    cmd = ("python tools/pickem/predict_stage.py --roster %s "
           "--n %d --k %d --seed %d%s%s --json out.json"
           % (roster_file, n, k, seed, " --uncertain" if unc else "",
              " --random-ties" if rt else ""))
    # 冻结件：赛前封存用。哈希按**规范化** roster（排序键、无多余空白）算，
    # 于是“同一份输入”可由任何人独立重算哈希对拍。
    canon = json.dumps(roster, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    freeze = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "stage": rep.get("stage"), "source": rep.get("source"),
              "n": n, "k": k, "seed": seed, "arm": rep.get("arm"),
              "tie_break": rep.get("tie_break"),
              "roster_sha256": hashlib.sha256(canon.encode("utf-8")).hexdigest(),
              "command": cmd}
    return {"report": rep, "report_text": buf.getvalue(), "command": cmd,
            "freeze": freeze, "elapsed_s": time.time() - t0,
            "roster": roster, "roster_file": roster_file}


def do_predict(payload):
    """POST /api/predict：给**已经写好的 roster**（含 矩阵 源）跑。"""
    roster = payload.get("roster")
    if roster is None and payload.get("fixture"):
        roster = json.loads(_fixture_text(str(payload["fixture"])))
    if roster is None:
        raise ValueError("需要 roster（对象/字符串）、fixture 名字或 teams（见 /api/pickem）")
    roster = _doc_from(roster, "roster")
    n, k, seed, unc, rt = _params(payload)
    return _run_roster(roster, n, k, seed, unc, rt,
                       roster_file="%s.json" % _slug(roster.get("stage"), "roster"))


def do_pickem(payload):
    """POST /api/pickem —— **主入口**：16 个队名 → 查强度库 → roster → 十格。

    这是「玩家只填队名」那一条路：强度由 `fixtures/team_strength.json` 预计算提供，
    查不到的队会被 `build_roster` 点名告警并按默认强度兜底（不静默）。
    """
    teams = payload.get("teams")
    if teams is None:
        raise ValueError("需要 teams：16 个队名的数组（也接受逗号/换行分隔的字符串）")
    if isinstance(teams, str):
        teams = [t.strip() for t in teams.replace("\n", ",").replace(";", ",").split(",")]
    if not isinstance(teams, (list, tuple)):
        raise ValueError("teams 应为数组")
    source = str(payload.get("source") or "rank_points")
    lib = strength_library()
    try:
        roster, warnings = PS.build_roster(teams, lib, source=source,
                                           stage=payload.get("stage"),
                                           date=payload.get("date"))
    except ValueError as exc:
        raise ValueError("建 roster 失败：%s" % exc)
    n, k, seed, unc, rt = _params(payload)
    out = _run_roster(roster, n, k, seed, unc, rt,
                      roster_file="%s.json" % _slug(roster.get("stage"), "pickem"))
    out["warnings"] = warnings
    out["library"] = {"asof": lib.get("asof"), "teams": len(lib["teams"]),
                      "source": lib.get("source")}
    return out


def do_score(payload):
    truth = SC.truth_from_doc(_doc_from(payload.get("truth"), "truth"))
    card = payload.get("card") or payload.get("picks")
    if card is None:
        raise ValueError("需要 card（十格对象 / predict_stage 报表）")
    card = _doc_from(card, "card")
    k = int(payload.get("k", 5))
    return {"stage": truth["stage"], "k": k,
            "results": SC.score_document(truth, card, k=k)}


def do_calibrate(payload):
    truth = CAL.truth_from_doc(_doc_from(payload.get("truth"), "truth"))
    doc = CAL.report_marginals(_doc_from(payload.get("report"), "report"))
    return {"result": CAL.calibrate(truth, [doc])}


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "pickem-gui/" + VERSION.split()[-1]
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):        # 单行、带时间、不发散
        sys.stderr.write("[gui %s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def _err(self, code, msg):
        self._json(code, {"error": str(msg)})

    def _drain(self, n):
        """把没打算处理的请求体读完再回错。

        不读就回 413，客户端还在发 → TCP 被重置（实测 WinError 10053），
        于是「服务端明明回得很好」却变成一个连接错误：又一种静默失败。
        """
        left = n
        self.close_connection = True
        while left > 0:
            chunk = self.rfile.read(min(65536, left))
            if not chunk:
                break
            left -= len(chunk)

    def _payload(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            self._drain(n)
            raise OverflowError("请求体过大（%d > %d 字节）" % (n, MAX_BODY))
        raw = self.rfile.read(n) if n else b"{}"
        try:
            doc = json.loads(raw.decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("请求体不是合法 JSON：%s" % exc)
        if not isinstance(doc, dict):
            raise ValueError("请求体应为 JSON 对象")
        return doc

    # ---- 路由
    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                page = PAGE.replace("__VERSION__", VERSION).replace("__MODE_NAV__", "")
                return self._send(200, page, "text/html; charset=utf-8")
            if path == "/api/status":
                return self._json(200, {"ready": True, "version": VERSION,
                                        "port": self.server.server_address[1],
                                        "fixtures": sorted(_fixtures()),
                                        "endpoints": ["/api/status", "/api/teams", "/api/demo",
                                                      "/api/fixtures", "/api/fixture",
                                                      "/api/pickem", "/api/predict",
                                                      "/api/score", "/api/calibrate",
                                                      "/api/vrs/status", "/api/vrs/refresh"]})
            if path == "/api/vrs/status":
                return self._json(200, _vrs_status())
            if path == "/api/fixtures":
                return self._json(200, _fixtures())
            if path == "/api/teams":
                return self._json(200, _teams_payload())
            if path == "/api/demo":
                return self._json(200, _demo_stage())
            if path == "/api/fixture":
                q = parse_qs(urlparse(self.path).query)
                name = (q.get("name") or [""])[0]
                return self._json(200, {"name": name, "text": _fixture_text(name)})
            if path in POST_ENDPOINTS:
                return self._err(405, "该端点只接受 POST：%s" % path)
            return self._err(404, "没有这个路径：%s" % path)
        except ValueError as exc:
            return self._err(400, exc)
        except Exception as exc:                       # noqa: BLE001 — 兜底也要是 JSON
            return self._err(500, "服务端异常：%s: %s" % (type(exc).__name__, exc))

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            payload = self._payload()
        except OverflowError as exc:
            return self._err(413, exc)
        except ValueError as exc:
            return self._err(400, exc)
        handlers = {"/api/predict": do_predict, "/api/pickem": do_pickem,
                    "/api/score": do_score, "/api/calibrate": do_calibrate,
                    "/api/vrs/refresh": _vrs_refresh}
        if path not in handlers:
            return self._err(404, "没有这个端点：%s" % path)
        try:
            return self._json(200, handlers[path](payload))
        except ValueError as exc:
            return self._err(400, exc)               # 非法输入：rc/状态码一律拒绝
        except Exception as exc:                      # noqa: BLE001
            return self._err(500, "服务端异常：%s: %s" % (type(exc).__name__, exc))


def port_busy(host, port):
    """端口是否已被占用。

    为什么必须自己查（**Windows 上的静默失败**）：`HTTPServer` 继承了
    `allow_reuse_address = 1`，而 Windows 的 `SO_REUSEADDR` 语义是「**允许抢绑同一端口**」
    （与 Linux 不同）：第二个进程会**成功**绑上已有人在 listen 的端口，于是同一端口上
    有两个服务、请求随机落到其中一个 —— 不报错、结果不可预期。
    实测：8790 已有人在 listen 时，本模块照旧打印就绪、rc=0。
    故在绑定前先**主动探一次连接**（探得通即视为被占）。局限：只绑了别的接口的监听
    探不到，但本工具默认只绑本机，这个差异可接受。
    """
    if not port:
        return False                     # port=0 = 让内核分配，不存在「占用」问题
    probe = host if host not in ("", "0.0.0.0", "::") else "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((probe, int(port))) == 0


def serve(host="127.0.0.1", port=DEFAULT_PORT, quiet=False):
    """起服务并返回 (httpd, thread)。测试用 port=0 拿随机空闲端口。

    端口被占一律抛 OSError（调用方报 rc=1），**不做静默回退**。
    """
    if port_busy(host, port):
        raise OSError("端口 %d 已被占用（%s）：换一个 --port，或先停掉旧进程"
                      % (int(port), host))
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    th = threading.Thread(target=httpd.serve_forever, name="pickem-gui", daemon=True)
    th.start()
    if not quiet:
        sys.stderr.write("[pickem-gui] http://%s:%d/（Ctrl+C 结束）\n"
                         % (host, httpd.server_address[1]))
    return httpd, th


# ---------------------------------------------------------------- 自证
def _selftest():
    import hashlib
    import subprocess
    import urllib.error
    import urllib.request

    ok = 0
    old_disable = os.environ.get("VRS_SYNC_DISABLE")
    os.environ["VRS_SYNC_DISABLE"] = "1"      # 自检不抓网络

    def chk(cond, msg):
        nonlocal ok
        if not cond:
            raise AssertionError("FAIL: " + msg)
        ok += 1

    httpd, _th = serve("127.0.0.1", 0, quiet=True)
    port = httpd.server_address[1]
    base = "http://127.0.0.1:%d" % port

    def get(path):
        try:
            with urllib.request.urlopen(base + path, timeout=120) as r:
                return r.status, r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8")

    def post(path, body):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(base + path, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.status, r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8")

    def jpost(path, body):
        st, txt = post(path, body)
        return st, (json.loads(txt) if txt else {})

    try:
        st, html = get("/")
        chk(st == 200 and "Pick'Em 十格预测" in html and "<script>" in html,
            "GET / 应返回页面")
        st, body = get("/api/status")
        chk(st == 200 and json.loads(body)["ready"] is True, "/api/status ready=true")
        st, body = get("/api/vrs/status")
        vj = json.loads(body)
        chk(st == 200 and vj.get("ok") is True and vj.get("target")
            and vj.get("latest") and vj.get("library_asof"),
            "GET /api/vrs/status 给出目标日 / 最新快照 / 强度库日期")
        chk("08:00" in (vj.get("target_rule") or ""),
            "VRS 状态说明北京时间 08:00 日期规则")
        st, vj = jpost("/api/vrs/refresh", {})
        chk(st == 200 and vj.get("started") is False and vj.get("reason") == "disabled",
            "POST /api/vrs/refresh 在自检环境不抓网络")
        st, body = get("/api/vrs/refresh")
        chk(st == 405 and "只接受 POST" in body, "GET /api/vrs/refresh → 405")
        st, html = get("/")
        chk('id="vrsp"' in html and "function vrsKick" in html,
            "页面带 VRS 状态与刷新触发")
        st, body = get("/api/fixtures")
        fx = json.loads(body)
        chk(st == 200 and {"budapest2025s1_elo", "budapest2025s1_uncertain",
                           "example_matrix"} <= set(fx), "内置 fixture 清单")
        chk(all(v["teams"] == 16 for v in fx.values()), "fixture 都应是 16 队")
        # 名单里不能混进非 roster：强度库（255 条对象）与示例队名清单（字符串数组）
        chk("team_strength" not in fx and "demo_stage_s1" not in fx,
            "非 roster 文件（强度库 / 队名清单）不进 fixture 清单")

        st, body = get("/api/fixture?name=budapest2025s1_elo")
        chk(st == 200 and len(json.loads(json.loads(body)["text"])["teams"]) == 16,
            "GET /api/fixture 应返回 roster 原文")
        st, _b = get("/api/fixture?name=..%2F..%2Fsecrets")
        chk(st == 400, "路径穿越/未知 fixture 应 400")
        st, _b = get("/api/fixtures/../secrets")
        chk(st in (400, 404), "非法路径应 400/404")
        st, body = get("/nope")
        chk(st == 404, "未知路径应 404")
        st, _b = get("/api/predict")
        chk(st == 405, "对 POST 端点发 GET 应 405")
        st, _b = get("/api/pickem")
        chk(st == 405, "/api/pickem 只收 POST（GET 应 405）")

        # 端口占用必须**硬失败**：Windows 上 allow_reuse_address=1 的语义是「允许抢绑」，
        # 不查就变成两个服务同时 listen 同一端口（静默失败，实测发生过）。
        holder, _th2 = serve("127.0.0.1", 0, quiet=True)
        busy_port = holder.server_address[1]
        chk(port_busy("127.0.0.1", busy_port) is True, "已占用端口应被判为忙")
        try:
            serve("127.0.0.1", busy_port, quiet=True)
            chk(False, "占用端口上再起服务应抛 OSError")
        except OSError as exc:
            chk("已被占用" in str(exc), "占用端口应抛 OSError（%s）" % exc)
        finally:
            holder.shutdown()
            holder.server_close()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]
        chk(port_busy("127.0.0.1", free_port) is False, "空闲端口不应被判为忙")
        chk(port_busy("127.0.0.1", 0) is False, "port=0（内核分配）不判忙")
        # CLI 面同样要硬失败：起一个真占用端口的进程，再让 CLI 去抢同一个端口
        blocker, _th3 = serve("127.0.0.1", 0, quiet=True)
        bp = blocker.server_address[1]
        try:
            p = subprocess.run([sys.executable, "-X", "utf8", os.path.abspath(__file__),
                                "--port", str(bp)], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=60)
            chk(p.returncode == 1 and "已被占用" in (p.stderr or ""),
                "CLI 在占用端口上应 rc=1 且说明原因（rc=%s）" % p.returncode)
        finally:
            blocker.shutdown()
            blocker.server_close()

        # 端到端：小 n 出十格
        st, r = jpost("/api/predict", {"fixture": "budapest2025s1_elo", "n": 2000, "seed": 7})
        rep = r.get("report") or {}
        chk(st == 200 and len(rep.get("picks", {}).get("padv", [])) == 6, "小 n 预测出十格")
        chk(st == 200 and 0.0 <= rep.get("p_ge_k", -1) <= 1.0, "P(≥k) ∈ [0,1]")
        chk(st == 200 and "predict_stage.py" in (r.get("command") or ""),
            "返回可复跑的 CLI 命令")
        chk(st == 200 and "P(命中≥5)" in (r.get("report_text") or ""),
            "返回与 CLI 同源的报表文本")
        chk((fx.get("example_matrix") or {}).get("source") == "matrix",
            "fixture 清单含 matrix 源示例")

        # 冻结件：哈希必须能由**独立重算**验证（端点封存靠这个）
        fz = r.get("freeze") or {}
        roster_doc = json.loads(json.loads(get("/api/fixture?name=budapest2025s1_elo")[1])["text"])
        canon = json.dumps(roster_doc, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
        chk(set(fz) >= {"utc", "stage", "source", "n", "k", "seed", "arm", "tie_break",
                        "roster_sha256", "command"}, "冻结件字段齐")
        chk(fz.get("roster_sha256") == hashlib.sha256(canon.encode("utf-8")).hexdigest(),
            "冻结件哈希可由规范化 roster 独立重算")
        chk(fz.get("n") == 2000 and fz.get("seed") == 7 and "--n 2000" in fz.get("command", ""),
            "冻结件记录 n/seed 与可复跑命令")

        # 发布档：HTTP 面必须逐位复现 0.2106
        st, r = jpost("/api/predict", {"fixture": "budapest2025s1_elo", "n": 20000,
                                       "seed": 20261012})
        chk(st == 200 and abs(r["report"]["p_ge_k"] - 0.2106) < 1e-12,
            "HTTP 面复现 P(≥5)=0.2106（逐位）")

        # ---- 主入口 /api/pickem：16 个队名 → 十格（「一填就出结果」那一条） ----
        st, body = get("/api/teams")
        libp = json.loads(body)
        chk(st == 200 and len(libp.get("teams") or []) >= 200 and libp.get("asof")
            and libp.get("_provenance") and libp.get("counts"),
            "GET /api/teams 给出强度库（队长表 + asof + provenance）")
        chk(all(t.get("name") for t in libp["teams"]), "强度库每队都有 name")
        chk(any(t.get("aliases") for t in libp["teams"]), "强度库带别名（简称可查）")
        st, body = get("/api/demo")
        dj = json.loads(body)
        chk(st == 200 and len(dj.get("teams") or []) == 16, "GET /api/demo 给 16 个队名")

        st, r = jpost("/api/pickem", {"teams": dj["teams"], "n": 2000, "seed": 7})
        rep = r.get("report") or {}
        chk(st == 200 and [len(rep.get("picks", {}).get(c, [])) for c in ("p30", "padv", "p03")]
            == [2, 6, 2], "/api/pickem 出十格 2/6/2")
        chk(st == 200 and len(r["roster"]["teams"]) == 16
            and len(r.get("warnings") or []) == 1 and "RED Canids" in r["warnings"][0],
            "示例阶段 16 队全建出；RED Canids 无 VRS 积分 → 点名告警（默认源 rank_points）")
        chk(st == 200 and r["roster_file"].endswith(".json")
            and r["roster_file"] in r["command"],
            "命令里的 roster 文件名 = 可下载件名")
        # 与同一实现逐位对拍：build_roster + PS.predict 必须给出 HTTP 面同一个数
        _roster2, _warn2 = PS.build_roster(dj["teams"], strength_library(), stage=dj["stage"])
        _rep2 = PS.predict(_roster2, n=2000, seed=7)
        chk(_rep2["p_ge_k"] == rep["p_ge_k"] and _rep2["picks"] == rep["picks"],
            "HTTP 面与同参数直跑逐位一致（P(≥5)=%.4f）" % rep["p_ge_k"])
        # 简称与逗号分隔字符串两种输入方式
        st, r_alias = jpost("/api/pickem", {"teams": dj["teams"][:15] + ["navi"],
                                             "n": 500, "seed": 1})
        chk(st == 200 and any(t["name"] == "Natus Vincere" for t in r_alias["roster"]["teams"]),
            "简称 navi 解析到 Natus Vincere")
        st, r_str = jpost("/api/pickem", {"teams": ",".join(dj["teams"]), "n": 500, "seed": 1})
        chk(st == 200 and len(r_str["report"]["picks"]["padv"]) == 6,
            "teams 也接受逗号分隔字符串")
        st, r_again = jpost("/api/pickem", {"teams": dj["teams"], "n": 2000, "seed": 7})
        chk(r_again["report"]["p_ge_k"] == rep["p_ge_k"]
            and r_again["freeze"]["roster_sha256"] == r["freeze"]["roster_sha256"],
            "/api/pickem 同参两次逐位一致")
        # 未收录的队：仍出结果，但必须点名（静默兜底是本项目明令禁止的失败模式）
        st, r_unk = jpost("/api/pickem", {"teams": dj["teams"][:15] + ["Zzzz Not A Team"],
                                           "n": 500, "seed": 1})
        chk(st == 200 and any("Zzzz Not A Team" in w for w in (r_unk.get("warnings") or [])),
            "/api/pickem 未收录队 → 200 但点名告警")

        # 矩阵 源 + σ 臂走 HTTP
        st, r = jpost("/api/predict", {"fixture": "example_matrix.json", "n": 1000, "seed": 1})
        chk(st == 400, "fixture 名带后缀应被拒（白名单按名字）")
        st, r = jpost("/api/predict", {"fixture": "example_matrix", "n": 1000, "seed": 1})
        chk(st == 200 and r["report"]["source"] == "matrix", "matrix 源可跑")
        st, r = jpost("/api/predict", {"fixture": "budapest2025s1_uncertain",
                                       "uncertain": True, "n": 1000, "seed": 1})
        chk(st == 200 and r["report"]["arm"] == "uncertain", "σ 臂可跑")
        st, r = jpost("/api/predict", {"fixture": "budapest2025s1_elo",
                                       "random_ties": True, "n": 1000, "seed": 1})
        chk(st == 200 and r["report"]["tie_break"] == "random", "平局键可切")

        # 负样本
        st, e = jpost("/api/predict", {"roster": {"source": "elo", "teams": []}})
        chk(st == 400 and "roster 无效" in e.get("error", ""), "非法 roster 应 400")
        st, e = jpost("/api/predict", {"fixture": "budapest2025s1_elo", "n": 10})
        chk(st == 400 and "n 应在" in e.get("error", ""), "n 越界应 400")
        st, e = jpost("/api/pickem", {"teams": dj["teams"][:15]})
        chk(st == 400 and "恰好 16" in e.get("error", ""), "/api/pickem 15 队 → 400")
        st, e = jpost("/api/pickem", {"teams": [dj["teams"][0]] * 16})
        chk(st == 400 and "同一支队" in e.get("error", ""), "/api/pickem 重复队 → 400")
        st, e = jpost("/api/pickem", {})
        chk(st == 400 and "teams" in e.get("error", ""), "/api/pickem 缺 teams → 400")
        st, e = jpost("/api/pickem", {"teams": dj["teams"], "source": "bogus"})
        chk(st == 400 and "source" in e.get("error", ""), "/api/pickem 未知 source → 400")
        st, e = jpost("/api/pickem", {"teams": dj["teams"], "uncertain": True,
                                       "n": 500, "seed": 1})
        chk(st == 400 and ("exposure" in e.get("error", "")
                           or "仅支持 source=elo" in e.get("error", "")),
            "/api/pickem 强度库 roster + σ 臂 → 400（σ 臂只在 Elo 单位验证过）")
        def raw(path, data):
            req = urllib.request.Request(base + path, data=data,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return r.status, r.read().decode("utf-8")
            except urllib.error.HTTPError as e:
                return e.code, e.read().decode("utf-8")

        st, txt = raw("/api/predict", b"{ not json")
        chk(st == 400 and "合法 JSON" in txt, "坏 JSON 应 400")
        st, txt = raw("/api/predict", b"[1,2]")
        chk(st == 400 and "JSON 对象" in txt, "非对象请求体应 400")
        big = json.dumps({"roster": "x" * (MAX_BODY + 10)})
        req = urllib.request.Request(base + "/api/predict", data=big.encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                code = r.status
        except urllib.error.HTTPError as e:
            code = e.code
        chk(code == 413, "超大请求体应 413")

        # 计分与校准
        truth = {"stage": "SYN", "records": {"T%02d" % i: [1, 3] for i in range(1, 17)}}
        for t, wl in (("T01", [3, 0]), ("T05", [3, 0]), ("T14", [0, 3]), ("T16", [0, 3])):
            truth["records"][t] = wl
        for t in ("T02", "T03", "T07"):
            truth["records"][t] = [3, 1]
        for t in ("T04", "T06", "T08"):
            truth["records"][t] = [3, 2]
        perfect = {"p30": ["T01", "T05"],
                   "padv": ["T02", "T03", "T04", "T06", "T07", "T08"],
                   "p03": ["T14", "T16"]}
        st, r = jpost("/api/score", {"truth": truth, "card": perfect, "k": 5})
        chk(st == 200 and r["results"]["card"]["hits"] == 10, "计分：完美卡 10/10")
        st, r = jpost("/api/score", {"truth": truth,
                                     "card": {"baselines": {"optimizer": {"picks": perfect}}}})
        chk(st == 200 and r["results"]["optimizer"]["hits"] == 10, "计分：报表形状可评")
        st, e = jpost("/api/score", {"truth": json.dumps(truth),
                                     "card": dict(perfect, p30=["ZZZ", "T05"])})
        chk(st == 400 and "阶段错配" in e.get("error", ""), "计分：阶段错配应 400")
        st, e = jpost("/api/score", {"truth": dict(truth, records=dict(truth["records"],
                                                                      T16=[3, 0])),
                                     "card": perfect})
        chk(st == 400 and "应恰" in e.get("error", ""), "计分：坏真值表应 400")

        marg = {t: {"p30": 0.1, "padv": 0.4, "p03": 0.1} for t in truth["records"]}
        st, r = jpost("/api/calibrate", {"truth": truth,
                                         "report": {"stage": "SYN", "source": "demo",
                                                    "marginals": marg}})
        chk(st == 200 and set(r["result"]["reports"][0]["categories"]) == {"p30", "padv", "p03"},
            "校准：三类指标齐")
        st, e = jpost("/api/calibrate", {"truth": truth, "report": {"stage": "SYN"}})
        chk(st == 400 and "marginals" in e.get("error", ""), "校准：缺 marginals 应 400")
        chk(all("error" in json.loads(t) for _s, t in
                [post("/api/score", {"truth": truth})]), "错误一律 JSON 形状")
    finally:
        httpd.shutdown()
        httpd.server_close()
        if old_disable is None:
            os.environ.pop("VRS_SYNC_DISABLE", None)
        else:
            os.environ["VRS_SYNC_DISABLE"] = old_disable

    print("PREDICT_GUI_SELFTEST_OK (%d checks)" % ok)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pick'Em 本地 Web 界面")
    ap.add_argument("--host", default="127.0.0.1", help="绑定地址（默认仅本机）")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help="端口（默认 8790；014 的 GUI 占 8765）")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if not (1 <= a.port <= 65535):
        print("[ERR] 端口非法：%d" % a.port, file=sys.stderr)
        return 1
    try:
        httpd, _th = serve(a.host, a.port, quiet=a.quiet)
    except OSError as exc:
        print("[ERR] 起服务失败：%s" % exc, file=sys.stderr)
        return 1
    if not a.quiet:
        print("PICKEM_GUI_READY http://%s:%d/" % (a.host, httpd.server_address[1]))
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[pickem-gui] 已停止")
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(_selftest())
    sys.exit(main())
