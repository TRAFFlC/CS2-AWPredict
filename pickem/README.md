# pickem

**Issue-015** · 依赖 `numpy`（其余为纯标准库）· 只读

## 它回答一个问题

> 给定一个 Major 阶段（16 队瑞士轮）的**队伍名单与种子**，
> 10 格 Pick'Em（2×3:0 + 6×晋级 + 2×0:3）该怎么填？
> **P(答对 ≥5)** 是多少？

这不是逐场预测。十格的正确事件由**同一批瑞士轮结果**驱动、彼此强相关，
目标又是一个**阈值概率** ⇒ 本质是组合选择（投资组合）问题，不是十个独立分类问题。
所以本工具直接模拟整条瑞士轮路径、在联合场景上优化十格。

## 最短路径（页面：填 16 个队名 → 出十格）

```powershell
python tools/pickem/predict_gui.py            # 打开 http://127.0.0.1:8790/
```

页面**只有一件事**：把本阶段的 16 支队伍按种子顺序填进 16 个输入框，点「预测十格」，
下面直接给出 **10 个位置**：`3:0` 2 格 ｜ `3:1 · 3:2` 6 格 ｜ `0:3` 2 格（每格带 P 与条形）。

## 独立项目接入（交接边界）

Issue-015 的唯一公开、可移植实现目录是 **`tools/pickem/`**。该目录本来就按独立工具组织并纳入 Git；
后续从独立项目接入时，整体复制/引入此目录即可，不要从 `data/public/` 取文件（该目录在本仓库被 Git 忽略），
也不要把 `issues/015-major-pickem-prediction.md` 当作运行时依赖。

- **预测核心**：`predict_stage.py`、`pickem.py`、`sources.py`、`swiss.py`、`uncertain.py`。
  Python Web 后端可直接调用 `predict_stage` 的 `load_strength_library()`、`build_roster(names, lib, source=...)`
  和 `predict(roster, n=..., k=..., seed=...)`；`names` 必须按官方种子顺序传入。`build_roster` 返回
  `(roster, warnings)`，集成方应把告警展示给用户；`predict` 返回可 JSON 化的预测报表。`build_roster` 的
  `source` 支持 `rank_points`（默认，VRS 积分）/ `elo`；若要用 `matrix`，须自行组装带 `matrix` 字段的 roster。
  导入前需把 `tools/pickem/` 目录加入 Python 模块搜索路径（模块间按同目录脚本名导入）。
- **依赖与数据**：预测核心需要 `numpy`；按队名查强度时还需要 `fixtures/team_strength.json`。
  若新 Web 项目不是 Python 后端，应由服务端封装/调用这套 Python 代码，不要把它当作浏览器端 JS 模块。
- **可选本地演示页**：`predict_gui.py` 与 `fixtures/demo_stage_s1.json`，用于本工具自带的本机 Web 页面；
  它不是供其他 Web 项目直接挂载的页面，也不提供远程服务鉴权。
- **可选赛后能力**：`scoring.py`、`calibration.py`；前者需要外部真值表，二者都不是赛前预测必需项。
- **开发验证**：`selftest.py` 和 `fixtures/` 中的示例/合成 fixture。正式预测不需要携带自测脚本或 Budapest 示例 roster；
  若希望保留可复现回归与演示，则建议整目录带走。
- **外部胜率源**：若独立项目已有逐场模型，可按 README 下文的 `source: "matrix"` roster 接入两两胜率；
  当前仓库只提供输入接口，不会自动加载 issue-014 的 Predictor。

因此给你的具体交接路径是 **[`tools/pickem/`](.)**；Issue-015 的问题背景与研究时间线仍留在
[`issues/015-major-pickem-prediction.md`](../../issues/015-major-pickem-prediction.md)，运行时不需要搬动它。

- **强度不用你填**：每队水平由公开强度库 `fixtures/team_strength.json` 预先算好（见下节），
  输入框里带实时强度标签（`VRS #3 · 1899`）。
- **支持简称**：`navi` → Natus Vincere、`nip` → Ninjas in Pyjamas、`vp` → Virtus.pro（点候选即选中）。
- **查不到的队会被点名告警**并按默认强度兜底，不静默（该队结果不可信）。
- 默认 `n=20000`，普通机器上 **约 4 秒**出结果（实测 3.4～4.5 s）；参数、手填 roster、计分、校准都在「高级」折叠区里，
  默认看不见。
- **开箱可见**：页面加载时自动**预填示例阶段**（Budapest S1 的 16 队名）⇒ 不填任何东西也能立刻看懂结果长什么样；
  换名单就改输入框，或点「清空」从零填。

## 用法

```powershell
# 页面：填 16 个队名 → 十格（默认 127.0.0.1:8790）
python tools/pickem/predict_gui.py

# 默认：示例 roster（同时是 issue-015 §5.1 Elo 行的复现输入）
python tools/pickem/predict_stage.py --roster tools/pickem/fixtures/budapest2025s1_elo.json

# 复现已公布数字（P(≥5)=0.2106）
python tools/pickem/predict_stage.py --roster tools/pickem/fixtures/budapest2025s1_elo.json --n 20000 --seed 20261012

# σ 不确定性传播臂（冷门队；需 roster 带 exposure/games[/rank] 或 sigma）
python tools/pickem/predict_stage.py --roster tools/pickem/fixtures/budapest2025s1_uncertain.json --uncertain --n 20000 --seed 20261012

# 同 Buchholz 并列改均匀随机抽签（issue §5.4；默认按种子升序）
python tools/pickem/predict_stage.py --roster tools/pickem/fixtures/budapest2025s1_uncertain.json --random-ties --n 20000 --seed 20261012

# 自己的 roster
python tools/pickem/predict_stage.py --roster my_stage.json --json out.json

# 事后：真实命中计分 + 逐队边际校准（真值表留私有）
python tools/pickem/scoring.py     --truth truth.json --card out.json
python tools/pickem/calibration.py --truth truth.json --report out.json
```

| 选项 | 默认 | 说明 |
|---|---|---|
| `--roster` | 必填 | roster JSON 路径 |
| `--n` | 20000 | 模拟场景数 |
| `--k` | 5 | 优化目标过关线（Pick'Em 官方口径 = 5） |
| `--seed` | 20261013 | 随机种子（同 seed 结果逐字节可复现） |
| `--uncertain` | 关 | 改跑 **σ 后验臂**（见下）；仅 `source=elo` |
| `--random-ties` | 关 | 同 Buchholz 并列的平局键改 **均匀随机抽签**（默认种子升序，见限制第 3 条） |
| `--json` | — | 把完整结果写入该路径 |
| `--quiet` | 关 | 只落盘不打印 |

退出码：`0` 成功；`1` 用法 / roster 非法 / 读文件失败。**非法 roster 一律拒绝，不出结果。**

一键自证：`python tools/pickem/selftest.py`（64 组；内含「只有 16 个队名 → 十格」那条主用法）

**基线对照默认开**（零新增模拟，只多两次打分）：报表末给出三路十格在同一批场景上的对比——
`优化器` / `naive 序`（按种子直填 1-2 / 3-8 / 15-16）/ `热门基线`（逐格取边际最大，再按剩余集合填）。
每路给 `P(≥k)` 与期望命中格数，并给出优化器相对后两路的**按类别命中分解**；
若某路与优化器选出同一批格子，报表会直接点出（`⚠ 本实例…`）。
JSON 侧落在 `baselines` 键（每路含 `p_ge_k` / `expected_hits` / `expected_by_cat` / `picks`）。
十格推荐行按**边际概率降序**显示（JSON 的 `picks` 保留优化器原始顺序，同一集合）。

## 强度库（`fixtures/team_strength.json`）—— 「只填队名」靠它

公开工具**不读 `data/`**，但「玩家只填 16 个队名」要求强度是现成的 ⇒ 强度在**私有侧算一次**，
落成这个带 provenance 的公开 JSON；公开侧**只查表**。

```jsonc
{
  "asof": "2026-10-28",          // VRS 快照日
  "elo_asof": "2026-10-18",      // Elo 语料切点（不是 VRS 日）
  "elo_matches": 2923,
  "source": "rank_points",
  "_provenance": "rating = 公开语料重算的 Elo；rank/points 刷新自 2026-10-28 全球 VRS 快照；VRS 概率尚未重新校准。",
  "vrs_source": {                // 用来判断「同日榜单是否被重算」
    "date": "2026-10-28",
    "teams": 350,
    "content_sha": "25f4492938ff…",
    "fetched_at": "2026-10-29T04:04:46+08:00",
    "revision": 2
  },
  "counts": { "teams": 391, "elo": 179, "vrs": 350, "both": 151 },
  "teams": [ { "name": "Spirit", "rating": 1753.0253, "rank": 5, "points": 1877.0, "aliases": [] } ]
}
```

- `vrs_source.content_sha` 只对 `(rank, name, points)` 排序后做 sha256，不含
  `fetched_at` / `revision`。**同一天、同样队数、但积分被 HLTV 重算**时哈希会变，
  根服务的 `vrs_sync.py` 据此重写强度库，不会漏掉「只改积分」的更新。
- 同一日期的快照可能有多版：`revision` 每变一次加 1，旧版归档在
  `data/cache/vrs_history/`（gitignored）。页面显示「第 N 版 · X 分钟前抓取」。

- `rating` = **2923 场公开赛果重放的 Elo**：`data/csapi_matches_all.jsonl` 2800 场
  （2025-10-14..2026-09-05）+ `data/groundtruth/new_records_*.jsonl` 123 场增量
  （2026-09-08..2026-10-18），K=24、初始 1500、按日期逐场累计；切点写在 `elo_asof`。
  斜率用公开侧 `sources.FITTED['elo']`。
  `rank`/`points` = 2026-10-28 全球 VRS 快照；`aliases` 保留原库内容。新加入的 VRS 队伍不伪造 Elo，`rating=null`。
- 当前覆盖：391 支队伍，179 支有 Elo、350 支有 VRS、151 支两者都有；**199 支只有 VRS，没有 Elo**，另有 13 支暂时两者都没有。
  在 `source=rank_points`（**默认**）下无 VRS 积分的队按 `250` 兜底并**点名告警**；在 `source=elo` 下无 Elo 的队按 `1500` 兜底并告警。
- **默认强度源是 VRS 积分，不是 Elo。** 原因：Elo 没有赛事权重，三线队在自己的池子里刷分也能刷到一线数字。
  同一批近期赛果（2026-09-20 起 83 场）上，按 VRS 积分判高下命中 **79.5%**，按 Elo 只有 **62.7%**。
  Elo 保留为参考列，不再参与计算。
- **`rank_points` 的斜率已按 VRS 量纲重拟合。** 旧值（bo1 0.4045 / bo3 0.5584）是在 **HLTV 周排名 points** 上拟合的，
  直接套到 VRS 积分上会把整场压平（Vitality vs VRS #54 只剩 56.1%）；新值见 `sources.FITTED`，按 45 天半衰期时间加权。
- **重要：增量语料不是完整赛程。** 2026-10-19..2026-10-28 仍缺（且增量只覆盖部分赛事），所以 Elo 列仍可能滞后；
  页面同时显示 `VRS <asof>` 与 `Elo <elo_asof>`，不要把两者当成同一天。
- 重算 Elo（仓库根目录运行；只改 `rating`，保留 VRS 的 `rank`/`points`）：
  `python pickem/update_elo.py`（预览）→ `python pickem/update_elo.py --write`（写回）。
  这是**维护脚本**，会读 `data/`；页面/CLI 本身仍只查 fixture，不读 `data/`。
  强度库按 mtime 热加载；只有改 `predict_gui.py` 本身才需要重启页面进程。
- **换 VRS 快照**：`python pickem/update_vrs.py --write`（默认取 `data/vrs_rank_*.json`
  中日期最大的一份）按 name → 别名 → 规范化匹配，更新 `rank` / `points`；新队伍
  `rating=null`，掉榜队伍 `rank` / `points=null`，`asof` / `counts` / `_provenance` /
  `vrs_source` 一起重写。`content_sha` 与库内记录相同时直接跳过，不制造无意义的
  mtime / git 变化；`--force` 可强制重写。
  根服务的 `vrs_sync.py` 会在页面加载时自动跑这一步，并在快照超过 60 分钟时复核
  最新页（同一天榜单可能被赛后重算）；独立部署（只有 `pickem/`）时需手动执行或由
  自己的调度器调用。
- 首次生成强度库仍走私有侧 `build_strength.py`（`…/015-scratch/`，自带队数对账断言）。
- `GET /api/teams` 直接返回这张表（含 `asof` / `elo_asof` / `elo_matches` / `_provenance`），
  页面用它做补全与强度标签。

## roster 格式

```jsonc
{
  "stage": "StarLadder Budapest Major 2025 Stage 1",   // 仅用于显示
  "date": "2025-11-24",                                // 仅用于显示
  "source": "rank_points",                             // "rank_points"（默认）| "elo" | "matrix"
  "slopes": { "bo1": 4.8824, "bo3": 4.0943 },          // 可选，覆盖默认斜率
  "teams": [
    { "name": "FaZe", "seed": 1, "points": 1558.0 },     // source=rank_points 用 points（VRS 积分）
    { "name": "Legacy", "seed": 2, "points": 1689.0 },
    // source=elo 时改填 "rating": 1485.6957；--uncertain 还要 exposure/games/rank
  ]
}
```

**第三条源：`source: "matrix"`（外部模型的桥）** —— 直接把两两胜率（bo3）交给本工具，
于是 **014 已训模型/任何第三方预测器**都能接进来，而公共工具仍然不读语料、不依赖私有路径：

```jsonc
{
  "stage": "Singapore Major Stage 1",
  "source": "matrix",
  "alpha_bo1": 0.75,                     // 可选：bo1 折算系数（默认 0.75）
  "matrix": { "FaZe": { "Legacy": 0.63 },  // 只给上三角即可，反向取 1−p
              "Legacy": { "M80": 0.41 }, ... },
  "teams": [ { "name": "FaZe", "seed": 1 }, { "name": "Legacy", "seed": 2 }, ... ]
}
```

矩阵的硬校验（违反即 rc=1）：每对有序都要有落点（缺项会被模拟拿 0.5 兜底）、
概率严格落在开区间 **(0,1)**（0/1 会让 logit 折算发散）、两向都给时之和必须 ≈1（自洽）、
**不允许出现 teams 之外的队名**（拼错会让整行静默失效）。bo1 用 `sigmoid(α·logit(p_bo3))`。
示例见 `fixtures/example_matrix.json`（由示例 Elo 生成，真跑时应由外部模型产出）。

硬约束（违反即 rc=1）：**恰好 16 队**、队名不重复、`seed` 恰为 **1..16 的排列**、
`source` 对应的强度字段齐全。

`--uncertain` 额外的硬约束（违反即 rc=1）：`source` 必须是 `elo`；
每队必须给 `sigma`（直接指定 σ），**或**同时给 `exposure`（E）与 `games`（n）；
`rank` 可选（缺则该队不做均值回归，μ 保持自身 rating）。

**种子是关键输入。** 配对规则以种子为平局键，填错种子会得到错的赛程。
现实里种子的来源：官方公告 / 上一阶段战绩。数据源不可用时由人工提供。

**`rating` 的精度会影响结果。** 模拟对概率敏感，四舍五入到 1 位小数会让随机流偏移、
复现失败——请给全精度。

**`exposure` / `games` / `rank` 的语义**（与 issue-015 §5.5 一致）：
E = 赛前与 `rank≤30` 队伍的交手场数；n = 赛前总场数；rank = 赛前最近一次排名。
它们必须由**赛前**数据算出（含本赛事即泄漏）。公开工具不抓数据、不读语料——
缺口径时用 `sigma` 字段直接表达你的先验。

## 算法（三段 + 可选 σ 臂）

1. **U1 胜率层**（`sources.py`）：强度 → `pwin(a, b, best_of)`。
   - `EloSource`：`sigmoid(k · (r_a − r_b) / 400)`
   - `RankPointsSource`：`sigmoid(k · (log1p(pts_a) − log1p(pts_b)))`（HLTV 周排名 points）
   - `MatrixSource`：两两胜率直给（bo3），bo1 用 `sigmoid(α·logit(p_bo3))`，α 默认 **0.75**
   - bo1 / bo3 各一条斜率，默认值为 014 语料上的拟合结果（见 `FITTED`）。
2. **U1.5 σ 后验臂**（`uncertain.py`，**仅 `--uncertain` 时启用**）：
   冷门队（零一线交手）**不靠采集去认识**，而是把「不知道」显式建模成强度后验的方差。
   - σE 分桶（先验量级）：E=0 → **120**、E1-2 → **80**、E≥3 → **40**（Elo 单位）
   - 逐队合成 σ = √( σE(E)² + (150/√(1+n))² )
   - 低暴露队（E < **5**）的均值 μ 回归到 **rank 先验**（队伍内拟合
     `rating ~ a + b·ln(1+rank)`），修正「弱旅在弱对手身上刷分」的高估；E ≥ 5 用自身 rating
   - 每场景抽 θ ~ N(μ, σ) 后跑完整瑞士轮
3. **U2 瑞士轮模拟器**（`swiss.py`）：16 队、三胜晋级 / 三败淘汰、最多 5 轮；
   R1 固定种子 `1v9 … 8v16`；其后**同战绩配对**，组内按 **Buchholz（已交手对手当前胜场和）降序**
   分簇，簇内平局键**默认种子升序**（`--random-ties` 时改**均匀抽签**），再**最好对最差**；
   **同阶段不重赛**（回溯求完美匹配）。
   **所有直接决定晋级/淘汰的比赛为 bo3，其余 bo1 ⇒ 每阶段恰好 20 场 bo1 + 13 场 bo3。**
4. **U3 十格优化器**（`pickem.py`）：在场景矩阵上做随机重启 + 首次改进爬山，
   最大化 `P(命中 ≥ k)`；另给「独立假设」泊松二项闭式解作对照，量化忽略相关性的乐观偏差。
   **生产重启预算 `OPT_RESTARTS = 24`**（`predict_stage.py`）：模块默认 8 实测会在**平坦目标面**
   落谷（12 队穷举对拍：1/26 实例、最大差 0.0067），24 起在发现的那例上换 5 个优化种子 5/5 达最优；
   真实实例在 8~800 重启下**结果完全一致** ⇒ 已公布数字不受影响。**爬山仍不保证最优**（限制第 11 条）。

**σ≡0 时 σ 臂逐字节退化为点估计臂**（自证第 7 条）——这是两臂共用同一模拟路径的证明。

## 真实命中计分（`scoring.py`）

报表回答「该怎么填、P(≥5) 是多少」；**真实命中**回答「填了这套格子，赛后中几格」——
这是「模型到底有没有用」的唯一判据，也是端点揭盲时的必备件。

```powershell
# 1) 出十格（报表 JSON 里同时含优化器与三路基线）
python tools/pickem/predict_stage.py --roster my_stage.json --json card.json
# 2) 赛后拿真值表计分
python tools/pickem/scoring.py --truth truth.json --card card.json
python tools/pickem/scoring.py --truth truth.json --card card.json --k 6   # 换过关线
```

真值表（**留在私有区、不入库**；`data/` 永不公开）：

```jsonc
{
  "stage": "StarLadder Budapest Major 2025 Stage 1",
  "records": { "FaZe": [3, 2], "FlyQuest": [3, 0], "Lynn Vision": [0, 3] }  // 16 队各自的 (胜, 负)
}
```

`records` 必须是一条**完整 16 队瑞士阶段**（3 胜恰 8 支、其中 3-0 恰 2 支；3 败恰 8 支、其中 0-3 恰 2 支）。
不符即 **rc=1 拒绝**；十格一侧同样硬校验（2/6/2、三组互斥、队名必须在真值表内 ⇒ **顺手挡住阶段错配**）。
理由很直接：**错的真值表会静默算出错的命中数**。

`--card` 接受 `predict_stage.py --json` 的完整报表（一次评优化器与三路基线）、最简 `{p30,padv,p03}`
或 `{picks:{...}}`；形状认不出即 rc=1（**不许 rc=0 空跑**）。输出给「命中格数 / 分类命中 / 命中格 /
未中格（含实为哪一类）」，可落盘 JSON。

> **口径警告（重要）**：本工具内置基线 `naive_rank` 是**按 roster 的 `seed` 字段直填**
> （官方种子 1-2 / 3-8 / 15-16）；而 014/015 私有回测里的 `naive_rank` 是**按赛前 HLTV 排名直填**
> ——**同名异义**，两者在本阶段上是**两张不同的卡**（实测真实命中 **6/10 vs 3/10**）。引用时必须写清是哪一个（已登记 PENDING BF-6）。

**已实测**（Budapest S1，真值来自私有语料；十格用 `--n 20000 --seed 20261012` 现场生成）：

| 卡 | 真实命中 | 分类（3:0 / 晋级 / 0:3） |
|---|---|---|
| 优化器（与本例热门基线同解） | 3/10 不过 | 1/2 ｜ 2/6 ｜ 0/2 |
| 私有 `heuristic`（逐格取边际最大） | 3/10 不过 | 1/2 ｜ 2/6 ｜ 0/2 |
| 私有 `naive`（**按赛前 HLTV 排名**直填） | 3/10 不过 | 0/2 ｜ 3/6 ｜ 0/2 |
| 本工具 `naive_rank`（**按官方种子**直填） | **6/10 过** | 0/2 ｜ 5/6 ｜ 1/2 |

三点读法：① 本计分器与私有 `count_correct` 在**三张可比卡上逐卡一致**（3 / 3 / 3）
⇒ 两套独立实现对拍互证；② 按官方种子直填**在这个阶段过了关**而模型没过——**但这是单阶段小样本**，
只说明本例瓶颈在**强度源**（赛前只有 6 周历史，见限制第 1 条）而不在组合搜索，**不能反推「种子序更优」**；
③ **种子本身是时点安全、赛前可得的官方信息**（邀请/晋级分区），是尚未建模的候选特征。

## 校准检验（`calibration.py`）

逐队 P(3:0) / P(晋级) / P(0:3) 到底可不可信？判据是**能不能击败常数基线**
（常数基线 = 永远输出该类平均发生率）。依据 issue-015 §5.1：赛前排名源的 3:0 Brier
（0.1220）**比常数基线（0.1094）还差** ⇒ 该源在 3:0 上没提供信息。

```powershell
python tools/pickem/calibration.py --truth truth.json --report rep_elo.json
# 多份报表可同时给（比较源/臂）；也可重复给不同阶段
python tools/pickem/calibration.py --truth t.json --report point.json --report sigma.json
```

吃**任何含 `marginals` 的报表**（`predict_stage.py --json` 的产物，或私有回测
`backtest_budapest_*.json`——同名字段，所以私有公布的数字可直接复核）。输出 Brier / 常数基线 /
差 / LogLoss / 实测率 + **可靠性分箱**，并对劣于基线的类别直接标 `⚠`。

**已实测（Budapest S1，真值来自私有语料）**：

| 源 | 3:0 差 | 晋级 差 | 0:3 差 | 读法 |
|---|---|---|---|---|
| elo（点估计） | −0.0007 | **+0.0320** ⚠ | **+0.0056** ⚠ | 只有 3:0 勉强打平基线 |
| elo（σ 臂） | **+0.0050** ⚠ | **+0.0017** ⚠ | −0.0032 | σ 臂把晋级/0:3 改好，但 3:0 反而变差 |
| rank（赛前排名） | **+0.0126** ⚠ | −0.0173 | **+0.0007** ⚠ | 只在晋级上赢基线 |
| predictor（014，⚠ 泄漏） | **+0.0130** ⚠ | **−0.0495** | −0.0069 | 三项中两项赢基线，但**不能当成绩** |

⇒ 三个**时间安全**源在 3:0 上全部输给常数基线，winning 的类别最多一个；
**端点上 Predictor 不再泄漏**（state 停在赛前），所以「接 014 模型」是本 issue 目前
证据最强的下一步（对标 P0-②）。逐项数字与私有回测逐位一致（互证）。

## 本地 Web 界面（`predict_gui.py`）

对标 issue-014 的 `predict_gui.py`：**一个本地页面**，主视图只干一件事——
**填 16 支队伍（按种子顺序）→ 出十格**；其余能力收在「高级」折叠区里：
模拟参数（n / k / seed / 强度源 / 平局键 / σ 臂）、手填 roster（含 matrix 源）、
**计分**（粘真值表 → 真实命中，十格卡片按命中着绿/红）、**校准**（真值表 + 报表 → Brier/基线/分箱）。
纯标准库 `http.server`，无依赖、无 CDN；配色与交互对齐 014（单列 900px、卡片、pill、带元信息的补全下拉）。

**不用手工确认种子序**：页面上的「自动排种子」按钮会先识别「首轮对阵顺序」
（`1v2、3v4…`，即前 8 与后 8 交错），把它转成种子序；若输入前 8 正好是 VRS 前 8
（官方种子序可能因区域避让微调 9–16 位，例如 MIBR VRS #12 排到 14 号位），保留原顺序；
其余情况才按 VRS 排名升序排序。强度标签写作 `VRS #… · 积分`：`#` 是 **VRS 排名**；
切到 `source=elo` 时写作 `Elo … · VRS #…`，Elo 只是参考列，不参与默认计算。

```powershell
python tools/pickem/predict_gui.py                 # 默认 http://127.0.0.1:8790
python tools/pickem/predict_gui.py selftest        # 起服务 + 真实 HTTP 回环自证（58 项）
```

**统一入口**：014 根目录的 `predict_gui.py` 也提供本工具——`python predict_gui.py` 后，
`http://127.0.0.1:8765/` 是分数预测，`http://127.0.0.1:8765/major` 是本文的十格预测；
两个页面顶部共用场景切换条。统一入口下本工具的 API 前缀是 `/major/api/*`
（如 `/major/api/pickem`），字段与独立入口完全一致。`pickem/` 仍可整体搬运、
单独运行，不依赖根目录的预测器。

**端口被占 → rc=1，不静默回退。** 但**光靠 `OSError` 拦不住**：Windows 的 `SO_REUSEADDR`
（`HTTPServer.allow_reuse_address=1`）语义是「**允许抢绑**」——实测 8790 已有人在 listen 时，
第二个进程照旧打印就绪、rc=0，于是同一端口上两个服务、请求随机落过哪一个。
故 `serve()` 在绑定前先 `port_busy()` **主动探一次连接**（已实测：在 014 占着的 8765 上
`--port 8765` 得 **rc=1 + 明确原因**）。

接口：`GET /`、`GET /api/status`、`GET /api/teams`、`GET /api/demo`、`GET /api/fixtures`、
`GET /api/fixture?name=`、`POST /api/pickem`（**主入口**：16 个队名 → 十格）、
`POST /api/predict`（手填 roster）、`POST /api/score`、`POST /api/calibrate`（详见文件头注释）。
`/api/pickem` 与 `/api/predict` 字段一致：`report` / `report_text` / `command` / `freeze` /
`elapsed_s` / `roster` / `roster_file`，`/api/pickem` 另带 `warnings`（未收录队）与 `library`。
两者都返回**冻结件**（`utc` / `stage` / `source` / `n,k,seed` / `arm` / `tie_break` /
`roster_sha256` / `command`），页面上可一键下载——哈希按**规范化 roster**（排序键、无空白）算，
任何人可**独立重算对拍**；页面同时给出**等价 CLI 命令 + 同名 roster 下载**，
于是「页面上看到的数字」可以整行粘进终端复现。

## 端点 runbook（新加坡 Major 赛前 5 步）

1. **装 roster**（`seed` 用官方公告；Stage 2/3 可按 §5.2.1 取证过的惯例推导：8 支晋级队进种子 9–16；
   强度用排名归档或 014 Predictor 的 16×16 矩阵）。数据新鲜度是已知瓶颈：Elo 语料止 2026-10-18
   （2923 场），VRS 快照 2026-10-28；不确定种子序就点页面上的「自动排种子」。
2. **跑预测**：打开页面（已预填示例阶段）→ 把 16 个名字改成当日名单（或先「清空」）→ 点「预测十格」
   （或 `predict_stage.py --n 20000 --seed <当日> --json card.json`）；
   看十个位置、`P(≥5/6/7)`、逐队边际、三路基线，以及那句 ⚠（热门基线是否同解）。
   **强度库是快照**（`asof` 见页面右上角）⇒ 名单越新、库越旧，“强度”就越不可信，接口在 `warnings` 里说。
3. **封存**：页面「下载冻结件」或手工存 `card.json` + 冻结件 + `sha256sum` ⇒ 事后无人能说“改过”。
4. **赛后**：拿真绩表 `scoring.py --truth --card` 计分（公开出真命几格），`calibration.py` 看边际是否可信。
5. **回写**：把揭盲结果、实际命中、偏离原因写回 issue（含冻结件哈希）。

**把 014 模型接进来（桥的一行调用 + 矩阵）**：

```python
# 在私有区跑（那里才有 014 的 predict.py 与语料）；不在本公共工具内
from predict import Predictor              # 014-scratch/predict.py
P, teams = Predictor(), [t["name"] for t in roster["teams"]]
matrix = {a: {b: round(P.live_sample(a, b, 3)["pA"], 4) for b in teams if b != a} for a in teams}
roster.update({"source": "matrix", "alpha_bo1": 0.75, "matrix": matrix})   # → 交给本工具
```

`live_sample(A, B, bo, date)` 直返 P(A 胜)；**端点上它不泄漏**（state 停在赛前），
而**回测会泄漏**（`elo_of/winrate/form` 取当前 state，见限制第 1 条）。
安全边界：**只读 `fixtures/*.json` 白名单**（不做路径拼接）、请求体上限 1 MiB、
默认只绑本机、错误一律 JSON 而非堆栈。

> **★ 端口选择有环境事实**：`8765` 已被 **issue-014 的 `predict_gui.py` 占用**（实测 `ready:true`），
> 所以本工具默认改为 **8790**；两者可以同时跑。启动失败一律 **rc=1 报错，不做静默回退**。

## 验证状态

- **模块自证**：`swiss` 45 / `pickem` 16 / `sources` 36 / `uncertain` 26 / `scoring` 38 /
  `calibration` 24 / `predict_gui` 58 = **243 项断言**，聚合 `selftest.py` **64/64 组** @2026-10-15
  （含报表基线块、`baselines` 三路字段、naive 直填 1-2/3-8/15-16、**优化器不劣于任一基线**
  （后者已用变异体验过会红）、**穷举对拍** 3 项 + 穷举器健全性 3 项、`matrix` 源 7 项、
  **计分器** 6 项、Web 面 405/413/路径穿越等负样本）。
- **优化器最优性穷举对拍** @2026-10-15：12 队小实例全空间 **83,160 组**穷举作精确参照
  ——`restarts=8` 在 seed 13 上 **0.3167 vs 0.3233**（差 2.1%），定因是**爬山落谷**（最优解的格子
  在候选池内，非截断）；预算抬到 24 后通过，**报表除「评估次数」行外逐字节不变**、0.2106 不变。
  对照扫描 23 个小实例，`8` 全部达最优 ⇒ 约 **1/26** 的失败率量级。
- **端到端复现** @2026-10-13：用示例 roster 跑 `--n 20000 --seed 20261012`
  → **P(≥5) = 0.2106**，与 issue-015 §5.1「赛前 Elo」行**逐位一致**
  （`grid_p` 同时给出 P(≥6)=0.0681、P(≥7)=0.0164；相关性修正 −0.0116）。
  这是公开工具与私有回测「同一实现」的交叉证据。
- **σ 臂落地** @2026-10-14：私有一致性交叉验证——同一 fixture、同参
  （`--n 8000 --seed 20261013`）下，本工具点估计臂 **0.2139**（私有区公布 **0.214**）、
  σ 臂 **0.1634**（私有区 **0.166**）。私有脚本已独立复跑确认其数字。
  残差 0.0026 已归因：**θ 抽样流按 roster 队序分配**（乱序同名 roster 得 0.158~0.163，
  漂移同量级），非机制差异。
- **基线对照** @2026-10-15（Budapest S1，`--n 20000`，三个 seed 一致）：优化器 **0.2106** /
  naive 序 **0.0685** / 热门基线 **0.2106**（期望命中 3.40 / 2.67 / 3.40）。⇒ 教训不是「搜索没用」
  而是**「按排序直填」这个策略没用**：任何序（种子序或纯 rank 序）都远劣于按边际选格，
  而一旦按边际选格，本例中三路的格子集合与优化器**完全相同**（`--k 5` 下）。数字为
  同一模型目标下的相对差，**不是**对真实战绩的检验（见限制第 10 条 + issue-015 §7）。
  ★ **但真实命中给出了相反的量级提示**（见上一节）：按**官方种子**直填在本阶段真实命中
  **6/10（过关）**，而模型与另两路只有 3/10 ⇒ 「按排序直填没用」这句在**真实战绩**上不成立，
  该结论只在「本模型眼里」成立，根因是强度源缺历史（限制第 1 条）。
- **本地 Web 界面** @2026-10-15：`predict_gui.py selftest` **58 项 rc=0**（真实 HTTP 回环；
  含**端口占用必硬失败** 5 项：`port_busy` 对忙/闲端口与 `port=0` 的判定、占用端口上
  `serve()` 抛 `OSError`、CLI 在占用端口上 **rc=1 且说明原因**（实测跑在 014 占着的 8765 上）；
  含**冻结件哈希可独立重算**、**`/api/pickem` 出十格 2/6/2**、**未收录队必须点名**、
  **σ 臂在强度库 roster 上必须 400** 等断言）。
  浏览器面板实测（**主用法**，`fixtures/demo_stage_s1.json` 的 16 个队名，`n=20000`）：
  页面 **4.27 s** 出十格（10 个位置按 `3:0 / 3:1·3:2 / 0:3` 分栏编号）、`P(命中 ≥5)=0.2840`，
  与同一 roster 直跑 `PS.predict` **逐位一致**；补全下拉敲 `navi` 命中 Natus Vincere；
  整页无横向溢出（`scrollWidth == clientWidth`）。
  再看两块赛后面板：计分（合成真值，与 `scoring.py` 命令行**逐卡一致**：optimizer 8/10 过、
  1/2 ｜ 5/6 ｜ 2/2，卡片命中/未中着绿/红，未中格给出「填 X / 实为 Y」）；
  校准（三类 Brier / 常数基线 / 差 / LogLoss / 可靠性分箱均出数）。
  此前一轮：默认 fixture + `--n 20000 --seed 20261012` 的页面数字
  **0.2106 / 0.0681 / 0.0164 / 独立 0.2222 / 修正 −0.0116 / 评估 5664** 与 CLI 逐位一致；
  再拿私有真值表计分得到 optimizer **3/10**、naive_rank **6/10**，十格卡片命中着色 3 绿 / 7 红。
- **校准检验** @2026-10-15（`calibration.py`）：私有 `backtest_budapest_{elo,rank,predictor}.json`
  的 Brier 被**逐位复现**（如 rank 3:0 = 0.1220 / predictor 晋级 = 0.1849）⇒ 公共模块与私有回测互证。
- **matrix 源（014 桥）** @2026-10-15：`fixtures/example_matrix.json` 端到端 rc=0；
  违规矩阵（含 1.0 / 缺对 / 未知队名）各自 rc=1；`matrix + --uncertain` 也 rc=1（σ 臂只在 Elo 单位验证过）。
- **真实命中计分** @2026-10-15（`scoring.py`，真值来自私有语料）：
  ① 合成真值 10/10（公共 fixture，`selftest.py` 内跑）；
  ② **与私有 `count_correct` 三张卡逐卡一致**（优化器 3 / heuristic 3 / 按赛前排名直填 3）；
  ③ 四卡对拍时发现 **同名异义**：本工具 `naive_rank`（按官方种子）真实命中 **6/10**，
     私有 `naive_rank`（按赛前 HLTV 排名）**3/10**（已登记 BF-6）；
  ④ 开发过程中抓到一个**静默失败**：`card` 形状认不出时原先 rc=0 且零策略——已改为报错，
     并补了 2 条负样本（认不出的 card / baselines 里没有合法十格）。
  报表因此多出基线块（较此前多 7 行文本；命中同一批格子时多 1 行 ⚠ 提示），
  **P(≥5)=0.2106 等数字不变**。
- **规则实现已用真实赛事反证**（见 issue-015 §5.1/§5.2）：语料内唯一完整的
  Swiss 阶段 **Budapest 2025 Stage 1（33/33 场）** 实测 20 bo1 + 13 bo3、
  8 晋级 / 8 淘汰、恰 2 支 3-0、恰 2 支 0-3，逐条吻合。
- 复跑：`python tools/pickem/selftest.py`；
  `python tools/pickem/predict_stage.py --roster tools/pickem/fixtures/budapest2025s1_elo.json --n 20000 --seed 20261012`；
  `python tools/pickem/scoring.py --truth <私有真值表> --card <predict_stage 的 --json 产物>`

## 已知限制

1. **时间安全强度源判别力弱。** 014 语料起点 2025-10-14，Budapest 阶段赛前只有约 6 周
   （445 场）历史 ⇒ Elo 失真（Top 段多为三线队）、排名源弱到 **3:0 的 Brier（0.122）
   反比常数基线（0.109）更差**。要让回测有意义，需先把 014 模型**回滚到赛前状态**。
   ⚠ 这个 Brier 是在**旧斜率**（HLTV points 量纲，0.5584）上算的；2026-10-29 已按 VRS 量纲重拟合
   （3.14），该行数字**尚未重跑**，不要拿它当「VRS 不行」的证据。
2. **bo1/bo3 斜率差测不出**：两个口径的置信区间重叠 ⇒ 只能沿用拟合值（issue §5.1）。
3. **同 Buchholz 并列：默认仍按种子确定性排序**（已公布数字依赖此口径，不作废），
   但证据（issue §5.4）显示真实平局打破**倾向抽签**（Stage 2 上「R2 完美」与「R5 完美」的
   种子序**结构性互斥、交集 0**）⇒ 提供 `--random-ties` 切到均匀抽签。
   **实测代价**（Budapest S1，同 seed/n 仅换平局键）：点臂 ΔP(≥5)=**−0.0005**
   （多 seed −0.003~−0.008）、σ 臂 ΔP(≥5)=**−0.0111**（多 seed −0.009~−0.014），
   边际最大偏移 0.008 / 0.018 ⇒ 随机抽签**略偏保守**（原口径轻微乐观），十格在小幅内换位。
4. **σ 臂的 σE 是「先验量级」而非拟合值**（E=0 只有 6 个样本，定不了小数点）；
   队间独立噪声**低估**系统性平移（「二线队整体在一线变弱」）⇒ 结果偏保守。
   验证靶目前**只有 Budapest S1 单阶段**；Cologne S1/S2 对照臂**未跑**
   （需先在私有区建配套 CONFIG，已登记 PENDING）。
5. **σ 臂只支持 `source=elo`**（σE / 150 采样项 / rank 先验都在 Elo 单位上标定）；
   `rank_points` 源 + `--uncertain` 直接 rc=1，不做量纲换算以免伪装成已验证。
6. **未来阶段的名单与种子当前数据源全断**（见 issue-015 §5）；工具不抓数据，
   roster 必须由外部提供。
7. **只模拟瑞士阶段**，不含 playoff（8 队单败、决赛 bo5）。
8. `P(≥k)` 是**该十格**在模拟分布下的过关概率，不是模型整体准确率；
   场景数越大越稳（`--n`）。
9. **`numpy` 依赖**：`pickem.py` 用 numpy 做场景矩阵与打分；`uncertain.py` 用它做 θ 抽样；
   未提供纯标准库降级路径。
10. **基线对照是「同目标内的策略对比」，不是准确率检验**：三路都在**同一批模型场景**上打分，
   故它只回答「哪套十格在本模型眼里更好」，不回答「哪套真的会中更多格」。
   真实命中需要 §5.1 的回测口径（`优化器实际命中 3/10`），而语料里的 Stage 1 **实际战绩表**
   未落在本仓库 ⇒ 本工具**无法**给出三路的真实命中对照（已登记 PENDING BF-4）。
11. **优化器不保证最优**：`pickem.optimize` 是「随机重启 + 首次改进爬山」，**不是**精确求解。
   12 队穷举对拍实测：`restarts=8` 有 **1/26** 实例落谷（最大差 0.0067 / 相对 2.1%）；
   本 CLI 的生产预算已抬到 24（该例上 5/5 达最优），但**16 队全空间有 10,090,080 组、无法穷举**
   ⇒ 真实实例的最优性**仍未证明**，只能说「在本预算下与多次重启/放宽池的结果一致」（PENDING BF-5）。
12. **真实命中的样本只有 1 个阶段**，且该阶段的强度源已知失真（限制第 1 条）⇒ 上面那 4 行命中数
   只能作**量级示意**，不能当准确率，更不能用来选策略；要选策略得等端点揭盲或多阶段真值。
13. **`naive_rank` 同名异义**（本工具＝按官方种子直填；私有回测＝按赛前 HLTV 排名直填）。
   本工具暂不改名（改名会动已公布的 JSON 契约），以文档标注 + PENDING BF-6 挂账。
14. **真值表需外部提供**：工具不抓数据、不读语料，`data/` 永不公开 ⇒ 公共区只能给合成 fixture；
   真实赛事的真值表要由私有区/人工提供。
15. **Web 界面是本机工具**：默认只绑 `127.0.0.1`；`--host 0.0.0.0` 会把一个无鉴权的
   （能读fixture、能跑模拟的）接口暴露到局域网——如需共用自己加反代/鉴权。
   端口的既有占用情况见上（8765 = 014 GUI，本工具默认 8790）。
   **端口占用靠 `connect` 探测而非 `bind` 失败**（Windows 的 `SO_REUSEADDR` 允许抢绑，
   `bind` 不会报错）；只绑在别的接口上的监听探不到，这是已知边界。
16. **校准只校逐队边际，不校 P(≥k)**：过关概率的校准需要**多阶段**（每个阶段只有 1 个
   0/1 结果），目前只有一个可校阶段 ⇒ 不声称 P(≥5) 已校准（见 `calibration.py` 输出末两行）。
17. **强度库是快照，不是实时：** `fixtures/team_strength.json` 的 `asof` = VRS 快照日，
   `elo_asof` = Elo 语料切点（当前 2026-10-18，2923 场）；两者不是同一天。
   `rating` 来自只有约一年跨度的语料（同限制 1，Elo 会失真），且增量赛果不完整
   （2026-10-19..2026-10-28 缺，增量也只覆盖部分赛事）。名单越新、库越旧，「强度」就越不可信；
   更新 Elo 用 `python pickem/update_elo.py --write`；换 VRS 快照用
   `python pickem/update_vrs.py --write`（根服务页面加载时会自动跑）。
18. **库里两个源的覆盖不同（2026-10-28 快照）：** 391 支里 179 有 Elo / 350 有 VRS / 151 两者都有；
   **199 支只有 VRS、没有 Elo** ⇒ `source=elo` 时按 `1500` 兜底（接口在 `warnings` 里点名）。
   另有 28 支只有 Elo，及 13 支两种强度都缺；`rank_points`（默认）缺 VRS 积分时按 `250` 兜底并告警
   （低于榜尾 355，但不是 0 分悬崖——0 分会让胜率饱和到 1.0）。
19. **「只填队名」这条路跑不了 σ 臂：** σ 后验需要每队的 `exposure`/`games`，强度库不含
   ⇒ 用库建的 roster + `--uncertain` 一律 **400/rc=1**（不静默降级为点估计）；要 σ 臂请手填 roster。
20. **`aliases` 里有一部分是人工整理的简写**（`nip`/`vp`/`mouz` 等），不是从数据里学出来的，
   可能有遗漏（例：`NIP` 靠人工表，`Ninjas in Pyjamas` 靠语料名）；未收录的队一律告警，
   不会静默当作已知队。

## 来源

- 核心 `pickem.py`：**逐字节复制**自 Deepseek 私有区
  `data/Deepseek/Deepseek-V4.1 Flash/2026-10-07/015-scratch/`（经用户 2026-10-13 授权转公开），
  自证断言计数与私有区一致（16）；`cmp` 复验一致。
- `swiss.py`：**2026-10-14 起与私有区不再逐字节一致**——一处**有意、受限的分歧**：
  新增 `tie_break="seed"|"random"`（默认 `"seed"` ⇒ 行为逐字节等价），
  组内排序由「`sorted(-buchholz, seeds)`」改为「Buchholz 分簇 + 簇内排序/抽签」，
  并新增 13 项自证（30→43）。分歧仅限配对平局键，不涉 bo 判定 / 晋级淘汰规则；
  私有区文件**未被改动**，是否需要回移由该区承接人决定。
- `sources.py`：把私有区的**推理端**抽成公开版（保留公式与拟合默认值，
  去掉语料读取 / 拟合 / bootstrap / 泄漏对照）。
- `uncertain.py`：把私有区 `data/Qwen/Qwen-3.8 Flash/2026-10-13/uncertain_sim.py`
  的机制抽成公开版（σE 分桶 / σ 合成 / rank 先验 / θ 抽样与 RNG 布局同式），
  **去掉语料扫描**——E/n/rank 改由 roster 携带。
- `predict_stage.py` / `uncertain.py` / `selftest.py` / `fixtures/`：公共入口与自证。
  `predict_stage.build_roster` / `load_strength_library` / `resolve_team` 是 16 队名 → roster 那一步
  （纯逻辑，不读文件；库由调用方传入）⇒ CLI 与 Web（与将来的其他前端）共用同一份实现。
- `fixtures/team_strength.json`（**公开强度库**）：初始由私有侧 `015-scratch/build_strength.py` 生成，
  内容 = 全语料 Elo（K=24）+ VRS 快照（rank/points）+ 队名/别名；provenance 与覆盖数写在文件内。
  维护脚本 `pickem/update_elo.py` 负责把增量公开赛果（`data/groundtruth/new_records_*.jsonl`）
  重放进 Elo，只改 `rating` 并写 `elo_asof` / `elo_matches`；`pickem/update_vrs.py`
  用最新 VRS 快照更新 `rank` / `points`（不碰 Elo）；页面/CLI 仍只查表。
- `fixtures/demo_stage_s1.json`（示例**队名清单**，不是 roster）：页面开机预填用；
  只回答「填哪 16 支队」，强度不在里面（查库）。
- issue-015 全文：`issues/015-major-pickem-prediction.md`；
  过关线与 R5 归因：`data/public/findings/007`；冷门队先验：`data/public/findings/008`。
