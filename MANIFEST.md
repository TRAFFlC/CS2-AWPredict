# MANIFEST — Alpha 入库清单

本文件记录**哪些文件被放进仓库、哪些被刻意排除**，以及依据。
目的是让任何人都能复现这次筛选，而不是只能相信某个人说过。

> 生成日期：2026-10-14　·　版本：`v1.0.1-alpha`

---

## 1. 白名单是怎么定的

不是按文件名猜的，而是**用 AST 算导入闭包**：从 `predict.py` / `predict_gui.py`
沿 `import` 递归，得到运行时真正需要的本地模块——结果只有 **10 个**，
远少于目录里约 200 个文件。运行期再按源码里的字符串字面量反查读盘的数据文件。

```
入口 predict.py     -> 10 个本地模块
入口 predict_gui.py -> 10 个本地模块
```

再加入增量摄入四件套（`state_snapshot` / `record_schema` / `veto_rec_build` /
`ingest_new`）与前端 `predict_page.html`。

---

## 2. 入库内容

| 分组 | 文件数 |
|---|---|
| `.gitattributes/` | 1 |
| `.gitignore/` | 1 |
| `LICENSE/` | 1 |
| `README.md/` | 1 |
| `__pycache__/` | 9 |
| `_gen_manifest.py/` | 1 |
| `coldstart.py/` | 1 |
| `data/` | 87 |
| `decide.py/` | 1 |
| `experiments/` | 50 |
| `ingest_new.py/` | 1 |
| `map_pools.py/` | 1 |
| `model_v2.py/` | 1 |
| `pool_experiment.py/` | 1 |
| `predict.py/` | 1 |
| `predict_gui.py/` | 1 |
| `predict_page.html/` | 1 |
| `pyproject.toml/` | 1 |
| `record_schema.py/` | 1 |
| `scorer.py/` | 1 |
| `split.py/` | 1 |
| `state_snapshot.py/` | 1 |
| `veto_model.py/` | 1 |
| `veto_rec_build.py/` | 1 |

**总计 167 个文件 / 19.92 MB**

---

## 3. 黑名单（刻意排除）

### 3.1 第三方内容 —— 与本项目无关，且体量大

| 路径 | 规模 | 排除理由 |
|---|---|---|
| `/rankings/` | 603 文件 / 103.3 MB | 上游 HLTV 周更归档，含嵌套 `.git` |
| `/ranking/` | 41 文件 / 0.2 MB | 克隆的第三方仓库（含 `.git`） |

> **替代方案**：`data/rankings/` 是它的**精简派生**（剥离 players / logo，
> 只留 name/rank/points），**103.3 MB → 0.70 MB，缩减 139 倍**。
> `pool_experiment.py` 需要它做「比赛日前最近一期」的时序安全排名，
> 所以必须入库——但只入库派生版。

### 3.2 抓取中间产物

`cache/`（Liquipedia HTML）、`csapi_cache/`、`fixtures/`、`stier.html`、
`valvetier1.html`、`*.html.json` —— 均为可重新生成的抓取缓存。

### 3.3 运行时产物与临时文件

`__pycache__/`、`*.log`、`*.out`、`*.err`、`*.tmp`、`*.bak*`、
`gui.out`、`predict_gui.log`。

### 3.4 凭据相关

`hltv_cookies.tmp`、`hltv_dates_cookies.tmp`（Netscape cookie 文件；
经检查**只有注释、无 cookie 值**，但仍排除）。

---

## 4. 敏感信息扫描（push 前）

```
私钥 -----BEGIN…PRIVATE KEY-----      未命中
GitHub token gh[pousr]_<36 chars>      未命中
AWS key AKIA…                          未命中
Slack xox[baprs]-…                     未命中
赋值型 token/api_key/secret/password    未命中
敏感文件名 cookie/secret/.env/.key     未命中
大于 20MB 的文件（GitHub 硬限 100MB）  无
PREPUSH_OK
```

---

## 5. 构建过程中被验收抓到的 5 个真问题

记录在此，因为**它们都属于「看清单看不出来、只有真跑才会炸」**的类型：

| 问题 | 现象 | 根因 |
|---|---|---|
| scorer 放 `scorer/` 目录 | `ImportError: cannot import name Prediction from scorer (unknown location)` | 无 `__init__.py` 的目录被当作**命名空间包** |
| scorer 放 `014-scorer/` | `ModuleNotFoundError: No module named 'scorer'` | 源码写死 `os.path.join(HERE, "..", "014-scorer")`，模块在仓库根时 `..` 指向**仓库外** |
| 数据放 `data/` | `FileNotFoundError: csapi_matches_all.jsonl` | 运行时代码到处 `os.path.join(HERE, …)` |
| `_build_rank` 用局部 `here` | rank 条目数为 **0** | 重定向脚本只匹配 `HERE`，漏了 `here` |

还有一处 `.gitignore` 陷阱，**若未核对暂存清单就推送，仓库会表面上正常、
实际 import 即崩**：

> 规则 `rankings/`（无前导斜杠）会匹配**任意层级**，连要发布的
> `data/rankings/`（54 个文件）一起排掉。已改为 `/rankings/`（仅根目录）。

---

## 6. 完整文件清单与校验和

`<sha256 前 16 位>`，共 167 个文件。

| 文件 | 字节 | sha256[:16] |
|---|---|---|
| `.gitattributes` | 505 | `fa6b4ae130af2dd1` |
| `.gitignore` | 881 | `b78d58f11d318454` |
| `LICENSE` | 1685 | `41e0b840584cd273` |
| `README.md` | 6473 | `2b497b4e37506f48` |
| `__pycache__/coldstart.cpython-312.pyc` | 30370 | `b56c7120b4ef111e` |
| `__pycache__/decide.cpython-312.pyc` | 53447 | `e31399e40efb5c06` |
| `__pycache__/model_v2.cpython-312.pyc` | 93756 | `04fbc2cbd4c0ba9a` |
| `__pycache__/pool_experiment.cpython-312.pyc` | 38594 | `517f97c8f28b96ba` |
| `__pycache__/predict.cpython-312.pyc` | 42664 | `c3fc8ca5b46ed525` |
| `__pycache__/scorer.cpython-312.pyc` | 17862 | `3ec3106e95889305` |
| `__pycache__/split.cpython-312.pyc` | 38219 | `8cdb87c0f75189b7` |
| `__pycache__/state_snapshot.cpython-312.pyc` | 21428 | `36c1d31168fa30f2` |
| `__pycache__/veto_model.cpython-312.pyc` | 53342 | `c4e4a939b27e2165` |
| `_gen_manifest.py` | 5597 | `c00c0d65444f6d4e` |
| `coldstart.py` | 20159 | `8ec109d1d3bd64bf` |
| `data/csapi_matches_all.jsonl` | 1218650 | `2569b2cde365540c` |
| `data/csapi_stats_all.jsonl` | 14875768 | `e9f7b1a53a9f9b56` |
| `data/csapi_veto.jsonl` | 656950 | `cf89969ff85620cb` |
| `data/decide_results.json` | 1836 | `c4b7b803ac4150b5` |
| `data/groundtruth/fissure_2026-09-08_results.json` | 3880 | `be600c8a78c61646` |
| `data/groundtruth/fissure_2026-09-09_results.json` | 3882 | `9f9b2faa1ffb1919` |
| `data/groundtruth/fissure_2026-09-10_results.json` | 3886 | `3ecfd0e8edfb00b0` |
| `data/groundtruth/fissure_2026-09-11_results.json` | 963 | `1d0e7445e531f5c0` |
| `data/groundtruth/fissure_2026-09-12_results.json` | 961 | `2d0a3d60903af137` |
| `data/groundtruth/fissure_2026-09-13_results.json` | 538 | `3d00c651fc46a45c` |
| `data/groundtruth/fissure_2026-09-17_results.json` | 1470 | `1286bf08a090ec9a` |
| `data/groundtruth/fissure_2026-09-18_results.json` | 1462 | `523071fff8595678` |
| `data/groundtruth/fissure_2026-09-19_results.json` | 2415 | `2009bbbf80661d6b` |
| `data/groundtruth/fissure_2026-09-20-bo5_results.json` | 548 | `9fa5ede922dc4589` |
| `data/groundtruth/fissure_2026-09-20_results.json` | 1491 | `889b5314aca15900` |
| `data/groundtruth/fissure_cold1_results.json` | 3892 | `71df37c1719a92ee` |
| `data/groundtruth/fissure_cold2_results.json` | 9411 | `0c5b405ad22006e6` |
| `data/groundtruth/fissure_s3_results.json` | 3464 | `c8b7a1ec3feb3ddb` |
| `data/groundtruth/leaderboard_ledger.jsonl` | 28169 | `cc6761415d4797f0` |
| `data/groundtruth/lgc_predictions.json` | 8234 | `a9ee54baff151dbf` |
| `data/groundtruth/lgc_score.json` | 232 | `f28c011498a9b506` |
| `data/groundtruth/new_records_2026-09.jsonl` | 34947 | `3e1ae09450913079` |
| `data/groundtruth/records_epl_s24.jsonl` | 26434 | `ee975825301fd26c` |
| `data/map_pool_timeline.json` | 11359 | `37bc8346115d1fe8` |
| `data/model_v2_results.json` | 9819 | `9d0678febfbe2b59` |
| `data/model_v2_test_details.json` | 138978 | `84d16d5b3d71f4aa` |
| `data/pool_experiment_results.json` | 5448 | `455f2befc6b5f395` |
| `data/predict_snapshot.meta.json` | 528 | `84d8616ff7909be4` |
| `data/rankings/2025/2025-10-06.json` | 14181 | `dd36e964dbbb38ca` |
| `data/rankings/2025/2025-10-13.json` | 14064 | `17d1e444d1b5d323` |
| `data/rankings/2025/2025-10-20.json` | 14124 | `a057b02ffac0ead1` |
| `data/rankings/2025/2025-10-27.json` | 13898 | `930b72e9b3cdd6ed` |
| `data/rankings/2025/2025-11-03.json` | 14077 | `f43511614e70c058` |
| `data/rankings/2025/2025-11-10.json` | 14500 | `36861f259d89bec6` |
| `data/rankings/2025/2025-11-17.json` | 14803 | `1642f068cb8c2c9a` |
| `data/rankings/2025/2025-11-24.json` | 14642 | `a46210c92527e9ff` |
| `data/rankings/2025/2025-12-01.json` | 13913 | `77a04d91b2684f54` |
| `data/rankings/2025/2025-12-08.json` | 13584 | `395ae869092ad9ff` |
| `data/rankings/2025/2025-12-15.json` | 13467 | `b1616de6ec91f07f` |
| `data/rankings/2025/2025-12-22.json` | 14353 | `e8cf57b7680066ce` |
| `data/rankings/2025/2025-12-29.json` | 13785 | `412c72773d828ba7` |
| `data/rankings/2026/2026-01-05.json` | 14055 | `02e0ecaead21f76f` |
| `data/rankings/2026/2026-01-12.json` | 13768 | `fd5c96cf13a40c5b` |
| `data/rankings/2026/2026-01-19.json` | 13969 | `57e90098b62d94a5` |
| `data/rankings/2026/2026-01-26.json` | 13676 | `58332d16eb42112d` |
| `data/rankings/2026/2026-02-02.json` | 13425 | `7c69e846231511a5` |
| `data/rankings/2026/2026-02-09.json` | 13403 | `b14a493e6978cdeb` |
| `data/rankings/2026/2026-02-16.json` | 13562 | `845ad9eeb2a8ee22` |
| `data/rankings/2026/2026-02-23.json` | 13964 | `9624199cddb70e6b` |
| `data/rankings/2026/2026-03-02.json` | 14153 | `b2f2a3968c2d61f9` |
| `data/rankings/2026/2026-03-09.json` | 13723 | `faf1f25ed99c9a35` |
| `data/rankings/2026/2026-03-16.json` | 13939 | `569f65379bb36bc1` |
| `data/rankings/2026/2026-03-23.json` | 14538 | `1d4d40c89430c1c3` |
| `data/rankings/2026/2026-03-30.json` | 15527 | `8a908cf4da7d9bac` |
| `data/rankings/2026/2026-04-06.json` | 15556 | `d57dd6bd56f9743f` |
| `data/rankings/2026/2026-04-13.json` | 15326 | `862914c02c78810c` |
| `data/rankings/2026/2026-04-20.json` | 15165 | `9c8df31e2946362c` |
| `data/rankings/2026/2026-04-27.json` | 15281 | `41d2fd8be2fdb62e` |
| `data/rankings/2026/2026-05-04.json` | 14941 | `f328718b15b0e34a` |
| `data/rankings/2026/2026-05-11.json` | 14206 | `50b2741ae8b1f05b` |
| `data/rankings/2026/2026-05-18.json` | 13988 | `da498778af593539` |
| `data/rankings/2026/2026-05-25.json` | 13609 | `1ab7db19d089ee9c` |
| `data/rankings/2026/2026-06-01.json` | 13308 | `2704051ec7c195d2` |
| `data/rankings/2026/2026-06-08.json` | 13381 | `ccf4fa3cabaa0859` |
| `data/rankings/2026/2026-06-15.json` | 13582 | `2df20b171bd507ad` |
| `data/rankings/2026/2026-06-22.json` | 13286 | `5a0e148301412a63` |
| `data/rankings/2026/2026-06-29.json` | 12957 | `d297a144c56e663d` |
| `data/rankings/2026/2026-07-06.json` | 12801 | `3b9ce1b330044a01` |
| `data/rankings/2026/2026-07-13.json` | 12221 | `ece4c44a1115e5ab` |
| `data/rankings/2026/2026-07-20.json` | 12551 | `9de55f10929d7048` |
| `data/rankings/2026/2026-07-27.json` | 12005 | `e4829189d3b91e1c` |
| `data/rankings/2026/2026-08-03.json` | 12092 | `6947c72a1a1fd2f6` |
| `data/rankings/2026/2026-08-10.json` | 11724 | `9b5f797abe1fbcaa` |
| `data/rankings/2026/2026-08-17.json` | 11756 | `2f8c436cbb8ed0d3` |
| `data/rankings/2026/2026-08-24.json` | 11670 | `553c424d6b2cdd2e` |
| `data/rankings/2026/2026-08-31.json` | 12422 | `71c8d15751d2c56d` |
| `data/rankings/2026/2026-09-07.json` | 12535 | `a8e32bb50e6699c6` |
| `data/rankings/2026/2026-09-14.json` | 13172 | `15585c58f66fcfc2` |
| `data/rankings/2026/2026-09-21.json` | 13501 | `e547b93033c6ddda` |
| `data/rankings/2026/2026-09-28.json` | 13584 | `b1015ee3a6c2edaa` |
| `data/rankings/2026/2026-10-05.json` | 13520 | `56ed40f5997b5243` |
| `data/rankings/latest.json` | 13516 | `991f723acd17510b` |
| `data/split_manifest.json` | 1950 | `d9d94ff471776f8c` |
| `data/state_2026-10.pkl` | 435681 | `0e10b7c2593029f2` |
| `data/truth_maps_B.jsonl` | 1540676 | `434a156e4480ac9b` |
| `data/veto_model_results.json` | 2414 | `001835aa0d4e2b4f` |
| `data/vrs_rank_2026-10-13.json` | 13795 | `d35d4cb024efc74c` |
| `decide.py` | 38744 | `50f40d9212834ea7` |
| `experiments/diag_map_coef.py` | 3116 | `10a8b35dcc35a2ba` |
| `experiments/diag_mvp_ceiling.py` | 2948 | `06ef1c8a1556af9d` |
| `experiments/diag_score_ratio.py` | 4690 | `70435744d35e5c60` |
| `experiments/eval_tier.py` | 4861 | `07b82adb047d8c5a` |
| `experiments/eval_update_holdout.py` | 5757 | `93b38c8bebacbffb` |
| `experiments/exp_item_ceiling.py` | 4169 | `a3b4a6d9d2f27f32` |
| `experiments/exp_mapelo.py` | 8068 | `44298457c7ea0fe7` |
| `experiments/exp_mapelo2.py` | 5127 | `037cdc5ec8773bea` |
| `experiments/exp_mapinfo.py` | 5947 | `ed23b9208bfd362d` |
| `experiments/exp_maps.py` | 5694 | `5e6d292460d48dfe` |
| `experiments/exp_maps_ceiling.py` | 5279 | `dcb987b25b993516` |
| `experiments/exp_maps_confound.py` | 5214 | `c2cdc98b60900e79` |
| `experiments/exp_maps_entropy.py` | 5961 | `59e8f72659a24580` |
| `experiments/exp_maps_richness.py` | 5365 | `f087ecf6b197d001` |
| `experiments/exp_mvp_confound.py` | 4315 | `4de435b8d0849b76` |
| `experiments/exp_mvp_map.py` | 3470 | `77535d7c119485c5` |
| `experiments/exp_mvp_prior.py` | 5995 | `ba9bef826e9e0249` |
| `experiments/exp_mvp_scoring.py` | 8065 | `845c75f97e2f1f6d` |
| `experiments/exp_mvp_tiebreak.py` | 9722 | `f5354bb505187fec` |
| `experiments/exp_nnue.py` | 10664 | `f54f5671ac81b9c3` |
| `experiments/exp_nnue_ablation.py` | 3744 | `aa5f658f63bd48ef` |
| `experiments/exp_nnue_add.py` | 7196 | `5458e036b73cc3f3` |
| `experiments/exp_nnue_diag.py` | 6022 | `6dc99bd51d8f30d4` |
| `experiments/exp_oppweak.py` | 4455 | `17a9a581951d0b8a` |
| `experiments/exp_roster.py` | 10403 | `b038a1c825d1102a` |
| `experiments/exp_score_bo.py` | 4121 | `7b6ab63b4bb59fd5` |
| `experiments/exp_score_diag.py` | 6095 | `a0f185ef69e1ce99` |
| `experiments/exp_score_product.py` | 7221 | `50cfcd2dbe1a6056` |
| `experiments/exp_score_threshold.py` | 4341 | `cde02ecfca2db359` |
| `experiments/exp_tianji.py` | 3061 | `58775860ee7790a9` |
| `experiments/exp_tianji_mvp.py` | 6550 | `86460d10cc13f128` |
| `experiments/exp_tune.py` | 6064 | `363b86a37c1ec51f` |
| `experiments/exp_veto_order.py` | 6794 | `0be40c7cf5e4867b` |
| `experiments/exp_walkforward_winner.py` | 10342 | `b568b2d84d180fee` |
| `experiments/exp_win_ens.py` | 2911 | `c64a047e49ef98c6` |
| `experiments/exp_win_sel.py` | 1185 | `c5375fb107e50611` |
| `experiments/exp_winner_rule.py` | 5292 | `66c3ecb18f4f968b` |
| `experiments/export_new_records.py` | 6631 | `08ce6328209564c3` |
| `experiments/export_records.py` | 5921 | `94d8beb2a9b146d5` |
| `experiments/predict_fissure.py` | 7752 | `1a6e814051281f84` |
| `experiments/predict_lgc.py` | 6015 | `ebbeea3409c2705a` |
| `experiments/record_schema.py` | 9293 | `6fa5b0739c135a53` |
| `experiments/replay_lgc_mvp.py` | 5218 | `78c99484ff295e94` |
| `experiments/replay_mvp_rule.py` | 5121 | `59b09a486317cf4b` |
| `experiments/replay_rank_winner.py` | 5121 | `e9fa50fbf6dcb8f5` |
| `experiments/score_batch.py` | 9018 | `c5b9a946d0297719` |
| `experiments/score_fissure.py` | 12920 | `81589f1abc45ef5a` |
| `experiments/score_lgc.py` | 8495 | `2a9b3c599e8208de` |
| `experiments/verify_records.py` | 2596 | `10bb94b9b8663448` |
| `experiments/verify_snapshot.py` | 4386 | `2aec641585ab8808` |
| `ingest_new.py` | 7580 | `79160ad855861389` |
| `map_pools.py` | 36497 | `babbae4222addec4` |
| `model_v2.py` | 67220 | `217e8d5637dc71f1` |
| `pool_experiment.py` | 25960 | `98ed568befe4a002` |
| `predict.py` | 33800 | `1bba4c142aaa5d17` |
| `predict_gui.py` | 28464 | `43fde639da39c0a7` |
| `predict_page.html` | 23622 | `b013f5dd3105f96f` |
| `pyproject.toml` | 1145 | `667e2594f1997435` |
| `record_schema.py` | 9293 | `6fa5b0739c135a53` |
| `scorer.py` | 15523 | `a5eb88514d1eeea6` |
| `split.py` | 26003 | `f1fd6b2be045aaae` |
| `state_snapshot.py` | 15790 | `9b333c753d70f510` |
| `veto_model.py` | 38196 | `539fd320f717207f` |
| `veto_rec_build.py` | 3013 | `71d44dc4ab91e11d` |

---

## 7. 复现构建

本清单由 `_build_repo.py` + `_verify_repo.py` 生成，这两者**未入库**（属构建工具）。
验收标准是 `REPO_OK`：`decide.py selftest` 25 项 + `predict.py selftest` 10 项
+ 仓库内独立跑通真实预测 + `scorer` 可导入 + 黑名单自检全过。
