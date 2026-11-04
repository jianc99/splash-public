# Splash 1.6.0 交接文档

> 状态截止：2026-11-03 约 19:55。
> 对应分支：本 fork 的 `release/1.6.0-final` 和 `release/1.6.0-cleanup`（`5fc74205`）。`release/1.6.0-final` 的代码头是 `3874a2f1`；本文档作为一个只含文档的提交放在它上面，所以分支尖端是那个文档提交，下文说"最终头"时都指代码头 `3874a2f1`。
>
> 本文依据两类来源写成：
> - git 历史，可以在仓库里复核；
> - 开发期间的内部工作笔记：设计稿、评审记录、状态日志、测量与调查记录。**这些笔记不在仓库里。**
>
> 两类来源都无法确认的内容标为 **未核实**。数字全部取自这两类来源。标"（计算）"的百分比由来源里的原始数字算出。

**阅读约定**

- **提交和代码位置。** 提交哈希取前 8 位。代码位置写作 `路径:行号`，未注明时指 `3874a2f1`。为了简短，常用文件只写文件名：`Engine.{hpp,cpp}`、`EngineMemory.cpp`、`EngineAllocate.hpp`、`Cache.{hpp,cpp}`、`KvPool.*`、`KvTier.hpp`、`Types.hpp`、`MemoryPlan.hpp`、`MemoryLedger.*`、`MemoryGovernor.*`、`MemoryControl.cpp`、`Scheduler.*`、`CostModel.*`、`Status.*`、`Protocol.hpp`、`FdTransport.cpp`、`NativeRuntime.cpp`、`Bootstrap.mm`、`RuntimeResources.mm`、`StreamedMemoryPlan.*`、`WriteBehind.*` 在 `runtime/engine/`；`Runtime.cpp`、`RuntimeArenas.hpp`、`Model.hpp`、`CommandCost.hpp`、`ElasticPool.hpp`、`ModelPools.hpp`、`ExpertStore.*`、`KvBlockCache.*`、`KvHistoryStream.*`、`SlotRecycler.*`、`QwenState.*` 在 `runtime/model/`；`KernelSelection.*`、`KernelCalibration.hpp`、`LinearGguf.cpp` 在 `runtime/ops/`；`*_test.cpp`、`*_test.mm`、`TestRegistry.hpp`、`RegrowthWatch.hpp` 在 `dev/tests/engine/`。
- **机器。** 只用以下代号：M6（24 GB）、M5 Pro（64 GB）、Studio（36 GB）、mini-M4（16 GB，Apple9 GPU，10 核）。"Studio 18G"指 Studio 加 `--max-memory 18G`；"ample"指不限内存。
- **树名**，与 `dev/benchmarks/serving/matrix-1.6.0.json` 一致：

  | 树名 | 指什么 |
  |---|---|
  | `r150` | 1.5.0 |
  | `main` | `031a25a6` |
  | `pass` | `0bd50301` |
  | `ntie` | `87382087`，near-ties |
  | `intp` | `1da46c67`，intpart |
  | `cand` | 1.6.0 候选 |
  | `floorcap` | cand 加 `--expert-memory` = 存储上限，即专家存储永不收缩 |

  表格里还会出现几个只在终测前录制参照数据时用过的树名：`integration`（11-01 的集成树 `trial/final` = `ec0de9c4`，即 intpart 的内容加内存计划步骤 2-4 和几个 PR；跑名里记作 mFN/pFN/sFN 等）、`R4`（integration 加"竞争的流式 lane 最少 4096 行"，plan 1，`a428081f`）、`KR`（integration 加"没有 lane spill 时不给读者预留"，plan 2，`bb668ed7`）、`passes-final`（多 lane 分层 pass 分支在 `0bd50301` 之前的一轮；M6 上录制的 pFa/pFb 就是它，构建是 `7e7b1a21`，见 §4.2.1）、`spillmem`（memres + spill-wait）。跑名形如 `sMNb`：第一个字母是机器（m = M6、s = Studio、p = M5 Pro），中间是树，最后是顺序 a/b。"共同 tag"指比较时只取几棵树都完成了的那些请求（按 tag 对齐），这样 TTFT 和延迟的均值才可比；不同比较的共同 tag 集合不一样，数字不能跨表直接比。
- **模型。**

  | 简称 | 型号 | 量化（按机器） |
  |---|---|---|
  | FN | Qwen3.8-Flash-Next（GGUF） | M6、Studio：Q2_0；M5 Pro：IQ3_S |
  | 27B | Qwen3.8-27B，稠密 | M6：UD-IQ3_XXS；Studio：UD-Q4_K_M |
  | 35B | Qwen3.6-35B-A3B | M6：UD-Q2_K_XL；M5 Pro、mini：UD-Q4_K_M |
  | Flint | Flint 40B-A2.5B MoE | UD-Q4_K_XL |
  | 122B | Ornith-1.5-122B（Qwen3.5-122B-A10B），没有 HiSparse 的流式 MoE | M5 Pro：GGUF UD-Q4_K_XL，74.2 GiB（只在 M5 Pro 上流式跑） |

  所有模型都带各自的 DFlash2 草稿。前四个的版本固定在 `matrix-1.6.0.json` 里；122B 的单元在 M5 Pro 的终测批次里（job 40-47）。
- **时间。** 一律为开发记录里使用的本地时间。11-01 凌晨 02:00 钟点回拨了一小时（夏令时结束），跨这一刻算时长时要注意；git 时间戳带时区偏移，可以对照。
- **失效引用。** 提交信息和代码注释引用了 `amend/M.md`、`amend/S.md`、`DESIGN §…`、`AMENDMENTS`、`DATA.md`、"owner 11-03 hh:mm"等。这些都是内部笔记，不在仓库里，链接已失效。本文已把其中需要的结论写进来；代码注释里常见的修订编号见附录 C 末尾的对照表。
- **内部编号。** 本文沿用了内部笔记的一些编号：A + 数字（内部尝试记录里的条目，例如 A53 = spill-wait 第一版）、D1-D13（推迟项，§8.4）、N1/T6/T7/F1 等（smoke 轮次的发现，附录 C）。同一个编号在设计修订里可能指另一件事，文中会注明。

---

## 0. 一页摘要

### 现状

- **当前发布版仍是 1.5.0。** tag `6838f255` 是注释 tag 对象，指向 commit `cabbad27`（10-28）。
- **上游 main 停在 `031a25a6`，没有发布。** 它是 1.5.0 + 29 个提交，其中有 12 个 PR，包括 #640 分块专家存储和 #645 弹性专家存储。
- **1.6.0 的最终代码在 `release/1.6.0-final`，代码头 `3874a2f1`**（分支尖端是在它上面加本文档的文档提交）。
  - 它 = 终测头 `a7dac91a` + 两个晚间修复 `e1310a4d`、`3874a2f1`。
  - 相对 main 有 434 个提交。
- **清理分支在 `release/1.6.0-cleanup` = `5fc74205`。** 它是 `a7dac91a` 上 7 个只做删除的提交，**没有**和两个修复合并。
- **上游没有合入任何 1.6.0 内容，也没有为它开 PR。**

### 能不能发：按现状不能

1. **M6 FN 压力单元：修复后追平了 1.5.0，但没有干净通过。**
   - `a7dac91a` 上默认（value）配置 G1 = 0.936、G2 latency = 1.126，都没过。
   - 修复 `e1310a4d` 后的确认跑 4 跑都跑完了（A3、B3、B4、A4；B3 因系统进程 CPU 占用被判 void，analyze 不计它）。19:55 的 `matrix.py analyze --build cand=src-784dbfc9`（即 `3874a2f1`）：
     - G1 work 对 1.5.0 = 1.004 [0.980, 1.029]，**PASS**；
     - G2 latency = 1.020 [0.987, 1.055]，INCONCLUSIVE；
     - G2 按请求 decode = 0.705，FAIL。这是 §4.1 说的误导性指标；按总量 decode 22.2-23.3 tok/s，1.5.0 是 22.7；
     - G3 对 main = 0.938，INCONCLUSIVE；G4、G8、G9 PASS；
     - 顺序 B 的 TTFT 包络在 2 个 tag 上 FAIL（61.97 对 61.18 s，109.07 对 107.36 s）；A4 的未记账内存增长 693 MB，超过 268 MB 的报告线（只报告，见 §8.1）。
2. **3874a2f1 上缺验证。**
   - 没跑完整的 Metal 套件。
   - M5 Pro 没有部署 `3874a2f1`。
   - 正确性批次全部搁置：test-real、http-real、pressure-real、故障注入、swift test、122B。
   - Studio 第 1、2 阶段的正确性 job（80-96）在 `a7dac91a` 上跑完了，但结果没有读出、没有记录（§1）。
3. **Studio 两项失败。**
   - Studio 18G 的 SSD 门 `disk_read_per_work` = 1.452，FAIL，原因未查。
   - Studio ample 有一次存储震荡，store line FAIL。
4. **托管 CI 在这个头上应该是红的。** CI 的第一步 `make check-source check-native-cpu` 里，`check-native-cpu` → `test-engine-cpu` 不带过滤器跑整个 engine-sim（`dev/native.mk:650`），会在两个已接受的已知失败 SIM-turns (b) 和 SIM-warning-M6 上失败，后面的步骤都到不了；即使到了，`test-sanitizers` 里带过滤器的 engine-sim（`native.mk:927-928`）也包含这两项。这是读 `ci.yml` 和 makefile 推断的，没有实际看到 CI 运行（§8.1）。
5. **单请求还有残留问题。**
   - Studio 上 300 token 的 TTFT 仍比 pass 树慢 3.1 %，约 30 ms；运行间噪声为 0.4-1.8 %。
   - Flint 的 G7 因基线测量工具出错，没法测。
6. **更根本的一条：** owner 的结论是，两个在线估计器应当先删掉再发布（见下文）。

### 终测证明有用的（owner 结论 1）

真正的收益来自两件简单的事。

- **分层 pass（layer passes）。**
  - Flash-Next 的预填充按层切分，不按行切，每条命令保持约 7,700 行。这样每层专家在一个大批次里只读一次。
  - 每预填充行读取的专家字节：M5 Pro 从 14.9 降到 5.7 MB（−62 %）；Studio 从 5.1 降到 3.1 MB（−39 %，owner 记为 −40 %）。
  - 这些数据只来自 M5 Pro 和 Studio；24 GB 的 M6 上 pass 的收益没有单独证明（§3.6）。
- **内存账本（ledger）的固定规则。**
  - 规则有五条：所有分配走 claim；decode 排在 prefill 前面；落地的内存留给等待者；只有落地内存足以覆盖需求时才等；有请求在等时，池不向空闲空间增长。
  - main 和 1.5.0 在 Studio、M5 Pro 上都有 110-430 s 的 decode 冻结。1.6.0 在同样单元的 14 次跑里没有任何超过 2.5 s 的客户端 token 间隔。

### 终测证明没用、反而让系统难以控制的（owner 结论 2）

两个在线估计器：

- **调度器的学习型代价模型 `CostModel`。** 有 7 个学习速率，预填充单元的上限是 max(500 ms, u/0.1)。
- **内存"价值模型"。** 它按"每字节值多少秒引擎工作"给内存定价，包括 victim 定价、增长定价和 A30 回长核算。

两者都没有测出收益。价值模型直接导致了 M6 的回退，还一修再修：N1（`a7dac91a`）之后又要 `e1310a4d`。大部分复杂度和大部分代码、测试膨胀都在这两个估计器上。

### 仍然需要的（owner 结论 3）

- **弹性专家存储仍然需要。** 主要用在小内存 Mac 上，尤其是没有 HiSparse 的 MoE 模型。Flash-Next 有 HiSparse，可以改为把 KV 溢出到 SSD。
- **为什么非弹性不行。** 没有弹性，只能二选一：
  - 启动时固定一个保守的存储：不排队、最大上下文更长，但永远更慢；
  - 固定一个满的存储：快，但要排队，或者上下文很短。
- **Flash-Next 可以用 floor = cap。** M6 数据：floor = cap 吞吐比 1.5.0 高约 2 %，TTFT 仍优于 1.5.0。

### 代码膨胀（owner 结论 4）

- main → `a7dac91a` 净增约 53.7k 行：
  - 产品代码净增约 13.2k，其中 `runtime/engine` 增长 32 %；
  - `dev` 净增约 39.7k。
- 测试钉住了实现细节，比如具体价格、拿走的是哪个 chunk。每改一条规则都要重写大量测试。
- 清理分支只删了死钩子，没有碰估计器代码。

### 过程教训（owner 结论 5）

- 估计一再滑坡：终测比原计划晚了约 27 小时，最后也没有发布。
- 一个托管在 Studio 上的代理 worker 约 9 小时内花掉约 $1,474（API 等价），只完成 11 个任务。
- 遇到问题反复打补丁，没有退一步重看设计。
- 简单规则的基线测得太晚。

### 建议的下一步（详见 §12）

1. 收尾测量：
   - M6 修复 1 的 analyze 已出（§4.7），G2 latency 和 G3 仍不确定，需要更多跑或在新头上重测；
   - 在 `3874a2f1` 上补跑完整 Metal 套件；
   - 部署 M5 Pro，跑正确性批次；读出 Studio 第 1、2 阶段 job 的结果。
2. 用固定规则加测量反馈替换两个估计器：
   - **调度器：**
     - 没人等待 → 整条命令跑完；
     - 有人等待且模型支持 pass → 保持行数，按层切，每个 pass 的层数由上一个 pass 的实测时间决定；
     - 模型不支持 pass → 像 1.5.0 那样切行（500 ms 切片）。
   - **内存：**
     - 固定的让出顺序；
     - 存储只在请求放不下时缩一个 chunk，在一段时间无人等待后回长一个 chunk；
     - 每个模型一个 floor；
     - 不到 Critical 的 macOS 压力不取存储。
   - **删除：** `CostModel`、`MemoryValue`、账本定价、A30 核算，以及钉价格的测试。
3. 合并清理分支，把测试精简成行为测试，让 CI 变绿。
4. 按每模型设置 floor：Flash-Next 按 M6 数据设为 floor = cap；35B 等没有 HiSparse 的模型保留弹性。
5. 按 §12.6 的单元在四台机器上重跑门，然后再谈发布。

---

## 1. 版本与分支

| 名称 | 引用 | 日期 | 内容 | 门禁状态 |
|---|---|---|---|---|
| 1.5.0 | tag `6838f255` → commit `cabbad27` | 10-28 03:52 | PR #612 的合并 | 当前发布版 |
| main | `031a25a6`（上游 origin/main） | 10-31 02:18 | 1.5.0 + 29 个提交：17 个普通提交 + 12 个 PR 合并（§6.1）。含 #640 和 #645 | 未发布；终测参照树 `main` |
| pass 树 | `0bd50301` | 10-31 / 11-01 | main + 75。包含当时各待合并分支的**较早轮次**：回收梯 #650、#644 的 KV 驻留、spill-wait、内存计划步骤 2-4、near-ties、A1-C 清理、prefill splits、#648/#649/#651，以及多 lane 分层 pass。这些分支的后续轮次（memres `75d48ac0`、spill-wait `6ea9f9e0`、#648 `a704ad87`、#649 `0e9289fb`、#651 `d3321f73`、#644-obs `6249a2b5`）和有毒叶子修复都不在里面。**没有账本** | G7 和修复测试的 ABBA 基线 |
| fork 基点 | `6fd54f61` | 11-01 21:34 | pass 树 + C0 合约 + `b0/catchup`（把 #648/#649/#651 的后续轮次 `2cc42cf1`、`1ac3c5dd`、`3ddb6985` 和有毒叶子修复 `9077b419` 补进来）+ 固定 cache-property 种子 9454 | — |
| 终测头 | `a7dac91a` | 11-03 11:55 | main + 432 | 四台机器终测（§4.3） |
| 最终头 | `3874a2f1`，`release/1.6.0-final` 的代码头（其上只有本文档的提交） | 11-03 18:03 | `a7dac91a` + `e1310a4d` + `3874a2f1`；main + 434（408 个普通提交 + 26 个合并） | 见下表 |
| 清理 | `5fc74205`，即 `release/1.6.0-cleanup` | 11-03 14:39 | `a7dac91a` + 7 个只做删除的提交；24 个文件，+160/−484 | M6 上部分门通过；没有和修复合并 |

**几点说明**

- **历史形状。** 历史基本是一条 cherry-pick 出来的直线：first-parent 链 319 个提交。first-parent 链上有三个集成点含真合并：
  - `trial/final`，11-01 03:48-03:49（10 个合并）；
  - `rel/1.6.0-base` 合入 `b0/catchup`（`ec3996e7`，11-01 21:32）；
  - `int/1.6.0`，11-03 03:24-03:33（11 个合并）。

  另外 4 个合并在侧分支上，经上面的合并进入历史：`wip/complete-tools`（11-02 20:05、20:40）、`wip/smoke-a`（22:59）、`wip/s-turnaround`（11-03 00:02）。合计 26 个合并。

  各条轨道分支（`t/*`、`earlyfix/*`、`m3afix/*` 等）都 rebase 到这条线上，大多数轨道尖端不是最终头的祖先。这些分支名只存在于本地，提交信息里会出现。
- **清理分支的行数。** 有笔记写的是"+152/−483"，那是前 6 个提交。第 7 个提交 `5fc74205` 又加了 +8/−1。
- **修复的行数。** 两个晚间修复合计：产品代码 +39/−18，测试 +229/−120。
  - `e1310a4d`：runtime +19/−11，测试 +171/−120，外加 `docs/internals.md` +3/−1。
  - `3874a2f1`：runtime +20/−7，测试 +58/−0。

  有说法称"修复 1 改了 39 行产品代码"，实际上 39 行是两个修复的合计。
- **1.6.0 线上带着、但上游没合入的 PR 内容：**
  - #644 相关工作（`e805f601`、`6776d9cd`、`ec0de9c4`）；
  - #648（含后续轮次 `2cc42cf1`）；
  - #649（`1ac3c5dd`）；
  - #651（`3ddb6985`）；
  - #650 的回收梯（`a5570f52`）。它先被带入，后来被 `7284fdb4` 删除。

**各头已跑的门**

| 检查 | `a7dac91a` | `3874a2f1` | `5fc74205`（在 `a7dac91a` 上） |
|---|---|---|---|
| `engine`（CPU） | 388/0 | 389/0（多了修复 2 的测试） | CPU 列表只有已知两项失败 |
| `memory-ledger` | 113/0 | 115/0 | 同上 |
| `native-engine-loop` | 47/0 | 47/0 | 同上 |
| `ragged-scheduler` / `cost-model` | 89/0 / 18/0 | 89/0 / 18/0 | 同上 |
| `engine-sim` | 8 过 / 2 败（已知两项） | 8 过 / 2 败（已知两项） | 已知两项 |
| `pressure-serving`（Metal） | 2/0 | 2/0 | 未单独记录 |
| ASan/TSan（`test-sanitizers`） | mini：只有已知两项，0 个 sanitizer 报告。Studio job 82 也跑了，结果没有记录 | **未跑** | M6：只有已知两项 |
| 完整 Metal 套件 | M5 的 `check-native-metal`（job 26）在 job 27 开始前就该跑完了，结果没有读出，**未核实** | **未跑** | M6：`test-engine-metal`、`test-sanitizers-metal`、`check-native-build` 通过 |
| `architecture-check` | 只开 R1、R2；Studio job 80 跑过，结果没有记录 | 只开 R1、R2（R3 未开）；未记录 | 通过，R3 已开 |
| `check-python` | Studio job 80 跑过，结果没有记录 | 未记录 | 1 个失败，已由 `5fc74205` 修好（`test_architecture` 19/19） |
| server、tools 的 Python 测试 | 未单独记录（job 80 的 check-python 包含它们）。最近一次有记录的是 `5d2d5348`：server 1035 OK；tools 除 `test_installer_restarts` 外全过 | 未记录 | 见上一行 |
| App（`swift test`） | Studio job 81 跑过，结果没有记录；M5 的 job 39 搁置 | 未跑 | 未跑 |
| `test-real` 真模型 oracle | M5 job 27（FN）17:30 仍在跑，结果**未核实** | 未跑 | 未跑 |
| Studio 第 2 阶段（90 恢复基准、92 PDF 渲染、95/96 驻留与 A10 读数） | 跑完了（Studio 队列 16:50 结束），结果没有记录；91 无存储 spill 跑不了（§4.3.5） | 未跑 | 未跑 |
| 服务矩阵 | 四台机器全量终测（§4） | 只有修复测试：Studio、mini 的 serve1 ABBA；M6 的 serve1 和 FN churn 4 跑（§4.7） | 未跑 |

工具测试 `test_installer_restarts` 有 4 个失败、2 个错误。这是在 `46f909fb` 就存在的旧问题，之后没有重跑。

"结果没有记录"的 job，输出可能还在各台测试机的作业日志里，没有读出来；接手人要向 owner 要这些日志，或者重跑。

---

## 2. 1.5.0 的问题与 1.6.0 的目标（设计时）

### 2.1 起因

1. **#645 合入 main（10-31 02:18）后，专家存储只缩不长。**
   - 场景：Studio `--max-memory 18G`，3 个并发 Flash-Next 请求。
   - 按请求的 decode 中位数 32.9 tok/s，#645 之前是 48.9 tok/s（这是按请求的指标，不是吞吐，见 §4.1）。同一组实验里 floor = cap 恢复到 45.6，说明损失来自存储收缩——这是结论（3）最早的线索。
   - 这组实验在 Studio 上用 `--max-memory` 模拟小内存，macOS 自己的 warning 是 0 %，而真 M6 上是另一回事：`--max-memory` 模拟不了 host 压力（§10.1）。
   - 原因：
     - 存储在 KV 和缓存压力下收缩；
     - 回长只在"没人等待、Normal 压力、上次收缩 30 s 后、留 2 GiB host 余量"时进行；
     - 固定的预算不知道存储已经缩了。
     - 根因追踪（A21）：收缩时存储排在缓存**前面**被让出，回长时却排在缓存**后面**，内存只会单向流走。
2. **10-31 owner 叫停点修补，批准了一个 4 步内存重设计：**
   1. 统一的回收梯，后来成为 PR #650；
   2. 把专家上限当作预留；
   3. 按实时空间准入；
   4. 提前 SSD 拷贝，以及约 256 MiB 的 PLE 行缓存。

   同时进行的还有 10-31 那轮 15 模块评审（62 个发现，确认 51 个，其中 8 个 P1）的修复组（A1、A2、B1、B2、C），以及多 lane 预填充 pass。这一轮和下面 11-01 的全仓评审 REPORT 是两次不同的评审。
3. **11-01 重跑 1.5.0 基线，发现 main 在内存压力下比 1.5.0 差。**
   - M6 差 6-8 %，Studio 18G 差 4-11 %。
   - 待合并的补丁链也没有追回：M6 的集成树差约 15 %。
   - 1.5.0 吞吐和延迟最好，TTFT 更差。
4. **全仓评审 REPORT（11-01）。**
   - 范围：约 280k 行，分 56 个单元，只读。40 个关键结论由独立代理复核：4 个完全成立，36 个核心成立、细节有修正，0 个被推翻。
   - 问题集中在策略层：调度器 C−，内存管理 C−/D，`Engine.cpp` D，Cache 和 `Runtime::Impl` 是上帝类。底层（KvPool、KV tier、Metal backend、ABI、kernel、ANE）评 B。
   - 严重特例的数量：1.5.0 有 59 个，main 新增 8 个，待合并分支新增 25 个。
   - 指出的根因：
     - 没有代价模型；
     - 没有统一的内存模型；
     - 模型边界暴露机制而不是能力；
     - 验证看不见这些问题；
     - App 用正则解析引擎日志来判断启动、失败和请求；重启策略在 Ready 后 60 s 内失败 3 次就永久停止。
   - **复核的修正：** REPORT 的结论由一份复核文档逐条核验（编号 T1-1、T1-2……）。其中 T1-2 指出，被评审的树（1.5.0 加待合并分支）里调度器**已经有**一个仿射代价估计（固定 + 每行），问题在于各条规则怎么用它，而不是缺一个代价模型。T1-2 自己给的建议是"处处只用一个代价模型和一条边际合批规则"。设计据此新建了 7 速率的学习模型，而没有在现有的仿射估计上统一规则——这是本文的解读。
5. **11-01 的决定：**
   - 15:46-16:05：不从 main 发布，1.5.0 保持为发布版；1.6.0 是"在现有框架内的重构，不是重写"，kernel 不动，只统一选择规则。
   - 17:58：全范围，不推迟任何东西（设计草稿里标为"1.7 再定""可选""后续"的条目都要在 1.6.0 里做完，或者带着测量理由放弃）。
   - 18:00 加了两条：
     - 先写完全部代码，期间不跑长测试，省下的时间用来评审；
     - 发布目标定为 11-02 周一晚上。

### 2.2 目标与门（设计稿 DESIGN-1.6.0 §1、§8.3，后经 owner 修改）

| 目标 | 依据数据 | 门 |
|---|---|---|
| 压力下吞吐、延迟不低于 1.5.0 | 1.5.0 在 M6 上 work 1023/995、latency 169/174 s；main 低 6-15 % | G1、G2 |
| 压力下 TTFT 不差于 main | main 在 M6 上 TTFT 92-102 s；Studio 18G 顺序 A 197-205 s，顺序 B 140-141 s | G3 |
| decode 间隔有界，不再出现 110-425 s 冻结 | main 在 Studio A 上 110-112 s；M5 Pro 424.8 s；1.5.0 ample 132.6 s（盲指标，见 §5.1） | G4 |
| 保住 pass 的并发 TTFT 收益（N=4 最差 15.3 → 5.2 s） | passes-final N=4：5152/5247 ms | G5 |
| 内存充足时不差于最好的树 | near-ties 1109/1235，TTFT 148/148 s | G6 |
| 单请求不变 | pass 树 N=1 3065 ms | G7 |
| 健康 | 无重启、无失败请求、无 Metal 失败；`loop.max_tick_ms` < 5000 | G8 |
| 账本自洽 | Σ 账户 = `charged_bytes`；空闲时 owed = 0 | G9 |
| 稠密、驻留 MoE、Flint、mini-M4 并发不回退 | 对比 pass | G10 |

owner 后来的修改：

- **11-01 20:30，5 % 规则。** 压力单元里 work@900 和 decode 最多可以比 1.5.0 低 5 %，前提是别处有明确收益（TTFT、冻结、并发 TTFT、完成时间），每项取舍都要给出数字。单请求速度不能拿来取舍（G7）。这条替换了原来 G1/G2 的 2 % 容忍。
- **11-01 20:45，OQ2：内存受限时 G3 的附加条件。** 18G 上限的单元里，G3（TTFT 对 main）只有在 G2 通过、G4 成立、并且 cand 的 TTFT 不差于"该单元里保持 decode 不停的最好那棵树"（M5 Pro 上是 ntie，Studio 上是 intp）时才算通过，否则在唯一的修复轮里按 FAIL 处理。仓库里的 `docs/validate.md` G3 行写的就是这条。
- **11-03 13:25，放宽 G4。**
  - G4a、G4b（按运行推导出的界：A09 的界 × (1 + e95) + 本次跑的 p50 间隔）改为只报告。
  - 阻塞项改为：G1/G2（5 % 规则）、G3、G7、G8，以及 G4 的原始形式：客户端 token 间隔最大 ≤ 10 s、p99 ≤ 5 s。
  - **仓库里的文字还没跟上。** `docs/validate.md:495` 的 G4 行仍是按运行推导的界 `max(decode_gap_target_ms, u/0.1)` 加 p99 和机器 floor，`matrix.py:1298` 仍按 `gap_bound(..., pass_overhead_share)` 判定。13:25 的放宽只存在于内部笔记里，要写进 `validate.md` 和 `matrix.py`。

### 2.3 设计原则（DESIGN-1.6.0 §2）

1. 模型边界只报告能力和结构量，不报告机制，也不报告猜测的时间。
2. 一个代价模型 + 一条公平规则：7 个学习速率，一个时间常数 G = 500 ms，一个预算 `kPassOverheadShare = 0.1`。
3. 一个内存账本：
   - 真实的预留；
   - 有所有者的在途字节和欠拷贝字节；
   - 让出和增长用同一条"每字节值多少秒引擎工作"规则；
   - 一条 host 规则。
4. 机制保留，策略换主人。每处删除都列出并有测试。
5. 先写合约（C0），再开并行轨道。轨道内只跑分钟级测试，长测试在合并后跑一次。
6. "测量，不要猜"。但 u（每个 pass 的周转时间）等几个量**最终没测**（§8）。
7. 新代码只允许出现在五处：
   - 模型报告的 GPU 固定代价结构；
   - 内存管理里的真实预留和在途、欠拷贝字节；
   - 验证设施（假内存、带 GPU 固定部分的假计时、自动矩阵、新的 `/status` 字段）；
   - Ready 里的能力记录；
   - 解除"一条命令只能整段流式读取历史"的限制。

**这段历史要讲清楚。** 10-31 设计之前，owner 批准的是一个**固定的**回收梯：

> 空闲 → 立即可离开的缓存 → 需要 SSD 拷贝的缓存 → [HiSparse] Resident lane 转 Spill → 专家 chunk → 在用缓存 → 挂起

随后 REPORT 和设计稿用"按价值定价"替换了固定顺序：`escalate` 被删，设计稿里有一节专门论证固定规则不行。终测之后，owner 的结论（2）又回到了固定让出顺序。

### 2.4 轨道与合并顺序

| 轨道 | 负责 |
|---|---|
| C0 | 所有共享头文件、签名、状态字段。没有调用者的函数体抛 `std::logic_error("<track>")` |
| F1 | 测试假件：`FakeMemory`、`FakeElasticPool`、`TimingProfile`、`EngineHarness` |
| K | 设备描述、kernel 校准表、kernel 选择 |
| I-model / I-engine | 模型边界（`commandLimits`、`prefillWork`、`sample()`、`pools()`、带范围的历史流）；`WeightResidency` |
| S | 调度器和 `CostModel` |
| M1 / M2 / M3a / M3b | 账本核心；Cache 作为 victim 来源；引擎接入账本；收尾 |
| C | 服务端和 App（Ready v12、StartFailed、重启策略、隔离、serve record） |
| V-tools / V-sim | 服务基准和 CI；测试迁到 harness，场景模拟 |
| cleanup | 删除遗留钩子，打开 R3 |

合并顺序：

> `0bd50301` → `b0/catchup` → C0 → fork → F1 → K → I-model → I-engine → S → M2 → M1 → M3a → M3b → C → V-sim → cleanup → 终测

设计稿自己的日程是：C0 约 1.5 天，各轨道并行 2-5 天，合并 1-2 天，加起来约 6-10 天工作量。实际排期被压到一晚（"合约今晚、轨道通宵、周一早上终测"）。

### 2.5 设计时的预期和已承认的风险

- **预期**（DESIGN §9）：
  - Studio 18G "大概率达到 1.5.0，余量比草稿说的小"；
  - M6 是"最难的单元"，当时测过的变体全都低于 1.5.0；
  - 内存方面造成冻结的原因由 M 轨道消除；
  - 单请求在噪声内不变，启动多一次预热（流式模型约 2 s）。
- **已承认的风险：**
  - 新假件可能"证明"新的错误信念；
  - 7 个速率在单一尺寸的竞争场景下不可辨识；
  - "价值输入可能让存储缩得比每 GiB 收益允许的更多——这正是 M6 的风险"；
  - 测试会锁死行为变化；
  - 矩阵噪声和机器时间。
- **价值模型的自注。** 设计稿写道，这些标定只固定了公式的形状，测量输入能否在真机上复现每 GiB 收益，要看矩阵。终测给出的答案是：不能（§4.3.1、§5.4）。

### 2.6 有约束力的决定（按时间）

内部笔记里 owner 的决定散在各处。下表按时间收拢，接手人改规则时要知道哪些是 owner 定的。"协调者"指 owner 03:55 之后授权代做决定的会话。

| 时间 | 决定人 | 内容 |
|---|---|---|
| 11-01 15:46-16:05 | owner | 不从 main 发布；1.6.0 是框架内重构；kernel 只统一选择规则 |
| 11-01 17:58 / 18:00 | owner | 全范围，不推迟；代码写完前不跑长测试；目标 11-02 晚发布 |
| 11-01 19:45 | owner | 有 HiSparse 时专家对 KV 按价值决定（OQ1 的前身） |
| 11-01 20:05 | owner | Q5：加 spill decode 项（每个 Spill lane 每条 decode 命令约 9 ms）；Q6：不改 IO 优先级 |
| 11-01 20:15 | owner | Flash-Next 的 floor 由终测里的 floor = cap 对照决定，不加新开关 |
| 11-01 20:30 | owner | 5 % 规则（§2.2），取代 G1/G2 的 2 % |
| 11-01 20:45-20:50 | owner | OQ1：断言价值模型的结果，前提是模型准确；OQ2：G3 附加条件（§2.2）；OQ3：每台机器约 5 小时、压力单元优先；OQ4：不用 CI 的自托管 runner 测 kernel |
| 11-01 21:05 / 21:45 / 23:40 | owner | 批准从别的引擎借鉴的 9 项（§3.18）；DFlash2 永远开着；TensorFold 的东西一项都不进 1.6.0 |
| 11-02 02:30 | owner | 终测四台机器同时开始 |
| 11-02 16:00 | owner | S-P5：冷启动定价"不低估、在 2 倍以内" |
| 11-02 19:15 | owner | decode 也按价值，不设"decode 一定保留专家"的硬规则 |
| 11-03 00:10 | owner | 三条调度取舍：稠密长上下文慢约 14 % 接受；u 辨识不出时上界保持 G；A103 启动规则（§3.5）的 −1 % 接受 |
| 11-03 00:35 | owner | 三条账本取舍：Start 可以释放空闲的可重建缓存；放置保持 Start 类；存储 miss 按边际读价计价 |
| 11-03 02:45 | 协调者（owner 交给它定） | 回退 VerifyBudget 的 0.25 成本下限（`42f998f5`） |
| 11-03 03:10 | owner | ANE 行下限 |
| 11-03 03:20 | owner | 只修 P0/P1/用户可见的问题，其余记入推迟表（§8.4） |
| 11-03 03:55 | owner | 回长按 A30 定价；之后授权协调者代做剩余决定（推送、PR、GitHub 操作除外） |
| 11-03 06:10、09:17、10:07 | 协调者 | A30 的计数放宽；a30-host 合入、否决 `430b6d60`；start-hold（后被 11:00 取代） |
| 11-03 11:00 | owner | 一条简单规则："有请求的 claim 在等时池不向空闲空间增长"，取代 start-hold 和 resume-price；A30 的定价保留 |
| 11-03 11:35 | owner | T6、N1、清理都进 1.6.0 |
| 11-03 13:25 | owner | G4a/G4b 只报告（§2.2） |
| 11-03 14:40 | owner | M5 Pro 被标 void 的跑不作废 |
| 11-03 17:30 | owner | 先停下修两个问题（§5.3、§5.4） |
| 11-03 18:40 | owner | 交接 |

### 2.7 owner 的长期规则（接手人要遵守）

这些规则在整个项目里一直有效，散见于内部记录：

- **代码质量。** 不要 hack，不加特判；所有实现要清晰、通用、可维护，不能把代码弄乱。每一行都要在生产里用到，不加推测性的保护。改之前先读懂现有机制。
- **参数。** 不拍脑袋定参数；阈值要推导出来并写明出处。
- **取舍。** 只收端到端有明确收益、代码又简单的改动。例子：verify 切分在 kernel 上快 22 %，端到端只有 +0.5 %，所以不做。
- **硬件与模型。** kernel 大多在各种硬件上实测过，不大动，只统一选择规则。不支持 M1、M2；不为 16 GB 机器专门优化。合理的差异（Apple9 和 Apple10、流式专家）由模型以数据报告，不算特判。DFlash2 永远开着，不为关掉草稿的情况规划、调参或测量。
- **owner 点名的方法照做**，不要另辟蹊径。
- **仓库文字。** 提交、PR、GitHub 上的文字用英文，不加 AI 署名。
- **对外操作。** 推送、开 PR、在 GitHub 上留言之前都要先问 owner。
- **测试机。** 用 GPU 锁；计时跑只在空闲的机器上跑；只比较固定到达顺序的 churn 跑；永远不要往可能链到模型缓存（HF 缓存、`install/models`）的路径写；不碰别的会话的进程和目录；协调机本地不编译。
- **汇报。** 先说现状；作业超过预期时间约 2 倍就报告；不在后台悄悄改代码。

---

## 3. 1.6.0 做了什么（按机制）

先看总表，再看各节。每节都按"作用 / 位置 / 来历 / 测量 / 建议"来写。

| § | 机制 | 主要位置 | 测量说明了什么 | 建议 |
|---|---|---|---|---|
| 3.1 | 内存账本与 claim | `runtime/engine/MemoryLedger.*`、`EngineAllocate.hpp`、`EngineMemory.cpp` | 冻结修复靠这套骨架承载 | 保留骨架，删掉定价 |
| 3.2 | claim 排序与持有 | `MemoryLedger.hpp:74-80`、`MemoryLedger.cpp:224-265` | **有用**：结束了冻结 | 保留 |
| 3.3 | host 规则、governor、控制 pass | `runtime/engine/MemoryGovernor.*`、`MemoryControl.cpp` | 骨架必需；按价值取存储导致 M6 回退 | 保留骨架，固定化；转换问题待 owner 决定 |
| 3.4 | 内存价值模型 | `MemoryValue.hpp`、账本定价、Cache 价格查询 | **没用，有害** | 删除 |
| 3.5 | 代价模型与调度器 | `CostModel.*`、`Scheduler.*` | **没用**，收益来自 pass | 删除 `CostModel` 和定价规划，换固定规则 |
| 3.6 | 分层 pass | `Runtime.cpp:3043`、`Engine::PassCommand` | **有用**：最大的收益来源 | 保留 |
| 3.7 | GDN 状态检查点、replay 点 | `Engine.cpp`、`EngineMemory.cpp:660-840` | 必需机制 | 保留；可选 claim 的收益定价改固定规则 |
| 3.8 | HiSparse spill 与热块缓存 | `Cache.cpp`、`model/KvBlockCache.*`、`model/KvHistoryStream.*` | 机制必需；转换释放的内存很少 | 保留；放置改固定规则 |
| 3.9 | KV 层级与 WriteBehind | `KvPool.*`、`KvTier.hpp`、`WriteBehind.*` | 必需 | 保留；需求窗口改成常量 |
| 3.10 | 弹性专家存储与池 | `model/ExpertStore.*`、`model/ModelPools.hpp` | 需要（小内存、无 HiSparse） | 保留；固定缩放规则；按模型设 floor |
| 3.11 | StreamedMemoryPlan | `runtime/engine/StreamedMemoryPlan.*` | 原样搬家 | 保留 |
| 3.12 | WeightResidency | `runtime/engine/WeightResidency.*` | 可用 | 保留 |
| 3.13 | 状态与指标（schema 7） | `runtime/engine/Status.cpp`、`server/metrics.py` | 诊断部分很有用 | 保留诊断；删估计器字段 |
| 3.14 | kernel 选择与校准 | `runtime/ops/KernelCalibration.hpp`、`KernelSelection.*` | 行为不变；mini-M4 上有测量收益 | 保留 |
| 3.15 | 服务端与 App | `server/supervision.py`、`server/serve_record.py`、`app/` | 可用 | 保留 |
| 3.16 | 其他保留的运行时改动 | 多处 | 各有测量 | 保留 |
| 3.17 | 测试与工具 | `dev/` | 部分有用 | 保留行为测试；删钉价格的测试 |
| 3.18 | 从别的引擎借鉴的 9 项 | 多处 | 诊断类有用；定价类随估计器删 | 见该节 |

### 3.1 内存账本与 claim

**作用**

- **每次分配都属于一个 claim。** 不在 claim 里的分配直接抛 `std::logic_error`（`b2e78aad`）。
  - 每个请求一个 claim：`Engine::ClaimHandle`（`Engine.hpp:364`，`EngineMemory.cpp:842-851`）。
  - 引擎自己的分配用临时 claim。
- **claim 的类别和时机。**
  - 类别 `ClaimClass{Floor, InService, Start, Optional, Growth, Host}`（`MemoryLedger.hpp:68`）。
  - 时机 `NeedTiming{CanWait, Immediate}`（`:70`）。
  - `ClaimSpec` 在 `:83-110`。其中 `holdSeconds`、`benefitSeconds` 属于价值模型。
- **账户。**
  - 账户从各自的所有者拉取：`kv` 和 `states` 在 `RuntimeResources.mm:1212-1213` 注册；模型池在 `Bootstrap.mm:436-440` 注册。
  - 另有一个 `fixed` 账户承担余数，所以账面按构造恒等。
- **分配流程。**
  - `Engine::allocate<>` 在 `EngineAllocate.hpp:75-170`：
    1. 每次尝试都在一个 `MemoryGovernor::Scope` 里进行。
    2. 被拒时记下一个 `Need`；就地部分（页、状态 buffer）由 `inPlace()` 加上。
    3. 依次调用 `governor.need` 和 `governor.free`。
    4. 重试，直到满足，或者再也释放不出东西。
  - 调用点：
    - `EngineMemory.cpp`：`growLane`（538）、`reserveReads`（548）、`growJoiners`（564）、`refuse`（594）、`reserveCaptures`（660）、`stateClaim`（773）、`reserveCapture`（798）、`spillLanded`（439）、`placement`（73）；
    - `Engine.cpp`：`prepare` 的窗口裁剪、`promotedCopy`（1229）。
- **规则主体** `freeFor`（`MemoryLedger.cpp:699-821`）：
  1. 可用空间 = 空闲 + 可无尺寸取走的部分 − 持有。
  2. 如果落地内存加上自己欠的拷贝能覆盖需求，就等（Pending）。
  3. 否则先做 `dryRun`（381-609），再 `take`（611）。取走是全有或全无；只覆盖一部分时就等剩下的。
  4. `available()`（825）是只看不取的 dry run，供放置和 `hostPass` 使用。
- **victim 来源。**
  - Cache：在统一回收遍历 `nextUnit` 上实现 `peek/take`（`0bae515d`）。旧的 9 个回收入口删除前，做过 1,000 个普通种子加 1,000 个窗口种子的差分比对。
  - `EngineVictims`：空闲 buffer、lane 转 Spill、pass 命令。
  - 模型池。
- **各类别能取什么。** 看 `dryRun` 的 `mayTake`（`MemoryLedger.cpp:400-415`），以及 `poolUnits`、`rebuiltUnits`、`pendingUnits`（430-436）。
- **最先取走零代价的空闲 buffer：** `takeUnsized`（667）。

**来历**

- T10（11-02）：
  - `3bdf8e93` 建立账户、claim、"那一条规则"及其价格；
  - `ce9eb6f1` 按类别和 rank 准入，加入 host 规则。
- M3a：
  - `7284fdb4` 让所有分配走 claim，并删除了：压力策略 pass、exchange、冷却期、增长信用、#650 回收梯、spill-wait。
  - `b2e78aad`、`7d31d72e`、`ca20c739` 继续删除和收尾。
- 之后是 53 个后续提交（`d726e59f^..e9626060` 里的非合并提交），其中 28 个只有标题、没有正文。

**测量**

- 冻结消失（§5.1）是 claim、排序、持有三者共同的结果。
- G9（账面自洽）在 M6 终测 job 01-03 的 `matrix.py check` 中通过。
- M3a 首次集成时引擎测试挂了 66 个。从 11-02 14:05 到 22:24，失败数 66 → 17 → 5 → 0。
- cache-property 种子找出多个回收遍历 bug（窗口种子 2484、9349、25949、28462，现都设为默认种子）。闲时的大规模扫描又找出 3 个 `ReclaimPlan`/`takeLeaf` 缺陷（`m3afix/sweep`）。

**建议**

- **保留骨架：** claim、类别、rank、持有空间、落地内存、全有或全无、floor、"claim 外分配即错误"。
- **删掉 `dryRun` 的价格排序：** `cost.at(holdSeconds) / lack.share`、benefit 上界、`Lacking::share`。改为固定的来源顺序（§12.2）。
- **待修 D12：** 等待中的 claim 的字节被持有两次，一次在 `tryReserve` 的空间里，一次在 `heldPages` 里。
- **注意一处不一致：**
  - `freeFor` 把被 host 拒绝的需求标为 `canWait = false`（`MemoryLedger.cpp:722`）。
  - 但 `dryRun` 只对 `Immediate` 跳过稍后才落地的单元（`:387`、`:572`）。
  - 这正是 §5.3 那个问题的根源，而修复 `3874a2f1` 只处理了 replay 点这一种情况。

### 3.2 claim 排序与持有（owner 认定的两大收益之一）

**作用**

- **排序。**
  - `ClaimRank{priority, decoding, completedTokens, earliness}` 按字典序比较（`MemoryLedger.hpp:74-80`）。
  - `Engine::rankOf`（`EngineMemory.cpp:853-862`）对处于 Decode 或 WaitingMask 的 lane 置 `decoding`，所以同优先级下 decode 排在 prefill 前面。
  - 抢占用同一顺序：`yieldsBefore`（873）、`laneToYield`（880）。
- **为高 rank 的等待者持有空间。** 等待中的 claim 对所有 rank 更低的所有者持有空间：
  - 字节在 `MemoryGovernor::tryReserve`（`MemoryGovernor.cpp:219-259`）里扣；
  - 页在 `KvPool::acquirePages(count, form, held)`（`KvPool.cpp:58`）里扣；
  - 相关函数：`heldByClaims`、`heldBytes`、`heldPages`（`MemoryLedger.cpp:224-255`）。
- **落地内存先付给等待者。**
  - 来源是 `Cache::returning` / `returningFor`（`Cache.cpp:563-619`）。
  - 自己欠的拷贝先付给自己（测试 `testLandingPaysItsOwnerFirst`；设计修订里的编号是 M-P3）。
- **有请求在等时，池不向空闲空间增长。**
  - `holdsFreeRoom()`（`MemoryLedger.cpp:261-265`）只把带 `requestId`、正在等待或被 host 拒绝后等待的 claim 算进来（`ccc89af3`）。
  - `grow()` 在 876、977 检查 `!waiting`。
  - 来历：`2b326a37`（11:02）加了"有 claim 在等时池不向空闲空间增长"，`ccc89af3`（11:24）收窄到只看请求的 claim。11:00 owner 要一条简单规则，它取代的是当时提出的 start-hold 和 resume-price 两个方案，**不是** A30。A30 的回长定价和核算原样留在代码里（`MemoryLedger.cpp:306-342`、`RegrowthWatch.hpp`），仍待删除（§3.4、§12.2）。
- **等待记账。**
  - `Engine::syncClaims`（`EngineMemory.cpp:243-276`）设置 waiting、hostRefusedWaiting 和持有上限。
  - 上限是 `resourceWaitTimeoutMilliseconds` = 30 s（`Engine.hpp:37`）。
  - 它还设置 `inDemand`，这一项只给 A30 用。

**来历**

这条规则最早出现在 spill-wait："回来的内存留给被拒绝的那个 lane"。bug 是 10-31 夜里在二分中发现的，修复提交是 `88f4dbba`（11-01 01:28）+ `2ddc977d`。

- 场景：Studio，FN Q2_0，18G，3 个 churn 客户端，480 s 固定顺序的跑（A53，测的是第一版补丁；之后经评审压成上面两个提交）。
- 最长 decode 停顿：顺序 A 从 111 s（单靠回收梯是 104 s）降到 2-10 s；顺序 B 从 370 s 降到 2 s。
- 顺序 B 的最长延迟从 531 s 降到 272 s。

1.6.0 把它推广成了一般规则。

**测量**

- 14 次 cand 跑里没有超过 2.5 s 的客户端间隔（§4.6）。
- 同一个 390 MiB 的热块 claim 现在约 1 s 内就能满足。

**建议**

- 保留。
- 注意：claim 等待唯一的硬上界是 30 s 资源等待（`EngineMemory.cpp:259-261`）。"等待都很短"目前只有测量证据。

### 3.3 host 规则、MemoryGovernor 与控制 pass

**作用**

- **估计 host 内存。**
  - `estimateHostMemory`（`MemoryGovernor.cpp:59`）= 空闲页 + 文件页 + 可清除页，再加压缩折扣。
  - `HostRule::sample`、`admits`、`deficitBytes`（353-389）。
  - 进出"带"（band）：余量低于 1 GiB（`kHostMarginBytes`）时进入；余量回到 margin + 最大增长单元以上才退出。
- **准入。**
  - `tryReserve`（219）依次查引擎上限、持有空间、host 规则。
  - 拒绝原因记为 EngineBudget 或 HostPressure。
  - `unseenGrowthBytes`（211）覆盖 1 s 的沉降期（`kHostSettleMilliseconds`）。
- **相当于 1.5.0 的 "serving mark" 的豁免。**
  - 位置：`MemoryGovernor::free`（415-424）。
  - Floor 或 InService 类别的需求被 host 拒绝且结果为 Short 时，置 `hostExempt` 并返回 Met，重试即可越过 host。
- **控制 pass。**
  - `controlPass`（480-516）：
    - Critical：先 `dropToFloors`（`MemoryLedger.cpp:1013`），再 `hostPass(max, critical)`；
    - 否则：没有赤字时先 `grow`；在 Warning 下（取到 margin 为止）或有赤字时（取 min(margin, deficit)），每 1 s 最多做一次 host pass；有赤字时之后再 `grow`。
  - `hostPass`（442-454）发出一个 `ClaimClass::Host` 的 claim，hold 为 60 s。
  - 启动时用 `growToTargets`（518）。
- **驱动方式。**
  - `runtime/main.mm:54-83` 设了 500 ms 定时器和 macOS 内存压力事件源。
  - `FdTransport::run` 只在没有命令时调用处理函数（`FdTransport.cpp:303-330`）。
  - 处理函数是 `MemoryControl::run`（`MemoryControl.cpp:31-46`），顺序为：
    1. `releaseIdleWeights`
    2. `setPressure`
    3. `collectLandings`
    4. `refreshMemoryRates`（价值模型）
    5. `governor.evaluate`
    6. 状态上报
    7. `controlPass`
- **`e1310a4d`（§5.4）。** 不到 Critical 时，Host claim 不取专家存储和热块缓存的单元，只能取：
  - 缓存；
  - 空闲 buffer；
  - rebuild-only 池（视觉 arena、嵌入行、空闲 staging）；
  - **转换**。

**一个误读，值得记住**

M6 上"memory_pressure 有 89-99 % 是 warning"说的是引擎自己的带状态：host 余量 p50 为 978-1131 MiB，低于当时 2 GiB 的预留。macOS 自己报 warning 的时间只有 0-2 %。早期把"M6 存储不回长"归因于"macOS warning 97 %"，是错的（§10）。

注意这些数字来自 11-01 在 1.5.0、main、pass 上的跑，那时带的门槛是 2 GiB。1.6.0 的带在余量低于 `kHostMarginBytes` = 1 GiB（`MemoryLedger.hpp:29`）时进入；按 978-1131 MiB 的 p50 余量，1.6.0 在 M6 上有多少时间在带里，**没有测过**。

**建议**

- 保留骨架。
- **owner 待决：** 不到 Critical 时，host 赤字仍可按价值把一个 Resident lane 转成 Spill（`mayTake`，`MemoryLedger.cpp:410-412`）。
  - 去掉这一点要改一个条件，并重写 7 个引擎测试（§5.4）。
  - owner 提出的固定让出顺序把"HiSparse 模型上长 lane 转 Spill"放在专家存储之前，可作参考。

### 3.4 内存价值模型（建议删除）

**作用**

用"引擎工作秒"做统一货币给内存定价。

- **单元代价。**
  - `UnitCost{once, rate}`（`MemoryLedger.hpp:142-146`）在申请者的 hold 上求值：`once + rate × hold`，再按字节比较。
- **hold 和 benefit 怎么取。**

  | 申请者 | hold 或 benefit |
  |---|---|
  | 请求 claim | 剩余预填充价格 + 预计 token 数 × 每 token decode 毫秒（`remainingSeconds`，`EngineMemory.cpp:202-220`） |
  | Host claim | `kValueHorizonSeconds` = 60 s |
  | Growth | benefit = rate × 视野 − `cycleSeconds` |
  | 放置 | benefit = spill 罚金 |
  | Optional | 检查点的重算价值，或捕获切分的代价 |
- **池的速率。**
  - `rate = ghost 命中/秒 × miss 秒 × min(u, ghostBytes)/ghostBytes + 扫描读字节/秒 × 每字节读秒 × u/(wholeBytes − currentBytes)`。
  - `cycleSeconds` = 回填 miss 时间 + 压实时间。
  - 这些都是 EWMA，跨 release/restore 保留。
- **缓存单元。** `once = r × (已拷贝 ? bytes/tierRead : tokens × 每行预填充秒)`。
- **spill 罚金。** 剩余预填充命令"有 Index 页"与"没有 Index 页"两种情况的价差，加一个 decode 项：每条 decode 命令 9 ms 先验。

**位置**

- `runtime/engine/MemoryValue.hpp`：205 行，全新。
- 账本里的定价：
  - `UnitCost`、`MemoryRates`（hpp:149-165）、`Victim::cost`、`ClaimSpec::holdSeconds/benefitSeconds`；
  - `setRates`、`observePools`（cpp:163-174）；
  - A30 的 `takesPerSecond`、`lanesInDemand`、`took`、`lanesTakes_`（306-342、459-463）；
  - `grow()` 的价值部分（874-1011）；
  - `fill()` 的价值列（1045-1057）。
- 引擎输入：
  - `spillPenaltySeconds`（`EngineMemory.cpp:173-200`）；
  - `remainingSeconds`；
  - `sampleSpilledReads`（222-235，`kSpilledReadPriorMilliseconds = 9`）；
  - `refreshMemoryRates`（1124-1155）；
  - `reserveCaptures` 的 `splitSeconds`（743-761）；
  - `stateClaim` 的 benefit（783-791）；
  - `promotedCopy`（`Engine.cpp:1229-1247`）；
  - `Engine.hpp:1245-1251` 的 `reuse_`、`tierReads_`。
- Cache 里的价格查询：
  - `cachedCost`（968-974）、`victimOf` 的 cost（977-998）、`ReclaimPlan::price`（2323-2338）；
  - `peek`、`peekReclaim`、`weighs` 都带着 `MemoryRates` 参数（2340-2435）；
  - `ReclaimPlan` 本身从 1044 行开始，约 1,360 行。
- 模型池计数：
  - `model::PoolValue`（`ElasticPool.hpp:33-54`）；
  - 各适配器的 `value()`（`ModelPools.hpp:105-120`、`159-168`）；
  - `ReleaseCount`、`RebuildCount`；
  - `ExpertStoreStats::nearGhostHits`、`tailHits`、`marginBytes`；
  - `SlotRecycler` 的计数；
  - `KvTier::throughput`。
- 状态输出：`value_per_gib`、`grow_value_per_gib`、`shrink_value_per_gib`（`Status.cpp:295-306`）。
- 测试：
  - `memory_value_test.cpp`（449 行）；
  - `memory_ledger_test.cpp` 里的价格测试；
  - `RegrowthWatch.hpp`（124 行）；
  - 录制夹具 `dev/tests/fixtures/recorded-loads/*.tsv`；
  - engine-sim 的 SIM-value 和 SIM-35b-churn。

**来历：一条补丁链**

1. `3bdf8e93` 引入价格。
2. `ec6825da`：对 M6 固定存储，`value_per_gib` 给出 3.79 和 3.25 ms/token/GiB；实测范围是 1.1-3.6。
3. capfix 修复轮（合并 1 之后，附录 C）里的一组：`570a7db3`、`f0020c29`、`a5e571fc`、`d5011264`。
4. `44f2fe7c`、`65cd5edd`：存储 miss 改按边际读价计价（0.130-0.155 ms），不再用均值 0.2095 ms。
5. `722d2146`、`f0e06b43`、`75674b20`、`7a160991`。
6. A30 系列：
   - `f0724a9d`："一个单元只有在它省下的读取在视野内不会被别的 claim 收回时才回长"；
   - `09836db7`；
   - `24a3d865`：smoke b 中 host claim 为了 5 MiB 的余量拿走一个 1,023 MiB 的 chunk，5 s 后回长，8 s 后又被拿走；
   - `89856d75`、`d9eef1cc`、`38995965`（RegrowthWatch）、`c4e73577`、`2cd99b39`、`ee6cdc6c`、`ffca336c`。
7. `2b326a37` / `ccc89af3`：11:00 owner 认为方案越来越复杂，要一条简单规则，于是加了"有请求的 claim 在等时，池不向空闲空间增长"。它取代的是当时提出的 start-hold 和 resume-price，A30 的回长定价和核算**没有被替换**，仍在代码里（`MemoryLedger.cpp:306-342`、`RegrowthWatch.hpp`）。
8. `a7dac91a`（N1）：dry run 原来把每个 chunk 都按顶部 chunk 的大小计价，导致 272 MiB 的 host claim 拿走了 1,202 MiB。
9. `e1310a4d`：终测里 327.7 MiB 的 host claim 仍然拿走了 179.3 + 1,023 MiB。

**测量**

- 没有任何测量显示它带来收益。
- M6 上 value 默认配置比 floor = cap 差：B17 判定 work 1.088、latency 0.890，结论 FLOOR = CAP。根因就在价格（§5.4）。
- 测试 `testValuePerGiBReproducesThePinnedStore` 的边界来自模型自己的输入，属于部分循环的标定，不是行为测试。
- 常量的选法：
  - 时间常数 τ（`kRateTimeConstantSeconds`）：回放规则在所有候选值上都不满足，60 s 是排除部分样本后选的（`MemoryLedger.hpp:41-56`；设计稿写的是 30 s）；
  - 视野 60 s 用 Kaplan-Meier 选出。

**建议**

- 全部删除，清单见 §12.2。
- `Cache::ReclaimPlan` 只在一种情况下保留"按单元大小"的部分：`available()` 和全有或全无的取走仍然需要它。否则改成一次回收一步（`reclaimStep`、`evictOne`、`takeReclaim`，`Cache.cpp:2553-2671`），然后重试。

### 3.5 代价模型与调度器（建议删除代价模型）

**作用**

- **`CostModel`。** `CostModel.hpp` 182 行 + `.cpp` 366 行，全新，来自 `c131b9be`。
  - 7 项（hpp:23-31）：
    - `Units`：每条命令或每个 pass 一次，即周转时间；
    - `ChunkLayers`、`RowLayers`、`AttentionPairs`、`IndexPairs`；
    - `WeightReadBytes`、`SpilledKvReadBytes`。
  - 拟合：衰减岭回归，系数非负，`kDecay = 0.99`（83）。每个 pass 是一个权重 1/passes 的样本，样本截到价格的 [0.5, 2] 倍。
  - 种子：在 2048 行预热之后再跑一次 256 行预热（`Bootstrap.mm:263-271`），流式模型启动因此多约 2 s。SSD 两项用 `loadReads()` 做种子。
- **调度规则**（`Scheduler.cpp`）。完整的定价策略写在 `Scheduler.hpp:156-183` 的类注释里。
  - `bound()` = max(G, u/0.1)（589-591）。u 还没辨识出来之前，上界就是 G。
  - `passesFor`（593-623）：
    1. 选出能让每个 pass 价格 ≤ bound 的最少 pass 数；
    2. 做不到就放宽到 2 × 固定部分；
    3. 还不行且有竞争时，才把行对半切（`bound_cuts++`）。
  - 其余定价规则：
    - `groups`（548）；
    - join 规则（690-700）；
    - 按价格算剩余代价的老化和 overdue（33-36）；
    - `supersedes`、`ranPastItsBound`、`supersedeGain`（415-499）；
    - `prefillAdmissionOrder`（234-336）；
    - A103 启动规则（设计修订里的 A103 条，`3521d94e`；和尝试记录里的 A103 = intpart 不是一回事）：排队的提示词只在三种情况下启动——它是领先者；它已超期；或者它前面的每个 lane 都在这条命令里结束提示词，且没有一个是 Spill lane。
  - ANE 行下限：`minimumUnitRows`。
- **固定、可保留的部分：**
  - `next`（362-398）：优先级 → decode 债 → 交替；
  - `nextDecode`（735）；
  - `contended`（580）；
  - `withinPairs`（625），即每层注意力对数上限；
  - `commit`（766）；
  - decode 债（约 920-938）；
  - `abandoned`（942）；
  - `lastPassMilliseconds_`（约 885-887），就是 owner 要的测量反馈。

**来历**

- `0bd50301` 已经有一个最小二乘的"固定 + 每行"命令估计。
- T8（11-02 01:54-07:16，17 个提交）用 7 项学习模型替换了它。
- 之后的修补提交：`b2f1a0a2`、`1fc4cf33`、`8eb174f8`、`ffcbc4db`、`58657778`、`b2071ded`、`2e45dd88`、`37497bd9`、`535d36a4`、`2732cc87`、`86b6995a`、`0b0309ef`、`3521d94e`、`8b9d444a`、`751690bf`、`16d03e2c`、`36c5ead0`、`a78183aa`、`9b73d34d`、`5ade28cb`、`eca7f1f9`、`70624a8f`、`de36ea42`、`5190eff2`、`7153bbeb`、`21268d5e`、`0549a2fb`、`66557140`、`7e9063d2`。

**测量与暴露的问题**

- **收益：** 没有能归到代价模型身上的测量收益。终测的预填充收益来自分层 pass（§4.5），而 pass 树用的只是简单估计。
- **S-P5，冷启动不准。**
  - 冷启动种子把 M6 第一个竞争的 300 行请求定价为 4,738.7 ms，真值 3,226.8 ms，高估 47 %。
  - 设计要求误差在 25 % 以内。owner 在 11-02 16:00 接受了"从不低估、在 2 倍以内"。
- **u（周转）辨识不出来。**
  - M6 smoke 用同一形状的启动样本拟合出 u = 301 ms，而硬件实际约 10 ms。
  - 结果第一个竞争命令的上界约 3 s，decode 间隔约 3.65-3.7 s。`751690bf`、`5190eff2` 改成：u 未辨识前上界保持 G。
  - 27B 上周转被定价为 153 ms，下一个竞争单元的上界就成了 1.53 s（`b2071ded`）。
  - D10：运行中 M6 读到的 u 是 50-60 ms，实测约 10 ms，所以单元约 0.6 s 而不是 0.5 s。
  - u 和 δ 从来没有测过。δ 是设计修订 A49 条定义的量：一个 Spill pass 的第一个注意力层要从头启动历史流，δ 就是这样每多一个 pass 多出的时间（用 1、8、24 个 pass 跑同一条 8192 行命令，Spill 与 Resident 对减得到）。树里没有能按指定 pass 数跑一条命令的工具（D7）。
- **自我 supersede 活锁**（修复 `8b9d444a`）。
  - `supersedes()` 给空的 joiner 集合也定了价，结果命令在每个 pass 后都放弃自己。
  - M6 上 24 个 pass 的命令有 91-95 % 被放弃。
  - smoke a 里 306/336 个被放弃，约 32 % 的预填充白做，TTFT 175 s，正常约 65 s。
- **同级 supersede 扔掉已经开始的行**（smoke b，M6）。
  - 一个 300 行的请求切断了 6,144 行的命令，扔掉 5-6.7 s 的预填充。
  - 修复 `7e9063d2`：同优先级的新到达等下一条命令。
- **27B 加 ANE。**
  - 打开 ANE FFN 切分时 `maximumPasses = 1`（`RuntimeArenas.hpp:121-123`），上界只能靠切行来满足。
  - 15 分钟的 M6 27B churn 里，有 2,797 次切到 ANE 的 512 行阈值以下，预填充行数比 pass 少 17 %。
  - 修复：ANE 行下限（`9b73d34d`、`5ade28cb`，owner 11-03 03:10 批准）。
- **`kDecay = 0.99` 实际是平局规则选的。** 它是回放 8 次录制 churn 选出的，但所有候选衰减值的 p95 价格误差都在 0.335-0.365 之间。
- **稠密长上下文。** 27B 的 64K 提示词在两个 decoder 旁边预填充，比单独跑慢 13.9 %（engine-sim 的 SIM-long-prompt；更早一轮是 15.1 %）。owner 11-03 00:10 接受约 14 %。
- **A103 启动规则的代价：** SIM-churn work −1.0 %，已接受。

**建议**

- **删除：**
  - `CostModel`；
  - 种子预热；
  - 定价规划：`bound` 里的 u/0.1、`passesFor`、`groups`、`remainingMilliseconds`、按价格老化、join 的价格比较、`supersedeGain`。
- **保留：**
  - 上面列出的固定部分；
  - ANE 行下限；
  - "同级新到达不切已经开始的行"这条规则。
- `runtime/SmallSolve.hpp` 被 `model/VerifyBudget.cpp` 共用，保留。
- 替代规则见 §12.1。

### 3.6 分层 pass（owner 认定的两大收益之一）

**作用**

一条预填充命令按层分成多个 pass。每个 pass 只跑一段连续的层（`model::passLayers`，`CommandCost.hpp:19-21`），pass 之间引擎可以插入 decode 命令。因为行数不变，每层专家对整批行只读一次。

**位置**

- `Runtime::prefillAsync`（`Runtime.cpp:3043`）：
  - 每个 pass 的 begins/ends 在 3051-3052；
  - 命令内状态捕获的窗口在第一个 pass 之前复制一次（3070-3077）；
  - 最后一个 pass 之前的 pass 不提交（3117-3127）；
  - ANE 切分的重跑用全部层（3100-3115）。
- `Engine::PassCommand{progress, items, captures}`（`Engine.hpp:576-580`）：
  - 后续 pass 复用 items 和 captures，只刷新页表（`Engine.cpp:1582-1596`）；
  - `apply` 推进 pass（2033-2036）；
  - pass 之间发生取消或到截止时间：`abandonPasses`、`settleEndedBetweenPasses`、`yieldsEndedLanes`（2063-2107）。
- 上限：
  - 一条命令最多 8,192 行，即 `kPrefillGroupChunks = 4` × 2,048（`RuntimeArenas.hpp:43、84、122`；适用于流式专家或 spill 目标）；
  - pass 数最多等于层数；打开 ANE FFN 切分时只能是 1。

**来历**

1. 1.5.0 已经能把单个领头 lane 的命令分层跑（`cabbad27 Scheduler.cpp:15-28`，`kContendedPrefillMilliseconds = 500`）。
2. `102ac46e` 允许多 lane 命令分 pass。runtime oracle 在 M5 Pro 上用 Flash-Next IQ3_S、3 个 pass 验证，结果逐位一致。
3. `0bd50301` 让流式目标的所有 lane 都按命令分 pass。
4. 1.6.0 进一步：
   - 每个目标的 `maximumPasses` 都等于层数；
   - 竞争单元先分 pass，再考虑切行；
   - 流式 lane 可以和别的 lane 合批分 pass；
   - 无存储的 spill lane 也能合组分 pass。

**测量**

- `0bd50301`（M6，FN Q2_0）：
  - N=4 同时发送时，TTFT 从 2.8/6.2/9.6/13.1 s 变成四个都约 5.1-5.7 s；
  - 预填充 GPU 时间从 11.3 s 降到约 5 s；
  - N=1 不变，约 3.0 s。
- 终测：

  | 指标 | Studio：1.5.0 → 1.6.0 | M5 Pro：1.5.0 → 1.6.0 |
  |---|---|---|
  | 每命令行数 | 3,702 → 7,313 | 2,104 → 7,667 |
  | 每预填充行专家字节 | 5.1 → 3.1 MB | 14.9 → 5.7 MB |

  - M5 Pro 900 s 内的预填充专家读取：1.5.0 是 5,157 GB（5.7 GB/s，接近 SSD 饱和），1.6.0 是 2,917 GB，同时还多预填充了 47 % 的行。
- 归因（§4.5，按与中间树的对比推断）：在没有冻结的中间树基础上，pass 优先的调度又带来预填充 +21-26 %、decode +18-20 %、TTFT −21 %。
- **24 GB 的 M6 上没有这类证据。** 终测没有给出 M6 的每命令行数和每行专家字节；M6 FN churn 上 value 配置相对 1.5.0 的预填充是 −7.8 %（修复 1 之前；修复后约持平），floor = cap 只有 +2.2 %。更早的 A66 在 M6 churn 上测到 pass 的总 decode −2 到 −7 %（噪声还是真实没定），mP4a/b、pFa/pFb 几跑一直没分析。owner 结论（1）的数据只来自 M5 Pro 和 Studio，pass 在 24 GB 机器上的收益**没有证明**。

**要注意的地方**

- **一个 pass 不切分一层。** Metal 的命令时间限制是按层算的，所以 GPU 时间上界必须在两个 host 步之间成立，不能按逻辑 pass 算。
  - `c305f2de` 的"按 pass 算的注意力上界"被 `5f16b109` 回退；122B 在 YaRN 深度会超出 1.6-1.9 倍。
- **pass 之间结束、挂起、放弃 lane 是崩溃高发面。** A71：第 4 轮修复引入了 2 个 P1 崩溃，pass 一度被移出发布。

**建议**

保留。每个 pass 跑多少层，改为按上一个 pass 的实测时间来定（`lastPassMilliseconds_` 已经有了）。

### 3.7 GDN 状态检查点与 replay 点

**作用**

Qwen 混合模型（Flash-Next 是 36 个 GDN 层 + 12 个注意力层）的递归状态不能从 KV 重建，所以预填充沿途要捕获状态。

- **检查点和边界。**
  - `kPrefillCheckpointTokens = 2 × draftContextTokens`（`Engine.hpp:30`）；`plannedCheckpoints`（94-106）；
  - `Request::StateBoundary{tokens, disposable}`（376-384）；
  - `replayStateBoundary`、`promptReplayBoundary`（`Engine.cpp:846-857`）；
  - `addStateBoundary`（1277）、`retireCheckpoint`（1354）、`publishReachedStateBoundaries`（1374）。
- **状态类别**（在 `reserveCaptures` 里分配，`EngineMemory.cpp:660-771`）：
  - `Checkpoint`：可丢弃；
  - `InUse`：提示词的 replay 点；
  - `Ordinary`：junction 或共享前缀。
- **对应的 claim**（`reserveCapture` 798-840、`stateClaim` 773-796）：
  - replay 点：InService，reach 为 InUse；
  - 其他：Optional/Immediate，benefit 被定价；
  - 越过边界的 KV：Optional 的 `laneNeeds` claim（743-768）。
- **跨 pass。** 命令内的状态捕获在多 pass 下仍然正确：窗口在 pass 0 之前复制，captures 跨 pass 复用。
- **存放位置。** 缓存的状态在 `runtime/engine/StateCache.*`，模型状态池在 `runtime/model/QwenState.*`。

**相关修复**

- `3874a2f1`：见 §5.3。
- `2f6411fd`（T7，smoke b）：越过命令第一个状态边界的 KV 改按 lane 增长判断。M6 上单独的 12K、18K 提示词恢复成和 1.5.0 一样的命令。
- smoke c 的单个 64K 提示词：cand 93.30 s，1.5.0 94.57 s；两者都是 8 条命令（7 × 8192 + 6743），0 次捕获切分。

**建议**

- 保留机制。
- Optional 检查点和 junction 的 benefit 定价改为固定规则。例如：Optional claim 只取空闲空间和零代价单元。**这是建议，未验证。**
- 测试 `testReplayPointAdmittedInMarginBand`（`engine_memory_test.cpp:5815-5821`）写的还是旧规则。它能通过，只是因为夹具里没有正在拷贝的状态。

### 3.8 HiSparse spill 与热块缓存

"HiSparse"是团队对 Flash-Next 块稀疏注意力的叫法，对应代码里的 `kv::Reads::SelectedBlocks`、`SparseKvShape` 和 Index 页。代码只在测试名里用这个词。

**作用**

- **KV 模式。** `KvMode{Resident, Spill}`（`Types.hpp:36`）。
- **Spill lane 在内存里留什么。**
  - 只留 Index 页。Index 页每页 24,576 B，是 Full 页（423,936 B）的 5.8 %。
  - K/V 只在 SSD 槽里。
  - 例外：前 64 页（稠密历史）和最近 64 页保持 Full。
- **怎么读。**
  - 预填充通过 `KvHistoryStream` 在后台读溢出的历史：每个注意力层、每个 Index 页 33,280 B。
  - decode 通过热块缓存 `KvBlockCache` 读选中的块。每层是 ARC 槽，miss 同步从 SSD 读。
- **热块缓存 floor。**
  - 第一个在 Index 页上 decode 的 lane 要 390 MiB，之后每多一个 lane 加 195 MiB，最多 975 MiB（`Model.hpp:490-504`；`KvBlockCache.cpp:95-96`）。
  - floor 是全有或全无的：`KvBlockCache::Impl::grow`（`KvBlockCache.cpp:173` 起）一次申请所缺的全部 extent，被拒就全部退回。
- **放置。**
  - `Engine::placement`（`EngineMemory.cpp:73-109`）：Resident KV 不超过 Spill 峰值时选 Resident；超过时，看这个 claim 能以低于 `spillPenaltySeconds` 的代价腾出的内存（`governor().available(spec, need)`）够不够：够就选 Resident，不够才选 Spill。这依赖价值模型。
  - `placeCandidates`（111）在每个账本状态下只算一次。
- **转换。**
  - `EngineVictims::conversions`（941-1001）按**从大到小**提供 Resident lane，受落地空间、配额和写预算限制。
  - `peek/take`（1003-1059）的顺序：空闲 buffer → pass 命令 → 转换。
  - Cache 侧：`convertToSpill`（3019）、`writeThrough`（3072、3099）、`moveLandedPages`/`spillWritten`（3164-3243）、`indexLandingPages`（621）。
- **相关标志和文档。** `--sparse-kv-memory`；`--kv-spill always` 仅用于验证。文档见 `docs/server.md` 的"KV on the SSD"一节。

**测量（详见 §5.2）**

- **转换能释放多少：**

  | lane 长度 | 结果 |
  |---|---|
  | 64K | Resident KV 828 MiB；Spill 为 97 MiB 页 + 390 MiB 热缓存 = 487 MiB；释放 341 MiB |
  | 128K | 释放 1.09 GiB |
  | 32K | 反而多用 49 MiB |
  | 约 37K 以下 | 不释放内存 |
- **decode 代价：** M6 单 lane 每条 decode 命令约 9.8 ms（设计估计 7.0-11.5 ms）。
- **终测里一次转换都没观察到。** M6 默认配置的跑里 Spill 请求为 0；floorcap 每跑 1 个，从一开始就放成 Spill。
- Studio 18G 的 A1 跑里，有 379 个样本出现 Spill decoder，那个 390 MiB 的 claim 约 1 s 内满足。

**建议**

- 保留机制。
- 放置改为固定规则，比如"Resident KV 在不取专家存储的前提下放不下，就选 Spill"。这是建议。
- 转换本来就是"从大到小"的固定顺序，只需去掉 cost 字段。
- 可改进的杠杆：
  - **热块缓存 floor 按最坏情况定**：8 行 × 512 块 × (lanes + 1)。要缩小它，必须先给 `resolve` 加回退：miss 超过空闲槽时它会抛 `logic_error`（`KvBlockCache.cpp:269-272`）。安全下限没有测过。
  - **放置对 Spill lane 多估了 195 MiB**：它假设保留 640 个 Full 页，实际只有 128 个（`MemoryPlan.hpp:344-348`）。

### 3.9 KV 层级与 WriteBehind

**页与 extent**

- 页形态 `kv::PageForm{Full, Index, Slim}`（`runtime/ops/PagedKv.hpp:56`）。
- extent 池 `KvPool`：`acquirePages`（58）、`movePages`（165）、`reclaimEmptyExtents`（211）、`compactExtent`（317）。
- 窗口裁剪到 Slim：`Cache::trimWindows`（3277）。

**SSD 层**

- `KvTier`，写入类别 `WriteClass{Reclaim, Persist, Spill}`。
- `BasicKvPageTier`：64 路写，其中 1/4 给 persist、1/2 给 spill；4,096 路恢复。
- 槽文件 `runtime/model/SlotFile.*`。
- 回收：`Cache::demoteKv`（3550）、`reclaimKvLeaf`（2774）、`pollTierTransfers`（3739）。

**WriteBehind**

- 常量：`kDelayMilliseconds = 10 s`，`kMinimumTokens = 2048`。
- 作用：提前把最新的恢复点拷到 SSD。
  - 持久层总是拷。
  - 临时层只在有"拷贝需求"时拷。需求由 `Engine::copyDemandBytes`（`EngineMemory.cpp:278-305`）给出，它基于 `ledger.copyDemand()` 的 60 s 视野。
  - 有 claim 在等时不拷。
- 来历：
  1. `ee58d77d`：在需要之前写出，用 5 分钟窗口；
  2. `e30c0b50`、`7512380b`：改成按内存需求；
  3. `7d31d72e`：删掉时间窗。
- 测量：M6 上带提前拷贝的树每跑读 3.9-7.9 GB 的 KV，不带的读 0.1-1.6 GB。

**#644 相关工作（`e805f601`）**

- 队列的驻留集只放持有请求页的 extent。
- 提交信息给的测量：
  - 8 GB 空闲的 Addressed buffer 让 Studio 上的 Flash-Next decode 周期慢 3.5-5.5 %；
  - 接近上限的 8.4 GiB KV 池慢 45 %。
- 这两个数字没有原始数据，提交信息自己也写了"未证明能修复 #644"。
- `6776d9cd` 在 `/status` 增加了 `kv.extents_active` 和 `kv.active_bytes`。

**有毒叶子泄漏的修复（`9077b419`）**

main 上就有这个泄漏，是 cache-property 用种子 9454 发现的。

**待查**

Studio 18G 的 `disk_read_per_work` = 1.452，FAIL（§8）。有一条线索，**未核实**：长提示词跑 Spill 的树通过历史流读取的 KV 层字节多 5-60 倍，因为预填充每个 chunk 都要重读溢出的历史（64K 时约 767 MB）。

**建议**

保留。`copyDemand()` 的 60 s 视野换成固定常量。

### 3.10 弹性专家存储与池

**main 已有的（#640、#645）**

- **分块。**
  - floor chunk 占 0.85 的槽，其余是约 1 GiB 的弹性 chunk。
  - kernel 通过编码时写入的地址表访问 chunk。
  - 驻留方式：每个 encoder 用 `useResources`。**不能**用队列驻留集，那会让 Studio 的 decode 慢 10 倍。
  - chunk 数的代价（Studio，17 GiB 存储）：36 个 512 MiB 的 chunk 让流式 decode 慢 2.3 %，9 个慢 0.6 %，1 个不慢。
- **弹性。**
  - ARC 淘汰最冷的专家，命令之间用 memcpy 压实存活的专家。
  - 扫描检查保证没有槽表指向已归还的槽。
  - `--expert-memory SIZE` 设 floor，默认取存储的 0.85。

**1.6.0 加的**

- **池接口。**
  - `model::ElasticPool`（`ElasticPool.hpp:61-72`）：`state`、`value`、`release(bytes)`、`grow(admit)`；`PoolState` 在 11-29。
  - 适配器：`ExpertStorePool` "experts"（`ModelPools.hpp:90-138`）、`HotBlocksPool` "hot_kv"（147-181）。
  - 运行时池 `kv_staging`、`vision_arena`、`embeddings` 都是 `rebuildOnly`（`Runtime.cpp:526-704`）。
- **存储操作。**

  | 函数 | 位置 | 行为 |
  |---|---|---|
  | `releaseAboveFloor` | `ExpertStore.cpp:1453` | 每次归还一个 chunk |
  | `releaseToFloor` | 1461 | 一次退到 floor，用于 Critical |
  | `growTowardCap` | 1483 | 每次取回一个带空槽的 chunk |
  | `release` / `restore` | 1383 / 1411 | 空闲释放与恢复 |
- **启动和恢复。**
  - 启动时存储建在 floor，在 Ready 之前长到目标（`ca20c739`、`d4b62d90`）。
  - 恢复只带回 floor（`beb1ca92`）。
  - 工具（runtime oracle、decode-profile 等）里，`--expert-memory SIZE` 同时设 cap 和 floor，存储固定在 SIZE（floor = cap = SIZE，`9fce7d5c`）。之前工具的存储从槽位的 85 % 起步、不会增长，给 8,413M 只拿到 7,210 MiB。
  - **用户可见的变化：** 服务端的 `--expert-memory SIZE` 在 1.5.0 里是"专家用这么多内存"，从 main 起变成"专家至少保留这么多内存"（floor），存储可以长到上限、再退回 floor。升级说明要写（§6.7）。

**每模型 floor 没有接通**

- 钩子 `StreamedMemoryInputs::expertFloorShare`（`StreamedMemoryPlan.hpp:53`）**没有调用者**（`RuntimeResources.mm:791-811`），所以所有模型的 floor 都是 0.85。
- 现在要 floor = cap，只能传 `--expert-memory <cap>`。终测的 `cand-floorcap` 单元就是这么跑的（`matrix-1.6.0.json`，`--expert-memory {experts_cap}`）。

**回长规则的演变**

1. #645：30 s 冷却期、2 GiB host 余量、Normal 压力、没人在等。
2. `a5570f52`：回长失败后退避时间翻倍，最长 4 分钟。
3. `277f3e86`："exchange"。
4. M3a：账本按价值决定增长。
5. A30 系列。
6. `2b326a37`：有请求在等时不向空闲空间增长。

**测量**

| 场景 | 收缩 / 回长 | 说明 |
|---|---|---|
| Studio 18G | 每跑 5.2 次 / 6.8 次 | |
| M5 Pro | 每跑 3.0 次 / 4.8 次 | |
| M6 value | t = 42 s 收缩 2 次，之后 14 分钟回长 0 次 | §5.4 |
| M6 floorcap | 不收缩 | 吞吐比 value 高 8.8 % |

其他事实：

- 35B（没有 HiSparse）在 18G 下只有 55-64 % 的专家驻留，存储是它唯一的杠杆。
- 专家内存每 GiB 的价值：24 GB Mac 上 decode +11-23 %，36 GB 上 +2-3 %。
- 压实约 1 GiB 的 chunk 需要 59.7-92.9 ms。

**建议**

保留弹性。收缩和回长改用 §12.3 的固定规则，并接通每模型 floor。

### 3.11 StreamedMemoryPlan

- **作用。** `planStreamedMemory`（`StreamedMemoryPlan.hpp:115`，`.cpp:283`）在模型加载前确定：
  - 专家存储的 cap 和 floor；
  - PLE 表；
  - spill floor（`planSpill`，`.cpp:99`）。

  调用方是 `RuntimeResources::create`（`RuntimeResources.mm:736-815`）。
- **来历。**
  - `db3d8b5b` 把它从 `MemoryPlan.hpp` 原样搬出（117 + 306 行）。
  - `ca20c739` 删除了 `a9881fd5` 的做法：按启动时的一次 host 采样定存储大小。单次采样会终身压低上限：一台 64 GB 的机器在 18 % 空闲时启动，TTFT 达到 772 s 和 1,720 s。
- **PLE。**
  - `--ple-memory auto` 的上限是 128K 提示词需要的行数，Flash-Next 为 237 MiB。这是 `a9881fd5` 保留下来的部分。
  - 24 GB Mac 上的 test-real 需要 `--ple-memory 237M`（`4ba625e3`）。
- **测试：** `engine-memory-plan`。
- **建议：** 保留。

### 3.12 WeightResidency

- **作用。** `runtime/engine/WeightResidency.{hpp,cpp}`（75 + 78 行）。
  - 空闲释放和恢复移到了引擎里（`5bccba22`）。
  - `restore` 每次只恢复一部分，期间可以回应帧。
  - 被拒绝的恢复每个 tick 重试（`39ec87f3`）。
  - 恢复只带回存储的 floor（`beb1ca92`），其余靠增长。
  - 引擎侧：`releaseIdleWeights`、`advanceWeights`（`Engine.cpp:2234-2306`）。
- **测量。**
  - 流式模型启动后恢复：0.86 s。
  - 驻留模型完整重载：6.17 s，分步进行，`max_tick_ms` 174 ms。
  - smoke b：空闲释放后第一个请求 6.6 s，专家在 10.4 s 后回来，`max_tick` 124.8 ms。
- **建议：** 保留。

### 3.13 状态与指标（schema 7）

**版本**

- `kStatusSchemaVersion` 从 6 升到 7。
- 线协议 `kProtocolVersion` 从 11 升到 12（`Protocol.hpp:21、25`；`server/protocol.py:23`）。

**新增字段**

| 字段 | 内容 | 来历 |
|---|---|---|
| `metrics.decode_gap_ms` | 记录每个 token 间隔 | `Status.hpp:150-165` |
| `metrics.ttft_parts_ms` | `{own_prefill, claim_wait, decode, other_prefill, rest}` | `3da98911` |
| `decode_gap_causes` | `{prefill_unit, claim_wait, ssd_read, pool_release, replan, other}` | `7160912b`、`9330b6f4` |
| `memory_governor` | `{accounts, reserved_bytes, owed, claims, host_telemetry}` | — |
| `scheduler` | `{bound_cuts, index_page_conflicts, pass_commands, abandoned_pass_commands, prefill_units, contended_prefill_ms, replans, estimate}` | — |
| `loop.tick_host_ms` | — | — |
| `kv.extents_active`、`kv.active_bytes` | — | `6776d9cd` |

- `/metrics` 由 `server/metrics.py` 映射，有双向合约测试。
- `status_golden.json` 只能重新生成，不能手改。

**删除字段**

`kv_spill.history_stream.contended_commands` 和 `kv_spill.history_stream.decode_gap_ms`。后者是旧的盲指标（§5.1）。

**测量**

TTFT 分解和间隔原因正是诊断冻结、诊断短提示词 TTFT 用的工具。

**注意**

- 间隔分类器先判 `prefill_unit`，再判 `claim_wait`（`Engine.cpp:532-537`），所以预填充间隔里的 claim 等待被藏起来了。
- 分类阈值用的是 `scheduler_.bound()`，删掉代价模型后要改成固定的 G。

**建议**

保留诊断字段。删除：`value_per_gib*`、`scheduler.estimate`、`splash_scheduler_prefill_estimate_error_p95_ratio`、`decode_gap_causes.prefill_predicted_ms`。

### 3.14 kernel 选择与校准

**重构（行为不变）**

- `fcca1921` 新建：
  - `runtime/ops/KernelCalibration.hpp`：473 行，每个测量常量都标了 GPU 类别和出处；
  - `KernelSelection.{hpp,cpp}`：81 + 294 行，含 `selectProjection`、`selectFloatTile`、`selectMoe`、`splitsFor`。
- 它从 `LinearGguf.cpp` 搬出约 460 行。`DeviceCapabilities::gpuClass()` / `plannedGpuCores()` 成为唯一的设备描述。
- `2c37a597` 加了选择 golden：FNV-1a，3,537 行生成文件。重构前后逐位一致。
- `2a5ebfb1`：Flint 跑在 Qwen3.5 目标上，删除 `FlintTarget`。
- `2dbf78aa`：注意力按 `kv::Layout::reads` 选择。

**有测量的规则改动**

| 提交 | 改动 | 测量 |
|---|---|---|
| `449b4a8f` | Apple9 ≤10 核：预填充 chunk 最多 64 行走 staged | 快 12-22 % |
| `5bf60946` | 同类 GPU 每 64 页切分 | — |
| `36617147` | 保留 Apple9 寄存器 tier 3 | — |
| `f658e7c8` | 记录 M4 的边界和 wave 规律 | — |
| `0e5e0d21` | 融合投影的预填充 chunk 任意高度一次 dispatch | Studio 8-32 行快 16-56 %；M4 快 3-11 % |

**没做的**

- K3（MoE 每块格式）：在任何机器上都不改变计划。
- K4（Apple9 >10 核的 MoE tile）：没有硬件可测。

矩阵里的三台 Apple10 Mac（M6、Studio、M5 Pro）上 kernel 计划不变。kernel 着色器只有琐碎改动，涉及 6 个文件。

**建议**

保留。#653（20 核 GPU 的 M16 tile）需要改写成校准表里的一行（§6.3）。

### 3.15 服务端与 App

**监督（C 轨道，`4077e079`）**

- **Ready 带能力。** Protocol 12 的 Ready 带 `vocabularySize`、`stopTokens`、`maskRows`。
- **新的 StartFailed 帧 `0x0109`。** 携带 `StartShortfall{memoryBound, required, budget}`，类别包括 `StartFailureKind::CommandMemory`。
- **重启策略。** 退避 0、5、10、20、40、60 s，不再永久停止。只有以下情况才停：Configuration、Device、CommandMemory、可执行文件缺失。
- **隔离。** 连续两次失败时都在跑的请求被隔离，返回 400 `request_quarantined`。
- **Retry-After。** 给出正在进行的启动的下一个界。
- **serve record。** `--serve-record` 写 `serve-PORT.json`。
- **新文件。**
  - `server/supervision.py`：187 行。
  - `runtime/Supervision.hpp`：64 行。常量：

    | 常量 | 值 |
    |---|---|
    | `kGpuWaitLimit` | 5 s |
    | `kCommandTimeout` | 120 s |
    | `kLoopStall` | 30 s |
    | `kFlush` | 6 s |
    | `kStop` | 15 s |
    | `kProbation` | 60 s |
    | `kStartRetryWindow` | 30 s |
  - `server/serve_record.py`：40 行。

**其他服务端修复**

| 提交 | 修复了什么 |
|---|---|
| `6272a216` | 退避一直翻倍，连续 1,026 次失败后 float 溢出，恢复线程死掉。现在停在上限 |
| `54b5eb52` | 资源装配时 GPU 内存不足归为 CommandMemory，不再重试。之前服务端会无限重启 |
| `38eaccfe` | 严格参数 union 的 JSON 语法原来按平方增长 |
| `15ada069` | Retry-After 按预期重启时间给出，最多 59 s |
| `62249dcf` | 分词缓存按上下文长度定大小，262K 时约 25 MB |
| `4413e356`、`1ba6d3b0`、`fdd2a67b`、`01bdf96b`、`b2bea9dd`、`2b5cc832`、`cd3f1c84`、`e05600e5`、`1930f937` | 其他修复 |

**带进来的 PR 内容**

- **#648 思考预算：** `ab1562e0`、`7457cbb3`。后续 `2cc42cf1` 修了一个 P1：`frontend.py` 会清空思考预算。
- **#649 PDF worker 沙箱：** `286c2de3`、`a7e3cf0d`、`00a7221c`。其中 `a7e3cf0d` 堵住了通过 `kern.procargs2` 泄漏 `SPLASH_API_KEY` 的路径。后续轮次 `1ac3c5dd`。
- **#651 App：**
  - 终端与日志抽屉（SwiftTerm 1.20.0）、平滑流式、工具栏和侧栏、有界的设备检查、停止 App 自己启动的服务、图片边框、强调色；
  - 后续轮次 `3ddb6985`：抽屉的 SIGHUP/waitpid、非美式键盘、遗留服务。

**App 侧文件**

- 新文件：`Drawer.swift`（406 行）、`WebPage.swift`（217 行），以及 `Transcript/page.js`、`markdown.js` 等。
- serve record 的措辞：`8d189c9f`。
- 测试：DrawerTests、MainViewTests、FakeRuntime。
- App 增长约 +2.9k 行，基本都是 #651。

**测量**

smoke b 的 kill -9：引擎在 0.54 s 后退出，11.0 s 后重启；恢复后的 TTFT 为 1.84 s，12,088 个 token 中 12,032 个命中缓存。

**建议**

保留。注意 #654（Stop 帧）需要线协议 v13，因为 v12 已被 1.6.0 占用。

### 3.16 其他保留的运行时改动

| 提交 | 内容 | 测量 |
|---|---|---|
| `fd6e6952` | 稠密预填充注意力按 chunk 看到的历史切分 | M5 Pro 35B 的 warm-suffix TTFT：66.7K −7.6..−9.9 %，134K −10.4 %；Studio 35B −4.7 %；冷预填充不变；27B 不显著 |
| `f4236596`、`759a7b62`（#621） | 33-64 行的 chunk 用 64 行 staged tile | 33-64 行 0.48 ms（之前 33 行要 0.58 ms）；27B 投影快 6-43 % |
| `87382087` | near-ties：长度相近的提示词里最老的领先 | 规则本身后来被 S 轨道的排序替换；这个树作为参照 `ntie` 保留 |
| `42f998f5` | 回退 `6b0370bf` 的 VerifyBudget 0.25 成本下限，回到 0.05 | Studio ABBA（FN + DFlash2，4 客户端）：0.984 / 0.990，区间 0.977-0.997，判 FAIL |
| `4e68300c`、`a33eb475`、`0f9c9538` | 截止时间统一在一处处理；请求离开时归还历史 staging；落地的页在准入前 spill | — |
| `9b73d34d`、`5ade28cb` 等 | ANE 行下限 | smoke b：86/86 个竞争单元都是 512 行（1,178-1,408 ms）；27B 预填充对 pass 在 60/120/180 s 时为 ×1.22/1.03/1.01，原来的 −17 % 消失 |
| `5174565b` | 前缀仍在缓存里的长追问，只需要还没持有的那部分 Resident KV。之前会被放成 Spill | — |
| `75674b20` | Start 可以释放空闲的 rebuild-only 模型缓存 | 修复一个 P1：图像请求要等文本请求的整段 decode |
| `2f6411fd`、`7e9063d2` | T7、T6，见 §3.7、§3.5 | — |

### 3.17 测试与工具

**测试拆分（`26522460`）**

- 删除 `engine_test.cpp`（16,291 行）和 `native_engine_loop_test.cpp`（2,817 行）。
- 拆成：
  - `engine_{requests,checkpoints,commands,memory,spill,disk,memory_rules}_test.cpp`；
  - `native_loop_{memory,weights,wire}_test.cpp`。
- 新增 `TestRegistry.hpp`：命名的测试注册表，取代原来包住 349 个调用的那一个 `try`。

**假件（F1）**

| 假件 | 说明 |
|---|---|
| `FakeMemory` | 按字节计的预算，就是 governor 读的那份 Metal 统计 |
| `FakeElasticPool` | 假的弹性池，带释放代价 |
| `TimingProfile` | 按机器和模型拟合录制的命令。误差：M6 最差 20 %、平均 8 %；Studio 10 % |
| `EngineHarness` | 事件时钟，每个 tick 做检查 |

55 个测试迁到了 harness 上（`afc4a3b2`）。

**engine-sim（`0e8ffe0e`，`engine_sim_test.cpp`，1,705 行）**

- 组成：真实的 `NativeRuntime`、`MemoryGovernor`、`MemoryControl`、`MemoryLedger`，加上假模型、假 host、假专家池。
- 用虚拟时钟，只跑 CPU，一个场景几秒钟。
- 场景：SIM-churn、SIM-churn-order、SIM-warning-M6、SIM-low-host、SIM-value、SIM-35b-churn、SIM-turns、SIM-turns-dense、SIM-long-prompt、placement。

**pressure-serving（`cf99a82f`，Metal）**

合成一个流式专家 MoE，走 `RuntimeBootstrap::start`。脚本化两轮 macOS warning/critical，3 分钟内跑完。

**服务基准（`8e330b47` 等）**

- 脚本：`dev/benchmarks/serving/{clients,run,matrix,stats}.py`，规格 `matrix-1.6.0.json`。
- `test-pressure-real`（`79fb8271`）：35B GGUF，18G，5 分钟 churn，已加入发布门。
- backend-benchmark 改用生产引擎配置。

**其他**

- Metal TSan 扩展到 expert-store、kv-block-cache、kv-history-stream。metal-backend 后来又移出了 TSan（`2e9b30eb`）。
- TSan 发现了一个真实的 MetalBackend 数据竞争（`260093d0`）和一处 autorelease 泄漏（`d70dff3e`）。
- kernel 选择 golden。
- 架构检查（`dev/tools/check_architecture.py`）：

  | 规则 | 内容 | 状态 |
  |---|---|---|
  | R1 | 引擎策略文件的标识符里不出现模型家族名（词表由 `install/families.py` 生成） | 已开 |
  | R2 | 引擎只包含白名单里的模型头文件 | 已开 |
  | R3 | 引擎策略不出现机制词汇（expert、indexer、ple、history stream 等） | **只在清理分支上开**（`d9585b59`） |

**有用的**

- cache-property 多种子扫描：找到了 main 上的有毒叶子泄漏和多个遍历 bug。
- pressure-serving。
- engine-sim 的行为场景。
- 服务基准的客户端间隔统计和有效性规则。
- TTFT 分解。

**没用或有害的**

- 钉价格、钉"拿了哪个 chunk"的测试。每次规则变化都要重写，修复 1 就重述了 10 个以上。
- 以录制价格回放为基础的标定测试：部分循环。
- `TimingProfile` 可能"证明"错误的信念，设计稿自己也承认了这一风险。

详见 §11、§12.4。

**仓库内文档。** `docs/internals.md` 自 1.5.0 起改了 +629/−206 行，其中"Admission and eviction"一节（第 1204-1634 行）是新人读架构的主要参考，但全文提到 value、price 或 cost model 的行有 47 处，删估计器时必须一起重写。`docs/server.md`（+298/−97）、`docs/api.md`（+66/−23）、`docs/validate.md`（+240/−20）也随 1.6.0 改过，前两者涉及用户可见的变化（§6.7），后者是服务矩阵和门的说明（G4 一行已过时，§2.2）。

### 3.18 从别的引擎借鉴的 9 项

11-01 晚做过几轮只读的同类引擎调研：Strata、oMLX、mlx-serve、vLLM、SGLang 和几个 MoE 专家卸载系统，另外深读了 Strata 的 PC 引擎，还看了 TensorFold。owner 批准了 9 项（21:05 批准 1-6，21:45 批准 7-9）；TensorFold 一项都没进。调研结论之一：各家的 prefill/decode 平衡都在"prefill 跑 1 s 欠 decode 约 0.5 s"附近，和我们的默认值一致。

| # | 内容 | 在最终头上的状态 | 删估计器后 |
|---|---|---|---|
| 1 | 专家存储的边际定价计数（最新 ghost 和驻留尾部的命中） | 已落地（`ExpertStoreStats::nearGhostHits`、`tailHits` 等） | 只服务价值模型，删 |
| 2 | host 内存遥测：`phys_footprint`、压缩字节、文件页、`vm_page_filecache_min` | 已落地（`memory_governor.host_telemetry`）。11-02 19:31 的完整性检查时还缺，之后补上 | 保留 |
| 3 | 六个回归测试 (a)-(f)，例如"被抢占的 spill 预填充从持久前缀恢复，不从 0 开始" | (a) 在 `engine_spill_test.cpp:2401` 附近，(d) 是 `testLaggingHostReadingsCauseNoSecondRelease`；其余逐项**未核实** | 保留行为部分 |
| 4 | 每个超界 decode 间隔归一个原因；每个账本决定一行日志 | 已落地（`decode_gap_causes`、账本日志） | 保留；去掉 `prefill_predicted_ms` |
| 5 | 提前拷贝的量的上界 | 是否按原文落地**未核实**（WriteBehind 现在按 `copyDemand()` 拷） | 换成固定常量（§3.9） |
| 6 | 代价样本清洗（只在样本被大量截断时才做） | 没做，只有 `CostModel` 的 `clamped` 计数 | 随代价模型删 |
| 7 | 每请求 TTFT 分成五段 | 已落地（`metrics.ttft_parts_ms`，19:31 时还缺） | 保留 |
| 8 | 矩阵报告预填充空洞比例、SSD 忙比例 | 已落地（`matrix.py` 的 `prefill_hole_share`、`ssd_busy_share`，只报告） | 保留 |
| 9 | 未记账内存 = `phys_footprint` − 各账户之和，随上下文增长超过约 256 MiB 就报告 | 已落地（`matrix.py`，`unbooked_growth_bytes` = 268,435,456）。M6 修复 1 的 A4 跑报告了 693 MB 的增长（§4.7），按原决定要在修复轮里把它记成一个固定账户或每 token 的 KV 项，**还没做** | 保留 |

---

## 4. 实验与测量结果

### 4.1 方法与读数约定

**churn 场景**

- 3 个客户端，每次回答 1,500 token，提示词逐轮增长。每 4 轮重发一次整个对话。
- 跑 15 分钟，每 2 s 采一次 `/status`。
- 首批请求按固定到达顺序发出：
  - A = (2,0,1)，B = (0,1,2)，间隔 0.6 s；
  - **只有同一顺序的结果才能相互比较。**

**work@900**

- 定义：t = 900 s 时的 prefill 行数 / R + 输出 token 数 / D。
- 系数：M6 R = 585、D = 46；Studio 和 M5 Pro R = 1000、D = 75。
- 由运行均值换算的例子：M6 上 1.5.0 为 364.6 × 900/585 + 22.7 × 900/46 ≈ 1005。

**区间规则**

- 取对数比值，做单侧 90 % 区间。
- 每台机器每次跑的噪声 σ：M6 2.6 %，Studio 1.55 %，M5 Pro 1.1 %。
- 判定：
  - 好的一侧边界越过容忍线 → PASS；
  - 坏的一侧边界越过 0 → FAIL；
  - 其他 → INCONCLUSIVE。
- 能分辨的差异：
  - G1 约 1.3-2.3 %；
  - G2 约 3 %；
  - G3 只有约 6-9 %；
  - 整个矩阵大约只能分辨 4-6 % 的差异。

**G7 单请求**

- 对比的是 **pass 树 `0bd50301`**，不是 1.5.0。
- 用 `dev/benchmarks/abba.py` 做 ABBA：差异不超过 max(2 %, 2 × 噪声) 判 PASS，噪声超过 5 % 判 INCONCLUSIVE。
- 和 1.5.0 唯一的直接单请求对比，是 smoke c 里单个 64K / 10K 提示词：没有差别。

**终测是压缩过的**

- churn 单元只跑 `cand` 和 `floorcap`，参照树（r150、main、pass、ntie、intp）用之前录制的数据。
- G7、single 和价格单元则把 pass 树一起跑：serve1 的 ABBA（pass-1/pass-2，重跑 pass-3 到 pass-6）、价格 pass（p-35b-price-pass-1、m6-27b-price-pass-1、Flint 价格 pass）和 s-single 的基线。
- 4 个压力单元每种顺序跑 2 次，其他单元 1 次。
- 录制的参照跑和终测的原始输出都只在测试机上，不在仓库里。接手人要么向 owner 要这些结果目录（按 §13.2 的标签找），要么在自己的机器上重录参照：每台机器每个压力单元每棵树每种顺序至少 2 跑，一跑 15 分钟，四棵参照树约需每台机器一整晚（估算）。

**两种 decode 读数，别混用**

- G2 decode 是按请求算的 decode 速率的几何均值。并发 lane 数不同时它有误导性：lane 多了，每个请求分到的 decode 就少。
- 比较 decode 应该看运行总量，即 900 s 内的 decode tok/s。
- 下文两种都给出。

### 4.2 终测前录制的参照数据

#### 4.2.1 M6，FN Q2_0 churn，默认内存

| 树 | Work A / B | 完成数 A / B | TTFT A / B (s) | 延迟 A / B (s) | 最大延迟 A / B (s) |
|---|---|---|---|---|---|
| 1.5.0 | 1023 / 995 | 17 / 17 | 141 / 120 | 169 / 174 | 290 / 325 |
| main（2 跑） | 962、922 / 920、894 | 15-16 | 97-102 / 92-100 | 184-194 / 193-198 | 306-364 |
| intpart（2 跑） | 941、913 / 927、883 | 14-16 | 94 / 93-95 | 186-196 / 191-206 | 292-392 |
| integration（2 跑） | 864、884 / 839、886 | 15 | 86-96 / 84-86 | 201-205 / 198-209 | 286-305 |
| R4（被否决） | 949 / 941 | 16 / 16 | 79 / 73 | 193 / 194 | 249 / 248 |
| KR（实验） | 927 / 963 | 16 / 16 | 83 / 76 | 193 / 191 | 259 / 278 |
| pass（实际是 `7e7b1a21`）† | 905 / 908 | — | 69 / 82 | 190 / 194 | 261 / 257 |

**说明**

- † M6 上录制的 FN "pass" 跑用的构建是 `7e7b1a21`，不是 `0bd50301`。两者是同一个改动（"流式目标的所有 lane 都按命令分 pass"）的前后两轮，在 `Scheduler.*` 上差 +21/−12 行；它的 TTFT 和延迟按它自己那次比较的共同 tag 算（顺序 A 13 个、B 14 个），和上面几行不是同一组 tag。
- 1.5.0 重跑可复现：顺序 A 1023 → 1016。
- integration 掉分的原因：Resident KV 上限从 1514 降到 1368 MiB，第一个 45K 提示词改跑 Spill，预填充时间从 72-79 s 变成 123-163 s。
- SSD 读取：
  - 所有树每 900 s 读 3.77-4.02 TB 专家，至少 4.2 GB/s。
  - KV 层的 `disk.read_bytes`：1.5.0 为 0.12-1.59 GB，pass 为 7.38-7.55 GB（历史流造成）。
- 1.5.0 固定存储值多少：
  - 它有 5722 个槽，floor 时的 main 只有 4856 个，多 1.18 GiB。
  - decode 每 token 读 97.6-101.5 MB，main 为 123.4-128.1 MB。
  - 折合 M6 work +6.0-8.5 %，即每 GiB 1.1-3.6 ms/token。

**M6 TTFT 测试（G5 的参照）**

N 个并发请求，共享 8K 缓存前缀，各带约 300 token 后缀。单位 ms，N > 1 时为中位 / 最大。

| 树 | N=1 | N=2 | N=4 |
|---|---|---|---|
| 1.5.0 | 3043 | 4979 / 7154 | 8753 / 14829 |
| main | 3061 | 4541 / 6713 | 8641 / 15287 |
| passes-final | 3065 | 4510 / 6976 | 5152 / 5247 |

#### 4.2.2 Studio 18G

| 树 | Work A / B | 完成数 A / B | TTFT A / B (s) | 延迟 A / B (s) | 最大延迟 (s) |
|---|---|---|---|---|---|
| 1.5.0 | 906 / 1012 | 13 / 14 | 175 / 136 | 250 / 226 | 484 / 544 |
| main（2 跑） | 853、814 / 972、966 | 11-13 / 14 | 197-205 / 140-141 | 274-305 / 238-241 | 438-754 / 548-565 |
| intpart（2 跑） | 972、984 / 1009、1004 | 14 / 15 | 198 / 165 | 230-232 / 208-209 | 342-358 |
| integration | 857 / 928 | 13 / 14 | 217 / 176 | 272 / 229 | 467 / 503 |
| R4 | 1028 / 1059 | 15 / 15 | 165 / 142 | 229 / 207 | 377 / 422 |

当时的"decode 间隔"盲指标给出 1.5.0 为 2.9-3.2 / 4.4-4.6 s。**这个数是错的**，1.5.0 实际冻结了 295-412 s（§4.6）。

#### 4.2.3 Studio ample

| 树 | Work A / B | TTFT A / B (s) | 延迟 A / B (s) | 最大 decode 间隔（盲指标） |
|---|---|---|---|---|
| 1.5.0 | 970 / 1218 | 166 / 131 | 242 / 185 | 132.6 s / 58.7 s |
| main | 1027 / 1212 | 156 / 143 | 200 / 179 | 4.1 s |
| integration | 1133 / 1073 | 147 / 188 | 179 / 217 | 3.3 s |
| near-ties | 1109 / 1235 | 148 / 148 | 176 / 176 | 2.6 s |

#### 4.2.4 M5 Pro 18G，FN IQ3_S

| 树 | Work A / B | 完成数 A / B | TTFT A / B (s) | 延迟 A / B (s) |
|---|---|---|---|---|
| 1.5.0（4 跑） | 344、339 / 524、527 | 5 / 8 | 322-324 / 308-309 | 580-589 / 552-558 |
| main（2+2 跑） | 330-337 / 534-535 | 5 / 8 | 324-335 / 293-296 | 581-596 / 540-545 |
| integration | 354 / 517 | 6 / 8 | 464 / 307 | 576 / 431 |
| near-ties（2 跑） | 516-518 / 508-517 | 8 / 8 | 259-263 / 307-324 | 391-395 / 431-446 |
| R4 | 549 / 559 | 8 / 8 | 215 / 244 | 413 / 404 |

M5 Pro 的 1.5.0 基线是 4 跑（11-01 晚补跑的 p150a/b、a2/b2；最早那组被重启打断）。和 M5 Pro 上所有计时跑一样，它们在空闲机器规则下都带 void 标记，owner 裁定照算（§4.3.6）。

#### 4.2.5 流式专家下一条预填充命令的代价

这就是分层 pass 要摊薄的固定成本。

| 行数 | M6 | Studio 18G |
|---|---|---|
| 512 | 2.73 s（n=271） | — |
| 1024 | — | 2.23 s（n=44） |
| 2048 | 4.6 s（拟合） | 2.95-2.99 s（n=600） |
| 4096 | 7.0 s（拟合） | 3.68 s（n=68） |
| 8192 | 11.98 s（n=15） | 4.97 s（n=127） |

M6 的拟合是 2.11 s + 1.204 ms/行。一条命令 2.9-11.7 s，其中 host 等待只占 0.65-1.7 s。约 300 行的 Flash-Next 命令不管几行，都要读约 16-17 GB 专家。

### 4.3 终测（`a7dac91a`，11-03 12:06 起，四台机器同时）

#### 4.3.1 M6：m6-fn-churn（FN Q2_0，15 分钟，value 4 跑 + floorcap 4 跑）

**900 s 运行均值**

| 树 | 预填充 rows/s | decode tok/s | 完成数 | 相对 1.5.0（预填充 / decode） |
|---|---|---|---|---|
| 1.5.0 | 364.6 | 22.7 | 16.3 | — |
| main | 329.7 | 21.6 | 15.7 | −9.6 % / −4.8 %（计算） |
| 1.6.0 value（默认） | 336.2 | 21.7 | 16.0 | **−7.8 % / −4.4 %** |
| 1.6.0 floor = cap | 372.5 | 23.2 | 16.5 | **+2.2 % / +2.2 %** |

**排队情况（全部请求）**

| 树 | 有请求等内存的样本比例 | TTFT 均值 | TTFT p90 | 延迟均值 |
|---|---|---|---|---|
| 1.5.0 | 16.4 % | 138.7 s | 266.4 s | 191.2 s |
| main | 4.2 % | 111.6 | 240.7 | 202.5 |
| 1.6.0 value | 0.3 % | 111.5 | 243.3 | 201.7 |
| 1.6.0 floorcap | 20.0 % | 131.6 | 242.4 | 190.4 |

**判定（17:05，4 + 4 跑）**

| 门 | 值 | 结论 |
|---|---|---|
| G1 work 对 r150 | 0.936 [0.921, 0.952] | **未过**。区间全在 1.0 以下；5 % 容忍要求别处有明确收益，而 G2 latency 同时 FAIL |
| G2 latency | 1.126 | FAIL |
| G2 decode（按请求） | 0.566 | FAIL。按请求的指标有误导性；按总量是 −4.4 % |
| G3 TTFT 对 main | 0.963 | INCONCLUSIVE |
| B17：floorcap 对 value 的 work | 1.088 [1.068, 1.108] | 结论 FLOOR = CAP |
| B17：floorcap 对 value 的 latency | 0.890 [0.865, 0.915] | 结论 FLOOR = CAP |
| floorcap 对 1.5.0 | work 约 +1.8 %，latency 约 1.00 | — |
| G4、G8 | job 01-03 的 `matrix.py check` | PASS |

间隔：

- value：p99 0.81-0.82 s，最大 0.88-0.91 s；
- floorcap：p99 0.61-0.74 s，最大 0.70-0.92 s。

**value 为什么输了**

- 存储起步是 3/3 个 chunk，cap 8.22 GiB。
- t = 42 s 时归还 2 个 chunk，降到 0.85 的 floor（7.04 GiB）。之后 15 分钟**再没回长**：收缩 2 次，回长 0 次，期间压力为 Normal。
- 有请求等内存的样本：value 的 A1 为 578 个中 0 个，B1 为 581 个中 3 个；floorcap 的 A1 为 596 个中 152 个。
- 结论：收缩买来了并发（TTFT 95 s，floorcap 112 s），代价是 decode（按请求约 16 tok/s，floorcap 约 26）和吞吐（−9 %）。
- 存储小了 1.18 GiB，Flash-Next 在 24 GB 机器上就跌下了工作集的悬崖。
- 根因见 §5.4。

**13:35 的早期对比**

每种顺序各 1 跑 cand，对比录制基线，用共同 tag 的均值：

| 树 | work@900 | TTFT（s） | 延迟（s） |
|---|---|---|---|
| 1.5.0 | 约 1005 | 约 123 | 约 178 |
| main | 约 930 | 约 92 | 约 189 |
| value | 941（G1 0.936 [0.913, 0.960]） | 95 | 192 |
| floorcap | 1028（比 1.5.0 +2 %） | 112 | 174 |

这一组和上面 4 跑的"全部请求"均值口径不同，不要混用。

#### 4.3.2 M6 其他单元

| 单元 | 结果 |
|---|---|
| 11 m6-27b-churn A1 | G4 PASS：最大间隔 2.78 s，p99 1.73 s；上界 2.6 s，来自 ANE 行下限 |
| 12 m6-27b-churn B1 | 17:05 时还在跑，G10 等它；结果没有记录 |
| 09 m6-fn-ttft（G5） | 按队列顺序应已跑完，但**没有记录结果** |
| 13 m6-single 27B | 17:30 仍在跑；结果没有记录 |
| 14 m6-single 35B、19-22（空闲时的 cand / r150 / main，kill） | 搁置在修复后的队列里 |
| 15-18 m6-serve1（G7） | `a7dac91a` 上**没跑**。`3874a2f1` 上的结果见 §4.7 |

#### 4.3.3 Studio：s-fn-churn-18g（FN Q2_0，18G）

**900 s 运行均值**

| 树 | 跑数 | 预填充 rows/s | decode tok/s | 完成数 | TTFT 均值 / p90 (s) | 延迟 (s) | 每命令行数 | 每预填充行专家 MB |
|---|---|---|---|---|---|---|---|---|
| 1.5.0 | 3 | 803 | 18.1 | 13.3 | 166.5 / 349 | 248 | 3702 | 5.1 |
| main | 4 | 776 | 16.9 | 13.0 | 170 / 334 | 262 | — | — |
| intpart | 录制 | 841 | 19.6 | — | 169 | 211 | 4402 | 4.5 |
| **1.6.0** | 4（有效） | **1016** | **23.0** | 16.5 | **132.8 / 268** | **189** | **7313** | **3.1** |

1.6.0 对 1.5.0（计算）：

- 预填充 +26.5 %，decode +27 %，完成数 +24 %；
- TTFT −20 %，TTFT p90 −23 %，延迟 −24 %；
- 每行专家字节 −39 %。

1.6.0 对 main：预填充 +31 %，decode +36 %，TTFT −22 %，延迟 −28 %。两次因后台进程被判 void 的跑，数字和上面一样。

**判定（16:50）**

| 门 | 值 | 结论 |
|---|---|---|
| G1 work 对 r150 | 1.244 | PASS |
| G2 latency | 0.841 | PASS |
| G2 decode（按请求） | 0.964 | INCONCLUSIVE（按总量是 +27 %） |
| G3 TTFT 对 main | 0.713 | PASS |
| SSD `expert_read_per_work` | 0.724 | 没问题 |
| SSD `disk_read_per_work` | **1.452** | **FAIL**，原因没查（§8） |
| G4b | 4 次有效跑都超出它的窄上界 | 只报告 |
| G8 | 全部 | PASS |

**每次跑**

| 跑 | 有效性 | G4b 客户端 p99 / 上限 (ms) | decode_gap 最大（全部间隔） | 其他 |
|---|---|---|---|---|
| A1 | 有效 | 1029 / 965 | 1.231 s | 2 个 claim 等待间隔，合计 1.215 s；最老 claim 353 ms |
| B1 | void（后台进程 114.6 % CPU） | 784 / 721 | 1.149 s | — |
| B2 | void（116 %） | 857 / 785 | 1.163 s | — |
| B1-2 | 有效 | 762 / 759 | 1.035 s | 4 次孤立的收回 |
| B2-2 | 有效 | 823 / 772 | 1.150 s | 3 次孤立的收回；最老 claim 2,509 ms |
| A2 | 有效 | 未记录 | 1.096 s | — |

**其他事实**

- 超界间隔的原因（A1/B1/B2）：主要是"history wait"（548 / 181 / 115），然后是 replan、压实、存储改大小。
- 客户端间隔全部 ≤ 2.5 s，98.7-99.9 % ≤ 1 s。
- 每跑存储活动：收缩 5.2 次，回长 6.8 次。

#### 4.3.4 Studio：s-fn-churn-ample（不限内存，cand A1、B1）

**900 s 运行均值**

| 树 | 预填充 rows/s | decode tok/s | 完成数 | TTFT / p90 (s) | 延迟 (s) |
|---|---|---|---|---|---|
| main | 955 | 21.7 | 16 | 154 / 325 | 194 |
| ntie | 1002 | 22.5 | 16.5 | 158 / 310 | 186 |
| **1.6.0** | **1388** | **28.3** | 20 | 131.5 / 235 | 160 |

**结论**

- G6 对 ntie：work 1.353、latency 0.804、TTFT 0.729，PASS。
- **B1 的 store line FAIL：** 一个回长的单元在之前已有一次收回之后又被收回，按 `store_churn` 规则算震荡。
- 最大 decode 间隔 0.682 s（A1）/ 0.731 s（B1），客户端间隔全部 ≤ 1 s。
- 这个单元的 SSD 计数（`disk_written_per_work`）结果没有记录。

#### 4.3.5 Studio：单请求、G7、G10 及其他

| 单元 | 结果 |
|---|---|
| s-serve1 第 1 组 | 2048 和 14336 PASS；**300 token 的 TTFT +30 %，decode +4.1 %，判 INCONCLUSIVE**（噪声 7 % / 20 %） |
| s-serve1 第 2 组（空闲机器重跑） | **确认真回退**：300 token 的 TTFT 对 pass 为 +26.2 %，约多 250 ms（pass 约 0.98 s，cand 约 1.23 s）；300 token decode +2-4 %；2048 TTFT −0.4 / −3 %；14336 TTFT −0.2 / −8 %（17:15 的记录，每格两个数）。按单请求规则阻塞发布 → §5.3。和 §4.7 "修复前"一列的出入见 §4.10 |
| s-fn-cold、s-warm（G7，35B 66.7K / 134K） | PASS，差 0.1-2.3 % |
| s-single 27B、35B | PASS |
| s-single Flint | 无法测：**基线**（pass 树）的 backend-benchmark 在 short-transcripts 场景报 "prompt token is out of vocabulary"，cand 一侧根本没跑。对 r150 重跑也一样 |
| s-27b-ttft（G10） | N=4 0.958 PASS；**N=2 0.946 INCONCLUSIVE**；N=1 没记录 |
| G8 | 全部 PASS |
| 91 store-less spill | EXIT=5 "NOT RUNNABLE"：FN Q2_0 加 `--expert-memory off` 启动要 38.9 GB，Studio 的 Metal 预算只有 29.1 GB（M5 Pro 为 60.3e9 对 55.7e9）。我们手上没有一台机器能跑 |
| 80-82、90、92、95、96（第 1、2 阶段） | 80 = check-native-cpu、architecture-check、check-native-build、check-python；81 = check-app（swift test）；82 = test-sanitizers；90 = 恢复基准；92 = PDF 渲染；95/96 = 驻留和 A10 读数。都在 `a7dac91a` 上跑完（队列 16:50 结束），**结果没有读出记录** |

#### 4.3.6 M5 Pro（18G；FN IQ3_S 和 35B）

**有效性**

M5 Pro 的每次计时跑都没通过空闲机器的有效性规则（外来 CPU 占用、睡眠断言等），被工具标为 void。owner 裁定不作废、照算。分析的看法是这类干扰只会让 1.6.0 显得更差（这是推断，不是 owner 的理由）。这一单元没有正式的 analyze 判定。

**FN IQ3_S（p-fn-churn-18g），900 s 运行均值，cand 6 跑**

| 树 | 跑数 | 预填充 rows/s | decode tok/s | 完成数 | TTFT 均值 / p90 (s) | 延迟 (s) | 每命令行数 | 每预填充行专家 MB |
|---|---|---|---|---|---|---|---|---|
| 1.5.0 | 4 | 385 | 7.25 | 6.5 | 314 / 515 | 566 | 2104 | 14.9 |
| main | 4 | 391 | 6.82 | 6.5 | 308 / 542 | 560 | — | 14.8 |
| ntie | 录制 | 452 | 9.03 | — | 313 | 437 | 3614 | 9.4 |
| **1.6.0** | 6 | **568** | **10.87** | 9.0 | **246 / 431** | **361** | **7667** | **5.7** |

1.6.0 对 1.5.0（计算）：

- 预填充 +47.5 %，decode +49.9 %，完成数 +38 %；
- TTFT −21.7 %，延迟 −36.2 %；
- 每行专家字节 **−62 %**。

其他事实：

- 900 s 内预填充专家读取：1.5.0 5,157 GB，1.6.0 2,917 GB。
- 每跑最大 decode 间隔 1.37-1.90 s。
- 每跑存储活动：收缩 3.0 次，回长 4.8 次。
- M5 Pro 上没有 floor = cap 对照。

**35B UD-Q4_K_M（p-35b-churn-18g）**

这是唯一一个没有 HiSparse 的压力单元。

| 树 | 跑数 | 预填充 rows/s | decode tok/s | 完成数 | TTFT / p90 (s) | 延迟 (s) |
|---|---|---|---|---|---|---|
| 1.5.0 | 2 | 774 | 125.5 | 78 | 26.9 / 65.8 | 34.6 |
| main | 2 | 775 | 176.3 | 108 | 10.8 / 31.4 | 24.4 |
| 1.6.0 | **1**（被标 void） | 777 | 181.2 | 110 | 8.7 / 29.0 | 23.9 |

1.6.0 和 main 差不多：decode +2.8 %，TTFT −19 %。相对 1.5.0 的收益主要来自 main 在 1.6.0 之前的工作。

**没有结果的 M5 Pro job**

- 07-09（35B 其余顺序）、10-12（Flint 价格和 churn）、13-14（p-single）、15-18（p-serve1）、26（check-native-metal）：按队列顺序，它们在 job 27 开始（17:30 前）之前就在 `a7dac91a` 上跑完了，但结果没有读出，**未核实**。
- 27（test-real-fn）：17:30 时还在 `a7dac91a` 上跑。
- 28-47：搁置。

#### 4.3.7 mini-M4（16 GB，35B UD-Q4_K_M）

| job | 结果 |
|---|---|
| 10 cpu-list | 只有 engine-sim 两项失败；其余 84 条命令全过（engine 388/0、memory-ledger 113/0、ragged 89/0） |
| 11 test-sanitizers | 只有两项已知失败（ASan、TSan 都是）；0 个 sanitizer 报告 |
| 12 cache-property 种子 | PASS |
| 13 engine-sim-seeds（ASan + TSan） | 只有两项已知失败；0 个报告（TSan 跑了 57 分钟） |
| 50-51 mini-35b-churn（G10，对 pass） | work 1.693，TTFT 0.721，PASS；最大间隔 0.6 s。work 高出 69 % 的原因，来源里没有解释 |
| 30-33 mini-serve1 第 1 组（G7） | INCONCLUSIVE：基线噪声 40 % |
| 30b-33b 第 2 组 | 300 token 的 TTFT −10 / −13 %；**2048 token 的 TTFT +16.9 / +13.5 %**（约多 0.7 s），原因与 Studio 相同（§5.3）；14336 −2.2 / +2.4 % |
| 20 Apple9 Metal 套件、40-41 mini-35b-ttft | 没有记录 |

### 4.4 门判定汇总（`a7dac91a`）

| 单元 | G1 | G2 lat | G2 dec | G3 | G4 | G5/G6/G10 | G7 | G8 | SSD | 存储 |
|---|---|---|---|---|---|---|---|---|---|---|
| m6-fn-churn（value） | 0.936 未过 | 1.126 FAIL | 0.566 FAIL* | 0.963 INC | PASS | — | — | PASS | 未报告 | 未报告 |
| m6-fn-churn floorcap | 对 1.5.0 约 +1.8 % | 约 1.00 | — | — | PASS | B17：FLOOR = CAP | — | PASS | — | — |
| m6-27b-churn | — | — | — | — | A1 PASS | G10 未报告 | — | — | — | — |
| s-fn-churn-18g | 1.244 PASS | 0.841 PASS | 0.964 INC* | 0.713 PASS | G4b 只报告 | — | — | PASS | **disk_read 1.452 FAIL** | 孤立收回 4 + 3 次 |
| s-fn-churn-ample | — | — | — | — | 最大 0.68-0.73 s | G6 PASS | — | PASS | 未报告 | **B1 FAIL（震荡）** |
| s-serve1 | — | — | — | — | — | — | 300 token TTFT **FAIL（真回退）** | PASS | — | — |
| s-fn-cold、s-warm、s-single 27B/35B | — | — | — | — | — | — | PASS | — | — | — |
| s-single Flint | — | — | — | — | — | — | 无法测 | — | — | — |
| s-27b-ttft | — | — | — | — | — | N=4 PASS，N=2 INC | — | — | — | — |
| p-fn-churn-18g | 只有数字（§4.3.6） | | | | 最大 1.37-1.90 s | | | | | |
| mini-35b-churn | — | — | — | — | 最大 0.6 s | G10 PASS | — | — | — | — |
| mini-serve1 | — | — | — | — | — | — | 2048 token TTFT +13.5-16.9 % | — | — | — |

\* 按请求的 decode 几何均值。并发 lane 数不同时有误导性，见 §4.1。

修复后（`3874a2f1`）的 m6-fn-churn（value）：G1 1.004 PASS、G2 latency 1.020 INC、G2 decode 0.705 FAIL*、G3 0.938 INC、G4/G8/G9 PASS；m6-serve1 的 G7 全部 PASS（§4.7）。

### 4.5 收益归因

对比对象是"没有冻结"的中间树，单位为 900 s 运行均值。

| 机器 | 树 | 预填充 rows/s | decode tok/s | 每命令行数 | 每行专家 MB | TTFT (s) | 延迟 (s) |
|---|---|---|---|---|---|---|---|
| M5 Pro FN 18G | 1.5.0 | 385 | 7.25 | 2104 | 14.9 | 314 | 566 |
| | ntie | 452 | 9.03 | 3614 | 9.4 | 313 | 437 |
| | 1.6.0 | 568 | 10.87 | 7667 | 5.7 | 246 | 361 |
| Studio FN 18G | 1.5.0 | 803 | 18.1 | 3702 | 5.1 | 166.5 | 248 |
| | intpart | 841 | 19.6 | 4402 | 4.5 | 169 | 211 |
| | 1.6.0 | 1016 | 23.0 | 7313 | 3.1 | 133 | 189 |

读法（16:10 的归因；这是按与中间树的对比做的**推断**，不是隔离实验）：

1. **冻结修复贡献 +5-17 % 预填充。** ntie / intpart 里已经有了这部分，归因于 spill-wait 的"落地内存留给等待者"。但 ntie、intpart 和 1.5.0 的差别远不止 spill-wait：还有 main 的 #640/#645、回收梯、#644 的 KV 驻留、prefill splits；ntie 另有内存步骤 2-4 和 near-ties。
2. **pass 优先的调度再贡献一截。** 每命令行数约翻倍、每行专家读取约 −40 %，带来预填充 +21-26 %、decode +18-20 %、TTFT −21 %。

这就是 owner 结论（1）的证据。终测里没有一跑把估计器单独拿掉（§9.1 第 2 条：固定规则的基线从没测过），所以只能说**没有测到**估计器的贡献，不能排除它有贡献。

### 4.6 decode 冻结：1.5.0、main 和 1.6.0 在同样单元里的对比

旧指标 `kv_spill.history_stream.decode_gap_ms` 只记录"中间有流式预填充命令"的间隔，所以大部分冻结都没被看到（§5.1）。下表的"真实停顿"来自 `status.jsonl` 里每请求的 token 计数，以及服务端的客户端间隔直方图。

| 跑（树、单元、顺序） | 盲指标最大值 | 真实停顿 |
|---|---|---|
| main，Studio 18G，A（2 跑） | 112.2 / 109.9 s | 请求 3：110 / 108 s |
| main，Studio 18G，B（2 跑） | 4.9 / 4.9 s | **请求 2：416 / 418 s** |
| 1.5.0，Studio 18G，A（2 跑） | 4.6 / 4.6 s | **请求 3：295 / 305 s** |
| 1.5.0，Studio 18G，B | 4.4 s | **请求 2：412 s** |
| main，Studio ample，B | 2.4 s | 请求 2：143 s |
| 1.5.0，Studio ample，A | 132.6 s | 请求 3：139 s |
| 1.5.0，Studio ample，B | 58.7 s | **请求 2：275 s** |
| main，M5 Pro 18G，A（3 跑） | 403.0 / 424.8 / 419.0 s | 请求 3：415 / 435 / 434 s |
| 1.5.0，M5 Pro 18G，A（2 跑） | 403.4 / 417.9 s | **请求 3：414 / 430 s** |
| main、1.5.0，M5 Pro 18G，B（4 跑） | 3.8-5.7 s | 请求 2：207-215 s |
| main，Studio ample，A | 4.1 s | 没有 ≥ 20 s 的 |
| **1.6.0，14 次 cand 跑**（Studio 18G × 6、ample × 2、M5 Pro 18G × 6） | 0.68-1.90 s（统计全部间隔） | **没有超过 2.5 s 的客户端间隔** |

M6 上 1.5.0 和 main 的冻结情况**没有检查过**。owner 说的"结束了 110-430 s 冻结"，在 Studio 和 M5 Pro 上成立。

### 4.7 修复测试（`3874a2f1`）

**设置：** cand `3874a2f1` 对 pass `0bd50301`，同一会话内 ABBA，取中位数。"修复前"列是同样单元在 `a7dac91a` 上的结果。

| 机器 / 指标 | 修复前（`a7dac91a`） | 修复后（`3874a2f1`） |
|---|---|---|
| Studio FN Q2_0 serve1，300 token TTFT | +26.2 % | **+3.1 %**（992 对 962 ms，还差约 30 ms；运行间噪声 0.4-1.8 %） |
| Studio 300 token decode | +2.1 % | −0.5 % |
| Studio 2048 TTFT | −3.3 % | −0.7 % |
| Studio 14336 TTFT | −6.2 % | −9.1 %（pass 噪声大） |
| mini-M4 35B，2048 token TTFT | +13.5 %（cand 6.59 s） | **−9.1 %**（cand 5.27 s） |
| mini-M4 300 token TTFT | −13.2 % | −13.6 % |
| mini-M4 14336 TTFT | +2.4 % | +2.3 %（pass 30.1-32.9 s，噪声大） |
| mini-M4 cand decode | — | 约 28-29 ms/token，pass 为 43-97（噪声大） |
| **M6 FN Q2_0 serve1（G7，第一次跑）** | 没跑 | 300 TTFT −0.3 %，2048 +0.5 %，14336 −1.0 %；decode −1.1 / +0.1 / −0.5 %：**没有变化**，19:55 判 G7 全部 PASS、G8 PASS |

Studio 的"修复前"三格和 §4.3.5 里 17:15 记的第 2 组对不上，见 §4.10。

**M6 FN churn 的修复 1 确认跑**

value 默认配置，4 跑（A3、B3、B4、A4）。19:12 时的原始数字：

| 跑 | 预填充 rows/s | decode tok/s | 有请求等内存的样本比例 | 说明 |
|---|---|---|---|---|
| A3 | 370.7 | 23.32 | 18.0 % | 有效 |
| B3 | 359.3 | 22.30 | 3.6 % | 被 `matrix.py check` 判 void（有系统进程 CPU 占用超阈值），数字保留并标注 |
| 修复前的 value 跑 | 336.2 | 21.67 | 0-0.9 % | — |
| floorcap 跑 | 368-377 | 23.0-23.3 | 13-34 % | — |
| 1.5.0（§4.3.1） | 364.6 | 22.7 | 16.4 % | — |

- A3 和 B3 的均值是 365.0 / 22.81（计算），相对 1.5.0 为预填充 +0.1 %、decode +0.5 %，相对修复前的 value 为 +8.6 % / +5.3 %。
- 等内存的比例上升，和 floorcap 的情形相似。据此推断修复 1 后默认配置保住了存储、用一部分排队换了吞吐；但来源里没有这几跑的存储收缩、回长次数，这一点**未核实**。

19:55 的 `matrix.py analyze --build cand=src-784dbfc9`（= `3874a2f1`；B3 被 check 判 void，用 A3、A4、B4）：

| 门 | 值 | 修复前（`a7dac91a`） |
|---|---|---|
| G1 work 对 r150 | **1.004 [0.980, 1.029]，PASS** | 0.936 |
| G2 latency | 1.020 [0.987, 1.055]，INCONCLUSIVE | 1.126 FAIL |
| G2 decode（按请求） | 0.705 [0.578, 0.861]，FAIL（顺序 A 0.597、B 0.834） | 0.566 FAIL |
| 按总量 | decode 22.2-23.3 tok/s（1.5.0 22.7）；预填充 350-371 rows/s（1.5.0 364.6） | 21.7 / 336.2 |
| G3 TTFT 对 main | 0.938，INCONCLUSIVE | 0.963 INC |
| TTFT 对 r150 | 0.774 | — |
| `expert_read_per_work` | 对 r150 0.932，对 main 0.856 | 未报告 |
| G4、G8、G9 | PASS（间隔最大 0.81-0.86 s） | PASS |
| 顺序 B 的 TTFT 包络 | 2 个 tag FAIL（61.97 对 61.18 s，109.07 对 107.36 s） | — |
| 未记账内存 | A4 报告：增长 693 MB，报告线 268 MB（24K 上下文） | — |
| 等内存的样本比例 | 2.4-18 % | 0-0.9 % |

按 5 % 规则，吞吐和总量 decode 已经和 1.5.0 持平；G2 latency 和 G3 还不确定，G2 的按请求 decode 是误导性指标（§4.1）。未记账内存的增长要按借鉴第 9 项处理（§3.18）。

**`3874a2f1` 上的门（Studio）**

| 检查 | 结果 |
|---|---|
| `engine` | 389/0 |
| `memory-ledger` | 115/0 |
| `native-engine-loop` | 47/0 |
| `ragged-scheduler` | 89/0 |
| `cost-model` | 18/0 |
| `memory-governor` | 通过 |
| `engine-sim` | 只有两项已知失败 |
| `pressure-serving` | 2/0（`MTL_SHADER_VALIDATION=1`） |

**没跑的：** 完整 Metal 套件、M5 Pro、真模型单元。

### 4.8 终测之前的早读与 smoke 轮次（摘要）

| 时间 | 构建 | 结论 |
|---|---|---|
| 11-02 23:12 起 | `c47bec51`（early a，没有调度器修复，没有 M3b） | M6 FN A：work 952（对 1.5.0 −5.8 %），TTFT −46 %，延迟 +9 %，最大间隔 3.67 s（启动时 u 引起的停顿）。floorcap A：1016.5（+0.6 %）。A+B 合计：cand −4.4 %，floorcap +2.0 %。27B 对 pass：预填充行 −17 % |
| 11-03 01:52 起 | `972503f3`（early b，加了调度器修复） | FN A：work −4.7 %，TTFT −34 %，最大间隔 1.76 s。27B 发现 ANE 问题：0 个 pass 命令，`bound_cuts` 2,797 |
| 11-03 06:10-07:45 | `f0724a9d`（smoke b） | M6 10 项过 9 项（27B G4 的上界没考虑 ANE 行下限，后由工具修正）；mini 3 项过 2 项（16 GB 上的 runtime-resources，`6e08b1ad` 修复）。发现 smoke-b/F1（P1，不是 F1 测试假件轨道）：回长的 1,023 MiB 单元 8.1 s 后被 host claim 收回 |
| 11-03 10:37-11:10 | `23b84284`（smoke c） | M6 4/4 PASS：64K + 2 个 decoder 最大间隔 0.82 s；单个 64K 提示词 cand 93.30 s 对 1.5.0 94.57 s。发现 N1：272 MiB 的 host claim 拿走 1,202 MiB |
| 11-02 晚 | smoke a | 64K 服务检查答对了，但 decoder 每约 5.3 s 出现一次约 3.65 s 的间隔，220 个 pass 命令里 208 个被放弃（自我 supersede，§3.5）；M5 Pro 的 test-pressure-real 和 35B 18G churn 都报"60 s 内单元被收回" |

smoke b 其他通过项：

- kill -9 后 11.0 s 重启；
- 空闲释放与回长；
- 150K spill：TTFT 610.2 s，结果正确，decoder 最大间隔 0.944 s，spill 了 1.94 GB，0 次失败；
- 调度器修复：放弃数 0/12、1/40、2/51；
- 35B 2 分钟 churn 最大间隔 ≤ 0.41 s。

### 4.9 不确定、无法测、或没跑的

1. **Flint G7。** 基线的 backend-benchmark 拒绝了一个超出词表的提示词 token。要先修测量工具。
2. **无存储 spill（设计里列出的第 5 条行为变化：不建专家存储时 Flash-Next 也能 spill KV）。** 我们没有一台机器能用 `--expert-memory off` 加载 Flash-Next，只能靠单元测试。
3. **M5 Pro 的计时跑全部被标 void。** 35B 只有 1 跑 cand 被读出；Flint、p-single、p-serve1 在 `a7dac91a` 上跑了，结果没有读出。
4. **INCONCLUSIVE 的：**
   - M6 G3 0.963；修复后 G2 latency 1.020、G3 0.938（§4.7）；
   - Studio G2 decode 0.964；
   - s-27b-ttft N=2；
   - mini-serve1 第 1 组；
   - Studio s-serve1 第 1 组。第 2 组后来证明是真回退。
5. **跑了但没有记录的：**
   - M6：G5（m6-fn-ttft）、27B B1 和 G10、m6-single；
   - mini：35b-ttft、`a7dac91a` 上的 Apple9 套件；
   - Studio：第 1、2 阶段 job 80-82、90、92、95、96；
   - M5 Pro：07-18、26；
   - ample 单元的 SSD 规则。

   这些输出应该还在各台机器的作业日志和结果目录里，需要 owner 取出来；取不到就只能重跑。
6. **没在最终二进制上跑过的：**
   - M6 的空闲和 kill 单元（19-22）；
   - M5 Pro 正确性批次 28-47，含 122B；
   - 清理分支上的 test-real oracle；
   - `3874a2f1` 上的完整 Metal 套件；
   - 清理与修复合在一起的版本。
7. **请求结束后的回长：** smoke c 里没覆盖到，每次跑都在最后一个请求结束后 2-5 s 就停了。

### 4.10 来源之间的出入

- **M6 TTFT 有两种口径：**
  - 13:35 是每种顺序 1 跑的共同 tag 均值：123 / 92 / 95 / 112 s；
  - 14:20 是 4 跑全部请求的均值：138.7 / 111.6 / 111.5 / 131.6 s。

  不要混用。
- **G2 decode 判定**（M6 0.566 FAIL、Studio 0.964 INCONCLUSIVE）用的是按请求的指标。按总量分别是 −4.4 % 和 +27 %。
- **M6 value 跑时的压力：** 终测记录是 Normal；smoke c 在 `23b84284` 上看到的是引擎带状态 warning，从 42 s 起暂停增长。
- **Studio serve1 的"修复前"数字：** §4.7 修复前一列是 decode +2.1 %、2048 −3.3 %、14336 −6.2 %；§4.3.5 里 17:15 记的第 2 组是 decode +2-4 %、2048 −0.4 / −3 %、14336 −0.2 / −8 %。两者对不上，来源没有解释，两组照录。
- **早读 TTFT：** early a 的 cand A1，一处记 67 s（13 个 tag），另一处记 73.8 s；early b 一处记 84.9 s（14 个 tag），另一处记 82 s。

---

## 5. 根因调查

### 5.1 main 和 1.5.0 的 decode 冻结

**范围**

- 共 43 次 ≥ 50 s 的 decoder 停顿，分布在 19 次 churn 跑里。树是 main（`031a25a6`）和 1.5.0；机器是 Studio 和 M5 Pro；内存为 18G 上限或不限。
- 复核方按"token 数不变的段"计数，得到 45 段；把被一次挂起切成两半的段算作一段，就是 43。
- **每一个被卡住的请求都是 `kv_mode: spill` 的长提示词。**
- 1.5.0 冻结得和 main 一样严重：`Engine.cpp:1622-1705` 在两棵树里逐字节相同。

**为什么当初没发现：指标是盲的**

- 设计的基线写着"1.5.0 decode 间隔最大 4.4-4.6 s""1.5.0 ample B 58.7 s"。
- 这个指标是 `kv_spill.history_stream.decode_gap_ms`。main 和 1.5.0 里，它只在间隔内跑过**流式**预填充命令时才记录（`031a25a6:runtime/engine/Scheduler.cpp:640-645`；1.5.0 同样规则在 `Scheduler.cpp:642-643`）。旁边跑的是驻留预填充时，什么都不记。
- 被它藏起来的有：Studio 顺序 B 的所有冻结、1.5.0 在 Studio 18G 的所有冻结、M5 Pro 顺序 B 的冻结、1.5.0 ample B 的大部分冻结。
- main 顺序 B 的 416 s 饥饿在内部尝试记录 A57 里出现过一次，但没有进入基线数据，也没有进入设计的目标表。
- 1.6.0 改为记录每一个 token 间隔（`Status.hpp:150-165`），超界的间隔还标出原因（`Engine.cpp:524-549`）。

**机制（复核后修正过的版本）**

1. **热块缓存的 floor。**
   - Spill decoder 每条 decode 命令之前，都要预留热块缓存的 floor：390 MiB（`031a25a6:Engine.cpp:1658-1672` → `Runtime.cpp:3159-3164`）。
   - 这个 floor 是**全有或全无**的（`031a25a6:runtime/model/KvBlockCache.cpp:162-171` 的 `Impl::grow`），约等于 3.6-4 个 103.5 MB 的 KV extent。空闲空间从来没有这么多。
   - 一次落地只给 +103.5 MB，要连续约 4 次落地都没被别人抢走，decoder 才能拿到。
2. **拒绝之后的回收要等写盘。**
   - 被拒后，引擎驱逐缓存的 KV。有 SSD 层时，驱逐就是降级到 SSD，内存要等写落地才回来。
   - 调用链：`allocate` → `reclaimForState` → `Cache::reclaimForMemory` → `evictOne` → `reclaimKvLeaf`。
   - 有传输在途时，这一步报 `pending`。
3. **`pending` 同时堵死了两个泄压阀。**
   - 归还存储 chunk：只有 main 有，`if (!reclaimed.madeProgress && !reclaimed.pending && !forLater) releaseExpertChunk()`。
   - 挂起预填充：两棵树都有，门限是 `!entry.denial.pending`。
   - 30 s 的资源等待截止也从不触发，因为只要更早的 lane 有进展，它就重新计时。
4. **等待中的 decoder 不可见。**
   - `resourceWaitSnapshot` 只统计 `Phase::WaitingResources` 和 `refusedResources && !waiting`。
   - prepare 阶段的拒绝路径从不设置 `refusedResources`。
   - 于是新请求被插到等待者前面：sMNb 里，请求 3-7 都开始并完成了，而请求 2 等了 416 s。
   - 存储的增长检查同样看不见这个等待者。
5. **有两种模式，不是一种。**
   - **驻留预填充模式。** 包括两台机器上所有顺序 B 的冻结，以及 Studio 顺序 A 从 t ≈ 223 起的部分。特征：降级次数和 extent 分配同步飙升，write-through 不动。
   - **Spill 预填充模式。** 包括 M5 Pro 顺序 A 的 415-435 s 冻结。特征：decoder 旁边是一个正在 write-through 的 Spill 流式预填充。main 的回收把 Spill 的 write-through 算作"在回来路上的内存"。M5 Pro 上到底是什么让拒绝一直 pending，**没有查清**。
6. **冻结怎么结束的。**
   - main 的 10 次跑里，有 5 次最长停顿在一次存储归还后结束。
   - M5 Pro 顺序 A 的那次归还放行的是请求 2，请求 3 要等请求 2 完成才动。

**设计提出的三个原因，复核后的结论**

| 设计的原因 | 复核后 |
|---|---|
| 落地的内存被先重试的 lane 拿走 | **和数据一致，但属于推断。** main 和 1.5.0 都没有记录决策的日志行。"单这一条就足够"**没有支持**：持有落地内存的补丁（A53）只在回收梯树上测过，而单靠回收梯仍会停 104 s |
| `escalate` 的欠拷贝一级要等完整个 write-through | **对 M5 Pro 的冻结不能排除。** `escalate` 不在 main、1.5.0 或 `a7dac91a` 里 |
| 存储要等冷却期才回长 | **确认不是冻结原因。** main 的存储卡住是**空间**问题：引擎余量从没超过 465-548 MB，而一个 chunk 要 1023 MiB |

复核还发现了设计漏掉的三个原因：盲指标；看不见的等待者；挂起阀共用 `pending` 门限。

**1.6.0 用固定规则怎么结束它**（引用 `a7dac91a` 的行号）

1. **热块 floor 变成账本 claim。** `EngineMemory.cpp:548-561`（`reserveReads`，`ClaimClass::InService`）→ `EngineAllocate.hpp:75-170`。等待的 decoder 现在是 `claims.waiting` 里看得见的一项。
2. **decode 排在 prefill 前。** `ClaimRank` 的 `.decoding` 字段（`EngineMemory.cpp:846`）。
3. **落地内存为更高 rank 的等待者持有。** `MemoryLedger.cpp:224-255`；测试 `testLandingPaysItsOwnerFirst`。
4. **只在落地内存足以覆盖时才等，否则立即取，包括存储。** `waits(false)`（`MemoryLedger.cpp:761`）；测试 `testOneConversionDoesNotStallTheStore`。
5. **有请求的 claim 在等时，不向空闲空间增长。** `holdsFreeRoom`；测试 `testAPoolDoesNotGrowWhileAClaimWaits`。它取代了 main 的 `MemoryControl::growExperts`，正是后者造成了 M5 Pro 冻结中途的那次回长。

**终测证据**

| 跑 | `decode_gap_ms` 最大 | 客户端间隔 | 超 500 ms 且由 claim 等待造成的间隔 | 最老的 `claims.oldest_ms` |
|---|---|---|---|---|
| Studio 18G，6 跑 | 1.035-1.231 s | 全部 ≤ 2.5 s；98.7-99.9 % ≤ 1 s | A1 两次，共 1.215 s；其他跑 0 次 | 353 ms（A1）、2,509 ms（B2-2） |
| Studio ample，2 跑 | 0.682 / 0.731 s | 全部 ≤ 1 s | 未读 | 未读 |
| M5 Pro 18G，6 跑 | 1.372-1.896 s | 全部 ≤ 2.5 s | A1-2、B1、B1-2 各 1 次（0.75-0.84 s） | 0 |

**对 owner 结论的含义**

- 冻结数据只要求两件事：
  1. 落地的内存给 rank 最高的等待 claim；
  2. 在等的 in-service claim 看得见，于是没有池会长进它要的空间，新的请求也不会插到它前面。
- 只加了简单 spill-wait 补丁的树（intpart、ntie）保留了冷却期和 `escalate`，在同样单元里也**没有超过 10 s 的客户端间隔**。不过这些机器上的构建没有做哈希校验，intpart 的提交也没有确认，所以只是参考。
- **冻结数据不支持把"删除冷却期"或"存储的价值与回长机制"当作冻结修复。**

### 5.2 HiSparse Spill 在 Flash-Next 上：内存里留什么，从 SSD 读什么

**模型结构**

- 48 层：36 个 GDN 层（线性注意力，没有 KV）和 12 个注意力层（每 4 层一个，`(l+1)%4==0`）。
- 12 个注意力层全部是 QSA 块稀疏：indexer 有 4 个 128 维的头，选出前 512 个 4-token 块，即 2,048 个 token。
- 没有滑动窗口。
- 历史 ≤ 2,048 token 的行做稠密注意力。
- 默认 `--kv-format int8`、`--state-format fp32`。

**页形态（int8）**

- 一页 32 个 token。
- Full 页 423,936 B = 12 层 × 33,280 B K/V + 24,576 B 索引。
- Index 页 24,576 B，是 Full 页的 5.8 %。
- SSD 槽按版本 1 格式排列：先放索引，再放 4,160 B 的（层，块）记录，每条一次 8 KiB 读取。

**Spill lane 在内存里留什么**

| 项目 | 内存里的形式 |
|---|---|
| 36 个 GDN 层的状态 | 完整保留，2 个单元 × 110.4 MiB（fp32）。Spill 不碰它 |
| 第 0-63 页（稠密历史）和最近 64 个已提交页 | Full（写穿到 SSD，但仍保持 Full）。128 页约 48.75 MiB K/V |
| 拷贝还没落地的页 | 落地前保持 Full |
| 其他所有页 | **只留 Index 页**，K/V 只在 SSD 槽里 |
| 热块缓存（所有 lane 共享） | 390 MiB 起，每多一个 lane 加 195 MiB，最多 975 MiB |
| 预填充 staging（只在 Spill lane 预填充时） | 一层 × 整条序列：64K 时 65-73 MiB，256K 时 268 MiB |
| runway | Spill 提示词预填充或有 Start 在等时，留一个空的 Full extent（108.5 MB）；另有一个 Index extent（6 MiB） |

**转换的账**

- 64K 时 Resident KV 828 MiB；转成 Spill 后是 97 MiB 页 + 390 MiB 热缓存 = 487 MiB，**释放 341 MiB**。
- 128K 释放 1.09 GiB。
- **32K 时转成 Spill 反而多用 49 MiB。**
- 引擎只在"释放 = 落地数 × (full − index) > 读取需求"时才提供转换（`EngineMemory.cpp:986-988`；在 `a7dac91a` 上是 974-976）。
- 计入的读取需求是 `laneShortfallBytes()`。热缓存在最后一个 lane 离开后仍保留 extent，所以后续一次 64K 转换会被估成 731 MiB。

**decode 读什么**

- 12 个注意力层各自：
  1. indexer 用内存里的索引给块打分；
  2. 每个 lane 合并 8 个 verify 行的前 512 个；
  3. 一个 host 步解析每个块：在 Full 页上的走页表，在 Index 页上的走热缓存，**miss 时从 SSD 同步读取**。
- 只读选中且 miss 的块，GDN 层什么都不读。每条命令要做 12 次 GPU→CPU→GPU 的交接。
- M6 测量（1 个 Spill lane，189 条 decode 命令）：
  - 每层每命令约 853 个不同的溢出块，命中率 93 %，696 次 miss，即每命令 3.32 MB；
  - 解析 8.41 ms，交接 1.38 ms，**合计约 9.8 ms/命令**。
- 这只是单 lane、单次跑的样本。没有 Studio、M5 Pro 或多 lane 的数据。

**预填充读什么**

- 一条命令或一个 pass 每跑一个注意力层，就后台读取每个 Index 页在这一层的 33,280 B。
- 每个 chunk 的所有 pass 加起来，相当于把 12 层的全部溢出历史读一遍：64K 时约 767 MB。
- 实测：106 条命令共 126 次层读取，合计 4.37 GB，约等于把 12 层重读了 10.5 遍；读取和计算重叠，总共只等了 66 ms。

**可能不必留的东西**

1. **热块缓存 floor 按最坏情况定。**
   - 1 个 lane 时是每层 8,192 个槽 = 390 MiB，相当于 32K token 的 K/V。这是最大的杠杆。
   - 但 `resolve` 在 miss 超过空闲槽时会抛异常，缩小前要先加回退。
   - 安全下限没测过（"大了约 10 倍"的说法没有证据支持）。
2. **放置把 Spill lane 多估了 195 MiB。** 这是估算问题，不是真实占用。
3. **空闲的 Full 页只能以整 extent（256 页，108.5 MB）为单位归还。**
4. **保留的 128 个 Full 页**（48.75 MiB）作为 sink 和近期窗口值不值，没测过。

**终测里 Spill 跑得多频繁**

- M6 默认配置：0 个 Spill 请求。
- floorcap：每跑 1 个，一开始就放成 Spill，从没转换过。
- 这和 owner 的结论（3）有关：在 Flash-Next 上，转 Spill 是 KV 一侧的泄压阀，但在约 37K token 以下几乎释放不了内存。

### 5.3 短提示词 TTFT 回退（修复 `3874a2f1`）

**现象**

`a7dac91a` 上 serve1 对比 pass 树：

| 机器 | 回退 | 被切的请求 | 没被切的 cand | pass |
|---|---|---|---|---|
| Studio（FN Q2_0 + DFlash2），300 token | TTFT +26.2 % | 1,202-1,447 ms | 804-904 ms | 935-983 ms |
| mini-M4（35B + DFlash2），2048 token | TTFT +13.5-17 % | 6.55-6.68 s | 5.17-5.25 s | 5.5-6.1 s |

**原因链**（在 8 次 Studio 跑和 8 次 mini 跑上逐一对应确认）

1. 一个单独的短提示词到达时，macOS 正处于 **Warning**，而前一个请求的缓存状态还在往 SSD 拷。
2. 它的 replay 点捕获（Studio 上是 291 token 的提示词，边界在第 256 token）要一个状态的 buffer：Studio 113 MiB，mini 63.7 MiB。
3. 这个 claim 是 InService、`NeedTiming::CanWait`（`EngineMemory.cpp:773-781`）。CanWait 是有意的：在引擎上限下，先拷缓存，再动存储 chunk。
4. host 在 Warning 下拒绝 InService（`MemoryGovernor.cpp:374-375`）。
5. `freeFor` 把需求标成 `canWait = false`，但 `dryRun` 只对 `Immediate` 跳过稍后落地的单元。于是账本选中那个还在拷贝的状态，结果是 **Pending** 而不是 Short。
6. `MemoryGovernor::free` 只在 **Short** 时给被 host 拒绝的 InService 豁免（415-424）。
7. 分配被拒，`reserveCaptures` 在边界处切开命令（`Engine.cpp:1721-1738`，`++captureCuts`，`Prepared::Replan`）。
8. 提示词被切成两条命令，第二条命令又把所有层的专家流读一遍：
   - Studio（291 token 的 Flash-Next 提示词）：变成 0..256 和 256..291，第二条为了 35 行又把 48 层的专家流读了一遍，比 pass 多 260-390 ms；
   - mini（2048 token 档的 35B 提示词，40 层）：多一遍 40 层的专家流读取（共 80 次加载），多 1.35-1.45 s，因为它的存储只装得下 22 % 的专家。

请求级的特征（Studio）：2 个 batch、96 次加载、`capture_cuts` +1，日志里出现 "in-service claim … 1 state buffers: pending"。mini 上是 80 次加载（2 × 40 层）。

**排除掉的原因**

- 等待：claim wait 为 0。
- 调度器按价格切分：`bound_cuts` 为 0。
- Spill：0 个 Spill 请求。
- 存储收缩：没有发生。
- 代价模型。

**来历**

- pass 树（`0bd50301:Engine.cpp:2117-2126`）在 HostPressure 下会用 serving mark 重试 replay 点，命令保持完整。
- `7284fdb4` 删掉了这次重试。
- 替代它的 governor host 豁免（`ce9eb6f1`）只在 Short 时生效。
- 测试 `testReplayPointAdmittedInMarginBand` 仍然陈述旧规则，它能过只是因为夹具里没有正在拷贝的状态。

**这是一个典型：用账本的结果逻辑替换一条简单规则，悄悄丢掉了一个行为。**

**修复**（`runtime/engine/EngineMemory.cpp:824-836`，在 `reserveCapture` 里）

触发条件：是 InUse 的 replay 点、被拒绝、拒绝结果是 Pending、失败原因是 HostPressure。此时：

1. `ledger.setHostExempt(claim, true)`；
2. 用 `allocate(attempt, spec, false)` 再试一次；
3. 成功则 `setWaiting(claim, false)`。

不变的部分：引擎上限仍然会在这里切开命令；检查点和 junction 的切分不变；KV 等待、全局 host 规则、调度器都不变。

**否决的替代方案**

- (a) 把 InUse 改成 `Immediate`：会在引擎上限下带回一个旧 bug。
- (b) 对所有被 host 拒绝的 InService 在 Pending 时都豁免：范围太大，会把 Warning 下长提示词的 KV 等待变成越过 host 的超额分配。
- (c) 让 lane 等拷贝完成：约 20-25 ms，但需要一条新的等待路径。

**测试**

- `testAHostRefusedReplayPointCutsNoCommand`（`dev/tests/engine/engine_memory_test.cpp:6290`）。
- 修复前它失败，报 "commands 64 1, 1 capture cuts"。

**结果**

见 §4.7：Studio +26.2 % → +3.1 %；mini +13.5 % → −9.1 %。

**还有一个没修的 Studio 效应**

- cand-1 r7（1,185 ms）只跑了一条命令，但多读了专家：3,434 次对 2,710 次，GPU 多 188 ms。
- 原因：r6 期间，host 的赤字 claim 拿走了两个存储 chunk；存储回长时是空的；r7 decode 时这两个 chunk 又被拿走。
- 推断（未测量）：`e1310a4d` 让 host 在 Critical 以下拿不到存储单元，应该也能防住这个效应，除非当时已是 Critical。Studio 剩下的 30 ms 可能是噪声，也可能是这个效应。

### 5.4 M6 回退与修复 1（`e1310a4d`）

**缺陷**

- M6，FN Q2_0，默认内存。macOS 余量比 host margin 少了 327.7 MiB。
- host 的 claim 拿走了存储的 chunk：
  > host claim needs 327.7 MiB: met, took experts unit 179.3 experts unit 1023.0 for 1.806 s (1202.3 MiB now, 0.0 later)
- 腾出的空间被 lane 的 KV 占了，存储在 t = 42 s 收缩两次后 14 分钟没有回长。
- 结果：吞吐 −6.4 %，延迟 +12.6 %（对 1.5.0，取自提交信息）。matrix 的口径是 G1 0.936、G2 latency 1.126。
- 价格的问题：日志里的 "for 1.806 s" 是 dry run 给这两个 chunk 定的**总价**（1.806 s 引擎工作），不是 hold；Host claim 的 hold 是 60 s（`kValueHorizonSeconds`，`a7dac91a:MemoryGovernor.cpp:442-445`）。在 60 s 的视野里这个价钱足够便宜，于是两个 chunk 被拿走；而腾出的空间被 lane 的 KV 占住，存储再也没回长，真实损失远超这个价。内部状态日志 14:00 和本文早先的版本都把它误读成"1.8 s 的 hold"。

**`a7dac91a` 上的路径**

1. `MemoryGovernor::controlPass`（480-515）测出赤字。
2. `hostPass` 发出 Host claim，hold 60 s。
3. `MemoryLedger::freeFor` → `dryRun`。在 `a7dac91a:MemoryLedger.cpp:426-428`，只要 host 有赤字，`poolUnits` 就把存储 floor 以上的单元算作可取。
4. `poolUnit` 给每个 chunk 定价，`take()` 释放它们。

**这是同一类错误的第二次出现。**

- 11:12 的 M6 早期检查就发现过 N1：272 MiB 的 host claim 拿走了 1,202 MiB，因为每个 chunk 都按顶部 chunk 的大小计价。`a7dac91a` 修了计价。
- 终测里仍然是 327.7 MiB 拿走 179.3 + 1,023 MiB。最后只能改规则。

**改动**（`MemoryLedger.cpp:431-435`，注释在 417-430）

- Host claim 只在 **Critical** 下取池单元。
- 有赤字但不到 Critical 时，只能取 `rebuiltUnits`：只花重建代价的池（视觉 arena、图像行、空闲 staging；`PoolState::rebuildOnly`），和 Start 的范围一样。
- 不按池的名字分支。
- 不变的部分：请求 claim、增长、带、Critical 路径、缓存与空闲 buffer、**赤字下的转换**。
- 满足不了的 host 赤字保持 Short：host 规则拒绝 Start，Start 被拒后等待，等待又会停住向空闲空间的增长。in-service 的工作仍然通过 Short 豁免。

**待决：转换**

- 不到 Critical 时，host 赤字仍可以**按价值把 Resident lane 转成 Spill**（`mayTake`，`MemoryLedger.cpp:410-412`）。
- 曾先试过连转换一起去掉，结果 7 个引擎测试失败：
  - `testSpillDecodeTermIsMeasured`
  - `testConversionsFitTheQuotaAndTheWriteBudgetTogether`
  - 三个 `testAConversion…IndexPages…`
  - 两个 `commands/testALane…RunsPrefillGroups`
- M6 churn 的 16-38K 提示词里，能转换的 lane 很少，因为约 37K 以下转换不释放内存；64K 的 lane 可以。
- **需要 owner 决定。**

**测试**

- 新增：
  - `testTheHostTakesNoStoreUnitShortOfCritical`（`memory_ledger_test.cpp:2173`），重放 M6 的情形；
  - `testTheHostTakesCacheNotTheStore`（:2207）。
- 按新规则重述的有 10 个以上，包括：
  - `testClassBoundsEligibility`、`testWarningAloneTakesNoPoolUnit`；
  - A30 的几个 host 模式测试；
  - 三个改名的测试；
  - SIM-churn 的 `testChurnRegrowsAfterTheHostEpisode`。
- 换了申请者或池的：`testConversionVsStoreFollowsValue`、`testAHostRefusedStartIsFreedForByThePass`、`testLaggingHostReadingsCauseNoSecondRelease`。
- 修复前，memory-ledger 115 个里有 9 个失败。

**这正是 owner 结论（4）的例子：** 测试钉住了价格和"拿哪个 chunk"，改一条规则就要改一大批测试。

**模拟器里的效果**

- SIM-churn：work 1,138.0 → 1,150.5（+1.1 %）；存储在 host 那段压力结束后 0 s 就回到原大小，修复前一直缺到结束。
- SIM-warning-M6 不变，仍然失败。

### 5.5 其他调查

- **11-01 的"速度差距"取证。**
  - decode 并没有变慢。旧吞吐脚本里的 `decode_agg` 在 decode 和 prefill 交替时会出偏差，按 decode GPU 秒算是 45.6 对 45.4。
  - 真正的损失有两处：
    - 竞争的 Spill 预填充命令被切到 512-2048 行，每条都要付几乎完整的一遍专家读取（main 就有，`68b9e706`，`Scheduler.cpp:431-432`）；
    - M6 上的 −6 % 来自 Resident KV 上限从 1514 MiB 降到 1368 MiB（`39541855`）。
- **"int/fixes 在 Studio 18G 上回退"（10-31）。** 二分后发现是 `churn.py` 的首请求到达顺序随机造成的。此后固定用 A、B 两种顺序。
- **M6 早读的 decode 不足（11-02 23:22，快照 `c47bec51`）。** 当时归因为"Warning 下存储卡在 floor"，看的是引擎的 `memory_pressure` 带状态。终测在 `a7dac91a` 上的 value 跑里压力是 Normal，存储是被 host claim 拿走的（§5.4）；早读那次的压力没有重新测量，"Warning 导致"这个归因**没有得到确认**。
- **"永不回长不算缺陷"（11-03 10:05，模拟复核）。** 结论是 lane 的 KV 占着空间、空闲的缓存提示词比这个单元更值钱。几小时后，同样的现象在终测里让 M6 吞吐少了约 6 %。
- **自我 supersede 与 u = 301 ms：** 见 §3.5。**27B 与 ANE：** 见 §3.5、§3.16。
- **超车取证（A05，11-01 05:05-05:38，M5 Pro 18G 顺序 A）。** 一个请求排队 176.8-602.5 s，期间所有准入和资源等待计数都是 0；552 s 时被一个只短 34 token 的后到请求超过，超车 3 次后才被强行放进来，成了第二条长 lane，两条 lane 都跑 Spill、3:1 交替，首 token 到 1019.9 s。Spill 预填充只有 279-292 token/GPU 秒，Resident 是 885-948；按预填充时间算 987 对 290 rows/s。这就是"到达顺序悬崖"（§8.8）。同一次取证发现 main 从 t = 381 s 到 766 s 没有 decode。
- **内存步骤 2-4 的端到端（A40，10-31 夜）。** M6 上和不带它们的树持平（完成数 12 对 12，总 decode 23.2 对 23.1 tok/s），但每跑约 6,700 次 KV 拷贝、SSD 读 7.5-11.6 GB（对照约 1.2 GB），多 6-10 倍。当时只报了"M6 上没有退步"，没有指出 SSD 读的增加——这些步骤后来被 M3a 删掉。

### 5.6 开发期间找到并修好的其他真缺陷

除了 §5.1-§5.4 和 §3 各节已经讲过的，下面这些是评审、smoke 和扫描找到、在最终头上已修好的真缺陷。只影响测试工具的问题（例如 §10.1 的 `matrix.py` bug）不列，唯一的例外是 smoke a 的 oracle 失败那一行：它是测试的问题，但退出码容易被误读成引擎崩溃。

| 缺陷 | 怎么发现 | 修复 |
|---|---|---|
| **P0：转换落地到可能被回收清空的 Index 页**（`indexLandingPages`；`bd2d0b1f` 上真实存在，修复前 4 个测试失败） | 评审发现 #11(a)（M3b 也没修到） | `aa0ed449`（从 `1b010b14` cherry-pick），Critical 下的守卫 `b77c187d` |
| MetalBackend 数据竞争；一处 autorelease 泄漏 | 合并 1 之前的复核评审（V-tools）和扩展后的 Metal TSan | `260093d0`；`d70dff3e` |
| 三处不在 claim 范围内的生产分配（L7 规则下会抛异常）：状态池没有空闲单元时 `pollRestores` → `promoteState`；一条命令里检查点之后的 replay 点（8192 行 Spill/pass 命令在 4096 之后结束提示词、状态池为空时）；`Trim` 单元在尝试的范围外被拿走 | M3a 的 L7 审计（11-02 16:58） | M3a-rest 的 G4、G1 修复组（先写会失败的测试；具体提交未逐个核对） |
| main 上的有毒叶子泄漏 | cache-property 种子 9454 | `9077b419` |
| 回收遍历的多个 bug | cache-property 窗口种子 2484、9349、25949、28462 | M3a 稳定化各组；种子已设为默认 |
| `ReclaimPlan`/`takeLeaf` 三个缺陷：A（中）一个等待的 claim 可能为永远不会来的 Slim/Index extent 持有需求；B（低）叶子离开时父节点的新旧顺序和 `KvCache::inherit` 不一致；C（低）计划保留了一个 `dropPoisoned` 已删掉的恢复点 | 闲时大规模扫描（11-03 02:10） | `m3afix/sweep`（`699c11ef` 等） |
| smoke a 里 Flash-Next oracle 的 spill 跑失败（fn3 "spill 跑的页被拒"、fn5 "热块缓存 floor 被拒"；fn4/fn6 的 exit 70 是 `lockf` 的 EX_SOFTWARE，表示跑被外部停掉） | smoke a | 诊断为测试的内存规划问题，不是生产缺陷：`8b06a726`（spill 检查先把峰值放进 governor 的空间）、`2716511a`（oracle 的权重读取只算专家存储） |
| smoke-b/F1（Studio 报告）：FN 18G 在预填充末尾因热块缓存只预留一个 extent 而 `capacity_exhausted` | smoke b | m3b-findings 的修复（可能是 `cadbf9b2`，**未核实**） |
| smoke-b/T1（P1）；smoke-b/T2：一个客户端取消会让别的请求丢掉它们的 pass | smoke b 的场景追踪 | `m3afix/smoke-b`（`b64f62ba`） |
| m3b-findings 的 P1 1-4 | M3b 的独立高风险评审 | `m3afix/m3b-findings`（`8dcd7a87`，7 个提交） |
| 自我 supersede 活锁；u 从同形状样本拟合成 301 ms | M6 smoke a | `8b9d444a`；`751690bf`、`5190eff2`（§3.5） |
| 同级 supersede 扔掉已开始的行（T6）；越过状态边界的 KV 按 Optional 申请、在带内被拒（T7） | smoke b | `7e9063d2`；`2f6411fd` |
| 图像请求要等文本请求的整段 decode（P1） | smoke a #20 | `75674b20`（Start 可以释放空闲的 rebuild-only 缓存） |
| 退避 float 溢出；GPU OOM 后无限重启；工具 schema 语法平方增长 | 深度审计 1 等评审 | `6272a216`；`54b5eb52`；`38eaccfe`（§3.15） |

---

## 6. Issue 与 PR

提交信息里**没有用过任何关闭关键词**（Fixes、Closes、Resolves），唯一的引用形式是 "Refs incoai/splash#641"。所以 GitHub 不会自动关闭任何 issue，接手人需要逐个到 GitHub 上核对状态。上游仓库是 `incoai/splash`。

### 6.1 1.5.0 → main 合并的 PR（12 个，按合并顺序）

| PR | 内容 | 合并提交 | 要点 |
|---|---|---|---|
| #622 | 只剩一个 Spill lane 时，在内存墙前等自己的拷贝 | `bedd8891` | 修复 issue #613：120K token 的 Flash-Next 请求 16 s 后报 `capacity_exhausted`。Studio 18G：main 6 次失败 2 次，修复后 6 次全过 |
| #620 | 资源失败的控制台行写明引擎给的原因 | `33d5faef` | 外部贡献（引用 #613）；非分配类的 `capacity_exhausted` 不再建议调大 `--max-memory` |
| #616 | 旧引擎退出后才启动新引擎 | `98c24dda` | 外部贡献；`_stop_process` 等到调用方截止时间 |
| #624 | 滚动测试等转录页画完 | `dbc46102` | 修复 CI 上 App 测试的抖动 |
| #631 | 保留输入法正在组合的内容 | `05da3714` | 修复 issue #623 |
| #632 | 没有函数名的工具调用按文本读 | `7f2825ee` | Qwen 和 GLM 两种布局；1.6.0 里 `2b5cc832` 删掉了由此变成死代码的守卫 |
| #621 | Apple10 上 33-64 行预填充 chunk 用 64 行 staged tile | `a56988e7` | 外部贡献（引用 #588）；27B 投影快 6-43 %；同一 PR 修了 35B 恢复时 3-5 % 的回退 |
| #643 | 96 GB 及以上推荐 Flash-Next IQ3_S | `6c6ef4ec` | 64 GB 仍推荐 Q2_0 |
| #642 | 发送前显示原生聊天的上下文用量 | `1c17501b` | 外部贡献（关联 #641），owner 重写过 |
| #634 | 基准预热按"在服务请求所需内存"准入 | `ceb45357` | 修复预热时 "host memory reserve protected" |
| #640 | 流式专家按 chunk 存放，kernel 按地址访问 | `11e09ecb` | 见 §3.10 |
| #645 | 专家存储能归还内存、再取回 | `031a25a6` | 见 §3.10。它的回长规则在 1.6.0 里被替换 |

### 6.2 1.6.0 线上带着、但上游没有合入的

| 编号 | 内容 | 1.6.0 里的状态 |
|---|---|---|
| issue #644 | 有 43.6 GB 缓存 KV 时 decode 从 97 掉到 6-12 tok/s | `e805f601`、`6776d9cd`、`ec0de9c4`：处理了可能的机制（空闲 extent 在驻留集里），**没有复现，也没证明修好** |
| PR #648 | 思考预算 | `e223d655`（合并）和后续轮次 `2cc42cf1` |
| PR #649 | PDF worker 沙箱 | `94bfeb26` 和后续 `1ac3c5dd` |
| PR #651 | App：抽屉、流式、工具栏、遗留服务 | `035e70e8` 和后续 `3ddb6985` |
| PR #650 | 统一回收梯 | 先以 `a5570f52` 带入，后被 `7284fdb4` **删除**。代码仍是上游开着的 PR |

### 6.3 开着的 PR（在 `a7dac91a` 上分类，都从 `031a25a6` 分叉）

**结论：没有一个是关键的，也没有一个被移植进 1.6.0。**

| PR | 内容 | 结论与计划 |
|---|---|---|
| #650 | 回收梯 | 1.6.0 账本已覆盖并取代（它改写的代码在 runtime/engine 里一处都不剩）。1.6.0 上了 main 后按"已覆盖"关闭。注意：owner 最后的结论（2）又支持固定让出顺序，和 #650 的思路接近 |
| #653 | 20 核 GPU 上单 lane 4.5-bit decode 用 M16 tile | 它改的 `Linear::sixteenRowLane` 已在 1.6.0 删除（`fcca1921`），规则现在在 `KernelSelection.cpp:203-210` / `KernelCalibration.hpp:322、376`。1.6.1 再做，要有 20 核 M5 Pro 的实测数据写成一行校准表，不要写 `gpuCores_ == 20`。会改动 golden。约 2-3 小时 |
| #654 | Stop 帧：停止序列匹配算 Stop 而不是 Cancel，提前发 done | 现在停止字符串匹配会取消请求，`/status` 记为取消，done 要等在途命令（最多约 1.5 s）。11 个文件冲突；线协议 v12 已被占用。1.6.1 放到 v13。约半天，中等风险 |
| #659、#660 | 实验，没有加速（+1.59 % 对 8.1 % 下限；−0.73 % 对 3.5 % 下限） | 关闭或保留为草稿，不发布 |
| #662、#664、#666 | App 功能：API 连接信息、启动选项、首次运行引导 | 1.6.1。#664 和 #666 在 `AppState.swift` 冲突。三者共享的提交 `054d83d5`（与 locale 无关的上下文长度测试）可以单独 cherry-pick |

上表是 PR 分类时看的 8 个比 fork 基点新的开着的 PR。另有两个更早的、和本工作有关的开着的 PR：

- **#591**（开着，5 个提交）：一个可选的本地 safeguards 分类器，是解决 issue #589 的一条路（§6.4）。要不要收，需要 owner 决定。
- **splash-dev 仓库的 PR #1**：10-31 在 splash-dev 上开的回收梯 PR，11-01 时仍开着（之后的状态没有查）。当时 owner 决定引擎和服务端的共同修复只进 splash（splash-dev 共用这部分），所以它应当关闭或按 splash 的结果重做。

### 6.4 issue（62 个开着的 issue 连同评论都读过）

**结论：没有哪个 issue 必须进 1.6.0。**

下表里的 #613 和 #641 是已由 main 处理的相关 issue，列出来是为了完整，不在那 62 个开着的 issue 里；其余 17 行是 62 个里的缺陷和性能类，另外 45 个属于功能需求、提议或讨论。

| issue | 报告的问题 | 状态 |
|---|---|---|
| #613 | 120K 的 Flash-Next 请求报 `capacity_exhausted` | main 的 #622 修复，#620 让日志说明原因 |
| #623 | 切换模型或新建聊天后，中日韩输入法的组合内容丢失 | main 的 #631 修复（1.5.0 里没有）。模型切换那条路径没验证，只有发布清单里的拼音手工检查通过后才能写"已修复" |
| #641 | 聊天上下文用量 | main 的 #642 处理 |
| #644 | KV 池从不驱逐空闲会话 | **部分处理：** 可能的原因在 1.6.0 修了。发布说明写"处理了可能的原因，请重测"。其余（空闲释放 KV、清缓存端点）是功能需求；高水位补丁被否决（会驱逐可复用前缀，tick 里循环无时间上界） |
| #588 | 恢复的提示词会重读最多 36 个缓存 token | main 的 #621 让 33-64 行 chunk 变快，但边界规则和 1.5.0 一样，**没覆盖**。约占 TTFT 2 % |
| #387 | 混合模型在长会话深处编辑时，从 0 开始重新预填充 174K token（139.7 s，正常约 55 s） | **没覆盖，损失大。** 需要状态保留的设计，放 1.6.1 |
| #361 | 第一个兄弟轮次拿不到 replay 状态 | 没覆盖。owner 有一个提交 `a9c5070`（最后一条用户消息设检查点，+405/−20，加了一个线协议字段），不在 main 也不在 1.6.0。engine-sim 的 SIM-turns (b) 就是这一类。1.6.1 和 #654 一起放进 v13 |
| #370 | 一台 Mac 上开几个服务会慢 3-7 倍，外加 2 次 GPU 超时 | **没覆盖，损失大。** 草稿 PR #377 的 `--share-gpu` 要在新调度器上重做。1.6.1 |
| #598 | 专家流式读取把 SSD 读满（命中率 36 %） | 在那个上限下属于设计行为。1.6.0 的 pass 让每行专家读取减少 40-62 % |
| #589 | `safeguards` 没有透传 | 没覆盖。PR #591（可选的本地分类器）或等价方案可以解决，需要 owner 决定 |
| #332 | ANE 校准记住了繁忙时的结果（预填充 −20 %） | 没覆盖。1.6.0 首次启动会重新校准一次（key 里有 build id），这只是信息性的，不算修复 |
| #538、#326 | Q2 工具调用循环；思考重复 | 不是 Splash 的缺陷（旧版本和 llama.cpp 都能复现） |
| #441、#331、#328 | kernel 和 ANE 的性能余量 | 没覆盖。1.6.0 没改 kernel |
| #432、#471、#580 | `/compress` 连接错误、量化视觉塔的提示信息、开发基准的保真度 | 未知（没有日志）/ 1.6.1 改提示信息 / 只影响开发 |

另有 45 个 issue 属于功能需求、提议或讨论，跳过。

### 6.5 1.6.1 建议顺序（来自分类）

1. 一条统一的状态保留规则，覆盖 #387、#361、#588。从 `a9c5070` 开始，和 #654 一起进线协议 v13。
2. `--share-gpu`（#370）。
3. #332：先做测量。
4. #471 的提示信息。
5. #589：owner 决定。
6. #538：考虑推荐 IQ3。
7. #644 的残余问题：只有在 1.6.0 上复现了才做。
8. 开着的 PR：#653、#662、#664、#666 按 §6.3 处理。

### 6.6 发布说明措辞

- 可以写"已修复"：#623，前提是拼音手工检查通过。
- #644 写"处理了可能的原因，请重测"。
- **不要**写成已修复：#598、#589、#387、#588、#361、#370、#331、#328、#441、#538、#326。

11-03 凌晨写过一份约 57 KB 的英文发布说明草稿（带 `[FINAL: …]` 占位和 `[verify]` 标记，附一个发布前的合并与决定清单），不在仓库里。可以向 owner 要来作底稿，但里面的数字都要按最终测量重填。

### 6.7 用户可见的变化与升级说明（自 1.5.0）

发布说明里要写的、会影响现有用户或外部集成的变化：

| 变化 | 说明 | 来源 |
|---|---|---|
| `--expert-memory SIZE` 的含义变了 | 1.5.0：专家用这么多内存。main 起：专家**至少**保留这么多（floor），存储可以长到上限，有请求需要时退回 floor。默认 floor 是上限的 0.85 | `docs/server.md`；#645；§3.10 |
| 新端点 `POST /v1/chat/completions/count_tokens` | 不推理，返回 Chat 请求的提示词 token 数（含模板和图像），`{"input_tokens": N}` | main，`712cfbe8`（#642）；`docs/api.md` |
| 新错误和头 | 连续两次失败时都在跑的请求被隔离，返回 400 `request_quarantined`；启动中返回 `Retry-After`（最多 59 s） | C 轨道 `4077e079`、`15ada069`；`docs/api.md` |
| 引擎重启策略 | 退避 0/5/10/20/40/60 s，不再永久停止，只有配置、设备、命令内存、可执行文件缺失这几类才停 | §3.15 |
| `--serve-record` | 写 `serve-PORT.json` | §3.15 |
| 线协议 v12、状态 schema 7 | `/status` 增删了字段（§3.13）。读 `/status` 的外部工具要跟着改；删掉了 `kv_spill.history_stream.contended_commands` 和 `kv_spill.history_stream.decode_gap_ms` | `Protocol.hpp`、`server/protocol.py` |
| `--thinking-end-text` 的默认值 | 原来默认 `</think>`；现在不设时用聊天模板在推理之后写的结束文本 | #648；`docs/server.md` |
| 流式模型启动慢约 2 s | 为代价模型做种子的第二次预热（`Bootstrap.mm:263-271`）。删代价模型后这 2 s 会回来 | §3.5 |
| 稠密长上下文 | 27B 的 64K 提示词在两个 decoder 旁边预填充慢约 14 %（owner 接受） | §3.5 |

---

## 7. 做了但没用、失败与回退、被否决的方案

### 7.1 在 1.6.0 里建了、但测量证明没用（建议删除）

| 东西 | 为什么没用 |
|---|---|
| `CostModel` 和按价格的规划（T8 及之后约 30 个修补提交） | 没有测出收益：终测的预填充收益归因于 pass，而 pass 树用的是简单估计。冷启动高估 47 %；u 从同形状样本拟合成 301 ms；运行中的 u 读成 50-60 ms（硬件约 10 ms）；自我 supersede 活锁；同级 supersede 扔掉已开始的行；`kDecay` 是平局规则选出来的。复核 T1-2 早就指出被评审的树里已有仿射估计（§2.1） |
| 内存价值模型（`MemoryValue.hpp`、账本定价、Cache 价格查询） | 没有测出收益；直接导致 M6 回退（§5.4），同一类错误出现两次（N1、`e1310a4d`） |
| A30 回长核算链（`f0724a9d` → … → `ffca336c`、`RegrowthWatch.hpp`） | 前后修了 3 版，又补了 a30-host。11:00 owner 嫌整体太复杂，加了一条简单规则"有请求的 claim 在等时不向空闲空间增长"（`2b326a37`），但那条规则取代的是 start-hold 和 resume-price；A30 的定价和核算仍在代码里，没有测出收益，待删 |
| `supersedeGain`、按价格的放弃规则（`21268d5e`、`0549a2fb`、`66557140`） | 带来了活锁和扔掉已开始的行。最后保留的有用部分只是一条固定规则："同级新到达不切已开始的行"（`7e9063d2`） |
| 给代价模型做种子的第二次预热（`Bootstrap.mm:263-271`） | 流式模型启动多约 2 s，只为喂代价模型 |
| `value_per_gib`、`scheduler.estimate` 等状态字段 | 只服务于估计器 |

### 7.2 四天之内写完又删掉的

除最后一行外，这些都在 pass 树上：10-31 写成（`a5570f52` 10-31 上午，`ee58d77d`、`277f3e86`、`a9881fd5` 10-31 傍晚），11-02 中午就被 M3a（`7284fdb4`、`ca20c739`、`b2e78aad`、`7d31d72e`，12:27-13:30）替换，前后约 44-54 小时。最后一行 #645 的回长规则来自 main，不是 pass 树。

| 提交 | 内容 | 测量与问题 |
|---|---|---|
| `a5570f52`（PR #650） | `Engine::escalate`：所有拒绝走同一个顺序（先转 Spill，再一个存储 chunk，再在用内存）；回长失败退避到 4 分钟 | 35B 18G 3 个对话：存储抖动从 7 个周期降到 4 个，完成数从 106 升到 110（main 111），decode +2 %。但它**没法让 24 GB 机器上的存储回长**；35B 上抖动更多，SSD 读 8.3-12.2 GB，main 只有 0.74 GB；跨 lane 的几级在真实长提示词负载下崩溃，8 个代理的对抗评审找出了 2 类测试没覆盖的崩溃 |
| `277f3e86` | "exchange"：把存储上限当作预留，由引擎在请求旁边偿还 | SSD 写在途时很少触发；第 6 轮的"不写就释放"误删了一个保留的 20K 系统提示词检查点 |
| `a9881fd5` | 按启动时的 host 采样定存储大小 | 单次采样会终身压低上限：64 GB 机器 18 % 空闲时 TTFT 772 s、1,720 s |
| `ee58d77d` 的时间窗 | 提前拷贝用 5 分钟需求窗口 | 4 个拍脑袋常量；改成按需求 |
| `88f4dbba` / `2ddc977d` 的具体实现 | `rewaitHeld`、`memoryAwaitedUntil` | **思路保留**（落地内存给等待者），具体实现换成账本规则 |
| #645 的回长规则 | 30 s 冷却期、2 GiB 余量 | 24 GB 机器上持续 Warning 时永不回长 |

### 7.3 回退的提交

| 提交 | 回退提交 | 原因 |
|---|---|---|
| `6b0370bf`：VerifyBudget 成本下限提高到 0.25 | `42f998f5` | Studio ABBA（FN + DFlash2，4 客户端）0.984 / 0.990，FAIL。原来的依据只有 CPU 上的证据 |
| `c305f2de`：按 pass 算注意力上界 | `5f16b109` | 一个 pass 不切分一层；122B 在 YaRN 深度会超出 Metal 5 s 等待的 1.6-1.9 倍。后来设计又重犯了一次，被覆盖检查发现 |
| `68bde02b`：配额等待只拦会 spill 的启动 | `ef3d962f` | 任何 Resident 启动都能通过 3 条路径转成 Spill，抢走等待者的配额。改成：配额等待时拦住所有启动 |
| `fc3b7317`：比正在跑的优先级低的挂起请求像新请求一样等 | `bc925bcd` | 会永远停在那里，引擎一直处于一次放一个的恢复准入，还挡住存储回长 |
| `fix/passes` 第 4 轮 | 暂时移出发布 | 第 4 轮修复本身引入了 2 个 P1 崩溃（在 pass 之间结束的 lane 被挂起或收尾） |
| 侧分支上的"空闲空间持有" | 合入前回退 | 之后以 `2b326a37` 的形式合入，`ccc89af3` 收窄到只看请求 claim |

### 7.4 被否决的方案

**调度**

| 方案 | 否决原因 |
|---|---|
| R4 / plan 1（`a428081f`，竞争命令最少 4096 行） | 每台机器吞吐都最好（Studio 18G 1028/1059，main 853/972），但停顿翻倍（M6 7.3-8.3 s，M5 Pro 8.6-9.5 s），而且绕开了 decode 份额的平衡。**教训：** 竞争命令更大是最主要的吞吐杠杆，但固定行下限不是正确的做法，正确的做法是分层 pass |
| KR / plan 2（`bb668ed7`） | 消除了 M6 的 −6 %（work 927/963），但它是挂在静态阈值上的特例 |
| near-ties v1（`1ffbee36`，固定 4/3 带） | 跑 GPU 之前就被 v2 取代 |
| near-ties v2（`87382087`） | 只在内存竞争下大赢（M5 Pro 18G A：TTFT 464 → 259 s），内存充足时用最差 TTFT 做交换。设计删掉了这个带 |
| 并发预填充的定时批处理窗口 | 单请求绝不能等 |
| 每条命令第二个历史流 | 吞吐没有增加，每个流还多花 146-268 MiB staging |
| 按价格做不均匀的层切分 | 推迟 |
| 不在 1.6.0 做的 | 草稿关闭时的 decode 份额调优（DFlash2 永远开着）、提示词热度做种子、mlock、持久化代价模型、prefill 命令里带 decode 行、专家文件重排、按近期重用 |

**内存**

| 方案 | 否决原因 |
|---|---|
| 点修补 1-3（10-31） | 恢复是时序运气；一条路径一条路径地修只会追着路径跑 |
| 为 HiSparse 发布前把存储改成非弹性（10-31） | 当时的理由：要弹性，缺陷在单向流动。终测数据说 Flash-Next 用 floor = cap 更好，与 owner 结论（3）一致 |
| memres 评审轮 r1-r8（`75d48ac0`） | 从未收敛：每轮确认的发现 20/17/13/6/5/7 个，每轮 3.6-9.2M token。每轮的修复又产生下一轮的发现，这就是"补丁摞补丁"的证据 |
| 提前拷贝 v1（所有 Mac 都拷）/ v2（5 分钟窗口） | v1 磨损 SSD；v2 有 4 个拍脑袋常量 |
| `kvModeFor` 里按实时空间准入（`7ec53273`） | 重复计数，按构造约等于 0，还和回收梯重复 |
| 社区为 #644 提的高水位回收补丁 | 固定字节阈值会驱逐可复用的前缀；在 tick 里最多循环 1<<20 步且没有时间上界 |
| 30 s 冷却期的定时回长（设计时否决） | 用定时器代替测量 |
| 按传输记账的 `OwedTicket` | 约 230 行，属于会泄漏的一类 |
| 预留到上限 | — |
| 按价值门控的空闲空间增长 | 冷存储速率为 0，永远不会回长，等于重现 main 的 bug |
| `430b6d60`（take 的 EWMA > 0 后，有请求到达就不回长） | 持续 churn 下会让存储整场停在 floor |
| start-hold、resume-price | 11:00 被"请求优先"这一条规则取代 |
| 跨 release/restore 保留 ghost key | 每次回填 miss 都会被算成 ghost 命中 |
| 在 GPU 上或增量压实 | 实测每 GiB 59.7-92.9 ms，不值得 |
| PLE 做成弹性 | 超过约 256-512 MiB 价值约为 0 |
| 第 8 轮的检查点拷贝 | exchange 本身已删除 |

**其他**

| 方案 | 否决原因 |
|---|---|
| PLE 只预取一半读者（`1aa972f5`） | 只省 0.5-1 s，SSD 流量主要是专家重读 |
| QSA verify 行的块列表 | spill decode 的 SSD 读约占 decode 的 3.7 %，砍掉 30 % 也最多约 1 % E2E |
| verify 切分策略 | kernel 上 −22 %，E2E 只有 +0.5 % |
| 让引擎报告 ANE 秒数来给出精确的 Retry-After | 为一个罕见情况加新协议 |
| 改 SSD IO 优先级（Q6） | 上界 ≤ 2.5 %，实际约 0.4-0.8 % |
| K3（MoE 每块格式）、K4（Apple9 >10 核 tile） | 任何机器上都不改变计划 / 没有硬件 |
| 用文本做预热 | 多读 3.5-10 GB；保留 token-0 预热 |

### 7.5 集成阶段丢掉的

- **`d58e94f3`（第一版 A30 规则）。** owner 接受这个取舍之前就替换了原来的规则；它要求的回报约是周期的 20 倍（原规则是 0.95 倍）；测试没有给回长设下界。owner 03:55 接受 A30 后，以 `f0724a9d` 的形式重新落地。
- **`30f4297c`（发现 4）。** 它的测试修复前也能通过，所以丢掉。P1 "Index runway 回填按价格计" 一直没落地。
- **发现 8/18 的修复**（只为没定下来的拒绝记录 `allocationFailed_`）：没有测试，回退。
- **`m3afix/t6-n1`。** 11:20 冻结时被排除，11:35 owner 撤销冻结后，T6 和 N1 在本地重做并进了 `a7dac91a`（`7e9063d2`、`a7dac91a`）。
- **`rowsCounted` 恢复修复在三个分支里各有一份。** git 会"干净地"把它们合成一个双重移位，可能下溢。只能留一份。

---

## 8. 没做完的事与已知问题

### 8.1 发布前必须处理

1. **M6 修复 1 的确认：G1 过了，G2 latency 和 G3 还不确定。** 19:55 的 analyze 见 §4.7。要么加跑把区间收窄，要么在删估计器后的新头上重测。
   - 同一组跑里 A4 报告未记账内存增长 693 MB（报告线 268 MB）。按借鉴第 9 项（§3.18）的原决定，要把这部分记成一个固定账户或每 token 的 KV 项，不能靠加大 host margin 来吸收。还没做。
2. **M5 Pro 没部署 `3874a2f1`，正确性批次 28-47 搁置。** 包括：
   - test-real：35B、27B、Flint；
   - http-real、pressure-real；
   - 故障注入 a-e；
   - swift test；
   - 122B 单元。

   另外 M6 的 14、19-22 也搁置了。
3. **`3874a2f1` 上没跑完整 Metal 套件**（`make test-engine-metal`、`test-sanitizers-metal`）。
4. **清理分支没和两个修复合并。**
   - 文本上的重叠只有 `memory_ledger_test.cpp` 一个文件：清理改的注释在 `a7dac91a` 第 2198 行附近，离修复 1 最近的改动约 40 行。
   - 修复里没有用到被删的符号。
   - 预计能干净合并，但还没合，也没构建。
   - 清理分支上的 test-real oracle 没跑过。
5. **托管 CI 应该是红的。** 这是读 `.github/workflows/ci.yml` 和 makefile 推断出来的，没有实际看到 CI 在这个头上跑过，**未核实**。
   - 托管 job 的第一步是 `make check-source check-native-cpu`（`ci.yml:44-45`）。`check-native-cpu` 依赖 `test-engine-cpu`（`dev/Makefile:119`），它的 recipe 不带过滤器跑整个 engine-sim（`dev/native.mk:650`），会在两个已知失败上失败。`make` 在第一个失败的 recipe 行就停，所以其后的 fd-transport、native-arguments、runtime-metrics、memory-audit、memory-governor、memory-ledger、runtime-status、q8-cpu 都不会跑，后面的 CI 步骤也到不了。
   - 就算到了，`make test-sanitizers` 也用 `--filter $(ENGINE_SIM_PR_SCENARIOS)` 跑 engine-sim（变量在 `native.mk:889`，调用在 `927-928`）。过滤器按子串匹配（`TestRegistry.hpp:27、56`），`Turns`、`PersistentWarning` 正好包含两个已知失败。
   - 怎么核实：在本地跑 `make test-engine-cpu` 和 `make test-sanitizers` 看退出码，或者在 fork 上开一个草稿 PR 看 CI。
6. **两个已接受的 engine-sim 失败：**
   - **SIM-turns (b)，`testTurnsSettleAndShareACommand`。**
     - 报错："two turns together: worst TTFT 7333.66 ms against one turn's 3666.02 ms"。
     - 1.5.0 是 3.0 倍，所以不算回退。属于 #361 这一类。
   - **SIM-warning-M6，`testStoreRegrowsUnderPersistentWarning`。**
     - 报错："under macOS's warning the store stayed at 6 chunks of 7 for 2000 ms"。
     - 场景设置低于原来回长规则的前提条件。
     - **这意味着 24 GB 机器在持续 Warning 下的回长，在最终头上没有一个通过的模拟来钉住。**
7. **Studio 18G 的 SSD 门 `disk_read_per_work` = 1.452，FAIL，原因没查。**
   - 这是 KV 层 `disk.read_bytes` 除以 work，不是专家读取。
   - FAIL 的意思是：确定高于 r150 和 main 两个参照。
   - 线索见 §3.9（**未核实**）。
8. **Studio ample B1 的 store line FAIL：** 发生了一次震荡。
9. **Studio 短提示词还剩 +3.1 %（约 30 ms）。** 高于 0.4-1.8 % 的噪声。可能是 §5.3 末尾说的存储效应，也可能是噪声。
10. **owner 待决：** Critical 以下的 host 赤字是否还能按价值转换 lane（§5.4）。
11. **Flash-Next 默认 floor：**
    - B17 的数据结论是 FLOOR = CAP。按设计规则这需要每模型的 floor share，但 `expertFloorShare` 没有调用者。
    - 是否给 Flash-Next 设 floor = cap，以及在哪些机器上设，需要决定（§12.3）。
12. **跑了但没读出的结果：** Studio 第 1、2 阶段 job（80-82、90、92、95、96）、M5 Pro 的 07-18 和 26、M6 的 09、12、13、mini 的 20、40-41（§1、§4.9）。先向 owner 要日志，取不到就重跑。

### 8.2 测量和工具问题

- **Flint G7：** 基线工具报 "prompt token is out of vocabulary"，要先修 backend-benchmark 的 short-transcripts 场景。
- **无存储 spill 检查：** 我们的机器都跑不了，应从清单里删掉，或换成能跑的配置。
- **`test_installer_restarts` 4 个失败、2 个错误：** 从 `46f909fb` 起就有，之后没重跑。
- **可能抖动的测试：**
  - metal-backend 的 `firstCommandWithAnAwakeHostStep`：负载下会失败，基线上也会；
  - D4：`TranscriptTests` 在 M6 上抖动；
  - macOS 26.5.2 上的 `DrawerTests.theTerminalShownHasTheKeyboardAndWhatHidesGivesItUp`：在 `DrawerTests.swift:198` 失败，macOS 27 上通过。最终头上的状态**未核实**。
- **G4b 在 4 次 Studio 18G 有效跑里都超出窄上界 0.4-6.6 %。** 现在只报告。
- **G4 依赖代价模型。** `matrix.py` 和 `docs/validate.md` 里的 G4 上界是 `max(decode_gap_target_ms, u/0.1)` 加估计的 p95 误差。删掉代价模型后要改写；而且 11-03 13:25 的放宽（G4a/G4b 只报告，阻塞的是最大 ≤ 10 s、p99 ≤ 5 s）还没写进这两处（§2.2）。
- **`matrix-1.6.0.json` 的门常量绑着估计器。** 里面有 `value_horizon_seconds`、`pass_overhead_share`，同样要改。
- **`matrix.py analyze` 的 s-serve1 只算第 1 组。** 第 2 组是手算的。
- **u 和 δ（设计修订 A49 条定义的量，见附录 C）从没测过（D7）。** 没有按指定 pass 数跑一条命令的工具。删掉代价模型后，u 只在给 pass 定层数时有参考意义。

### 8.3 代码里已知、但没修的

- **D12：** 等待 claim 的字节被持有两次，一次在设备空间里，一次在 `heldPages` 里。计划修法：只把空间覆盖不了的部分按页持有。
- **D13：** `NativeRuntime.cpp:126` 把命令 OOM 算作持久缓存试用期失败（1.5.0 也一样）。要在 60 s 内出现两次 OOM 启动才会触发。
- **`freeFor` 和 `take()` 不一致：** `canWait = false` 与 Pending 的结果对不上（§3.1、§5.3）。修复只处理了 replay 点。
- **claim 等待只有 30 s 资源等待这一个上界。**
- **间隔分类器先判 `prefill_unit` 再判 `claim_wait`**（`Engine.cpp:532-537`）。
- **热块缓存 floor 按最坏情况定**，`resolve` 没有回退（§3.8）。
- **放置对 Spill lane 多估 195 MiB**（§3.8）。
- **集成阶段没落地的发现**（编号是 smoke a 之后 M3 修复轮的问题清单）：
  - 4：Index runway 回填和裁剪 claim 的定价，没找到能先失败的测试（删估计器后多半不再相关）；
  - 8/18：`allocationFailed_` 的归因（拒绝没定下来时记成什么）；
  - 12：命令中途 take 之后再决定的 claim；
  - 22 的路径 (a)（来源里没有更多描述）；
  - 27：池报告首个单元之后的单元大小。可能已被 N1 覆盖，**未核实**。
- **M3b 计划里没做的项**（`R` 是 M3b 计划的条目号）：
  - R6：借鉴第 5 项，提前拷贝的量钉在下一个 claim 的 2 倍、最便宜的先拷；
  - R8：转换的 Index runway 上界没有测试；
  - R9：两个上界测试；
  - R10：冷启动场景测试的重写；
  - R11：`testHeldLaneDoesNotSpin`；
  - R12：spill-wait 后续轮次 `6ea9f9e0` 的场景；
  - R13：捕获 KV 的那一对测试；
  - R14：HiSparse 的价值测试（删估计器后不需要）；
  - R15：设计稿 §7.8 欠下的规则测试；
  - R16：引擎层面的回长测试；
  - R17：基准预热要当作请求处理；
  - R25：重叠的 supersede 计划在命令准备好之前就放弃了它。
- 另有：m3b 发现 A/B/C（M3b 独立评审留下没修的三项，和 §5.6 的扫描缺陷 A/B/C 不是一回事）；ANE 切分退役后是否保留行下限；`t/m3a` 评审的 #8、#16、#19。这些都没开始。
- **设计稿 §7.12 的清理完成标准没达到，清理分支上也没达到。**
  - 检查命令（设计稿原文）：`grep -rnE "releaseOneCache|IdleMemory|reclaimIdleState|holdsSpilledReads|growSpilledReads|reserveHistoryStaging|releaseExpertChunk|releaseExpertsToFloor|expertGrowthBytes|growExperts|prefillGroupRows|prefillPasses|hostMilliseconds|streamMilliseconds|prefillTimingIsRepresentative|hostSeconds|expertAdmission" runtime dev`，结果应为空。
  - 命中数：`a7dac91a` 上 219 行，清理分支的 `d9585b59` 上 88 行（`5fc74205` 只改了一个 Python 测试，推断命中数相同）。
  - 剩下的主要是 `IdleMemory`、`Model::reclaimIdleState`、`Runtime::Impl::releaseOneCache`（88 行里占 81 行）。Metal oracle 有 57 行调用 `reclaimIdleState`，其中好几处（例如约 :974、:2815 附近）断言精确到一个单元。
  - 还有 `metal::CommandTiming::hostSeconds`，以及测试里约 19 个文件的内存模拟开关（`growthBlocked` 125 处、`budgetPages` 75 处等）。

### 8.4 推迟的低影响项（owner 11-03 03:20 的分类规则）

分类规则：修 P0（崩溃、错误输出、挂起、OOM、数据丢失、安全）、修 P1（相对 1.5.0 的速度回退、功能坏掉、饥饿、会放过坏构建的门）、修用户可见的行为。其他每条记一行。

| 编号 | 位置 | 内容 |
|---|---|---|
| D1 | runtime/engine claim 日志 | pending 的 claim 每个 tick 打一行相同的日志（M6 上 4 分钟 497 行） |
| D2 | Start claim 日志 | 原样打出 UINT32_MAX 哨兵值（"4294967295 state buffers"） |
| D3 | `install/hub.py` 的 `failure_kind` | gated 仓库的 403 被归为 not_found |
| D4 | App 测试 | `TranscriptTests/aReplyStreamsAtAnEvenPaceDrawingOnlyItsEnd` 在 M6 上抖动 |
| D5 | 终测工具 | 没有 AppFlow 的第 2 阶段 job |
| D6 | `backend_regression.py` | G7 没有为 K 计划设恢复和上下文场景 |
| D7 | 测量 | u 和 δ（A49）的每 pass 时间没测 |
| D8 | 内部设计文档 | 正文没有完全合并修订 |
| D9 | M3b 计划 | 8 个计划中的测试名找不到 |
| D10 | CostModel | 运行中 u 读成 50-60 ms，实测约 10 ms |
| D11 | 作业脚本 | 手动重排会覆盖同名的已完成日志 |
| D12 | MemoryLedger | 等待 claim 的字节被持有两次 |
| D13 | `NativeRuntime.cpp:126` | 命令 OOM 被算作试用期失败 |

### 8.5 文档和仓库卫生

- **失效引用。** 代码注释和提交信息引用了不在仓库里的设计文档和 owner 时间戳：runtime 和 server 里约 49 处 `amend/`，dev 里约 264 处设计引用。新接手的人没法追溯。建议在删估计器时一起清掉，换成可以自解释的注释。
- **已经推到两个公开分支上、涉及机器归属或机器目录布局的内容，在分享本文之前就应该用一个提交清理掉**（两个分支都有）：
  - `dev/benchmarks/serving/matrix-1.6.0.json:28、33、35`：机器归属的说明和机器上的 home 相对路径；
  - `dev/benchmarks/serving/run.py:59、62`：机器归属的说明；
  - `dev/tests/tools/test_serving_benchmark.py:2969`（机器归属的说明）、`3937、3945`（一个合成的 `/home/...` 绝对路径，照搬了测试机的目录布局）；
  - `docs/validate.md:446`：机器归属的说明；
  - `dev/tests/engine/fixtures/prefill_commands_*.json:2` 和 `dev/tests/fixtures/recorded-loads/*.tsv:1` 的表头 `source` 行：测试机上的 home 相对路径。
  - 代码注释里引用"owner 11-03 hh:mm"的地方是次要问题，可以在删估计器时顺手清掉。

  树里没有 `/Users/` 这类本机绝对路径，但上面这些 home 相对路径和合成路径会暴露测试机的目录布局。
- **改 `.github/workflows` 下的文件需要带 workflow 权限的凭据。**

### 8.6 发布前只能 owner 做的

- 拼音输入法手工检查。
- AppFlow 下载用例。
- 用 Homebrew Python 3.13 / 3.14 跑 PDF，可以通过发布 PR 的 CI 来做。
- 怎么关闭或更新 #648、#649、#651、#650、#591，以及 splash-dev 的 PR #1。
- 推送、开 PR、在 GitHub 上留言，都要 owner 明确同意。
- 清理公开分支上涉及机器的内容（§8.5）。

接手人做不了这些，只能提醒 owner。建议的顺序：先清理公开分支上的内容（§8.5），再分享本文；发布前做拼音和 AppFlow 手工检查；最后决定各 PR 的去留。

### 8.7 测试机上留下的东西

测试机上的临时文件、暂停的进程和临时工作目录由 owner 自己清理，不在本文范围内。接手人如果要用同样的机器，先向 owner 确认机器状态。

### 8.8 11-01 列出的未决问题，到最终头的状态

11-01 的尝试记录末尾列了一张"一直没解决的问题"清单。下面按能确认的程度给出状态；"仍在"表示代码里还能看到原样的机制，影响没有重测；"未核实"表示没有查。

| 问题 | 状态 |
|---|---|
| 竞争的流式 Spill 预填充被切到 512-2048 行，每条都付一遍专家读取（main 起就有） | **解决**：分层 pass，每命令约 7,300-7,700 行（§3.6） |
| decode 间隔指标是盲的 | **解决**：记录每个间隔并归因（§3.13） |
| main 上的有毒叶子泄漏 | **解决**（`9077b419`） |
| pass 状态有两个所有者 | **解决**（`Engine::PassCommand`） |
| 按 pass 算的注意力上界对混合注意力模型不对 | **解决**：改成每层的注意力对数上限（`withinPairs`） |
| 按启动时一次 host 采样定存储大小 | **解决**（`ca20c739`） |
| 重启策略 3 次失败后永久停止；严格工具 schema 语法平方增长 | **解决**（`4077e079`；`38eaccfe`） |
| M5 Pro 没有 1.5.0 基线 | **解决**：4 跑（带 void 标记，§4.2.4） |
| main 在压力下比 1.5.0 差 | Studio、M5 Pro 上 1.6.0 明显好于 1.5.0；M6 修复后 G1 持平（§4.7） |
| 24 GB 机器在持续 macOS Warning 下存储不回长 | **未解决**：SIM-warning-M6 在最终头上仍失败（§8.1） |
| 发布门和 CI 不覆盖流式专家、`--max-memory`、Flash-Next | 部分：自托管的发布门加了 `test-pressure-real`（35B，18G）；PR CI 仍然没有 GPU |
| 假件把内存建模成一堆布尔开关 | 部分：有了 `FakeMemory`，但约 19 个测试文件还在用内存模拟开关（§8.3） |
| 准入是串行的：有命令在途时 `Engine::tick` 直接返回 | 仍在（`Engine.cpp:221`、`270` 的 `if (pending_)`），影响没有重测 |
| 到达顺序的悬崖（第二个流式 Spill 提示词插进来，987 → 290 rows/s，A05/A100） | 部分：A103 启动规则让 SIM-churn-order 通过，真机上没有单独验证 |
| 短 25 % 以上的提示词仍触发 3:1 交替和可能的转换 | 未核实 |
| Metal 一次等待超过 5 s 让后端永久不健康 | 仍在（`MetalBackend.hpp` 的超时仍然标记不健康；`kGpuWaitLimit` = 5 s） |
| 服务端 50 ms 的状态探测在压力下可能触发 30 s 的引擎 kill | 未核实 |
| App 用正则解析日志 | 部分：Ready 帧带能力、新增 StartFailed 帧，但 `app/Sources/Splash/ChildProcess.swift:274、284` 仍按正则读子进程输出 |
| install 和 server 互相 import 不在 `check_architecture.py` 的检查里 | 未核实 |
| 回长后只靠 miss 回填；空闲恢复把 ARC 重置为冷 | 未核实 |
| Spill decode 只有 15-57 tok/s，Resident 是 120-134（A19），转换的代价没有按每字节量化 | 仍在（§3.8，M6 单 lane 约 9.8 ms/命令） |
| #648：在空白处截断需要改引擎；真模型 smoke 没跑过 | 未核实 |
| Apple9 大 GPU 的 MoE tile 没测；`MoeShape::valid` 接受 256 个专家以上的 sigmoid 路由 | 前者仍在（K4，§3.14）；后者未核实 |
| Flash-Next 的 MLX 计划需要 `dd23d6eb` 删掉的 bf16 norm 实例 | 未核实 |
| prefill splits 在 27B 128K 和 Flint 上没测 | 仍在 |
| VerifyBudget 的下限挡不住只有锚点的周期（只有 CPU 证据） | 仍在：0.25 下限被回退，问题本身没处理（§3.16） |
| #644 的崩溃从没复现 | 仍在（§6.2） |

---

## 9. 教训与经验

### 9.1 技术

1. **先用简单规则；估计器得先赢过简单基线才能留下。**
   - 这次的教训是：凡是不可逆的决定，比如把内存让出去、丢掉已做的工作，都应该用带余量的固定规则，再用上一次的实测结果做反馈，不要靠预测。
   - 要保留一个靠估计驱动的机制，前提是它在真实矩阵上的 A/B 里赢过简单规则的基线。而且这个 A/B 要尽早做，在任何东西建在它上面之前做。
   - 终测给出的证据：收益来自分层 pass 和账本的固定规则，两个估计器什么也没贡献。
2. **尽早对简单基线做 A/B。**
   - intpart（没有内存步骤 2-4）是 18G 下最好的树，加上步骤 2-4 反而把收益还了回去。这件事很晚才被注意到。
   - 1.6.0 开发期间从没测过"固定规则版本"的基线，所以直到终测才知道估计器没用。
3. **用测量反馈代替预测。**
   - u（每个 pass 的周转）在竞争场景下根本辨识不出来：拟合出 301 ms、50-60 ms，硬件实际约 10 ms。
   - 内部记录 D10 早就建议"直接测 pass 之间的间隔"。
   - 每个 pass 跑几层，应该由上一个 pass 的实测时间决定，而不是由学出来的速率决定。
4. **测试钉行为，不钉实现。**
   - 11-01 那份复核文档的 T6-5 条（和 smoke b 的 T6 发现无关）早就指出：钉死录制时间和 pass 数的场景测试，只是变化探测器。
   - 修复 1 改了不到 40 行产品代码，却重述了 10 多个测试；修复前 memory-ledger 有 9 个失败。
   - 价格、"拿哪个 chunk"这类断言，应该换成行为断言：不冻结、decode 不饿死、存储会回长、TTFT 有界。
5. **范围要小。** "全范围、什么都不推迟"加上"新代码允许出现的五处"，最后长出了 53.7k 行。owner 一开始就要求不能变乱，这一点没守住。
6. **别信盲指标。** 每个指标都要先审计，确认它能看到它声称测的东西。

   | 被误读的指标 | 实际情况 |
   |---|---|
   | `decode_gap_ms`（基线写"1.5.0 最大 4.4 s"） | 真实冻结是 295-430 s |
   | 按请求的 decode tok/s | 并发 lane 数一变就误导 |
   | 旧脚本的 `decode_agg` | decode 和 prefill 交替时有偏差 |
   | 引擎的 `memory_pressure` 带状态 | 不是 macOS 的压力级别 |
7. **替换一条简单规则时，把旧行为一条条列出来，逐个给出去处。**
   - pass 树里有一条"serving mark 下重试 replay 点"的规则，换成账本逻辑时被悄悄丢了，造成了短提示词 TTFT 回退（§5.3）。
   - 还有一个陈述旧规则的测试，只因为夹具里缺少触发条件才一直能过。
8. **碰其他 lane 的内存代码，要对照 prepare 和准入的每个阶段检查。** prepare 的阶段包括 KV 循环、spill 段、`reserveCaptures`、pass。对抗评审在这里找到过推理没发现的真实崩溃。
9. **GPU 时间上界必须在两个 host 步之间成立。** 一个 pass 不切分一层，Metal 的限制是按层算的（`c305f2de` 被回退；设计后来又重犯过一次）。
10. **补丁链本身就是信号。** 同一类问题修两次以上，通常说明模型错了，而不是参数错了：A30 修了三版加 host 补丁；N1 之后又要 `e1310a4d`。
11. **大的竞争命令是最主要的吞吐杠杆。**
    - R4 已经说明了这一点，但它用的是固定行下限，结果停顿翻倍。
    - 正确的做法是分层 pass：行数保持大，时间按层切。
12. **模型差异会以意想不到的方式冒出来。**
    - 27B 开着 ANE FFN 切分时没有层 pass（`maximumPasses = 1`），按时间切行会切到 ANE 阈值以下，悄悄失去 ANE。
    - 合理的差异（Apple9 和 Apple10、流式专家）应该由模型以数据的形式报告，不要用代理位来判断。
13. **DFlash2 永远开着。** 不要为关掉草稿的情况做规划、调参或基准。

### 9.2 过程

1. **估计一再滑坡。** 原计划 11-02 白天开始终测、当晚发布；实际 11-03 12:06 才开始终测，晚了约 27 小时，最后也没发布。

   | 给出估计的时间 | 承诺的终测开始时间 |
   |---|---|
   | 11-01 18:00 | 11-02 约 09:00-10:00，当晚发布 |
   | 11-03 01:20 之前 | 约 04:00 |
   | 11-03 01:20 | 约 04:45-05:00 |
   | 11-03 03:32 | 约 06:00-06:30 |
   | 11-03 04:05 | 约 07:30 |
   | 11-03 07:47 | 约 10:30-11:00 |
   | 11-03 08:25 | 约 12:00 |
   | 11-03 10:07 | 约 13:00 |
   | 11-03 11:20（硬冻结） | 12:20 前 |
   | **实际** | **11-03 12:06，没有发布** |

   记录下来的原因：
   - M3a 集成的失败数 66 → 17 → 5 → 0，耗时 8 小时以上；
   - 每次评审都还能找到真问题；
   - 每次合并带进来的分支远多于计划，例如 merge:M3a 一次带了 20 个 earlyfix 分支；
   - 每个里程碑之前都会加新步骤；
   - 设计稿按天估算的日程被压进了一个晚上。
2. **"代码写完前不跑长测试"推迟了证据。**
   - 这条规则让 GPU 空出来给模块测试用，但集成后的引擎直到 11-02 晚上 smoke a 才第一次跑真模型。
   - smoke 轮在终测前夜和当天上午还在找 P1。
   - 估计器没用，只有终测才看得出来。
3. **评审找到了很多真缺陷，但回答不了"这个估计器值不值"。**
   - 找到的真缺陷包括：P0 的 `indexLandingPages`、MetalBackend 数据竞争、claim 外分配会抛异常、自我 supersede 活锁（完整列表见 §5.6）。
   - "值不值"只有测量能回答。
   - 在补丁摞补丁的分支上做评审不会收敛：每一轮找到的，都是上一轮修复造出来的。
   - 记录下来的评审和检查规模：
     - 深度审计 1（11-02 18:30）：55 个代理，确认 12 个问题；
     - shadow-review-1（21:55）：85 个代理，终测工具（部署脚本、作业文件、`matrix.py`）两轮分别确认 53 个和 30 个问题；
     - M3a 深度评审：发现从 30 个收敛到 22 个（P0 归零）；M3b：39 → 9；
     - V-sim：三轮分别确认 12、8、3 个；
     - 11-02 19:31 的完整性检查：对照全部设计修订和 owner 决定的 161 项，88 项完成、55 项部分完成、13 项缺失、5 项只有计划。当时缺失的包括借鉴第 2、7、9 项（之后都落地了，§3.18）、122B 的 YaRN 深度检查，以及几条内部修订编号下的测量和记录（I-B13 D9、K-12、S-10、I-B10）。这些在最终头上的状态没有再整体核对过（122B 那项见 §8.1，测量类见 D7）。
4. **代理 worker 的代价。**
   - 一个托管在 Studio 上的代理 worker 从 11-03 01:45 跑到 11:23（约 9 小时），最多同时跑 6 个 max/xhigh 档的任务，到 11:00 共花掉约 $1,474（API 等价），只完成 11 个任务，另有 2 个被闹钟停掉。
   - 原因：最高档努力；03:05 限流之前每个工作流 12-40 个并行代理（之后最多 8 个）；每个代理每轮都重读约 30K 行的账本 diff 和设计文档。
   - 本地的代理也在 11-01 上午因为用量上限停转过约 2.5 小时。
   - 如果再用，规则是：一次一个任务、努力用 high、最多 4 个代理、一轮评审、每个任务后查一次成本。
5. **给代理的简报要带历史。**
   - 设计代理只拿到了代码和数据，没拿到尝试记录。
   - 覆盖检查（11-01 19:25，对照尝试记录逐项核对设计）的结果是 39 项已覆盖、69 项部分覆盖、4 项缺失、13 处和历史矛盾，例如重新提出已回退的按 pass 注意力上界。
6. **信 A/B 之前先证明测的是对的二进制。** 出过的错：
   - rsync 后 mtime 是旧的，make 跑了旧二进制；
   - -Werror 下的变异体跑的是旧二进制；
   - 端口上残留的旧服务给出 0.51 s TTFT；
   - 写死的树路径让 main 跑了 30 分钟；
   - `git archive` 按 export-ignore 丢掉了 `dev/tests`。
7. **每个里程碑都要有硬截止。** 预先决定哪些进、哪些带着重跑后进。估计里要包含一轮修复。
8. **owner 夜里委托决定时，把每个决定连同理由一起记下来。** 这次协调者代做的决定有：放宽 A30 的计数、否决 `430b6d60`、回退 VerifyBudget。其中 VerifyBudget 的回退一开始没先给 owner 看。
9. **单请求单元（G7）要在空闲机器上尽早跑；结果不确定就立刻重跑。** 这次的 G7 回退是在计划给结论前约 1 小时才确认的。
10. **写计划时就确认每个单元能跑：内存放得下、工具支持这个模型。** 这次：
    - 无存储 spill 早在 11-02 15:20 就知道跑不了，仍留在清单里，11-03 16:21 失败；
    - Flint G7 在终测当天之前没人试过。
11. **破坏性操作会被安全分类器拦下，需要 owner 批准。** 例如杀掉卡住的作业、停掉远端任务、删除目录。计划里要预留这段延迟。

---

## 10. 踩过的坑

### 10.1 测量与基准

| 坑 | 后果 | 怎么避免 |
|---|---|---|
| `decode_gap_ms` 只记录有流式预填充的间隔 | 设计不知道 1.5.0 和 main 会冻结几分钟 | 用客户端间隔直方图和 `status.jsonl` 的每请求 token 数做交叉验证 |
| 按请求的 decode 速率 | M6 的 G2 decode 0.566 被读成"decode 几乎减半"，实际总量只差 4.4 % | 并发不同时比总量（900 s 内的 tok/s）、work@900 和共同 tag |
| churn 首请求到达顺序随机 | 一次"回退"其实是到达顺序造成的 | 固定顺序 A = (2,0,1)、B = (0,1,2)，只在同一顺序内比较 |
| SSD KV 缓存按当天空闲磁盘定大小 | 空闲磁盘不同的跑不可比 | `matrix.py disk` 检查空闲磁盘窗口；必要时删目录或写占位文件把空闲空间调平 |
| 计时机器上有外来负载：别的程序、系统后台进程、睡眠断言 | 多次跑被判 void；有一次系统后台进程长时间占用约 115 % CPU | 计时跑前检查 top 进程和睡眠断言；计时作业持有本机 GPU 锁和所有构建锁。别一律作废：记下干扰，判断它的方向 |
| 用 `--max-memory` 在大内存机器上模拟小内存 | Studio 加 18G 时 macOS 报 warning 的时间是 0 %，真 24 GB 的 M6 上不是这样；host 压力下的行为（比如持续 Warning 下存储回长，SIM-warning-M6 那一类）在模拟里看不到 | `--max-memory` 只限制引擎自己的预算，模拟不了 host 压力；host 压力相关的结论必须在真的小内存机器上验证 |
| 监控脚本自己每 5 分钟 ssh 一次 | 引发网络睡眠断言，作废了一次 M5 Pro 跑 | 改成每 30 分钟轮询；监控本身也算干扰源 |
| serve1 的提示词生成器按每词 4 个 token 换算，实际是每词 8.09 个（Qwen 分词器把数字逐位切开） | 提示词实际是 662 / 4.2K / 29K token，不是 300 / 2K / 14K；s-warm 的 134K 前缀实际是 271K，超过了 `--max-context`（`a7ba7b81` 改成按每词 8.09 token 定长） | 用分词器实际计数核对场景参数 |
| G4a 用 log2 分桶读 p95 | p95 被读成 1,048.576 ms，smoke b 里 G4a 失败（`7153bbeb` 改成毫秒分桶） | 分位数用足够细的分桶 |
| G4 上界没考虑 ANE 行下限 | M6 27B 的 cand 跑全部会失败（工具的 kit-g4 修复：上界按 512 行下限的单元时间算） | 门的上界要反映模型自己报告的最小单元 |
| `matrix.py` 自身的 bug | work@900 重复计数；G4b 按 token 事件计；没检查被测构建；G7 json 过期；只给 serve1 第 1 组打分。shadow-review-1 的两轮对终测工具（部署脚本、作业文件、`matrix.py`）分别确认了 53 和 30 个问题 | 工具也要评审；`analyze` 必须带 `--build cand=BUILD_ID` |
| 参照树不一致 | M6 录制的 FN "pass" 跑实际用的是 `7e7b1a21`，不是 `0bd50301` | 每个参照跑都记下 build id，`analyze` 按 build 过滤 |
| 在繁忙机器上跑 G7 | 第一次判 INCONCLUSIVE（噪声 7 % / 20 %），真回退被掩盖 | 在空闲机器上尽早跑单请求单元 |

### 10.2 构建、同步与部署

| 坑 | 后果 | 怎么避免 |
|---|---|---|
| 用 `git archive` 复制树 | `.gitattributes` 的 export-ignore 丢掉了 `dev/tests` 和 `dev/benchmarks` | 按内容同步（`rsync --checksum`） |
| 同步时保留旧 mtime / 全量 rsync 刷新 mtime | 前者让 make 跑旧二进制；后者强制全量重建 | 做门之前先删掉要用的二进制再构建；记录 commit、空的 `git status --porcelain`、sha256、测试数 |
| 端口上残留旧服务 | 得到假的 0.51 s TTFT | `run.py` 会拒绝已经有响应的端口，还会校验 `/status` 的 pid 和 build_id |
| 往可能是符号链接的路径重定向输出 | 子代理把输出写进了 HF 模型缓存，损坏了 DFlash2 草稿 blob | 永远不要往可能链到模型缓存或 `install/models` 的路径写；`run.py` 会拒绝这类输出目录 |
| 删目录前只查进程和作业 | 参照树 `.venv` 的符号链接指向被删的目录 | 删之前先查有哪些符号链接指向它 |
| 同步脚本会删目标里的未跟踪文件 | 作业脚本和日志被删 | 作业脚本和日志放在被同步的树之外 |
| 作业脚本调用了目标机器 lib 里没定义的函数（`port_free`） | 等待循环永不结束，堵住 M6 的作业队列，部署被拒；杀掉它还要 owner 批准 | 部署前在目标机器上先跑一遍 lint 或 dry-run |
| BSD `sed -i` 不支持 `\b` | 一次编辑静默地什么也没改 | 改完用 diff 核对 |
| `git worktree add` 用相对路径；`check-source` 需要 `.git` | 失败 | 用绝对路径；需要 `.git` 的检查在完整的检出里跑 |
| mini-M4 没有 Metal 工具链 | 只能由 Studio 构建后转发二进制 | 按 `FINAL-STAMP` 的 sha256 核对转发的二进制 |
| 非交互 ssh 下 `make` 找不到合适的 Python | 构建失败 | 在 PATH 前面加上 Python ≥ 3.12 |

### 10.3 测试与 CI

| 坑 | 后果 | 怎么避免 |
|---|---|---|
| `make` 在第一个失败的 recipe 行就停 | `test-engine-cpu` 里 engine-sim 之后的测试都不会跑 | 已知失败要么修掉，要么移出 recipe；或者手动跑后面的步骤 |
| engine-sim 的 `--filter` 按子串匹配 | `test-sanitizers` 的过滤器也包含了两个已知失败（CI 在更早的 `test-engine-cpu` 就会失败，见 §8.1） | 过滤器用精确名字 |
| 测试夹具缺少触发条件 | 陈述旧规则的测试一直能过（`testReplayPointAdmittedInMarginBand`） | 每个行为修复都记录一次"修复前失败"的运行：用头上的测试配基线的产品代码构建，并记录那个二进制的哈希 |
| 测试钉价格和 chunk 选择 | 改一条规则就要重写 10 多个测试 | 断言行为 |
| `git merge` 合并三份相同的修复 | 合成双重移位，可能下溢 | 同一个 bug 只在一个分支里修 |
| 分支通配（`earlyfix/*`） | 不想合的分支被一起合进来 | 要排除的分支先改名（`basetest/…`、`hold/…`） |
| engine-sim 在 sanitizer 下很慢 | ASan 下 203 s 中有 188 s 花在 3 个场景上，TSan 下 1,111 s 中有 1,013 s | 已拆出 `test-sanitizers-sim` |
| 测试机内存不够 | Studio 上一个残留的 engine-sim 跑了 3.5 小时，把 swap 推到 62 GB、空闲磁盘只剩 7.7 GiB，污染了同机所有结果 | 每个代理任务设时间上限，结束时停掉自己启动的进程 |

### 10.4 引擎行为

| 坑 | 后果 | 怎么避免 |
|---|---|---|
| 用队列驻留集让专家 chunk 常驻 | Studio decode 慢 10 倍 | 每个 encoder 用 `useResources`（#640） |
| 空闲的 Addressed buffer 留在驻留集里 | decode 周期慢 3.5-5.5 %（提交信息里的数字，没有原始数据） | 只让活动的 extent 驻留（`e805f601`） |
| 存储 chunk 太多 | 36 个 chunk 让 decode 慢 2.3 % | 弹性 chunk 约 1 GiB |
| 静态的 Resident/Spill 阈值 | 一次 staging 记账改动（`39541855`）把 M6 的 Resident 上限从 1514 MiB 降到 1368 MiB，第一个 45K 提示词改跑 Spill，work −6 % | 放置按实时空间判断，并对边界附近的提示词做测量 |
| 启动时按一次 host 采样定上限 | 终身压低上限，TTFT 达 772 s、1,720 s | 存储建在 floor，之后再增长 |
| 退避一直翻倍 | 连续 1,026 次失败后 float 溢出，恢复线程死掉 | 退避到上限就停（`6272a216`） |
| 命令执行中 GPU OOM 后重试 | 服务端无限重启 | CommandMemory 不重试（`54b5eb52`） |
| 在 pass 之间结束、挂起、放弃 lane | P1 崩溃 | pass 状态只能有一个所有者（`Engine::PassCommand`）；为每个阶段写测试 |
| 给空集合定价 | 自我 supersede 活锁 | 删掉定价；如果保留，空集合的价格必须是 0（`8b9d444a`） |
| 用同一形状的样本拟合 | u 拟合成 301 ms，上界 3 s，decode 间隔约 3.7 s | 不要从无法区分各项的样本里学速率 |
| 用 `--expert-memory off` 跑 Flash-Next | 需要 37.9-38.9 GB，我们的机器都放不下 | 写计划时先算内存 |
| "macOS warning 97 %" | 是从一次跑得出的，而且看的是引擎的带状态 | 区分引擎的带状态（`memory_pressure`）和 macOS 自己的级别；16 次跑里 macOS warning 只占 0-2 % |

### 10.5 自动化与代理

| 坑 | 后果 | 怎么避免 |
|---|---|---|
| 恢复一个带并行阶段的工作流来改后面的步骤 | 只重放了缓存前缀，重跑了评审 | 从记录的状态启动一个续跑脚本 |
| 2 小时闹钟 | 两个"修复 + 评审 + 门"任务在修复轮中途被杀，没留报告 | 闹钟 ≥ 3 小时；要求在截止前 20 分钟写出报告 |
| 无头 CLI 的 print 模式默认在最后一轮 10 分钟后停掉后台工作流 | 第一次尝试没有任何输出，花了约 $59 | 关掉这个上限 |
| 监控按进程参数匹配 "engine-sim" | 误报（匹配到了任务自己的参数） | 过滤条件要精确 |
| 监控的过滤词太宽 | 漏看了一份报告（报告头里有 "drafts"） | 收紧过滤条件 |
| 评审工作流的文件过滤正则转义错了 | 静默跳过了 `Engine.cpp`、`Engine.hpp`、`Bootstrap.mm` | 打印实际被评审的文件列表 |
| 代理在本地协调机上编译、跑模拟器、跑 `python3 -` | 违反"本地不构建"规则 | 每个代理简报都写明；构建只走远端的构建锁 |

---

## 11. 代码量与测试膨胀

### 11.1 数字

| 范围 | 文件 | 增 / 删 | 净增 |
|---|---|---|---|
| 1.5.0 → main，全部 | 127 | +7,549 / −955 | +6.6k |
| 1.5.0 → main，产品代码（runtime、server、app） | — | +3,344 / −615 | **+2.7k** |
| main → pass 树，产品代码 | — | +5,964 / −2,322 | +3.6k |
| pass 树 → `a7dac91a`，产品代码 | — | +16,246 / −6,706 | **+9.5k** |
| main → `a7dac91a`，全部 | 417 | +81,954 / −28,259 | **+53.7k** |
| main → `a7dac91a`，产品代码 | — | +20,242 / −7,060 | **+13.2k** |
| main → `a7dac91a`，dev | — | +60,418 / −20,694 | **+39.7k** |
| main → `3874a2f1`，全部 | — | +82,086 / −28,259 | +53.8k |
| 两个晚间修复 | 10 | 产品 +39 / −18，测试 +229 / −120 | — |
| 清理分支（`a7dac91a` → `5fc74205`） | 24 | +160 / −484 | −324 |

几点观察：

- **产品代码的增长大部分发生在 pass 树之后**，也就是账本、代价模型、价值模型进来之后。
- runtime 从 77,084 行增到 86,507 行，其中 `runtime/engine` 从 22,780 行增到 30,048 行（+32 %）。
- runtime 和 server 新增的 16,142 行里有 5,051 行是注释（约 31 %，计算），另删了 2,164 行注释。注释里大量引用了不在仓库里的设计文档。
- dev 净增的构成：
  - `dev/tests/engine` 约 +26.9k；
  - `dev/benchmarks/serving` 约 +5.2k；
  - `dev/tests/tools` 约 +4.2k；
  - `dev/tests/server` 约 +1.8k；
  - 夹具约 +1.2k。
- main 之后的 408 个普通提交里，有 100 个只有标题、没有正文，集中在代价模型、模型边界计数器和 M3a/M3b 的账本后续修复。

**最大的产品文件变化**

| 文件 | main | 最终 |
|---|---|---|
| `runtime/engine/Cache.cpp` | 2,044 | 4,004（`ReclaimPlan` 约 1,360 行） |
| `MemoryLedger.{hpp,cpp}` | — | 466 + 1,123（`a7dac91a` 上是 1,582） |
| `EngineMemory.cpp` | — | 1,157（部分从 `Engine.cpp` 搬来；`Engine.cpp` 从 2,768 行减到 2,314 行） |
| `CostModel.{hpp,cpp}` | — | 182 + 366 |
| `StreamedMemoryPlan.{hpp,cpp}` | — | 117 + 306（搬家） |
| `Scheduler.{hpp,cpp}` | 263 + 819 | 445 + 1,086（净增 +449：hpp +182，cpp +267） |
| `MemoryValue.hpp` | — | 205 |
| `ops/KernelCalibration.hpp` | — | 473（搬家 + 出处） |
| app | — | 约 +2.9k（基本是 #651） |

**最大的测试和工具文件**

| 文件 | 变化 |
|---|---|
| `engine_memory_test.cpp` | +6.5k |
| `test_serving_benchmark.py` | +4.3k（共 4,264 行） |
| `memory_ledger_test.cpp` | +3.9k（共 3,946 行） |
| `kernel_selection_golden.txt` | +3.5k，生成文件 |
| `matrix.py` | +3.1k（共 3,148 行） |
| `engine_spill_test.cpp` | +2.8k |
| `engine_disk_test.cpp` | +2.4k |
| `ragged_scheduler_test.cpp` | 共 2,912 行（+2,799 / −1,536） |
| `engine_sim_test.cpp` | 1,705 行 |
| `cost_model_test.cpp` | 700 行 |
| `memory_value_test.cpp` | 449 行 |

引擎测试共 18,320 行，main 上的 `engine_test.cpp` 是 13,152 行。

### 11.2 原因

1. **两个估计器和它们的测试。**
   - 产品代码：`CostModel` 548 行、`MemoryValue.hpp` 205 行、账本和 Cache 里的定价，`ReclaimPlan` 的大部分。
   - 测试：`cost_model_test` 700 行、`CostSeeds.hpp` 91 行、`ragged_scheduler_test` 的定价部分、8 个录制命令夹具、`memory_value_test` 449 行、`memory_ledger_test` 里的价格测试、`RegrowthWatch.hpp` 124 行、录制负载夹具。
   - 还有为它们服务的模型边界计数器：ghost、near-ghost、tail、`PoolValue`、`ReleaseCount`、`KvTier::throughput`。
2. **测试钉实现，规则每变一次就要重写。**
   - 这在 1.6.0 之前就存在：评审时，待合并分支改了 8,114 行测试，只对应 3,190 行产品代码。
   - 1.6.0 期间更严重。M3a 一段（`8cb691c8..908c3701`）就是 runtime/engine +2,732 / −2,905，dev/tests +3,811 / −4,425。
3. **"补丁摞补丁"留下的层。** A30 系列、N1、`e1310a4d`、ANE 行下限、supersede 规则等，每一次修补都同时加产品代码和测试。
4. **大量设计理由写在注释里**，而且引用外部文档。
5. **验证设施本身。** 服务矩阵和它的测试（`matrix.py` 3,148 行，`test_serving_benchmark.py` 4,264 行）、假件和 harness、engine-sim、生成的 golden。其中有一部分是值得的（§3.17）。
6. **清理轨道的范围只是死钩子。** 24 个文件，−484 行，没碰任何活的估计代码（`runtime/engine`、`server`、`app` 都没碰）。

按 §12 的方案删掉估计器和钉价格的测试，是减少代码量最直接的办法。被删的大致范围：

- 产品代码：`CostModel` 548 行 + `MemoryValue` 205 行 + 账本、Cache、引擎、模型池里的定价。
- 测试：`cost_model_test` + `memory_value_test` + 价格测试 + 夹具。

具体能删多少，**未核实**。

---

## 12. 建议的下一步

本节是 owner 结论（2）-（4）的具体化。凡标"建议"的规则细节和常量，都要先测量再定，不能直接照搬。

### 12.1 调度器：固定规则替换代价模型

**规则**

1. **没人在等。** 没有同级或更高优先级的 lane 处于 Decode 或 WaitingMask，即 `Scheduler::contended()` 为假。
   - 整条命令一次跑完：行数取 min(剩余行, `commandRows`)，受 `withinPairs` 约束，跑全部层。
2. **有人在等，且模型支持 pass**（`CommandLimits::maximumPasses > 1`）。
   - 行数不变，按层切。
   - 每个 pass 的层数 = clamp(上一 pass 层数 × G / `lastPassMilliseconds_`, 1, 层数)，G = 500 ms。
   - 第一个 pass 的起始值，建议用同一目标上一条命令的实测"每层时间"；没有的话，用一个保守的固定值。**起始值要测量后再定。** 判定办法建议：在 M6 和 Studio 的 FN churn（m6-fn-churn、s-fn-churn-18g）上看第一个竞争命令的 `decode_gap_causes.prefill_unit` 间隔和 G4 的客户端间隔最大值；起始值让第一个 pass 不超过 2 × G、整跑客户端间隔最大 ≤ 10 s、p99 ≤ 5 s，并且 work@900 不低于上一个头，即可。
3. **有人在等，但模型不支持 pass**（例如 27B 开着 ANE FFN 切分）。
   - 像 1.5.0 那样切行，目标是 500 ms 的竞争切片（参考 `cabbad27` 的 `Scheduler.cpp`）。
   - 对半切，最低到 max(`kMinimumPrefillRows` = 64, `minimumUnitRows`)。后者就是 ANE 行下限，保留。
4. **合批（join）** 不再比较价格。只要合起来的行数不超过 `commandRows`，并满足 `lanesPastChunk` 和 `indexPageLanes`，就合批。
5. **排序**：优先级 → 提交顺序。overdue 改为按提交顺序加一个固定的让步次数，或直接删除，需要测量后决定（判定：SIM-churn-order 和 Studio/M5 Pro 的 FN churn 两种顺序，最差 TTFT 和 work@900 都不比保留 overdue 的头差）。A103 的启动规则（§3.5）里"领先"的判断改为按提交顺序。
6. **pass 之间：**
   - 同级的新到达不切已开始的行，等下一条命令（保留 `7e9063d2` 的规则）；
   - 只有更高优先级可以在 pass 边界放弃当前命令；
   - 取消和截止的处理保持不变。

**保留的**

- `next`、`nextDecode`、decode 份额和 decode 债；
- `contended`、`withinPairs`、`commit`、`abandoned`；
- `lastPassMilliseconds_`；
- `PassCommand`、Replan 机制；
- `bound_cuts`、`pass_commands`、`replans` 等诊断计数。

**删除的**

- `runtime/engine/CostModel.{hpp,cpp}`、`Scheduler::seedCost`，以及 `Bootstrap.mm:263-271` 的种子预热。删掉预热后，流式模型启动大约快 2 s。
- `Scheduler.cpp` 里：
  - `bound()` 的 u/0.1，改成固定的 G；
  - `passesFor`、`groups`、`remainingMilliseconds`；
  - 按价格的老化（33-36）、join 的价格比较（690-700）；
  - `supersedes`、`ranPastItsBound`、`supersedeGain`（415-499）；
  - `prefillMilliseconds`、`weightReadMillisecondsPerByte`、`decodeTokenMilliseconds`、`decodeTokensPerLaneCommand`。这几个只给价值模型用。
- 模型边界上只为代价模型存在的东西，删之前先确认没有别的用户：
  - `model::CommandWork`、`CommandSample`、`SsdReads`、`PrefillWorkSource`；
  - `model/WorkFormulas.hpp`、`Runtime::Impl::prefillWork`；
  - `ExpertStore::expectedPrefillReadBytes`、`Model::loadReads`。

  `passLayers` 和 `CommandLimits` 保留。
- 状态和指标：`scheduler.estimate`、`splash_scheduler_prefill_estimate_error_p95_ratio`、`decode_gap_causes.prefill_predicted_ms`。
- 测试：`cost_model_test.cpp`、`CostSeeds.hpp`、`dev/tests/engine/fixtures/prefill_commands_*.json`，以及 `ragged_scheduler_test.cpp` 里依赖价格的测试。
- 门：
  - `matrix.py` 和 `docs/validate.md` 里的 G4 上界改成固定 G 加 ANE 行下限的单元时间；
  - `matrix-1.6.0.json` 删掉 `pass_overhead_share`；
  - 间隔分类阈值改成固定 G。

### 12.2 内存：固定规则替换价值模型

**保留账本骨架**

claim、类别、rank、持有空间、落地内存、全有或全无、floor、"claim 外分配即错误"、`holdsFreeRoom`、Critical 下退到 floor。

**固定的让出顺序**

替换 `dryRun` 的价格排序：

1. 零代价的空闲 buffer。`takeUnsized` 已经最先取它们。
2. 前缀缓存，按现有回收遍历的顺序：已拷贝到 SSD 的先于没拷贝的，确定会被重用的最后。
3. 其他空闲 buffer 和 rebuild-only 池：视觉 arena、嵌入行、空闲 staging。
4. **只在有 Index 页的模型（HiSparse）上：** 把长 lane 转成 Spill。
   - 只转换真正能释放内存的 lane，保留 `freed > reads` 的门（`EngineMemory.cpp:986-988`），大约 37K token 以上；
   - 从大到小。
5. **最后才动专家存储。**

**存储的固定规则**

- **收缩：**
  - 只在一个请求的 claim（InService 或 Start）用完前面所有来源仍放不下时，才收缩；
  - 每次一个 chunk，最低到每模型的 floor；
  - 不到 Critical 的 macOS 压力不取存储（`e1310a4d` 已经做到）；
  - Critical 时退到 floor（现有行为）。
- **回长：**
  - 连续 Y 秒没有任何请求 claim 在等之后，回长一个 chunk；
  - 前提是引擎上限内有空间，且 host 规则准入；
  - 每次回长后重新等 Y 秒；
  - Y 要测量后定，#645 用的是 30 s 冷却期，可以作为起点。判定办法建议：在 m6-fn-churn（value 对 floorcap）和 s-fn-churn-18g 上，store line 不震荡（同一单元 Y 秒内不反复收缩、回长），请求结束后存储在 2Y 内回到上限，G1/G2 在 5 % 规则内。注意 smoke c 的跑在最后一个请求结束后 2-5 s 就停了，验证"请求后回长"要把跑延长。
  - **在 M6 上验证。** 11-01 在 1.5.0、main、pass 上测到 M6 有 89-99 % 的样本在引擎的带里（当时带的门槛是 2 GiB；1.6.0 的门槛是 1 GiB，比例没测过，§3.3）。如果回长条件要求"不在带内"，M6 很可能会和 main 一样永不回长。

**各类别能拿什么**

`mayTake` 去掉价格部分后保留：

- **Optional 和 Growth：** 只能拿空闲空间、零代价单元和 rebuild-only 池，不拿可重用的缓存，也不拿存储。**建议，未验证。**
- **Host 在 Critical 以下：** 能拿缓存、空闲 buffer、rebuild-only 池；转换与否由 owner 决定（§5.4）。

**放置**

Resident KV 在不取存储的前提下放不下时，选 Spill。**建议。** 去掉 `spillPenaltySeconds`。

**WriteBehind**

`copyDemand()` 的 60 s 视野换成固定常量。

**删除的**（按文件）

- `runtime/engine/MemoryValue.hpp`，整个文件。
- `MemoryLedger.hpp`：
  - `UnitCost`、`MemoryRates`（142-165）；
  - `ClaimSpec::holdSeconds`、`benefitSeconds`；
  - `Victim::cost`；
  - `ClaimState::openedSeconds`、`poolTakes`、`inDemand`；
  - `kValueHorizonSeconds`、`kRateTimeConstantSeconds`。
- `MemoryLedger.cpp`：
  - 52-71（`Lacking::share` 等）；
  - 306-342（take 速率）；
  - 381-609 里的价格排序；
  - 874-1011 的价值部分（只保留"host 准入时向空闲空间长一个单元"）；
  - 1045-1057（`fill()` 的价值列）；
  - `lanesTakes_`、`setRates`、`observePools`。
- `EngineMemory.cpp`：
  - 15-22（先验常量）；
  - 173-235（`spillPenaltySeconds`、`remainingSeconds`、`sampleSpilledReads`）；
  - 743-761、783-791（捕获和状态的 benefit）；
  - 1124-1155（`refreshMemoryRates`）；
  - 270-271（`setInDemand`）。
- `Engine.cpp:1229-1247`（`promotedCopy` 的 benefit）和 `Engine.hpp:1241-1251`。
  - `benefitSeconds = infinity / 0` 这种写法（`EngineMemory.cpp:459`、`Engine.cpp:1570`）实际是个伪标志，表示"能拿的都拿"还是"只拿零代价的"，改成布尔值。
- `Cache.cpp`：
  - `cachedCost`（968-974）、`victimOf` 的 cost（977-998）、`ReclaimPlan::price`（2323-2338）；
  - 所有 `MemoryRates` 参数。
  - `ReclaimPlan`（1044 起）如果 `available()` 和全有或全无的取走还需要"按单元大小"，就只保留那部分；否则改成一次回收一步再重试。
- 模型池：
  - `model::PoolValue`、各适配器的 `value()`、`PoolState::wholeBytes`（只给定价用）；
  - `ReleaseCount`、`RebuildCount`；
  - `ExpertStoreStats::nearGhostHits`、`tailHits`、`marginBytes`；
  - `SlotRecycler` 的对应计数；
  - `KvTier::throughput`。
- 状态：`value_per_gib`、`grow_value_per_gib`、`shrink_value_per_gib`；`MemoryAccountSnapshot::takesPerSecond`、`regrowthSeconds`。
- A30：`RegrowthWatch.hpp`、`PoolRate::takes`，以及 `matrix.py` 里依赖视野的 store line 规则（改成"同一单元在 Y 秒内反复收缩、回长"）。

**待修**

- D12：等待 claim 的字节被持有两次。
- `freeFor` 和 `take()` 的 Pending 不一致（§3.1）。

### 12.3 弹性专家和每模型 floor

1. **接通每模型 floor。** 由模型计划设置 `StreamedMemoryInputs::expertFloorShare`（`StreamedMemoryPlan.hpp:53`；现在 `RuntimeResources.mm:791-811` 没有设置）。`--expert-memory SIZE` 仍然作为精确覆盖。
2. **Flash-Next（有 HiSparse）：**
   - 按 M6 数据和 owner 的结论，默认 floor = cap。M6 上吞吐比 1.5.0 高约 2 %，TTFT 仍优于 1.5.0，代价是排队更多（等内存的样本 20 %）。
   - Studio 18G 和 M5 Pro 没有做过 floorcap 对照。在这两台上，弹性存储每跑收缩 / 回长 5.2 / 6.8 次和 3.0 / 4.8 次，收益也很大。改默认前先在这两台上补跑 floorcap 对照。
   - 如果有损失，就按机器内存决定 floor。
3. **没有 HiSparse 的 MoE（35B、流式 Flint、122B）：**
   - 保留弹性，默认 0.85，按 §12.2 的规则缩放。
   - 在 mini-M4（16 GB）和 M6（24 GB）上测出 floor：35B 在 18G 下只有 55-64 % 的专家驻留，存储是它唯一的杠杆。
4. **热块缓存和其他池不变。** 如果要缩热块缓存的 floor，必须先给 `resolve` 加回退（§3.8）。

### 12.4 精简测试

**删除钉价格、钉实现的测试**（先按名字核对，删之前确认）：

- 文件：`cost_model_test.cpp`、`memory_value_test.cpp`、`CostSeeds.hpp`、`RegrowthWatch.hpp`。
- 夹具：`prefill_commands_*.json`；`recorded-loads/*.tsv` 如果只给价值测试用，也删。
- `memory_ledger_test.cpp`：
  - `testCheapestPerByteFirstAcrossSources`
  - `testAUnitLargerThanTheNeedCostsItsWholeCost`
  - `testBenefitBoundsTheDryRun`
  - `testDisplacementNeedsValueAboveCycle`
  - `testTheMarginalPrice*`
  - `testTheLedgerValuesTheStoreByItsScans`
  - `testHiSparseOn*Weighs*`
  - `testDisplacementPaysForItsExpectedRetake`
  - `*PatternTakesNoRegrownUnitBack` 这一组
  - `testTheTakeRateDecaysWhileTheEngineIdles`
  - `testEachStoreUnitCountsItsOwnChunk`
- `engine_memory_rules_test.cpp`：
  - `testConversionVsStoreFollowsValue`
  - `testAMissedExpertByteIsPricedAtTheFittedRate`
  - `testACheckpointIsWorthTheRowsSinceTheLanesRecoveryPoint`
  - `testAJunctionTakesNothingWorthMoreThanItSaves`
  - `testCacheReuseFollowsTheRecentProbes`
- engine-sim：SIM-value、SIM-35b-churn，或者改写成行为断言。
- `ragged_scheduler_test.cpp` 里依赖价格的部分。

**保留行为测试**（不冻结、decode 不饿死、存储会回长、TTFT 有界）：

- engine-sim：SIM-churn、SIM-churn-order、SIM-low-host、SIM-long-prompt、placement。SIM-turns 和 turns-dense 改成固定上界；SIM-warning-M6 按新的回长规则修好或改写。
- pressure-serving；`native_loop_memory`。
- `engine_memory`：准入、挂起、replay 点的测试（含 `testAHostRefusedReplayPointCutsNoCommand`）；`testTtftPartsSumToTtft`；`testADecodeGapNamesItsCause`；pass 之间取消的测试。
- 账本：
  - `testWaitingClaimHoldsRoomForItsRank`
  - `testHeldPagesFollowRank`
  - `testLandingThenVictimsThenOwed`
  - `testOneHostRule`
  - `testCriticalDropsPoolsToFloors`
  - `testAPoolDoesNotGrowWhileAClaimWaits`
  - `testOnlyARequestsWaitingClaimHoldsFreeRoom`
  - `testTheHostTakesNoStoreUnitShortOfCritical`
  - `testPoolFloorBoundsEveryTake`
  - `testAccountsSumToCharged`

**让 CI 变绿**

- 两个已知的 engine-sim 失败修掉或移出过滤器。
- `test-engine-cpu` 的 recipe 要能全部跑完。

**原则**

- 每个行为变化都有一个修复前会失败的测试。
- 断言写行为，不写价格，也不写"拿了哪个单元"。
- 逐步把测试里的内存模拟开关（`growthBlocked`、`budgetPages` 等，设计里归 V6 这一项清理）换成 `FakeMemory` 加真实 governor。

### 12.5 合并清理分支

1. 把 `5fc74205` 的 7 个提交按原顺序 cherry-pick 到新头上：`cf1182e0`、`06c4077d`、`8b83266e`、`5642db3f`、`cba0dc84`、`d9585b59`、`5fc74205`。计划里原来的顺序（C2 在前）编译不过。
2. 预计只在 `memory_ledger_test.cpp` 一处注释上有重叠。合并后重跑门；engine 测试数在 `3874a2f1` 上是 389。
3. 删估计器时顺手完成设计稿 §7.12 的标准（§8.3）：
   - `IdleMemory`、`reclaimIdleState`、`releaseOneCache` 要先改写 Metal oracle 里调用 `reclaimIdleState` 的 57 行，其中好几处断言精确到一个单元；
   - `metal::CommandTiming::hostSeconds` 在 `metal_backend_test.mm:1875-1879` 有断言。

### 12.6 任何发布之前要跑的

**构建与门**

| 目标 | 内容 |
|---|---|
| `make check` | check-source（含 architecture-check，R3 开）、check-native-engine（CPU、Metal、build）、check-python、check-app |
| `make test-engine-cpu` | 全部步骤都跑完 |
| `make test-engine-metal` | — |
| `make test-sanitizers`、`make test-sanitizers-sim`、`make test-sanitizers-metal` | 0 个 sanitizer 报告 |
| `make test-real` | 每个家族；24 GB 机器用 `ORACLE_ARGS="--expert-memory 6656M --ple-memory 237M"` |
| `make test-http-real`、`make test-pressure-real`、`make release-check` | — |
| App | swift test、AppFlow |
| 故障注入 | 27B `--max-memory 1G` 的提示措辞；churn 下 kill -9 应 0 s 重启；60 s 内 4 次 kill 应按 0/5/10/20 s 重启；空闲释放后恢复的 `max_tick_ms` |
| owner 手工清单 | §8.6 |

**矩阵单元**

| 机器 | 单元 |
|---|---|
| M6 | m6-fn-churn（cand；floor 还不是 cap 时加 floorcap 对照）、m6-fn-ttft（G5）、m6-serve1、m6-single、m6-27b-churn |
| Studio | s-fn-churn-18g、s-fn-churn-ample、s-serve1、s-warm、s-fn-cold、s-single（Flint 要先修工具）、s-27b-ttft |
| M5 Pro | p-fn-churn-18g、p-35b-churn-18g、p-flint-churn-18g、p-single、p-serve1；最后做一次 122B 检查（Ornith-1.5-122B，即 Qwen3.5-122B-A10B，GGUF UD-Q4_K_XL，74.2 GiB，在 64 GB 的 M5 Pro 上流式跑；owner 11-01 定的是"Flash-Next 通过后再做"。它在 11-03 凌晨 smoke a 的 M5 Pro 检查里通过过，`job 40-47` 的终测批次搁置） |
| mini-M4 | mini-serve1、mini-35b-churn、mini-35b-ttft、Apple9 Metal 套件 |

**阻塞的门**

- G1/G2（5 % 规则）；
- G3；
- G4：客户端间隔最大 ≤ 10 s，p99 ≤ 5 s；
- G7：对 pass 树，同时报告对 r150 的结果；
- G8。

**另外**

- G9 自洽要通过。
- SSD 门要解释清楚，或由 owner 签字。
- store line 不能震荡。

### 12.7 建议顺序

1. **在 `3874a2f1` 上收尾。**
   - M6 修复 1 的 analyze 已出（§4.7）；
   - 跑完整 Metal 套件；
   - 部署 M5 Pro 跑正确性批次；读出 Studio 第 1、2 阶段和 M5 Pro 07-18、26 的结果；
   - 查 `disk_read_per_work`；
   - 把 A4 报告的未记账内存记成账户（§3.18 第 9 项）。

   这样就有了一个"已经往固定 host 规则走了一步"的基线。
2. **从 `3874a2f1` 开新分支**，合入清理分支的 7 个提交，跑门。
3. **删调度器代价模型，换 §12.1 的规则**，改写 G4，精简调度器测试。
   - 立刻在 Studio 和 M6 上对上一步的头做 serve1 和 FN churn 的 A/B。
   - 单请求必须不变。
4. **删价值模型，换 §12.2 的规则**，接通每模型 floor，精简账本测试。
   - 在 M6（value 对 floorcap）、Studio 18G、M5 Pro 上做 A/B。
5. **修 CI**（engine-sim 已知失败、recipe 中断）。
6. **全量矩阵（§12.6）**，然后发布。
7. **进入 1.6.1**（§6.5）。

每一步都要小、都要能单独回退，并且**在下一步开始之前**就对简单基线做 A/B。

---

## 13. 如何构建、测试、跑矩阵

### 13.1 构建与门

在仓库根目录用 `make`。根 `Makefile` 用 `-include dev/Makefile` 引入开发目标，构建规则在 `dev/native.mk`。`make` 会建 `.venv`，依赖在 `install/requirements.txt`，需要 Python ≥ 3.12。

| 目标 | 定义位置 | 做什么 |
|---|---|---|
| `test-engine-cpu` | `dev/native.mk:604-658` | 依次跑所有 CPU 测试二进制，含 kernel_selection、protocol、status、native_command、family_values 这些 golden。**engine-sim 失败后后面的步骤不会跑** |
| `test-engine-metal` | `native.mk:660-707` | Metal 测试，开 `MTL_SHADER_VALIDATION=1`，含 pressure-serving |
| `test-sanitizers` | `native.mk:893-941` | 多线程 CPU 测试的 ASan+UBSan 和 TSan 构建；engine-sim 带过滤器 |
| `test-sanitizers-sim` | `native.mk:943` | engine-sim 全部场景，在 sanitizer 下跑 |
| `test-sanitizers-metal` | `native.mk:958-966` | TSan 版 Metal 构建，在 `build/metal-tsan` 里跑 kv-block-cache、kv-history-stream、expert-store |
| `check-native-build` | `dev/Makefile:127` | 构建所有测试、基准和工具，不运行 |
| `check-native-cpu` / `check-native-metal` | `dev/Makefile:119-121` | — |
| `architecture-check` | `dev/Makefile:148` → `dev/tools/check_architecture.py` | 包含边界、算子 workspace 名、计时用的时钟、workflow 里的 HF token；R3 只在清理分支上开 |
| `check-python` | `dev/Makefile:138` | `dev/tests` 的 unittest、compileall、ruff check 和 format |
| `test-real` | `native.mk:730-743` | runtime oracle 跑 3 次，加视觉一致性检查 |
| `test-pressure-real` | `dev/Makefile:219` | 自己起一个服务，`--max-memory 18G`，churn 5 分钟，然后 `matrix.py check --streamed` |
| `test-performance-real` | `dev/Makefile:188` | backend benchmark，对 `BASELINE` 做 ABBA（G7 的 "single"） |
| `release-check` | `dev/Makefile:245` | 发布检查 |
| `check` | `dev/Makefile:151` | check-source + check-native-engine + check-python + check-app |

**CI**（`.github/workflows/ci.yml`）

- 托管 runner：check-source、check-native-cpu、check-native-build、test-sanitizers、check-app，以及跑 check-python 的 Python 矩阵。
- 自托管的发布门：check-native-metal、每个模型的 release-check、test-pressure-real。
- PR CI 没有 GPU，流式专家、`--max-memory`、Flash-Next 在压力下的情况，PR CI 都覆盖不到。

**主要测试二进制**

| 二进制 | 内容 |
|---|---|
| `engine` | 389 个测试，分布在 7 个 `engine_*_test.cpp` |
| `memory-ledger` | 115 个：账本 105 + 价值 10 |
| `native-engine-loop` | 47 个 |
| `engine-sim` | 10 个场景 |
| `ragged-scheduler` | 89 个 |
| `cost-model` | 18 个 |
| 其他 | `engine-cache`、`cache-property`、`persistent-cache`、`elastic-kv-pool`、`memory-audit`、`memory-governor`、`runtime-status`、`protocol`、`kernel-selection`、`model-boundary`、`command-limits`、`fake-memory`、`timing-profile`，以及 Metal 的 `expert-store`、`kv-block-cache`、`kv-history-stream`、`runtime-resources`、`runtime-bootstrap`、`pressure-serving` |

二进制和源文件的对应关系在 `dev/native.mk:89-221`、`374-376`、`819-874`。

**做门的步骤**

1. `rm -rf build/engine-tests`，然后重新构建。
2. 每个 make 和测试都在锁和闹钟下运行。
3. 核对同步过来的文件哈希和 git 一致。
4. 记录 commit、空的 `git status --porcelain`、sha256、`--list` 的测试数。
5. 每个行为修复都要有一次"修复前失败"的运行：用头上的测试配基线的产品代码构建，并记录那个二进制的哈希（必须和通过的那个不同）。

### 13.2 服务矩阵

工具在 `dev/benchmarks/serving/`，仓库内的完整说明见 `docs/validate.md` 的 "Serving benchmark" 一节。

**`run.py`：单次跑**

```sh
.venv/bin/python -m dev.benchmarks.serving.run --tree TREE --machine m6 \
  --model ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0 --revision <commit> \
  --draft-model DRAFT --scenario churn --order 2,0,1 --minutes 15 --label LABEL --out RESULTS
```

它会：

1. 用这棵树自己的安装器离线安装模型；
2. 起这棵树自己的服务；
3. 校验 `/status` 的 `instance.pid` 和 `identity.cache.build_id`；
4. 跑场景；
5. 先 SIGINT、再 SIGTERM 停服务。

输出 `LABEL.out`（最后一行是 `DONE` 才算完成）、`.meta.json`、`.events.jsonl`、`.status.jsonl`、`.server.log`。

**`clients.py`：场景**

| 场景 | 内容 |
|---|---|
| `churn` | 3 个客户端；顺序 A = (2,0,1) 或 B = (0,1,2) |
| `ttft` | N 个对话共享约 8K 前缀，同时发约 300 token 的后缀 |
| `serve1` | 一次一个冷提示词：300、2K、14K |
| `warm` | 不同缓存深度 |

**`matrix.py`**

| 子命令 | 作用 |
|---|---|
| `plan --machine M --tree NAME=DIR ...` | 每次跑写一个作业脚本 `NN-<label>.sh`，在本机 GPU 锁内运行 `run.py`。`interleave()` 交错各树，避免某棵树占住队首或队尾；`abba_runs()` 排成参照、cand、cand、参照 |
| `check RESULTS LABEL --minutes N` | 判单次跑 |
| `check --running QUEUE` | 列出超时的作业 |
| `disk` | 空闲磁盘会让 SSD 缓存不可比时拒绝运行 |
| `analyze --results DIR... --build cand=BUILD_ID --out DIR` | 写出 `verdict.json` 和 `verdict.md`。**必须带 `--build`**，否则会混入别的构建的跑 |

**`stats.py` 与 `dev/benchmarks/abba.py`**

- `stats.py`：单侧 90 % t 区间，合并 σ。
- `abba.py`：差异不超过 max(2 %, 2 × 噪声) 判 PASS，噪声超过 5 % 判 INCONCLUSIVE。

**`matrix-1.6.0.json`**

- 树：r150 = `cabbad27`、main = `031a25a6`、pass = `0bd50301`、ntie = `87382087`、intp = `1da46c67`、cand。
- 模型都固定了版本，机器带噪声 σ。
- 共 22 个单元。
- **换机器时要改什么。** 每台机器的条目（GPU 锁路径、端口、噪声 σ、G4 客户端间隔的机器 floor `g4c_floor_s`、空闲磁盘窗口）都是按矩阵里的四台机器定的。换成别的 Mac 时：
  - σ 要重新估：每棵树同一单元同一顺序至少跑 2-3 次，用 `stats.py` 的合并 σ；
  - 空闲磁盘窗口按参照跑 Ready 时的空闲磁盘和 SSD 缓存大小重新算（`matrix.py disk` 的逻辑），否则 SSD 缓存不可比、跑会被判 void；
  - 参照树要在新机器上重录（§4.1）。

**标签**

- 格式：`<cell>-<tree>-<order><repeat>`，或 `<cell>-<tree>-<repeat>`。
- **标签永不复用：** 重跑取下一个序号。
- serve1 的一组 ABBA 里有一跑被判 void，就整组重跑（pass-3、cand-3、cand-4、pass-4）。

### 13.3 有效性规则（`docs/validate.md`）

一次跑要算数，需要同时满足：

- 以 `DONE` 结束；
- 没有重启、没有失败请求（`context_length_exceeded` 除外）；
- 没有 serious 或 critical 的热状态样本；
- 跑的是这个单元的模型；
- CPU 前五名里没有非本次启动、占用 ≥ 24 % 的进程；
- 规格里标为可能有人使用的机器上，跑的期间没有输入活动、没有应用的睡眠断言；
- ample 单元在 Ready 时专家处于上限，且 macOS 压力为 Normal；
- churn 的 SSD 缓存在 2 % 以内可比。

被判 void 的跑仍然判 G4、G8、G9。不要一律作废：记下干扰，判断它的方向。

### 13.4 部署流程（通用描述，脚本不在仓库里）

**接手人能直接用的只有仓库里的 `run.py`、`matrix.py`、`stats.py`、`abba.py` 和 makefile 目标。** 下面描述的部署脚本、作业队列（gpuq）、`FINAL-STAMP`、`batch` 参数、同步和构建脚本都不在仓库里，接手人跑不了。最小的手工流程是：在每台机器上检出同一个 commit，`rm -rf build/engine-tests` 后 `make all`，记下 commit、空的 `git status --porcelain` 和二进制的 sha256；确认机器空闲、没有别的 GPU 作业；按 `matrix.py plan` 生成的命令一次跑一个单元，跑完用 `matrix.py check` 判有效性，最后 `matrix.py analyze --build cand=BUILD_ID`。下面的描述可以作为把这套流程脚本化的参考。

**每台机器的树**

- 一棵 `cand` 树，第一次用时从种子树做 APFS 克隆；
- 一棵 `pass` 树，作为 G7 基线；
- 1.5.0 和 main 的参照树。

**部署**（每台机器一个脚本，从协调机运行）

1. 要求本地 worktree 在目标 commit 上且是干净的。
2. 有作业在跑、有别的部署在跑、或者 GPU 锁被占用时，拒绝部署。
3. 在整个部署期间持有部署锁、一个构建槽和 GPU 锁。
4. 按内容同步树：`rsync --checksum`，排除 `.git`、`build`、`.venv`、`install/models`。
5. 删掉作业要校验的二进制，然后 `make all`，再构建 backend-benchmark 和 weight-digests。
6. `make verify-build-identity`。
7. 写 `FINAL-STAMP`：commit、干净状态、同步时间、make 状态、每个二进制的 sha256 和 mtime（mtime 必须晚于同步时间）、build id、venv。
8. 加 `batch` 参数时，把 commit 钉进队列的 COMMIT 文件，之后所有排队作业都会拒绝别的 commit 的 stamp。

补充：

- mini-M4 没有 Metal 工具链，由 Studio 构建，二进制经协调机 rsync 转发，再按 `FINAL-STAMP` 核对。CPU 测试在 mini 上单独的树里构建。
- Studio 的部署在后台跑 15-25 分钟，日志以 `DEPLOY OK` / `EXIT=0` 结尾才算成功。
- 同一份 Metal 源码在不同机器上编出来的 metallib 哈希可能不同，只能在同一台机器上比较哈希。

**作业队列**

- 每台 GPU 机器有 pending、running、hold 三个目录，runner 按文件名顺序执行。
- 排队：把作业从 hold 挪到 pending。暂停：挪回去，永远不要删。
- 只在两个单元之间部署。

**每个作业的步骤**

1. 校验 stamp 和钉住的 commit。
2. 取构建锁和 GPU 锁。
3. 对每个二进制、模型、选项组合，先跑一次不计时的 serve1 预热，让 Metal 管线在计时之外编译。mini-M4 上见过 21 s 的首次 dispatch 停顿。
4. 检查端口空闲；churn 还要检查磁盘窗口。
5. 开始和结束时各拍一次 ps、memory_pressure、热状态、磁盘、swap 的快照。
6. 照 `plan` 写的命令跑 `run.py`。
7. 跑 `matrix.py check`。void 的跑让作业失败，以便重跑。

`EXIT=3` 表示在锁内被拒绝，标签没被占用。

**单元顺序**

压力单元在前，cand 和 floorcap 交错；然后是 TTFT（G5）、G10；最后是单请求单元。

---

## 附录 A：代码地图（`3874a2f1`）

标记说明：

| 标记 | 含义 |
|---|---|
| **保留** | 固定规则或机制，留下 |
| **EST-C** | 只为调度器代价模型存在 |
| **EST-V** | 只为内存价值模型存在 |
| **简化** | 两者混合，主要因为估计器才需要现在的形状 |

### A.1 引擎主循环

| 位置 | 作用 | 标记 |
|---|---|---|
| `runtime/main.mm:54-83` | 500 ms 定时器 + macOS 内存压力事件源 → `FdTransport::controlNotifier` | 保留 |
| `runtime/engine/FdTransport.cpp:303-330` | 只在没有命令时运行控制处理 | 保留 |
| `runtime/engine/MemoryControl.cpp:31-46` | 控制 pass：释放空闲权重 → 设置压力 → 收集落地 → `refreshMemoryRates`（EST-V）→ 评估 → 上报 → `controlPass` | 保留（去掉 EST-V） |
| `runtime/engine/Engine.cpp:175-457` `Engine::tick` | 截止时间 → 轮询传输 → 恢复 → `syncClaims` → WriteBehind → 应用完成的命令 → 权重恢复 → `spillLanded` → `admitQueued` → 规划循环（`scheduler_.next` → pass 之间的放弃 / supersede → `prepare` → 提交，passes > 1 时建 `PassCommand`） | 保留 |
| `Engine.cpp:1529` `prepare`、`1827` `apply`、`2063-2107` pass 辅助函数 | — | 保留 |
| `Engine.cpp:66-77`、`Bootstrap.mm:436-440`、`RuntimeResources.mm:1212-1213` | 账本接线：victim 来源、池、账户 | 保留 |
| `Bootstrap.mm:263-271` | 为代价模型做种子的第二次预热 | EST-C |
| `Bootstrap.mm:489-493` | Ready 之前 `growToTargets` | 保留 |

### A.2 内存（`runtime/engine/`）

| 文件 | 行数 | 内容 | 标记 |
|---|---|---|---|
| `MemoryLedger.hpp` / `.cpp` | 466 / 1,123（新） | claim、类别、rank、持有、落地、`freeFor`、`dryRun`、`take`、`grow`、`available` | 骨架保留；定价与 A30 为 EST-V |
| `MemoryValue.hpp` | 205（新） | `namespace value`：`PoolRate`、`poolUnit`、`marginal`、`cycleSeconds`、`growthBenefit`、`spillDecodeSeconds`、`conversion` 等 | EST-V |
| `MemoryGovernor.hpp` / `.cpp` | 改动 +252/−178、+287/−169 | host 估计、`HostRule`、`tryReserve`、`free`（Short 豁免）、`controlPass`、`hostPass` | 保留 |
| `EngineAllocate.hpp` | 172（新） | `Engine::allocate<>` 模板 | 保留 |
| `EngineMemory.cpp` | 1,157（新，部分从 `Engine.cpp` 搬来） | `placement`、`admitSpill`、`convertToSpill`、`spillLanded`、`syncClaims`、`copyDemandBytes`、`growLane`、`reserveReads`、`reserveCaptures`、`reserveCapture`、`stateClaim`、`rankOf`、`claimFor`、`EngineVictims`、`refreshMemoryRates` | 主体保留；173-235、743-761、783-791、1124-1155 为 EST-V |
| `StreamedMemoryPlan.hpp` / `.cpp` | 117 / 306（搬家） | 启动前规划专家存储、PLE、spill floor | 保留 |
| `WeightResidency.hpp` / `.cpp` | 75 / 78（新） | 空闲释放和恢复权重 | 保留 |
| `MemoryPlan.hpp` | — | `SpillGeometry`、`EngineSpillConfig`（328-368）；Spill 峰值估计（344-348，多估 195 MiB） | 保留 |
| `Cache.{hpp,cpp}` | 4,004（main 上 2,044） | 前缀缓存、回收遍历 `nextUnit`、`returning`、spill、写穿、降级；`ReclaimPlan`（1044 起，约 1,360 行） | 机制保留；价格查询为 EST-V；`ReclaimPlan` 为简化 |
| `StateCache.*` | — | 缓存的 GDN 状态 | 保留 |
| `KvPool.*` | — | KV extent、页形态、按 rank 持有页 | 保留 |
| `KvTier.hpp`、`KvPageTier.hpp` | — | SSD 层（`throughput` 为 EST-V） | 保留 |
| `WriteBehind.*` | — | 提前拷贝 | 保留（需求窗口改成常量） |
| `RegrowthWatch.hpp` | 124（新，dev 侧） | A30 震荡规则 | EST-V |

### A.3 调度（`runtime/engine/`）

| 文件 | 行数 | 内容 | 标记 |
|---|---|---|---|
| `CostModel.hpp` / `.cpp` | 182 / 366（新） | 7 项衰减岭回归 | EST-C |
| `Scheduler.hpp` / `.cpp` | 445 / 1,086 | 固定部分：`next`、`nextDecode`、`contended`、`withinPairs`、`commit`、decode 债、`abandoned`、`lastPassMilliseconds_`。定价部分：`bound`、`passesFor`、`groups`、`remainingMilliseconds`、老化、join、`supersede*`、`prefillAdmissionOrder` | 混合 |
| `runtime/SmallSolve.hpp` | 40（新） | 小型最小二乘求解器，`VerifyBudget` 也在用 | 保留 |
| `Engine.hpp:576-580` `PassCommand` | — | pass 状态唯一的所有者 | 保留 |

### A.4 模型边界（`runtime/model/`）

| 文件 | 内容 | 标记 |
|---|---|---|
| `CommandCost.hpp`（97，新） | `passLayers`、`CommandLimits`（保留）；`CommandWork`、`CommandSample`、`SsdReads`、`PrefillWorkSource` | 混合（后四个为 EST-C） |
| `WorkFormulas.hpp`（70，新） | 注意力可达范围、期望的不同专家数 | EST-C（删前先确认没有别的用户） |
| `ElasticPool.hpp`（74，新） | `PoolState`（保留）、`PoolValue`（EST-V）、`ElasticPool` 接口 | 混合 |
| `ModelPools.hpp`（183，新） | `ExpertStorePool`、`HotBlocksPool`；`ReleaseCount` / `RebuildCount`（EST-V） | 混合 |
| `RuntimeArenas.hpp` | `commandLimits`（121-123：`maximumPasses = aneFfnRows ? 1 : layers`）、`kPrefillGroupChunks` | 保留 |
| `Runtime.cpp` | `prefillAsync`（3043）、`passLayers`（1246）、`prefillWork`（1254-1271，EST-C）、池（526-704）、`loadReads`（3480，EST-C）、spilled reads（3494-3530） | 混合 |
| `ExpertStore.*`、`ExpertChunkLayout.hpp`、`SlotRecycler.*` | 弹性专家存储（`kExpertFloorShare = 0.85`，`kExpertElasticChunkBytes = 1 GiB`）；near-ghost 和 tail 计数为 EST-V | 保留 |
| `KvBlockCache.*` | 热块缓存（floor 390 / 195 MiB，`resolve` 没有回退） | 保留 |
| `KvHistoryStream.*`、`SpilledKvReads.*` | 预填充读溢出的历史 | 保留 |
| `Model.hpp` | 模型接口：`commandLimits`、`prefillWork`、`sample()`、`pools()`、`releaseIdleBuffer`、`spilledReadShortfall` / `reserveSpilledReads`；遗留的 `reclaimIdleState`、`hostMilliseconds` 等（清理分支删掉了一部分） | 混合 |

### A.5 其他

| 位置 | 内容 |
|---|---|
| `runtime/engine/Status.cpp`、`Status.hpp`、`server/metrics.py` | schema 7 状态和指标（§3.13） |
| `runtime/engine/Protocol.hpp`、`server/protocol.py` | 线协议 v12，状态 schema 7 |
| `runtime/Supervision.hpp`、`server/supervision.py`、`server/serve_record.py` | 监督常量、重启策略、隔离、serve record |
| `runtime/ops/KernelCalibration.hpp`、`KernelSelection.*`、`DeviceCapabilities` | kernel 校准与选择 |
| `app/`（`Drawer.swift`、`WebPage.swift`、`Transcript/*`、`MainView`、`Server.swift`） | App |
| `dev/tests/engine/` | 引擎测试、harness、假件、engine-sim、golden |
| `dev/tests/fixtures/` | 录制负载（价值测试用）、服务夹具 |
| `dev/benchmarks/serving/` | 服务矩阵 |
| `dev/tools/check_architecture.py` | 架构规则 R1-R3 |

### A.6 `3874a2f1` 上新增的文件（main 上都没有）

| 类别 | 文件 | 行数 |
|---|---|---|
| runtime | `CostModel` | 182 / 366 |
| | `MemoryValue.hpp` | 205 |
| | `MemoryLedger` | 466 / 1,123 |
| | `StreamedMemoryPlan` | 117 / 306 |
| | `WeightResidency` | 75 / 78 |
| | `EngineMemory.cpp` | 1,157 |
| | `EngineAllocate.hpp` | 172 |
| | `KernelCalibration.hpp` | 473 |
| | `KernelSelection` | 81 / 294 |
| | `Supervision.hpp` | 64 |
| | `SmallSolve.hpp` | 40 |
| | `WorkFormulas.hpp` | 70 |
| | `CommandCost.hpp` | 97 |
| | `ElasticPool.hpp` | 74 |
| | `ModelPools.hpp` | 183 |
| server | `supervision.py` | 187 |
| | `serve_record.py` | 40 |
| 测试与工具 | `engine_sim_test.cpp` | 1,705 |
| | `pressure_serving_test.mm` | 304 |
| | `cost_model_test.cpp` | 700 |
| | `memory_value_test.cpp` | 449 |
| | `memory_ledger_test.cpp` | 3,946 |
| | `matrix.py` | 3,148 |
| | `check_architecture.py` | 293 → 443（已有文件，增长） |

---

## 附录 B：时间线

### 10-28 至 10-31

| 时间 | 事件 |
|---|---|
| 10-28 03:52 | 打 1.5.0 的 tag（`cabbad27`，PR #612） |
| 10-28 至 10-31 | 12 个 PR 合入 main（§6.1） |
| 10-31 02:18 | #645 合入（`031a25a6`） |
| 10-31 | 发现存储只缩不长（Studio 18G，3 并发，按请求 decode 中位数 32.9 对 48.9 tok/s）。owner 叫停点修补，批准 4 步内存重设计。回收梯成为 PR #650，是这期间代理开的唯一一个上游 PR（同期 owner 自己开了 #648、#649、#651） |
| 10-31 16:45 | 定下"今晚发布"的计划。10-31 的 15 模块评审（不是 11-01 的全仓评审 REPORT）有 62 个发现，确认 51 个（8 个 P1），分成修复组 A1、A2、B1、B2、C |
| 10-31 18:10 | 加入多 lane 预填充 pass |
| 10-31 约 23:10 | 测量后丢掉 QSA verify 行列表和 PLE 预取减半；保留 prefill splits |

### 11-01（周日）

| 时间 | 事件 |
|---|---|
| 00:00 | 二分发现"回退"来自随机到达顺序，改用固定顺序 A/B。发现 spill-wait 的真 bug：一个 decode lane 饿了约 506 s |
| 02:45 | #648、#649、#651 进发布 |
| 05:35 | 取证发现 main 在一次 M5 Pro 跑里从 t = 381 s 到 766 s 没有给 decoder 任何输出（指标的最大间隔 424.8 s） |
| 05:50 | pass 因第 4 轮的 2 个 P1 崩溃被移出 |
| 08:10 | 代理因用量上限停转约 2.5 小时 |
| 10:50 | pass、near-ties、内存步骤 2-4 全部放回 |
| 11:30 | 交接给新会话 |
| 12:00-15:45 | 重跑 1.5.0 基线：main 在压力下比 1.5.0 差，补丁链没追回 |
| 13:40 | 停掉所有改代码的代理，开始全仓只读评审（REPORT） |
| 16:05 | 决定：不从 main 发布；1.6.0 是框架内的重构 |
| 17:40 | 六份组件设计完成 |
| 17:55 | 补做尝试记录（ATTEMPTS） |
| 17:58 | 全范围，什么都不推迟 |
| 18:00 | "先写完代码，不跑长测试"；目标 11-02 晚上发布 |
| 19:00 | 设计稿批准 |
| 19:25 | 尝试覆盖检查：39 覆盖、69 部分、4 缺失、13 矛盾 |
| 19:45-21:45 | 一系列 owner 决定（§2.6）：HiSparse 按价值；Flash-Next 的 floor 由数据决定；5 % 规则；四个待答问题 OQ1-OQ4；DFlash2 永远开着；借鉴 9 项 |
| 20:20-23:40 | 同类引擎调研（Strata、oMLX、mlx-serve、vLLM、SGLang、MoE 卸载系统、Strata PC、TensorFold，§3.18） |
| 21:15 | C0 完成 |
| 22:00 | fork 基点 `6fd54f61`，各轨道开始 |

### 11-02（周一）

| 时间 | 事件 |
|---|---|
| 08:20 | 合并 1（`bb3d6fd2`） |
| 12:40 | capfix 完成（`8cb691c8`；capfix 是合并 1 之后对复核确认的 11 个问题做的修复轮） |
| 14:05 | M3a 集成，引擎测试挂 66 个 |
| 15:55 / 16:00 | 提出 S-P5（冷启动高估 47 %）；16:00 owner 接受"2 倍以内" |
| 18:30 | 深度审计 1：55 个代理，确认 12 个问题 |
| 19:15 | 第一次发现集成后的引擎还没跑过真模型 |
| 19:31 | 完整性检查：161 项，88 完成、55 部分、13 缺失、5 只有计划 |
| 20:15 | owner 问估计为什么一再滑坡；改用续跑脚本，评审轮数上限改为 2 |
| 21:45 | M6 smoke a：自我 supersede，95 % 的 pass 命令被放弃 |
| 22:24 | M3a 测试全部通过 |
| 22:40 | 诊断出 u = 301 ms 和自我 supersede |
| 23:32 | 第一次早读：cand 对 1.5.0 −5.8 % |
| 23:55 | floorcap 对照 +0.6 % |

### 11-03（周二）

| 时间 | 事件 |
|---|---|
| 00:12 / 00:36 | owner 接受三条调度取舍和三条账本取舍 |
| 01:45 | 在 Studio 上部署代理 worker |
| 02:55 | early b：27B 和 ANE 的问题 |
| 03:10 | ANE 行下限 |
| 03:20 | 只修 P0/P1/用户可见的问题，其余推迟（推迟表建立时是 D1-D11，D12、D13 后来补上） |
| 03:55 | owner 选定 A30，之后委托协调者决定 |
| 04:01 | merge:M3a（`bd2d0b1f`） |
| 06:08 | READY1（`f0724a9d`） |
| 07:44 | smoke b：M6 10 项过 9 项，发现 A30 被 host claim 收回 |
| 09:17 | a30-host 合入；否决 `430b6d60` |
| 10:46 | READY-FINAL（`5d2d5348`） |
| 11:00 | owner：太复杂，要一条"请求优先"的简单规则（`2b326a37` / `ccc89af3`），取代 start-hold 和 resume-price；A30 定价保留。Studio worker 已花约 $1,474 |
| 11:12 | smoke c 发现 N1 |
| 11:20 | 硬冻结 |
| 11:23 | 停掉 Studio worker 的全部任务，工作移回协调机 |
| 11:35 | owner 撤销冻结：T6、N1、清理都进 |
| 12:04 | READY-FINAL3 = `a7dac91a` |
| 12:06 | **终测开始** |
| 12:40 | PR 分类；清理分支写完 |
| 12:55 | issue 分类 |
| 13:25 | G4a/G4b 改为只报告 |
| 13:35-14:20 | M6 分析，找到 host claim 拿走存储的根因 |
| 14:45 | 清理分支在 M6 上跑门 |
| 14:50 | M5 Pro 收益的解释：每行专家字节 −62 % |
| 15:58 | 冻结调查和 HiSparse 内存调查完成 |
| 16:10 | 收益归因 |
| 16:50 / 17:05 | Studio、mini、M6 打分 |
| 17:15 | 确认 Studio 300 token TTFT 回退（G7，阻塞） |
| 17:30 | owner：先停下修 |
| 17:40 | 短 TTFT 的原因 |
| 18:03 | `e1310a4d` + `3874a2f1` |
| 18:17 | 部署到 Studio、M6、mini |
| 18:30 / 18:45 | Studio +3.1 %，mini −9.1 % |
| 18:40 | 交接；两个分支推到本 fork |
| 19:12 | M6 serve1 不变；FN churn A3/B3 回到 1.5.0 水平 |
| 19:55 | M6 修复 1 打分：G1 1.004 PASS，G2 latency 1.020 INC，G3 0.938 INC；serve1 G7 全部 PASS |

### 集成头的谱系

`0bd50301`（pass）→ `6fd54f61`（fork 基点）→ `bb3d6fd2` → `8cb691c8` → `t/m3a`（`92a0bd5e` → `908c3701`）→ `bd2d0b1f` → `f0724a9d` → `54150ebf` → `46f909fb` → `23b84284` → `5d2d5348` → `ccc89af3` → `a7dac91a` → `e1310a4d` → `3874a2f1`

清理分支是另一条：`a7dac91a` → `d9585b59` → `5fc74205`。

---

## 附录 C：术语表

| 术语 | 含义 |
|---|---|
| **账本（ledger）** | `MemoryLedger`。所有内存都记在账户上，所有分配都属于一个 claim；它决定一个需求是现在满足、等待（Pending）还是满足不了（Short），从哪些来源取内存，以及池什么时候增长 |
| **claim** | 一次内存需求的身份，带类别（Floor、InService、Start、Optional、Growth、Host）、rank 和时机（CanWait、Immediate）。每个请求一个，引擎自己的分配用临时 claim |
| **rank** | `ClaimRank{priority, decoding, completedTokens, earliness}`。等待中的 claim 为所有比它低的持有空间；decode 在 prefill 前面 |
| **landing（落地）** | 正在进行的传输（比如写到 SSD）完成后回到空闲的内存 |
| **owed（欠拷贝）** | 还没开始的拷贝 |
| **Returning** | `{landing, owed}`，"正在回来路上的内存" |
| **victim** | 可以被拿走以腾出内存的单元：缓存、空闲 buffer、lane 转换、pass 命令、池的 chunk |
| **dry run** | 只计算不取走的一轮挑选；之后的 `take` 是全有或全无 |
| **Resident / Spill** | 一个 lane 的 KV 模式。Resident：全部 K/V 在内存里。Spill：大部分页只留 Index 页，K/V 在 SSD |
| **HiSparse** | 团队对 Flash-Next 块稀疏注意力的叫法（`kv::Reads::SelectedBlocks`）。有了它，KV 才能被 spill |
| **Full 页 / Index 页 / Slim 页** | KV 页的三种形态。Full：完整 K/V 加索引，int8 时 423,936 B。Index：只有索引键，24,576 B。Slim：滑动窗口模型裁剪后的形态 |
| **热块缓存（hot block cache）** | `KvBlockCache`。Spill lane decode 时缓存选中的 K/V 块，按层组织 ARC 槽。floor 为 390 MiB 起，每多一个 lane +195 MiB |
| **历史流（history stream）** | `KvHistoryStream`。预填充时后台读溢出的历史，每条命令只有一个流 |
| **专家存储（expert store）** | 流式 MoE 模型在内存里缓存的专家权重，按 chunk 组织 |
| **floor / cap** | 专家存储的下限和上限。floor 默认是 cap 的 0.85，可用 `--expert-memory SIZE` 精确指定；floor = cap 即不收缩 |
| **chunk** | 专家存储的单元。floor chunk 加上约 1 GiB 的弹性 chunk |
| **rebuild-only 池** | 只花重建代价、没有在用者的池：视觉 arena、嵌入行、空闲 staging |
| **band（带）** | 引擎 host 规则的状态：host 余量低于 1 GiB margin 时进入，回到 margin + 最大增长单元以上时退出。**不是** macOS 的压力级别 |
| **Warning / Critical** | macOS 的内存压力级别 |
| **pass** | 一条预填充命令按层切出的一段。行数不变，每段跑一部分层 |
| **command（命令）** | 一次提交给 GPU 的工作：一批 lane 的预填充或 decode |
| **lane** | 一个请求在批里的位置 |
| **contended（有竞争）** | 有同级或更高优先级的 lane 在 Decode 或 WaitingMask，即有人在等 |
| **G** | decode 间隔的目标时间常数，500 ms |
| **δ** | 设计修订 A49 定义的量：Spill pass 第一个注意力层重新启动历史流带来的每 pass 额外时间。从没测过 |
| **共同 tag** | 比较几棵树时只取它们都完成了的请求（按 tag 对齐）算 TTFT 和延迟均值 |
| **u** | 代价模型里每个 pass 的周转时间。从没被直接测过 |
| **supersede** | 在 pass 之间放弃当前命令，去跑新的规划 |
| **replay 点** | 提示词末尾附近（`promptReplayBoundary`）的状态捕获。下一轮可以从这里恢复，不必从头预填充 |
| **检查点（checkpoint）** | 预填充沿途的 GDN 状态捕获，可丢弃 |
| **GDN** | 线性注意力层（gated delta net）。递归状态不能从 KV 重建 |
| **ANE FFN 切分** | 把 27B 的 FFN 放到 Neural Engine 上跑。开着时一条命令不能分 pass，并且有最小行数（M6 27B 上是 512） |
| **DFlash2** | 推测解码的草稿模型，永远开着 |
| **work@900** | t = 900 s 时 prefill 行数 / R + 输出 token 数 / D（M6：R = 585、D = 46；Studio 和 M5 Pro：R = 1000、D = 75） |
| **churn** | 3 个客户端持续发长提示词的压力场景 |
| **顺序 A / B** | churn 首批请求的固定到达顺序：A = (2,0,1)，B = (0,1,2) |
| **serve1** | 空闲服务器上一次一个冷提示词：300、2K、14K token |
| **ABBA** | 参照、候选、候选、参照的交错顺序，用来抵消漂移 |
| **void** | 因为干扰或有效性规则被判不算数的跑 |
| **r150 / main / pass / ntie / intp / cand / floorcap** | 矩阵里的树名，见开头的阅读约定 |
| **G1-G10** | 门。G1 吞吐对 1.5.0；G2 延迟和 decode 对 1.5.0；G3 TTFT 对 main；G4 decode 间隔（阻塞形式：最大 ≤ 10 s，p99 ≤ 5 s；G4a/G4b 只报告）；G5 并发 TTFT；G6 内存充足时；G7 单请求对 pass 树；G8 健康；G9 账本自洽；G10 稠密、驻留、Flint、mini-M4 对 pass |
| **B17** | floor 默认值的判定：floorcap 对 value 默认 |
| **SSD 门** | 每单位 work 的磁盘读写字节和专家读取字节，不能确定高于两个发布参照 |
| **store line** | 存储震荡检查：一个回长的单元在短时间内被反复收回就算失败 |
| **价值模型（value model）** | 用"每字节值多少秒引擎工作"给内存定价的那套机制（`MemoryValue.hpp` 等），建议删除 |
| **代价模型（cost model）** | 调度器在线学习的 7 项命令代价（`CostModel`），建议删除 |
| **A30** | 本文指"回长的单元在视野内不应被收回"的规则及其核算。内部尝试记录里另有一个同名编号，指 35B 的回长退避实验，别搞混 |
| **N1、T6、T7** | 终测前最后几轮 smoke 的发现编号：N1 是 chunk 按自己的大小计价（`a7dac91a`）；T6 是同级到达不切已开始的行（`7e9063d2`）；T7 是越过状态边界的 KV 按 lane 增长判断（`2f6411fd`） |
| **smoke-b/F1** | smoke b 的发现，两份报告各有一个 F1：M6 那份是"回长的 1,023 MiB 单元 8.1 s 后被 host claim 收回"（§4.8），Studio 那份是"热块缓存只预留一个 extent 导致 `capacity_exhausted`"（§5.6）。和 F1 测试假件轨道无关 |
| **G4a / G4b** | G4 的两个按运行推导的界：G4a 看竞争预填充单元的 p95，G4b 看客户端间隔的 p99；界 = A09 定的上界 × (1 + 估计误差 e95) + 本次跑的 p50 间隔。11-03 起只报告 |
| **OQ1-OQ4** | 11-01 20:45 设计修订后提给 owner 的四个问题（§2.6） |
| **capfix** | 合并 1 之后，对复核评审确认的 11 个问题做的修复轮（11-02 12:40，`8cb691c8`） |
| **kit-g4** | 终测工具的一个修复：G4 的上界加上 ANE 行下限的单元时间 |
| **复核 T1-2、T6-5** | 11-01 那份复核文档对 REPORT 结论的逐条核验编号，和 smoke 的 T6、T7 无关 |
| **D1-D13** | 推迟的低影响项（§8.4） |
| **smoke a / b / c** | 终测前在 M6 等机器上用真模型做的冒烟轮次 |
| **C0、F1、K、I、S、M1-M3、C、V** | 1.6.0 的开发轨道（§2.4） |

**代码注释里常见的设计修订编号**

代码注释里大约有 49 处 `amend/…` 引用，指向不在仓库里的设计修订。最常见的含义如下，读代码时可以对照；删估计器时建议把这些引用换成能自解释的注释。

| 引用 | 含义 |
|---|---|
| `amend/M.md B2` | 各类 claim 能拿哪些单元的表（例如 Start 原来只能拿 Ordinary 单元，00:35 改为也可以拿空闲的 rebuild-only 缓存） |
| `amend/M.md B3` | macOS Warning 下的 Host claim 规则 |
| `amend/M.md B4` | 池的增长公式（后来被 A30 取代，但带内在被拒的 Start 旁边回长存储这一点仍引用它） |
| `amend/M.md B5` | CanWait 的 claim 先用自己正在回来的字节，再取 victim |
| `amend/M.md B6` | replay 点的 `reserveCapture` |
| `amend/M.md B7` | `EngineVictims` 里转换的资格和守卫 |
| `amend/M.md B8` | HiSparse 的定价（含 spill decode 项） |
| `amend/M.md B9` | 提前拷贝什么时候允许 |
| `amend/M.md B13` | 放置规则 |
| `amend/M.md B14` | claim 的等待（`syncClaims` 的 waiting 定义） |
| `amend/M.md B17` | floor 份额作为计划参数（`expertFloorShare`）；B17 判定即 floorcap 对 value |
| `amend/M.md A38` | 检查点拷贝：否决，保留场景 |
| `amend/S.md A04` | 捕获切分 |
| `amend/S.md A10`、`D2`、`D4` | 调度常量要推导出来；同级重组；"先发不带 I 流 pass 的 S"（按数据放弃） |
| `amend/S.md A103` | 启动规则（§3.5） |
| `amend/S.md A77` | VerifyBudget 下限"测了再定、不行就回退" |
| `amend/V.md A09` | G4 的按运行推导的界 |
| `amend/V.md A30` | 回长的单元在视野内不应被收回 |
| DESIGN §0 的 O3、O4、O9、O13、Q5、Q6 | `kPassOverheadShare` = 0.1；启动预热的取舍（保留约 2 s 的种子预热，否决更长的）；Flash-Next 的 floor 看数据定；R3 剩下的名字改名而不设白名单；spill decode 项；不改 IO 优先级 |
