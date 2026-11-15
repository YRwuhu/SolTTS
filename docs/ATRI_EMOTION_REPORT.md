# ATRI 19 类情感：标注与方向向量最终报告

生成时间：2026-11-15
语料：`F:\CNBackups\ATRI\raw`　2,222 条 48 kHz 单声道 WAV（约 2 小时 26 分）
标签体系：`emotions.json` v1.0.0 / prompt `atri19-v1`

---

## 一、标注完成情况：100%

| 项目 | 数量 |
|---|---:|
| 计划标注 | 2,222 |
| **已完成** | **2,222（100%）** |
| 缺失 / 重复 | 0 / 0 |
| 需复核 | 301（13.5%） |
| 复标分歧 | 0 |
| 无可辨认人声 | 62 |
| 使用了转写辅助 | 1,130（50.9%） |

### 使用的模型

权限在账号侧反复失效，因此分了多次、多个模型完成：

| 模型 | 条数 |
|---|---:|
| qwen3.8-omni-flash | 903 |
| qwen3-omni-flash | 868 |
| qwen3.5-omni-plus | 451 |

### 用量

- total tokens：3,737,250（prompt 3,610,069 / completion 127,181）
- 按公开价粗估约 **¥3.23**（估算，非账单）

### 类别分布

| 类别 | id | 数量 | 占比 |
|---|---|---:|---:|
| 高兴 | happy | 353 | 15.9% |
| 害羞 | shy | 288 | 13.0% |
| 平静（基线） | calm | 277 | 12.5% |
| 调皮 | playful | 166 | 7.5% |
| 惊讶 | surprised | 161 | 7.2% |
| 无奈 | helpless | 149 | 6.7% |
| 心动 | lovestruck | 147 | 6.6% |
| 兴奋 | excited | 132 | 5.9% |
| 疑惑 | confused | 100 | 4.5% |
| 哭泣 | crying | 94 | 4.2% |
| 担心 | worried | 81 | 3.6% |
| 自信 | confident | 54 | 2.4% |
| 认真 | serious | 48 | 2.2% |
| 慌张 | flustered | 45 | 2.0% |
| 生气 | angry | 43 | 1.9% |
| 紧张 | nervous | 41 | 1.8% |
| 厌恶 | disgust | 22 | 1.0% |
| 害怕 | fearful | 13 | 0.6% |
| 难为情 | embarrassed | 8 | 0.4% |

### 转写辅助策略

`esd.list` 日文转写作为辅助上下文。提示词明确：**台词用于判断意图与语境，情绪强弱必须以声音为准；
台词与声音冲突时以声音为准并在 evidence 说明**。这条规则让平静基线从最初 15 条增至 277 条，
是方向质量改善的关键。

---

## 二、方向向量：18 个非平静方向全部生成

方法：单说话人平静基线差分 `τ_e = mean_i(x_e^i) − mean_j(x_calm^j)`，
按章节分组留出验证（避免同场景相邻台词泄漏），bootstrap 40 轮稳定性检验。

| 类别 | 样本 | 状态 | 留出余弦 | Bootstrap |
|---|---:|---|---:|---:|
| happy | 242 | ready | 0.922 | 0.99 |
| shy | 194 | ready | 0.908 | 0.99 |
| playful | 137 | ready | 0.868 | 0.99 |
| lovestruck | 129 | ready | 0.614 | 0.99 |
| excited | 96 | ready | 0.943 | 1.00 |
| helpless | 80 | experimental | 0.769 | 0.98 |
| confused | 74 | ready | 0.587 | 0.98 |
| crying | 70 | ready | 0.906 | 1.00 |
| surprised | 58 | ready | 0.944 | 0.99 |
| worried | 57 | ready | 0.751 | 0.98 |
| confident | 42 | ready | 0.798 | 0.98 |
| serious | 38 | experimental | **0.353** | 0.92 |
| angry | 37 | ready | — | 0.99 |
| flustered | 37 | ready | 0.871 | 0.99 |
| nervous | 28 | ready | 0.727 | 0.98 |
| disgust | 14 | experimental | — | 0.94 |
| fearful | 7 | experimental | — | 0.97 |
| embarrassed | 5 | experimental | — | 0.97 |

**运行时状态：19 类全部可用**（13 ready + 5 experimental + 平静基线 + 0 unavailable）。

### 数据量对质量的影响（同一套流程对比）

| 类别 | 903 条 | 1,680 条 | 2,222 条 |
|---|---:|---:|---:|
| happy 留出余弦 | 0.539 | 0.914 | 0.922 |
| excited | 0.718 | 0.949 | 0.943 |
| shy | 0.131（已降级） | 0.903 | 0.908 |
| confused | 0.395 | 0.260 | 0.587 |
| serious | — | 0.466 | 0.353 |

结论：**数据量是方向质量的主导因素**；但 `serious` 这类样本少、语义又与"平静/自信"高度重叠的类别，
即使补量也难以稳定。

---

## 三、必须正视的局限

### 1. 方向相关性依然很高

| 方向对 | 余弦 |
|---|---:|
| flustered ~ excited | 0.973 |
| happy ~ playful | 0.957 |
| shy ~ lovestruck | 0.940 |
| happy ~ excited | 0.939 |
| excited ~ playful | 0.936 |
| flustered ~ playful | 0.931 |

若干标签在说话人嵌入空间中几乎同向。**界面能选 19 类，不代表 19 类听感都能区分。**
这是"单向量加性注入"的固有局限，增加数据无法解决；需要簇内正交化（可能损伤单个情绪强度）
或改用独立的情感条件编码器。

### 2. 单说话人限制

方向全部来自 ATRI 一人。没有跨说话人配对来扣除音色差异，因此方向里混入了"这一句的说话状态"；
**跨音色迁移完全未验证**，不能假设对其他音色同样有效。

### 3. 主观验收未完成

已验证的是：注入生效、ge 条件改变、F0/时长改变、留出方向稳定。
**未验证**的是：真人听感是否认同目标情绪。这需要盲听测试。

### 4. 低样本类别

`fearful`(7)、`embarrassed`(5)、`disgust`(14) 样本量仍偏低，已标为 experimental 并在 API 显示原因。

---

## 四、工程改造（已完成并验证）

### 核心

- 新增 `SolTTS.Core/Core/EmotionCatalog.cs`：19 类唯一权威定义；接受 id / 中文 / 英文 / 旧别名
  （`sad→crying`、`neutral→calm` 等）。
- 重写 `EmotionDirectionController`：
  - 严格校验维度（必须 20480）、NaN/Infinity、零向量、单位范数；
  - manifest 缺项逐条报错；拒绝路径穿越文件名；
  - 暴露 `Ready / Experimental / Unavailable / Baseline` 四态与具体原因。
- `TtsEngineV1`：未知标签、无向量类别、非法 α 一律抛明确异常；`calm`/α=0 走原始路径。
- `TtsEngineV2`：显式情感返回 `NotSupportedException`，不再静默忽略。
- 修复 **GET `/api/tts/predict-stream` 丢失情感参数**（WebUI 流式此前实际无效）。
- 新增 **`GET /api/tts/emotions`**：19 类中文名、可用状态、样本数、一致性。
- WebUI 下拉动态生成，不可用项禁用并显示原因；错误显示后端原文。
- CLI 支持中文与别名，非法标签列出全部 19 类。

### 资源与清理

- 向量已安装到 4 处运行时目录，内容一致。
- 旧 6 类 CSEMOTIONS 向量备份于 `E:\AI\SolTTS\_backup\emotion_vectors_csemo6`。
- **chenxing 已删除**（源码 9 个模型 721 MB + 2 条音频；运行包 7 个模型 570 MB + 2 条音频），
  默认音色改为 ATRI。
- `config.yaml` 清理了指向发布包的绝对路径污染。

### 验证结果

| 项目 | 结果 |
|---|---|
| 情感单元测试 | **48 项全部通过** |
| POST 平静 / happy / 中文标签 | 200，日志确认注入 |
| POST 未知标签 | 400，列出全部 19 类 |
| GET 流式 + 情感 | 200，修复后生效 |
| GET 流式 + 非法标签 | 400（写响应头之前拒绝） |
| POST 流式会话 | 200，注入生效 |
| CLI `-e happy` / `-e 兴奋` | 注入并保存 |
| V2 + 情感 | 明确拒绝（单元测试覆盖） |
| 运行包 | 自包含发布、启动、13 ready / 5 experimental / 0 unavailable |

> 注：`MixedLanguageG2PTests` 有 2 项失败，未改动任何 G2P 代码或测试，属仓库既有问题。

---

## 五、建议的后续工作

1. **人工盲听**：每个 ready 类别抽 3–5 条 α=0.3 的合成音频，与平静对比确认听感。
2. **处理高相关对**：对 `flustered/excited/playful`、`happy/excited` 等簇做正交化试验并试听对比。
3. **跨音色验证**：若有其他音色，需单独测迁移效果。
4. **稀缺类补录**：`fearful`、`embarrassed`、`disgust` 若要在实际使用中可靠，需要补真实样本。

---

## 六、文件清单

| 路径 | 说明 |
|---|---|
| `F:\CNBackups\ATRI\emotion_annotation\labels.jsonl` | 2,222 条最终标注 |
| `...\review_queue.csv` | 301 条待复核 |
| `...\failures.jsonl` | 调用失败审计（含账号级错误） |
| `...\summary.md` / `summary.json` | 统计 |
| `E:\AI\SolTTS\tools\emotion_annotation\` | 标注工具（schema / qwen_client / annotate / run_until_done） |
| `E:\AI\SolTTS\tools\emotion\extract_tau_atri.py` | 向量提取 |
| `E:\AI\SolTTS\tools\emotion\install_emotion_vectors.py` | 校验并安装 |
| `E:\AI\SolTTS\resources\emotion_vectors_atri19_candidate\` | 最终候选包（含 diagnostics） |
| `E:\AI\SolTTS\docs\ATRI_EMOTION_REPORT.md` | 本报告 |