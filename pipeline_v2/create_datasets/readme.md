# create_datasets（临时说明文档）

> 位置：`pipline_v2/create_datasets/readme.md`
>
> 说明：这是当前阶段的临时说明文档，目标是帮助你快速跑通“从原始清单到 ChatML 导出”的可复现流程。

## 1. 目录用途

本目录用于构建**数据集重构流水线**（而不是训练脚本），重点是：

- 可复现：同输入 + 同配置 + 同模型参数 => 同输出
- 可打断：每个阶段都可以单独运行
- 可重跑：你手动决定从哪一步继续（例如改了 05，就重新跑 06/07/08/09）

## 2. 文件说明

- `common.py`：通用工具（稳定哈希、配置读取、SQLite 连接、路径归一化等）
- `schema.py`：SQLite 表结构定义与清表函数
- `config_template.json`：模板配置

阶段脚本（按顺序）：

- `01_ingest_to_sqlite.py`
  - 读取 `scp/text`
  - 统一为 `samples` 表
  - 生成稳定 `utt_id`

- `02_tokenize_ja.py`
  - 日语分词（优先 fugashi）
  - 写入 `tokens`

- `03_stage1_stats.py`
  - 统计 TF/DF/IDF
  - 阈值筛候选词
  - 写 `token_stats` 与 `utt_stage1`
  - 可选输出 IDF 图

- `04_stage2_llm_filter.py`
  - 对 Stage1 候选词做 LLM 精筛
  - 默认 `heuristic`，可切 `openai_compatible`
  - 温度参数来自 config（可设为 0）

- `05_stage3_merge_hotwords.py`
  - 合并 Stage1 与 Stage2（默认交集）
  - 输出最终词表与每条样本热词

- `06_split_hash.py`
  - 稳定哈希划分 `95% train / 5% eval_full`
  - 在 eval_full 内确定性选 `eval_fast_5k`

- `07_prepare_distractors.py`
  - 构建随机干扰词池
  - 构建音近干扰词候选

- `08_export_chatml.py`
  - 导出三份 ChatML JSONL：
    - `train_chatml.jsonl`
    - `eval_fast_5k_chatml.jsonl`
    - `eval_full_5pct_chatml.jsonl`
  - 导出时按 `profile` + `dataset_id` 拼绝对路径前缀
  - 训练集热词注入比例默认 30%（可配）

- `09_console_summary.py`
  - 控制台汇总，不落地报告文件

## 3. 运行示例

先复制模板配置：

```bash
cp pipline_v2/create_datasets/config_template.json pipline_v2/create_datasets/config_local.json
```

然后按顺序运行（你可以手动中断并从任意阶段继续）：

```bash
python pipline_v2/create_datasets/01_ingest_to_sqlite.py --config pipline_v2/create_datasets/config_local.json
python pipline_v2/create_datasets/02_tokenize_ja.py --config pipline_v2/create_datasets/config_local.json
python pipline_v2/create_datasets/03_stage1_stats.py --config pipline_v2/create_datasets/config_local.json
python pipline_v2/create_datasets/04_stage2_llm_filter.py --config pipline_v2/create_datasets/config_local.json
python pipline_v2/create_datasets/05_stage3_merge_hotwords.py --config pipline_v2/create_datasets/config_local.json
python pipline_v2/create_datasets/06_split_hash.py --config pipline_v2/create_datasets/config_local.json
python pipline_v2/create_datasets/07_prepare_distractors.py --config pipline_v2/create_datasets/config_local.json
python pipline_v2/create_datasets/08_export_chatml.py --config pipline_v2/create_datasets/config_local.json
python pipline_v2/create_datasets/09_console_summary.py --config pipline_v2/create_datasets/config_local.json
```

## 4. 关键规则（当前版本）

- 划分规则：`eval_ratio=0.05`（5%）
- 快速验证集：`eval_fast_size=5000`
- 训练集热词注入：`train_hotword_ratio=0.30`
- 干扰词比例：`50% 随机无关 + 50% 音近`

## 5. 注意事项

- 该版本默认 `speech_length=-1`，`text_length=len(transcript)`（简化模式）。
- 如果后续训练强依赖精确长度，可在导出阶段扩展“真实长度计算逻辑”。
- `04_stage2_llm_filter.py` 切到在线 LLM 时，建议固定模型版本并设置 `temperature=0`。
