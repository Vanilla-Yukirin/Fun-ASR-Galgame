"""
阶段 02：日语分词与读音提取。

本文件负责：
1) 从 samples 读取 transcript
2) 使用 fugashi 进行分词（若环境缺失则降级到规则分词）
3) 提取每个 token 的读音（用于后续音近干扰词）
4) 写入 tokens 表

设计特点：
- 可复现：固定排序 + 固定规则
- 可打断：批量提交，支持中断后重跑
"""

from __future__ import annotations

import argparse
import re

from tqdm import tqdm

from common import chunked_commit, json_dumps, load_config, now_iso, open_sqlite
from schema import clear_tables, ensure_schema


def create_tokenizer() -> tuple[str, object]:
    """
    尝试构建 fugashi 分词器。

    返回：
    - (tokenizer_name, tokenizer_obj)
    """
    try:
        import fugashi

        return "fugashi", fugashi.Tagger()
    except Exception:
        # 如果环境里没有 fugashi，则回退到规则分词
        return "regex_fallback", None


def regex_tokenize(text: str) -> tuple[list[str], list[str]]:
    """
    规则分词兜底方案。

    说明：
    - 仅在 fugashi 不可用时使用
    - 读音字段直接复用 token 本身（占位）
    """
    # 规则：
    # 1) 连续日文/汉字作为一个片段
    # 2) 连续字母数字作为一个片段
    # 3) 其他非空白字符单独保留
    pattern = r"[\u3040-\u30ff\u3400-\u9fff]+|[A-Za-z0-9]+|[^\s]"
    tokens = re.findall(pattern, text)
    readings = tokens[:]
    return tokens, readings


def token_to_reading(word) -> str:
    """
    从 fugashi token 中提取读音。

    不同词典字段名可能不同，这里做了多重兼容。
    """
    surface = str(getattr(word, "surface", ""))
    feature = getattr(word, "feature", None)

    if feature is None:
        return surface

    # 常见字段名兼容
    for attr in ("kana", "pron", "reading"):
        if hasattr(feature, attr):
            value = getattr(feature, attr)
            if value and value != "*":
                return str(value)

    # 再尝试按可迭代特征结构兜底
    if isinstance(feature, (tuple, list)):
        for value in feature:
            if value and value != "*":
                return str(value)

    return surface


def fugashi_tokenize(tagger, text: str) -> tuple[list[str], list[str]]:
    """使用 fugashi 分词并返回 (tokens, readings)。"""
    tokens = []
    readings = []
    for word in tagger(text):
        surface = str(word.surface).strip()
        if not surface:
            continue
        tokens.append(surface)
        readings.append(token_to_reading(word))
    return tokens, readings


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段02：日语分词")
    parser.add_argument("--config", required=True, help="配置文件路径（JSON）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    db_path = cfg["db"]["path"]
    batch_size = cfg.get("tokenize", {}).get("batch_size", 5000)

    tokenizer_name, tokenizer_obj = create_tokenizer()
    print(f"[02] 当前分词器: {tokenizer_name}")

    conn = open_sqlite(db_path)
    ensure_schema(conn)

    # 每次重跑阶段02，清空 tokens，保证输出可控
    clear_tables(conn, ["tokens"])

    cur = conn.cursor()
    cur.execute("SELECT utt_id, transcript FROM samples ORDER BY utt_id;")
    rows = cur.fetchall()

    insert_sql = """
    INSERT INTO tokens (utt_id, tokens_json, readings_json, tokenizer_name, updated_at)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(utt_id) DO UPDATE SET
        tokens_json=excluded.tokens_json,
        readings_json=excluded.readings_json,
        tokenizer_name=excluded.tokenizer_name,
        updated_at=excluded.updated_at;
    """

    batch = []
    inserted = 0

    for utt_id, transcript in tqdm(rows, desc="[02] 分词处理中"):
        text = transcript or ""

        if tokenizer_name == "fugashi":
            tokens, readings = fugashi_tokenize(tokenizer_obj, text)
        else:
            tokens, readings = regex_tokenize(text)

        batch.append((utt_id, json_dumps(tokens), json_dumps(readings), tokenizer_name, now_iso()))

        if len(batch) >= batch_size:
            inserted += chunked_commit(cur, conn, batch, insert_sql)

    inserted += chunked_commit(cur, conn, batch, insert_sql)
    print(f"[02] 分词完成，写入 tokens: {inserted}")

    conn.close()


if __name__ == "__main__":
    main()
