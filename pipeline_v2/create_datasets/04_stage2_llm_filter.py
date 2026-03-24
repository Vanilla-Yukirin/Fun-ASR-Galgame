"""
阶段 04：LLM 精筛（Stage2）。

本文件负责：
1) 读取 Stage1 候选词（token_stats.stage1_keep=1）
2) 使用“可配置后端”进行二次筛选
3) 输出 stage2_vocab（token 级别标签）

说明：
- 默认使用 heuristic 模式（完全离线、可复现）。
- 可选 openai_compatible 模式（温度等参数来自 config）。
- 你要求的 temperature 参数已保留在 config 中。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request

from tqdm import tqdm

from common import load_config, now_iso, open_sqlite
from schema import clear_tables, ensure_schema


JAPANESE_PARTICLES = {
    "は",
    "が",
    "を",
    "に",
    "で",
    "と",
    "も",
    "の",
    "へ",
    "か",
    "な",
    "ね",
    "よ",
    "ぞ",
    "さ",
}


def heuristic_classify(token: str) -> tuple[int, float, str]:
    """
    离线启发式分类器（确定性）。

    返回：
    - keep(0/1)
    - score(0~1)
    - reason(文本)
    """
    t = token.strip()
    if not t:
        return 0, 0.0, "empty"

    if t in JAPANESE_PARTICLES:
        return 0, 0.05, "common_particle"

    if re.fullmatch(r"[0-9]+", t):
        return 0, 0.05, "pure_number"

    if re.fullmatch(r"[\W_]+", t):
        return 0, 0.05, "pure_symbol"

    # 全平假名且长度很短，通常不是“专有热词”
    if re.fullmatch(r"[\u3040-\u309f]+", t) and len(t) <= 2:
        return 0, 0.2, "short_hiragana"

    # 常见“疑似专有词”特征
    has_katakana = re.search(r"[\u30a0-\u30ff]", t) is not None
    has_kanji = re.search(r"[\u3400-\u9fff]", t) is not None
    has_upper = re.search(r"[A-Z]", t) is not None

    if has_katakana or has_kanji or has_upper or len(t) >= 3:
        return 1, 0.9, "likely_domain_term"

    return 0, 0.3, "low_confidence"


def openai_compatible_classify(token: str, llm_cfg: dict) -> tuple[int, float, str]:
    """
    OpenAI-compatible 接口调用。

    注意：
    - 该模式依赖网络与外部服务。
    - 若调用失败，会自动回退到 heuristic。
    """
    endpoint = llm_cfg.get("endpoint", "")
    model = llm_cfg.get("model", "")
    api_key_env = llm_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env, "")

    if not endpoint or not model or not api_key:
        return heuristic_classify(token)

    prompt = (
        "你是日语ASR热词筛选器。"
        "请判断给定词语是否应作为垂直领域热词。"
        "仅返回JSON：{\"keep\":0或1,\"score\":0到1,\"reason\":\"简短原因\"}。"
        f"\n词语: {token}"
    )

    payload = {
        "model": model,
        "temperature": float(llm_cfg.get("temperature", 0.0)),
        "top_p": float(llm_cfg.get("top_p", 1.0)),
        "messages": [
            {"role": "system", "content": "你是一个严格输出JSON的分类器。"},
            {"role": "user", "content": prompt},
        ],
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=llm_cfg.get("timeout", 60)) as resp:
            body = resp.read().decode("utf-8")
        obj = json.loads(body)
        content = obj["choices"][0]["message"]["content"]

        # 允许模型返回纯 JSON 字符串
        result = json.loads(content)
        keep = int(result.get("keep", 0))
        score = float(result.get("score", 0.0))
        reason = str(result.get("reason", "llm_response"))
        return keep, score, reason
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, ValueError):
        return heuristic_classify(token)


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段04：LLM 精筛")
    parser.add_argument("--config", required=True, help="配置文件路径（JSON）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    db_path = cfg["db"]["path"]
    llm_cfg = cfg.get("llm", {})
    backend = llm_cfg.get("backend", "heuristic")

    conn = open_sqlite(db_path)
    ensure_schema(conn)

    # 每次重跑阶段04，清空旧结果
    clear_tables(conn, ["stage2_vocab"])
    cur = conn.cursor()

    cur.execute("SELECT token FROM token_stats WHERE stage1_keep=1 ORDER BY token;")
    candidate_tokens = [row[0] for row in cur.fetchall()]

    insert_sql = """
    INSERT INTO stage2_vocab (token, llm_keep, llm_score, llm_reason, updated_at)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(token) DO UPDATE SET
        llm_keep=excluded.llm_keep,
        llm_score=excluded.llm_score,
        llm_reason=excluded.llm_reason,
        updated_at=excluded.updated_at;
    """

    rows = []
    batch_size = llm_cfg.get("batch_size", 1000)
    keep_count = 0

    print(f"[04] backend={backend}, temperature={llm_cfg.get('temperature', 0)}")

    for token in tqdm(candidate_tokens, desc="[04] Stage2 精筛"):
        if backend == "openai_compatible":
            keep, score, reason = openai_compatible_classify(token, llm_cfg)
        else:
            keep, score, reason = heuristic_classify(token)

        keep_count += int(keep)
        rows.append((token, int(keep), float(score), reason, now_iso()))

        if len(rows) >= batch_size:
            cur.executemany(insert_sql, rows)
            conn.commit()
            rows.clear()

    if rows:
        cur.executemany(insert_sql, rows)
        conn.commit()

    print("\n[04] 完成")
    print(f"- Stage1 候选词数: {len(candidate_tokens)}")
    print(f"- Stage2 保留词数: {keep_count}")

    conn.close()


if __name__ == "__main__":
    main()
