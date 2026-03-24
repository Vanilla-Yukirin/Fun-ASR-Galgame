"""
阶段 03：统计学热词候选筛选（Stage1）。

本文件负责：
1) 从 tokens 表统计 TF / DF / IDF
2) 按阈值筛出“潜在热词候选”
3) 写入 token_stats 与 utt_stage1
4) 可选输出统计可视化图

说明：
- 这是“先大筛”的阶段，尽量减少后续 LLM 成本。
- 本阶段不区分 train/eval，统一处理全量数据。
"""

from __future__ import annotations

import argparse
import math
import os
import re
from collections import Counter

from tqdm import tqdm

from common import json_dumps, json_loads, load_config, now_iso, open_sqlite
from schema import clear_tables, ensure_schema


def is_candidate_token(token: str, min_len: int) -> bool:
    """
    判断 token 是否参与统计筛选。

    规则可后续再调，目前是稳健且通用的一版：
    - 长度 >= min_len
    - 不能全是标点
    """
    if not token:
        return False
    if len(token) < min_len:
        return False
    if re.fullmatch(r"[\W_]+", token):
        return False
    return True


def maybe_plot(token_rows: list[tuple], output_path: str) -> None:
    """可选绘制 IDF 分布图（如果环境有 matplotlib）。"""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[03] 未检测到 matplotlib，跳过可视化输出")
        return

    idf_values = [row[3] for row in token_rows]
    if not idf_values:
        print("[03] IDF 数据为空，跳过可视化输出")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(idf_values, bins=100)
    plt.title("Stage1 IDF Distribution")
    plt.xlabel("IDF")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[03] 可视化已输出: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段03：统计学热词候选筛选")
    parser.add_argument("--config", required=True, help="配置文件路径（JSON）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    db_path = cfg["db"]["path"]

    stage_cfg = cfg.get("stage1", {})
    min_df = int(stage_cfg.get("min_df", 5))
    max_df_ratio = float(stage_cfg.get("max_df_ratio", 0.2))
    min_idf = float(stage_cfg.get("min_idf", 0.0))
    min_token_len = int(stage_cfg.get("min_token_len", 2))
    enable_plot = bool(stage_cfg.get("plot", True))
    plot_path = stage_cfg.get("plot_path", "pipline_v2/create_datasets/stage1_idf_hist.png")

    conn = open_sqlite(db_path)
    ensure_schema(conn)

    # 清空当前阶段的目标表
    clear_tables(conn, ["token_stats", "utt_stage1"])
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM tokens;")
    total_docs = cur.fetchone()[0]
    if total_docs == 0:
        print("[03] tokens 表为空，请先执行 02_tokenize_ja.py")
        conn.close()
        return

    tf_counter = Counter()
    df_counter = Counter()

    # ---------- 第一遍：统计 TF / DF ----------
    cur.execute("SELECT tokens_json FROM tokens ORDER BY utt_id;")
    token_rows = cur.fetchall()

    for (tokens_json,) in tqdm(token_rows, desc="[03] 统计 TF/DF"):
        tokens = json_loads(tokens_json, default=[])
        filtered = [t for t in tokens if is_candidate_token(str(t).strip(), min_len=min_token_len)]

        tf_counter.update(filtered)
        df_counter.update(set(filtered))

    # ---------- 计算 IDF 并写 token_stats ----------
    token_stats_rows = []
    for token in sorted(df_counter.keys()):
        df = int(df_counter[token])
        tf = int(tf_counter[token])
        idf = math.log((total_docs + 1) / (df + 1)) + 1.0
        df_ratio = df / total_docs

        keep = int(df >= min_df and df_ratio <= max_df_ratio and idf >= min_idf)
        token_stats_rows.append((token, df, tf, idf, keep, now_iso()))

    insert_token_stats_sql = """
    INSERT INTO token_stats (token, df, tf, idf, stage1_keep, updated_at)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(token) DO UPDATE SET
        df=excluded.df,
        tf=excluded.tf,
        idf=excluded.idf,
        stage1_keep=excluded.stage1_keep,
        updated_at=excluded.updated_at;
    """
    cur.executemany(insert_token_stats_sql, token_stats_rows)
    conn.commit()

    keep_set = {row[0] for row in token_stats_rows if row[4] == 1}

    # ---------- 第二遍：生成每条样本的候选词 ----------
    cur.execute("SELECT utt_id, tokens_json FROM tokens ORDER BY utt_id;")
    utt_rows = cur.fetchall()

    insert_utt_stage1_sql = """
    INSERT INTO utt_stage1 (utt_id, candidates_json, updated_at)
    VALUES (?, ?, ?)
    ON CONFLICT(utt_id) DO UPDATE SET
        candidates_json=excluded.candidates_json,
        updated_at=excluded.updated_at;
    """

    batch = []
    batch_size = cfg.get("stage1", {}).get("batch_size", 5000)
    for utt_id, tokens_json in tqdm(utt_rows, desc="[03] 构建样本候选词"):
        tokens = json_loads(tokens_json, default=[])
        candidates = sorted(
            {
                str(t).strip()
                for t in tokens
                if is_candidate_token(str(t).strip(), min_len=min_token_len) and str(t).strip() in keep_set
            }
        )
        batch.append((utt_id, json_dumps(candidates), now_iso()))

        if len(batch) >= batch_size:
            cur.executemany(insert_utt_stage1_sql, batch)
            conn.commit()
            batch.clear()

    if batch:
        cur.executemany(insert_utt_stage1_sql, batch)
        conn.commit()

    # ---------- 可视化 ----------
    if enable_plot:
        maybe_plot(token_stats_rows, plot_path)

    keep_count = len(keep_set)
    print("\n[03] 完成")
    print(f"- 文档总数: {total_docs}")
    print(f"- 词表总数: {len(token_stats_rows)}")
    print(f"- Stage1 候选词数: {keep_count}")

    conn.close()


if __name__ == "__main__":
    main()
