"""
阶段 09：控制台汇总检查。

本文件负责：
1) 打印数据库关键统计
2) 打印导出文件的行数
3) 抽样展示若干条样本信息

说明：
- 按你的要求，这里只输出到控制台，不额外写报告文件。
"""

from __future__ import annotations

import argparse
import os

from common import count_file_lines, json_loads, load_config, open_sqlite, stable_hash_int
from schema import ensure_schema


def safe_count(cur, sql: str) -> int:
    cur.execute(sql)
    row = cur.fetchone()
    return int(row[0]) if row else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段09：控制台汇总")
    parser.add_argument("--config", required=True, help="配置文件路径（JSON）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    db_path = cfg["db"]["path"]

    export_cfg = cfg.get("export", {})
    output_dir = export_cfg.get("output_dir", "pipline_v2/create_datasets/exports")
    train_path = os.path.join(output_dir, "train_chatml.jsonl")
    eval_fast_path = os.path.join(output_dir, "eval_fast_5k_chatml.jsonl")
    eval_full_path = os.path.join(output_dir, "eval_full_5pct_chatml.jsonl")

    inject_cfg = cfg.get("inject", {})
    train_hotword_ratio = float(inject_cfg.get("train_hotword_ratio", 0.30))
    seed_inject = str(inject_cfg.get("seed_inject", "inject_v1"))

    conn = open_sqlite(db_path)
    ensure_schema(conn)
    cur = conn.cursor()

    sample_count = safe_count(cur, "SELECT COUNT(*) FROM samples;")
    token_count = safe_count(cur, "SELECT COUNT(*) FROM token_stats;")
    stage1_keep_count = safe_count(cur, "SELECT COUNT(*) FROM token_stats WHERE stage1_keep=1;")
    stage2_keep_count = safe_count(cur, "SELECT COUNT(*) FROM stage2_vocab WHERE llm_keep=1;")
    final_vocab_count = safe_count(cur, "SELECT COUNT(*) FROM final_vocab;")

    train_count = safe_count(cur, "SELECT COUNT(*) FROM splits WHERE split_tag='train';")
    eval_full_count = safe_count(cur, "SELECT COUNT(*) FROM splits WHERE split_tag='eval_full';")
    eval_fast_count = safe_count(cur, "SELECT COUNT(*) FROM splits WHERE is_eval_fast=1;")

    # 估算训练集“理论注入数”（同时要求该样本有热词）
    cur.execute(
        """
        SELECT s.utt_id, COALESCE(fh.hotwords_json, '[]')
        FROM splits s
        LEFT JOIN final_hotwords fh ON s.utt_id = fh.utt_id
        WHERE s.split_tag='train'
        ORDER BY s.utt_id;
        """
    )
    train_inject_eligible = 0
    train_inject_actual = 0
    for utt_id, hotwords_json in cur.fetchall():
        hotwords = json_loads(hotwords_json, default=[])
        if hotwords:
            train_inject_eligible += 1
            bucket = stable_hash_int(f"{utt_id}::{seed_inject}", modulo=10000)
            if bucket < int(train_hotword_ratio * 10000):
                train_inject_actual += 1

    print("\n========== [09] 数据集构建汇总 ==========")
    print(f"samples: {sample_count}")
    print(f"token_stats: {token_count}")
    print(f"stage1_keep: {stage1_keep_count}")
    print(f"stage2_keep: {stage2_keep_count}")
    print(f"final_vocab: {final_vocab_count}")
    print("----------------------------------------")
    print(f"train: {train_count}")
    print(f"eval_full(5%): {eval_full_count}")
    print(f"eval_fast(5000): {eval_fast_count}")
    print("----------------------------------------")
    print(f"train 有热词样本(可注入): {train_inject_eligible}")
    print(f"train 理论注入样本数: {train_inject_actual}")
    if train_count > 0:
        print(f"train 理论注入比例: {train_inject_actual / train_count:.4f}")

    print("----------------------------------------")
    print(f"导出文件行数 - train: {count_file_lines(train_path)} ({train_path})")
    print(f"导出文件行数 - eval_fast: {count_file_lines(eval_fast_path)} ({eval_fast_path})")
    print(f"导出文件行数 - eval_full: {count_file_lines(eval_full_path)} ({eval_full_path})")

    print("----------------------------------------")
    print("示例样本（最多3条）:")
    cur.execute(
        """
        SELECT s.utt_id, s.dataset_id, s.rel_path, COALESCE(fh.hotwords_json, '[]')
        FROM samples s
        LEFT JOIN final_hotwords fh ON s.utt_id = fh.utt_id
        ORDER BY s.utt_id
        LIMIT 3;
        """
    )
    for utt_id, dataset_id, rel_path, hotwords_json in cur.fetchall():
        hotwords = json_loads(hotwords_json, default=[])
        print(f"- utt_id={utt_id[:12]}..., dataset={dataset_id}, rel_path={rel_path}, hotwords={hotwords[:5]}")

    conn.close()


if __name__ == "__main__":
    main()
