"""
阶段 01：读取多数据集清单并导入 SQLite。

本文件负责：
1) 读取配置中的多个 (scp, text) 对
2) 按行配对并做基础校验（utt 是否一致）
3) 生成稳定 utt_id（基于 dataset_id + rel_path）
4) 写入 samples 表

设计特点：
- 可复现：utt_id 与路径归一化规则固定
- 可打断：批量提交，脚本中断后可重新执行
- 可重跑：本阶段每次执行前清空 samples
"""

from __future__ import annotations

import argparse
from itertools import zip_longest

from tqdm import tqdm

from common import (
    chunked_commit,
    load_config,
    now_iso,
    open_sqlite,
    parse_kv_line,
    stable_hash_hex,
    to_rel_path,
)
from schema import clear_tables, ensure_schema


def ingest_one_dataset(conn, dataset_conf: dict, batch_size: int) -> dict:
    """导入单个数据集，返回统计信息。"""
    dataset_id = dataset_conf["dataset_id"]
    scp_file = dataset_conf["scp_file"]
    text_file = dataset_conf["text_file"]
    source_prefix = dataset_conf.get("source_prefix", "")

    cursor = conn.cursor()
    insert_sql = """
    INSERT INTO samples (utt_id, dataset_id, rel_path, transcript, source_utt, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(utt_id) DO UPDATE SET
        dataset_id=excluded.dataset_id,
        rel_path=excluded.rel_path,
        transcript=excluded.transcript,
        source_utt=excluded.source_utt,
        created_at=excluded.created_at;
    """

    stats = {
        "dataset_id": dataset_id,
        "inserted": 0,
        "mismatch": 0,
        "bad_line": 0,
        "line_count": 0,
    }
    batch = []

    # 使用 zip_longest 是为了明确发现行数不一致问题
    with open(scp_file, "r", encoding="utf-8") as f_scp, open(text_file, "r", encoding="utf-8") as f_txt:
        pair_iter = zip_longest(f_scp, f_txt, fillvalue=None)

        for scp_line, txt_line in tqdm(pair_iter, desc=f"[01] 导入 {dataset_id}"):
            stats["line_count"] += 1

            if scp_line is None or txt_line is None:
                stats["mismatch"] += 1
                continue

            scp_utt, audio_path = parse_kv_line(scp_line)
            txt_utt, transcript = parse_kv_line(txt_line)

            if not scp_utt or not txt_utt:
                stats["bad_line"] += 1
                continue

            if scp_utt != txt_utt:
                stats["mismatch"] += 1
                continue

            # 把绝对路径转换为相对路径，便于跨环境导出
            rel_path = to_rel_path(audio_path, source_prefix=source_prefix)

            # 使用稳定哈希作为 utt_id，避免路径前缀变化带来的不一致
            utt_id = stable_hash_hex(f"{dataset_id}::{rel_path}")
            batch.append((utt_id, dataset_id, rel_path, transcript, scp_utt, now_iso()))

            if len(batch) >= batch_size:
                stats["inserted"] += chunked_commit(cursor, conn, batch, insert_sql)

    # 刷掉尾批
    stats["inserted"] += chunked_commit(cursor, conn, batch, insert_sql)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段01：导入数据到 SQLite")
    parser.add_argument("--config", required=True, help="配置文件路径（JSON）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    db_path = cfg["db"]["path"]
    batch_size = cfg.get("ingest", {}).get("batch_size", 5000)

    conn = open_sqlite(db_path)
    ensure_schema(conn)

    # 本阶段每次重跑都清空 samples，以保证输出唯一且可复现
    clear_tables(conn, ["samples"])

    all_stats = []
    for dataset_conf in cfg.get("datasets", []):
        stats = ingest_one_dataset(conn, dataset_conf, batch_size=batch_size)
        all_stats.append(stats)

    # 控制台输出汇总信息
    print("\n[01] 导入完成，统计如下：")
    total_inserted = 0
    for item in all_stats:
        total_inserted += item["inserted"]
        print(
            f"- {item['dataset_id']}: inserted={item['inserted']}, "
            f"mismatch={item['mismatch']}, bad_line={item['bad_line']}, "
            f"line_count={item['line_count']}"
        )
    print(f"[01] 总写入样本数: {total_inserted}")

    conn.close()


if __name__ == "__main__":
    main()
