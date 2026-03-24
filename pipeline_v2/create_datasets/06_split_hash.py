"""
阶段 06：基于稳定哈希做数据划分。

本文件负责：
1) 将样本划分为 train 与 eval_full（默认 95/5）
2) 在 eval_full 内部确定性选取 eval_fast_5k

说明：
- 不使用随机打乱，全部采用稳定哈希分桶。
- 同样本 + 同 seed => 划分结果完全一致。
"""

from __future__ import annotations

import argparse

from tqdm import tqdm

from common import load_config, now_iso, open_sqlite, stable_hash_int
from schema import clear_tables, ensure_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段06：哈希划分数据集")
    parser.add_argument("--config", required=True, help="配置文件路径（JSON）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    db_path = cfg["db"]["path"]

    split_cfg = cfg.get("split", {})
    eval_ratio = float(split_cfg.get("eval_ratio", 0.05))
    eval_fast_size = int(split_cfg.get("eval_fast_size", 5000))
    seed_split = str(split_cfg.get("seed_split", "split_v1"))
    seed_fast = str(split_cfg.get("seed_fast", "fast_v1"))

    # 以 10000 分桶做比例切分，便于精确控制
    eval_cutoff = int(eval_ratio * 10000)

    conn = open_sqlite(db_path)
    ensure_schema(conn)
    cur = conn.cursor()

    # 每次重跑阶段06，清空旧划分
    clear_tables(conn, ["splits"])

    cur.execute("SELECT utt_id FROM samples ORDER BY utt_id;")
    utt_ids = [row[0] for row in cur.fetchall()]

    insert_sql = """
    INSERT INTO splits (utt_id, split_tag, is_eval_fast, h_split, h_fast, updated_at)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(utt_id) DO UPDATE SET
        split_tag=excluded.split_tag,
        is_eval_fast=excluded.is_eval_fast,
        h_split=excluded.h_split,
        h_fast=excluded.h_fast,
        updated_at=excluded.updated_at;
    """

    rows = []
    batch_size = split_cfg.get("batch_size", 10000)

    for utt_id in tqdm(utt_ids, desc="[06] 划分样本"):
        h_split = stable_hash_int(f"{utt_id}::{seed_split}", modulo=10000)
        h_fast = stable_hash_int(f"{utt_id}::{seed_fast}")
        split_tag = "eval_full" if h_split < eval_cutoff else "train"
        rows.append((utt_id, split_tag, 0, h_split, h_fast, now_iso()))

        if len(rows) >= batch_size:
            cur.executemany(insert_sql, rows)
            conn.commit()
            rows.clear()

    if rows:
        cur.executemany(insert_sql, rows)
        conn.commit()

    # 从 eval_full 内部确定性取前 N 条作为 eval_fast
    cur.execute(
        """
        SELECT utt_id
        FROM splits
        WHERE split_tag='eval_full'
        ORDER BY h_fast ASC, utt_id ASC
        LIMIT ?;
        """,
        (eval_fast_size,),
    )
    fast_ids = [row[0] for row in cur.fetchall()]

    cur.execute("UPDATE splits SET is_eval_fast=0;")
    conn.commit()

    if fast_ids:
        cur.executemany(
            "UPDATE splits SET is_eval_fast=1, updated_at=? WHERE utt_id=?;",
            [(now_iso(), utt_id) for utt_id in fast_ids],
        )
        conn.commit()

    cur.execute("SELECT COUNT(*) FROM splits WHERE split_tag='train';")
    train_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM splits WHERE split_tag='eval_full';")
    eval_full_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM splits WHERE is_eval_fast=1;")
    eval_fast_count = cur.fetchone()[0]

    print("\n[06] 完成")
    print(f"- train 数量: {train_count}")
    print(f"- eval_full 数量: {eval_full_count}")
    print(f"- eval_fast 数量: {eval_fast_count}")
    print(f"- eval_ratio: {eval_ratio}, eval_fast_size: {eval_fast_size}")

    conn.close()


if __name__ == "__main__":
    main()
