"""
本文件用于维护 create_datasets 流水线的 SQLite 表结构。

说明：
- 采用“单库多表”模式，方便阶段化重跑。
- 每个阶段输出到独立表，便于后续手动重跑某一步。
"""

from __future__ import annotations

import sqlite3


TABLE_DDLS = [
    """
    CREATE TABLE IF NOT EXISTS samples (
        utt_id TEXT PRIMARY KEY,
        dataset_id TEXT NOT NULL,
        rel_path TEXT NOT NULL,
        transcript TEXT NOT NULL,
        source_utt TEXT,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS tokens (
        utt_id TEXT PRIMARY KEY,
        tokens_json TEXT NOT NULL,
        readings_json TEXT NOT NULL,
        tokenizer_name TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (utt_id) REFERENCES samples(utt_id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS token_stats (
        token TEXT PRIMARY KEY,
        df INTEGER NOT NULL,
        tf INTEGER NOT NULL,
        idf REAL NOT NULL,
        stage1_keep INTEGER NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS utt_stage1 (
        utt_id TEXT PRIMARY KEY,
        candidates_json TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (utt_id) REFERENCES samples(utt_id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS stage2_vocab (
        token TEXT PRIMARY KEY,
        llm_keep INTEGER NOT NULL,
        llm_score REAL NOT NULL,
        llm_reason TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS final_vocab (
        token TEXT PRIMARY KEY,
        updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS final_hotwords (
        utt_id TEXT PRIMARY KEY,
        hotwords_json TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (utt_id) REFERENCES samples(utt_id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS splits (
        utt_id TEXT PRIMARY KEY,
        split_tag TEXT NOT NULL,
        is_eval_fast INTEGER NOT NULL,
        h_split INTEGER NOT NULL,
        h_fast INTEGER NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (utt_id) REFERENCES samples(utt_id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS distractor_pool (
        token TEXT PRIMARY KEY,
        reading TEXT NOT NULL,
        freq INTEGER NOT NULL,
        pool_type TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS phonetic_neighbors (
        token TEXT PRIMARY KEY,
        neighbors_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
]


INDEX_DDLS = [
    "CREATE INDEX IF NOT EXISTS idx_samples_dataset ON samples(dataset_id);",
    "CREATE INDEX IF NOT EXISTS idx_splits_tag ON splits(split_tag);",
    "CREATE INDEX IF NOT EXISTS idx_splits_fast ON splits(is_eval_fast);",
    "CREATE INDEX IF NOT EXISTS idx_token_stats_keep ON token_stats(stage1_keep);",
]


def ensure_schema(conn: sqlite3.Connection) -> None:
    """初始化所有表与索引。"""
    cur = conn.cursor()
    for ddl in TABLE_DDLS:
        cur.execute(ddl)
    for ddl in INDEX_DDLS:
        cur.execute(ddl)
    conn.commit()


def clear_tables(conn: sqlite3.Connection, table_names: list[str]) -> None:
    """
    清空指定表。

    注意：
    - 这是“显式操作”，仅在当前阶段脚本内部使用。
    - 不做自动依赖判断，执行顺序由使用者手动控制。
    """
    cur = conn.cursor()
    for table_name in table_names:
        cur.execute(f"DELETE FROM {table_name};")
    conn.commit()
