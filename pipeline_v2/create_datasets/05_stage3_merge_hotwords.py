"""
阶段 05：融合 Stage1 与 Stage2，得到最终热词。

本文件负责：
1) 生成 final_vocab（默认：Stage1 与 Stage2 的交集）
2) 为每个样本生成 final_hotwords

说明：
- 这一阶段仍然是“全量样本统一处理”，不区分 train/eval。
- 每条样本热词可按 IDF 排序并截断上限。
"""

from __future__ import annotations

import argparse

from tqdm import tqdm

from common import json_dumps, json_loads, load_config, now_iso, open_sqlite
from schema import clear_tables, ensure_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段05：融合最终热词")
    parser.add_argument("--config", required=True, help="配置文件路径（JSON）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    db_path = cfg["db"]["path"]
    max_hotwords_per_utt = int(cfg.get("stage3", {}).get("max_hotwords_per_utt", 20))

    conn = open_sqlite(db_path)
    ensure_schema(conn)
    cur = conn.cursor()

    # 每次重跑阶段05，清空旧结果
    clear_tables(conn, ["final_vocab", "final_hotwords"])

    # ---------- 生成 final_vocab（交集） ----------
    cur.execute(
        """
        INSERT INTO final_vocab (token, updated_at)
        SELECT ts.token, ?
        FROM token_stats ts
        INNER JOIN stage2_vocab sv ON ts.token = sv.token
        WHERE ts.stage1_keep = 1 AND sv.llm_keep = 1
        ORDER BY ts.token;
        """,
        (now_iso(),),
    )
    conn.commit()

    # 读取 final_vocab 和 idf 映射，用于样本内排序
    cur.execute("SELECT token FROM final_vocab ORDER BY token;")
    final_vocab = {row[0] for row in cur.fetchall()}

    cur.execute("SELECT token, idf FROM token_stats ORDER BY token;")
    idf_map = {token: float(idf) for token, idf in cur.fetchall()}

    # ---------- 生成每条样本的最终热词 ----------
    cur.execute("SELECT utt_id, candidates_json FROM utt_stage1 ORDER BY utt_id;")
    rows = cur.fetchall()

    insert_sql = """
    INSERT INTO final_hotwords (utt_id, hotwords_json, updated_at)
    VALUES (?, ?, ?)
    ON CONFLICT(utt_id) DO UPDATE SET
        hotwords_json=excluded.hotwords_json,
        updated_at=excluded.updated_at;
    """

    batch = []
    batch_size = cfg.get("stage3", {}).get("batch_size", 5000)

    for utt_id, candidates_json in tqdm(rows, desc="[05] 合成每条样本热词"):
        candidates = json_loads(candidates_json, default=[])

        filtered = [w for w in candidates if w in final_vocab]
        # 排序规则：IDF 高的优先，其次按词典序保证稳定
        filtered.sort(key=lambda w: (-idf_map.get(w, 0.0), w))

        if max_hotwords_per_utt > 0:
            filtered = filtered[:max_hotwords_per_utt]

        batch.append((utt_id, json_dumps(filtered), now_iso()))

        if len(batch) >= batch_size:
            cur.executemany(insert_sql, batch)
            conn.commit()
            batch.clear()

    if batch:
        cur.executemany(insert_sql, batch)
        conn.commit()

    cur.execute("SELECT COUNT(*) FROM final_vocab;")
    final_vocab_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM final_hotwords;")
    final_hotwords_count = cur.fetchone()[0]

    print("\n[05] 完成")
    print(f"- 最终词表大小(final_vocab): {final_vocab_count}")
    print(f"- 样本热词行数(final_hotwords): {final_hotwords_count}")

    conn.close()


if __name__ == "__main__":
    main()
