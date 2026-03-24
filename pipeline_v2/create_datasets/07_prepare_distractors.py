"""
阶段 07：构建干扰词池与音近候选。

本文件负责：
1) 构建 distractor_pool（非最终热词）
2) 基于读音近似为最终热词生成 phonetic_neighbors

说明：
- 干扰词策略后续在导出阶段使用（50%随机无关 + 50%音近）。
- 为了可复现，构建过程全部按固定排序进行。
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from tqdm import tqdm

from common import json_dumps, json_loads, load_config, now_iso, open_sqlite
from schema import clear_tables, ensure_schema


def levenshtein(a: str, b: str) -> int:
    """
    计算编辑距离（Levenshtein）。

    说明：
    - 这里用于“读音近似”排序。
    - 实现为经典 DP，结果确定性。
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            repl = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, repl))
        prev = cur
    return prev[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段07：构建干扰词池")
    parser.add_argument("--config", required=True, help="配置文件路径（JSON）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    db_path = cfg["db"]["path"]
    top_k = int(cfg.get("distractor", {}).get("phonetic_top_k", 20))

    conn = open_sqlite(db_path)
    ensure_schema(conn)
    cur = conn.cursor()

    # 每次重跑阶段07，清空旧结果
    clear_tables(conn, ["distractor_pool", "phonetic_neighbors"])

    # ---------- 1) 收集 token -> reading 映射 ----------
    token_reading = {}
    cur.execute("SELECT tokens_json, readings_json FROM tokens ORDER BY utt_id;")
    for tokens_json, readings_json in tqdm(cur.fetchall(), desc="[07] 收集 token 读音"):
        tokens = json_loads(tokens_json, default=[])
        readings = json_loads(readings_json, default=[])

        # 兼容长度不一致场景：只遍历最短长度
        n = min(len(tokens), len(readings))
        for i in range(n):
            token = str(tokens[i]).strip()
            reading = str(readings[i]).strip()
            if token and token not in token_reading:
                token_reading[token] = reading if reading else token

    # ---------- 2) 构建 distractor_pool（非 final_vocab） ----------
    cur.execute("SELECT token FROM final_vocab ORDER BY token;")
    final_vocab = {row[0] for row in cur.fetchall()}

    cur.execute("SELECT token, tf FROM token_stats ORDER BY token;")
    token_stats = cur.fetchall()

    insert_pool_sql = """
    INSERT INTO distractor_pool (token, reading, freq, pool_type, updated_at)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(token) DO UPDATE SET
        reading=excluded.reading,
        freq=excluded.freq,
        pool_type=excluded.pool_type,
        updated_at=excluded.updated_at;
    """

    pool_rows = []
    for token, tf in tqdm(token_stats, desc="[07] 生成 distractor_pool"):
        if token in final_vocab:
            continue
        reading = token_reading.get(token, token)
        pool_rows.append((token, reading, int(tf), "general", now_iso()))

    if pool_rows:
        cur.executemany(insert_pool_sql, pool_rows)
        conn.commit()

    # ---------- 3) 为 final_vocab 构建 phonetic_neighbors ----------
    # 为了降低复杂度，先把非热词按 (首字符, 长度) 建桶
    cur.execute("SELECT token, reading FROM distractor_pool ORDER BY token;")
    non_hot_rows = cur.fetchall()

    buckets = defaultdict(list)
    for token, reading in non_hot_rows:
        rd = (reading or token).strip()
        if not rd:
            continue
        key = (rd[0], len(rd))
        buckets[key].append((token, rd))

    cur.execute("SELECT token FROM final_vocab ORDER BY token;")
    hot_tokens = [row[0] for row in cur.fetchall()]

    insert_neighbor_sql = """
    INSERT INTO phonetic_neighbors (token, neighbors_json, updated_at)
    VALUES (?, ?, ?)
    ON CONFLICT(token) DO UPDATE SET
        neighbors_json=excluded.neighbors_json,
        updated_at=excluded.updated_at;
    """

    neighbor_rows = []
    for token in tqdm(hot_tokens, desc="[07] 计算音近词"):
        rd = token_reading.get(token, token).strip()
        if not rd:
            neighbor_rows.append((token, json_dumps([]), now_iso()))
            continue

        # 候选桶：首字符相同，长度在 +/-1
        cand = []
        for L in (len(rd) - 1, len(rd), len(rd) + 1):
            if L <= 0:
                continue
            cand.extend(buckets.get((rd[0], L), []))

        # 计算编辑距离并排序
        scored = []
        for cand_token, cand_rd in cand:
            dist = levenshtein(rd, cand_rd)
            scored.append((dist, cand_token))

        scored.sort(key=lambda x: (x[0], x[1]))
        neighbors = [tok for _, tok in scored[:top_k]]
        neighbor_rows.append((token, json_dumps(neighbors), now_iso()))

    if neighbor_rows:
        cur.executemany(insert_neighbor_sql, neighbor_rows)
        conn.commit()

    print("\n[07] 完成")
    print(f"- distractor_pool 词数: {len(pool_rows)}")
    print(f"- phonetic_neighbors 词数: {len(neighbor_rows)}")

    conn.close()


if __name__ == "__main__":
    main()
