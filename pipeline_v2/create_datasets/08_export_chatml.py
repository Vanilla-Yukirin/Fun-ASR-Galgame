"""
阶段 08：导出 ChatML 数据集。

本文件负责：
1) 从 SQLite 读取 samples + splits + final_hotwords
2) 按配置决定是否注入热词与干扰词
3) 生成三份 JSONL：train / eval_fast_5k / eval_full_5pct
4) 导出时根据 profile 拼接绝对路径前缀

说明：
- 数据库里只保存相对路径，这里才做路径前缀映射。
- 训练集热词注入比例默认 30%，基于稳定哈希判定。
"""

from __future__ import annotations

import argparse
import json
import os

from tqdm import tqdm

from common import (
    json_loads,
    load_config,
    open_sqlite,
    resolve_audio_abs_path,
    stable_hash_int,
)
from schema import ensure_schema


def deterministic_pick_from_candidates(
    candidates: list[str],
    count: int,
    utt_id: str,
    seed: str,
    forbidden: set[str],
) -> list[str]:
    """
    从“候选列表”中确定性挑选若干词。

    方法：
    - 先去重、去禁用词
    - 再按稳定哈希排序取前 N
    """
    if count <= 0:
        return []

    uniq = sorted({c for c in candidates if c and c not in forbidden})
    uniq.sort(key=lambda w: (stable_hash_int(f"{utt_id}::{seed}::{w}"), w))
    return uniq[:count]


def deterministic_pick_from_pool(
    pool_tokens: list[str],
    count: int,
    utt_id: str,
    seed: str,
    forbidden: set[str],
) -> list[str]:
    """
    从“大词池”中确定性挑选若干词（避免每次全量排序）。

    方法：
    - 通过稳定哈希生成索引，多次尝试直到凑够数量
    - 该方法复杂度低，适合超大词池
    """
    if count <= 0 or not pool_tokens:
        return []

    picked = []
    n = len(pool_tokens)
    max_attempt = max(200, count * 50)

    for i in range(max_attempt):
        idx = stable_hash_int(f"{utt_id}::{seed}::{i}", modulo=n)
        token = pool_tokens[idx]
        if token in forbidden:
            continue
        if token in picked:
            continue
        picked.append(token)
        if len(picked) >= count:
            break

    return picked


def should_inject_by_ratio(utt_id: str, ratio: float, seed: str) -> bool:
    """基于稳定哈希按比例决定是否注入（确定性）。"""
    threshold = int(ratio * 10000)
    bucket = stable_hash_int(f"{utt_id}::{seed}", modulo=10000)
    return bucket < threshold


def build_prompt(base_prompt: str, template_prompt: str, context_words: list[str]) -> str:
    """按是否有词表构造 prompt。"""
    if context_words:
        return template_prompt.format(context_words="、".join(context_words))
    return base_prompt


def make_chatml_record(
    system_prompt: str,
    user_prompt: str,
    audio_abs_path: str,
    transcript: str,
    meta: dict,
) -> dict:
    """
    组装 ChatML 样本。

    注意：
    - speech_length/text_length 这里使用简化占位（可后续扩展为真实值）
    """
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{user_prompt}<|startofspeech|>!{audio_abs_path}<|endofspeech|>",
            },
            {"role": "assistant", "content": transcript},
        ],
        "speech_length": -1,
        "text_length": len(transcript or ""),
        "meta": meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段08：导出 ChatML")
    parser.add_argument("--config", required=True, help="配置文件路径（JSON）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    db_path = cfg["db"]["path"]

    prompt_cfg = cfg.get("prompt", {})
    base_prompt = prompt_cfg.get("base", "语音转写成日文：")
    with_context_template = prompt_cfg.get(
        "with_context_template",
        "请结合上下文词表进行转写。\n词表：{context_words}\n语音转写成日文：",
    )
    system_prompt = prompt_cfg.get("system", "You are a helpful assistant.")

    inject_cfg = cfg.get("inject", {})
    train_hotword_ratio = float(inject_cfg.get("train_hotword_ratio", 0.30))
    seed_inject = str(inject_cfg.get("seed_inject", "inject_v1"))

    distractor_cfg = cfg.get("distractor", {})
    random_ratio = float(distractor_cfg.get("random_ratio", 0.5))
    phonetic_ratio = float(distractor_cfg.get("phonetic_ratio", 0.5))
    max_hotwords_per_sample = int(distractor_cfg.get("max_hotwords_per_sample", 6))
    max_distractors_per_sample = int(distractor_cfg.get("max_distractors_per_sample", 6))
    seed_random_distractor = str(distractor_cfg.get("seed_random", "random_distractor_v1"))
    seed_phonetic_distractor = str(distractor_cfg.get("seed_phonetic", "phonetic_distractor_v1"))
    seed_context_order = str(distractor_cfg.get("seed_context_order", "context_order_v1"))

    export_cfg = cfg.get("export", {})
    output_dir = export_cfg.get("output_dir", "pipline_v2/create_datasets/exports")
    profile = export_cfg.get("profile", "wsl")
    path_prefix_map = export_cfg.get("path_prefix", {})

    # eval 是否注入热词可单独控制
    eval_use_hotwords = bool(export_cfg.get("eval_use_hotwords", False))

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_chatml.jsonl")
    eval_fast_path = os.path.join(output_dir, "eval_fast_5k_chatml.jsonl")
    eval_full_path = os.path.join(output_dir, "eval_full_5pct_chatml.jsonl")

    conn = open_sqlite(db_path)
    ensure_schema(conn)
    cur = conn.cursor()

    # 预加载：音近词映射
    cur.execute("SELECT token, neighbors_json FROM phonetic_neighbors ORDER BY token;")
    neighbor_map = {token: json_loads(nei_json, default=[]) for token, nei_json in cur.fetchall()}

    # 预加载：随机干扰词池（仅 token 列）
    cur.execute("SELECT token FROM distractor_pool ORDER BY token;")
    random_pool = [row[0] for row in cur.fetchall()]

    # 查询总量用于进度条
    cur.execute("SELECT COUNT(*) FROM samples;")
    total_samples = cur.fetchone()[0]

    join_sql = """
    SELECT s.utt_id, s.dataset_id, s.rel_path, s.transcript,
           sp.split_tag, sp.is_eval_fast,
           COALESCE(fh.hotwords_json, '[]')
    FROM samples s
    INNER JOIN splits sp ON s.utt_id = sp.utt_id
    LEFT JOIN final_hotwords fh ON s.utt_id = fh.utt_id
    ORDER BY s.utt_id;
    """
    cur.execute(join_sql)

    train_count = 0
    train_injected = 0
    eval_full_count = 0
    eval_full_injected = 0
    eval_fast_count = 0
    eval_fast_injected = 0

    with (
        open(train_path, "w", encoding="utf-8") as f_train,
        open(eval_fast_path, "w", encoding="utf-8") as f_eval_fast,
        open(eval_full_path, "w", encoding="utf-8") as f_eval_full,
    ):
        for row in tqdm(cur, total=total_samples, desc="[08] 导出 ChatML"):
            utt_id, dataset_id, rel_path, transcript, split_tag, is_eval_fast, hotwords_json = row
            hotwords_all = json_loads(hotwords_json, default=[])

            use_hotwords = False
            if split_tag == "train":
                # 训练集按 30%（可配）注入，使用哈希决定
                use_hotwords = should_inject_by_ratio(utt_id, train_hotword_ratio, seed_inject)
            else:
                # 评测集是否注入可配
                use_hotwords = eval_use_hotwords

            selected_hotwords = []
            selected_distractors = []

            if use_hotwords and hotwords_all:
                selected_hotwords = hotwords_all[:max_hotwords_per_sample]

                # 先按比例拆分“困难负样本/简单负样本”数量
                phonetic_num = int(round(max_distractors_per_sample * phonetic_ratio))
                random_num = max(0, max_distractors_per_sample - phonetic_num)

                # 1) 音近干扰词
                phonetic_candidates = []
                for hw in selected_hotwords:
                    phonetic_candidates.extend(neighbor_map.get(hw, []))
                selected_phonetic = deterministic_pick_from_candidates(
                    phonetic_candidates,
                    phonetic_num,
                    utt_id,
                    seed_phonetic_distractor,
                    forbidden=set(selected_hotwords),
                )

                # 2) 随机无关干扰词
                selected_random = deterministic_pick_from_pool(
                    random_pool,
                    random_num,
                    utt_id,
                    seed_random_distractor,
                    forbidden=set(selected_hotwords) | set(selected_phonetic),
                )

                selected_distractors = selected_phonetic + selected_random

            context_words = selected_hotwords + selected_distractors
            if context_words:
                # 为了不总是“热词在前”，再做一次确定性排序
                context_words = sorted(
                    context_words,
                    key=lambda w: (stable_hash_int(f"{utt_id}::{seed_context_order}::{w}"), w),
                )

            user_prompt = build_prompt(base_prompt, with_context_template, context_words)
            abs_audio_path = resolve_audio_abs_path(path_prefix_map, profile, dataset_id, rel_path)

            meta = {
                "utt_id": utt_id,
                "dataset_id": dataset_id,
                "rel_path": rel_path,
                "split_tag": split_tag,
                "is_eval_fast": int(is_eval_fast),
                "use_hotwords": bool(context_words),
                "hotwords": selected_hotwords,
                "distractors": selected_distractors,
            }

            record = make_chatml_record(system_prompt, user_prompt, abs_audio_path, transcript, meta)
            line = json.dumps(record, ensure_ascii=False, sort_keys=True)

            if split_tag == "train":
                f_train.write(line + "\n")
                train_count += 1
                if context_words:
                    train_injected += 1
            else:
                f_eval_full.write(line + "\n")
                eval_full_count += 1
                if context_words:
                    eval_full_injected += 1

                if int(is_eval_fast) == 1:
                    f_eval_fast.write(line + "\n")
                    eval_fast_count += 1
                    if context_words:
                        eval_fast_injected += 1

    print("\n[08] 导出完成")
    print(f"- train: {train_count}, 注入热词样本: {train_injected}")
    print(f"- eval_full: {eval_full_count}, 注入热词样本: {eval_full_injected}")
    print(f"- eval_fast: {eval_fast_count}, 注入热词样本: {eval_fast_injected}")
    print(f"- 输出文件: {train_path}")
    print(f"- 输出文件: {eval_fast_path}")
    print(f"- 输出文件: {eval_full_path}")

    conn.close()


if __name__ == "__main__":
    main()
