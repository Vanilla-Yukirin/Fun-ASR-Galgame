"""
本文件用于放置 create_datasets 流水线的通用工具函数。

设计目标：
1) 保证哈希可复现（不同机器/不同进程结果一致）
2) 保证路径归一化稳定（WSL/Autodl 统一处理）
3) 提供 SQLite 连接、JSON 编解码、通用文本解析能力
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def load_config(config_path: str) -> Dict[str, Any]:
    """
    读取 JSON 配置文件。

    说明：
    - 这里使用 JSON 而非 YAML，避免增加额外依赖。
    - 所有参数都通过该配置注入，保证可复现。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent_dir(file_path: str) -> None:
    """确保文件的父目录存在。"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    """返回当前 UTC 时间（ISO 格式，秒级）。"""
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def stable_hash_int(text: str, modulo: Optional[int] = None) -> int:
    """
    生成稳定哈希整数（可复现）。

    注意：
    - 不要使用 Python 内置 hash()，其结果在不同进程可能变化。
    - 这里使用 sha1，保证跨平台稳定。
    """
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    # 取前 16 位十六进制，已足够用于分桶和排序
    value = int(digest[:16], 16)
    if modulo is not None:
        return value % modulo
    return value


def stable_hash_hex(text: str) -> str:
    """返回稳定哈希的十六进制字符串（用于 utt_id）。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_path(path: str) -> str:
    """统一路径分隔符为 '/'，并去除首尾空白。"""
    return path.strip().replace("\\", "/")


def to_rel_path(audio_path: str, source_prefix: str = "") -> str:
    """
    将绝对路径转换为相对路径。

    规则：
    - 如果 audio_path 以 source_prefix 开头，则去掉该前缀。
    - 否则保留原路径，但去掉可能的前导 '/'.
    """
    path_norm = normalize_path(audio_path)
    prefix_norm = normalize_path(source_prefix).rstrip("/") if source_prefix else ""

    if prefix_norm and (path_norm == prefix_norm or path_norm.startswith(prefix_norm + "/")):
        rel = path_norm[len(prefix_norm) :].lstrip("/")
        return rel if rel else "."

    return path_norm.lstrip("/")


def parse_kv_line(line: str) -> Tuple[Optional[str], Optional[str]]:
    """
    解析形如 '<key> <value>' 的行。

    返回：
    - (key, value): 正常
    - (None, None): 空行或格式非法
    """
    line = line.strip()
    if not line:
        return None, None

    parts = line.split(maxsplit=1)
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def json_dumps(data: Any) -> str:
    """
    统一 JSON 序列化。

    说明：
    - ensure_ascii=False：保留中文/日文
    - sort_keys=True：键顺序稳定，便于比对
    """
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def json_loads(text: str, default: Any) -> Any:
    """
    安全 JSON 反序列化。

    如果文本为空或解析失败，返回 default。
    """
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def open_sqlite(db_path: str) -> sqlite3.Connection:
    """
    打开 SQLite 连接并设置常用 PRAGMA。

    这些设置用于兼顾稳定性与写入速度。
    """
    ensure_parent_dir(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def resolve_audio_abs_path(path_prefix_map: Dict[str, Dict[str, str]], profile: str, dataset_id: str, rel_path: str) -> str:
    """
    根据 profile + dataset_id + rel_path 组装导出时的绝对音频路径。

    这样可以做到：
    - 库里只存相对路径
    - 导出时按机器环境拼接不同前缀
    """
    profile_map = path_prefix_map.get(profile, {})
    prefix = profile_map.get(dataset_id, "")

    prefix_norm = normalize_path(prefix).rstrip("/")
    rel_norm = normalize_path(rel_path).lstrip("/")

    if not prefix_norm:
        # 没配前缀时，直接返回相对路径（不推荐，但保底）
        return rel_norm

    return f"{prefix_norm}/{rel_norm}"


def chunked_commit(cursor: sqlite3.Cursor, conn: sqlite3.Connection, batch: list, sql: str) -> int:
    """
    执行批量写入并提交，返回写入行数。

    统一封装是为了减少重复代码。
    """
    if not batch:
        return 0
    cursor.executemany(sql, batch)
    conn.commit()
    n = len(batch)
    batch.clear()
    return n


def count_file_lines(file_path: str) -> int:
    """统计文本文件行数（用于导出后快速检查）。"""
    if not os.path.exists(file_path):
        return 0
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count
