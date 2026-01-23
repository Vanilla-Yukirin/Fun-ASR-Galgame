#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Galgame数据集格式转换工具 (含数据集划分)
将解压后的音频文件转换为FunASR所需的scp和text格式
支持 offset/count 选择数据子集
"""

import os
import glob
import random
from pathlib import Path
from tqdm import tqdm

# ====== 配置变量 ======
# 解压后的数据目录（包含 000000, 000001, ... 等子文件夹）
DATA_DIR = r"/mnt/d/ML/datasets--litagin--Galgame_Speech_ASR_16kHz/data"

# 输出文件路径
OUTPUT_DIR = r"/mnt/d/ML/datasets--litagin--Galgame_Speech_ASR_16kHz"

# 音频文件扩展名
AUDIO_EXTENSION = ".ogg"
TEXT_EXTENSION = ".txt"

# ====== 脚本开始 ======

from multiprocessing import Pool, cpu_count

def _process_chunk(chunk):
    """Worker function to process a batch of files"""
    results = []
    missing_count = 0
    
    for audio_path in chunk:
        file_id = Path(audio_path).stem
        txt_path = audio_path.replace(AUDIO_EXTENSION, TEXT_EXTENSION)
        
        try:
            # 尝试直接读取，省去 os.path.exists 的开销
            with open(txt_path, 'r', encoding='utf-8') as t:
                text_content = t.read().strip()
            results.append((f"{file_id} {audio_path}\n", f"{file_id} {text_content}\n"))
        except FileNotFoundError:
            missing_count += 1
        except Exception as e:
            # 可以选择打印错误或忽略
            missing_count += 1
            
    return results, missing_count

def process_file_list(file_list, wav_scp_path, text_txt_path):
    """处理文件列表，生成scp和text文件 (多进程优化版)"""
    success_count = 0
    missing_text_count = 0
    
    # 根据 CPU 核心数决定进程数，保留一点余量
    num_processes = max(1, cpu_count() - 1)
    # chunk_size 设置较小，以便进度条能频繁更新
    # 虽然这会增加一点进程间通信开销，但用户体验更好
    chunk_size = 1000
    
    # 将文件列表切分为 chunks
    chunks = [file_list[i:i + chunk_size] for i in range(0, len(file_list), chunk_size)]
    
    print(f"I/O 优化: 启用 {num_processes} 个进程并发处理，总计 {len(chunks)} 个批次...")
    
    with open(wav_scp_path, 'w', encoding='utf-8') as f_wav, \
         open(text_txt_path, 'w', encoding='utf-8') as f_txt:
        
        with Pool(processes=num_processes) as pool:
            # 使用 imap_unordered 稍微提速，因为写入顺序不影响最终训练 (只要 wav和text 对应即可)
            # 但为了保险起见，如果需要严格对应顺序，可以用 map (不过这里 sort 过了，只要 wav/text 内部对齐就行)
            # scp 文件通常不强制要求全局有序，但为了整洁我们可以保持有序
            # 这里为了速度优先使用 imap
            
            for batch_results, batch_missing in tqdm(pool.imap(_process_chunk, chunks), total=len(chunks), desc="Processing"):
                missing_text_count += batch_missing
                for wav_line, txt_line in batch_results:
                    f_wav.write(wav_line)
                    f_txt.write(txt_line)
                    success_count += 1
                
    return success_count, missing_text_count


def get_user_input(total_count):
    """获取用户输入的offset和count"""
    print(f"\n数据选择 (总计 {total_count} 个文件)")
    print("=" * 40)
    
    # 获取 offset
    while True:
        offset_str = input(f"请输入起始位置 offset (0~{total_count-1}, 按Enter默认0): ").strip()
        if offset_str == "":
            offset = 0
            break
        try:
            offset = int(offset_str)
            if 0 <= offset < total_count:
                break
            print(f"错误：offset 必须在 0~{total_count-1} 之间")
        except ValueError:
            print("错误：请输入有效的整数")
    
    remaining = total_count - offset
    
    # 获取 count
    while True:
        count_str = input(f"请输入处理数量 count (1~{remaining}, 输入0或按Enter表示全部={remaining}): ").strip()
        if count_str == "" or count_str == "0":
            count = remaining
            break
        try:
            count = int(count_str)
            if 1 <= count <= remaining:
                break
            print(f"错误：count 必须在 1~{remaining} 之间")
        except ValueError:
            print("错误：请输入有效的整数")
    
    return offset, count


def main():
    print("====== Galgame数据集格式转换工具 (含数据集划分) ======\n")
    
    if not os.path.exists(DATA_DIR):
        print(f"错误：数据目录不存在: {DATA_DIR}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"正在扫描目录: {DATA_DIR}")
    pattern = os.path.join(DATA_DIR, "**", f"*{AUDIO_EXTENSION}")
    audio_files = glob.glob(pattern, recursive=True)
    audio_files.sort()  # 保证顺序一致
    
    total_count = len(audio_files)
    print(f"找到 {total_count} 个音频文件")
    
    if total_count == 0:
        print("错误：未找到任何音频文件")
        return

    # 获取用户输入的 offset 和 count
    offset, count = get_user_input(total_count)
    
    # 切片选择数据（先选择范围）
    selected_files = audio_files[offset : offset + count]
    print(f"\n已选择数据范围: [{offset}, {offset + count}) 共 {len(selected_files)} 个文件")
    
    # 询问是否打乱（默认不打乱）
    shuffle_choice = input("是否打乱选中的数据? (y/N, 按Enter默认不打乱): ").strip().lower()
    if shuffle_choice == 'y':
        print("正在打乱选中数据...")
        random.seed(42)  # 固定种子保证可复现
        random.shuffle(selected_files)
        shuffle_tag = "_shuffled"
    else:
        print("保持原始顺序")
        shuffle_tag = ""
    
    # 8:2 划分训练集和验证集
    split_idx = int(len(selected_files) * 0.8)
    train_files = selected_files[:split_idx]
    val_files = selected_files[split_idx:]
    
    print(f"\n训练集数量: {len(train_files)} (80%)")
    print(f"验证集数量: {len(val_files)} (20%)")
    
    # 生成带范围标识的文件名
    range_tag = f"{offset}_{offset + count}{shuffle_tag}"
    
    # Process Train
    print("\n生成训练集列表...")
    train_wav_scp = os.path.join(OUTPUT_DIR, f"train_wav_{range_tag}.scp")
    train_text_txt = os.path.join(OUTPUT_DIR, f"train_text_{range_tag}.txt")
    t_suc, t_miss = process_file_list(train_files, train_wav_scp, train_text_txt)
    
    # Process Val
    print("\n生成验证集列表...")
    val_wav_scp = os.path.join(OUTPUT_DIR, f"val_wav_{range_tag}.scp")
    val_text_txt = os.path.join(OUTPUT_DIR, f"val_text_{range_tag}.txt")
    v_suc, v_miss = process_file_list(val_files, val_wav_scp, val_text_txt)
    
    print("\n====== 转换完成 ======")
    print(f"训练集: 成功 {t_suc}, 缺失 {t_miss}")
    print(f"验证集: 成功 {v_suc}, 缺失 {v_miss}")
    
    print(f"\n输出文件位于: {OUTPUT_DIR}")
    print(f"文件范围标识: {range_tag}")
    
    # 生成下一步命令
    train_jsonl = os.path.join(OUTPUT_DIR, f"train_{range_tag}.jsonl")
    val_jsonl = os.path.join(OUTPUT_DIR, f"val_{range_tag}.jsonl")
    
    print("\n" + "=" * 50)
    print("下一步操作建议 (PowerShell):")
    print("=" * 50)
    
    print("\n# 1. 生成训练集 JSONL (日文Prompt)")
    cmd_train = f'python tools/scp2jsonl.py "++scp_file={train_wav_scp}" "++transcript_file={train_text_txt}" "++jsonl_file={train_jsonl}" "++prompt=\'语音转写成日文：\'"'
    print(cmd_train.replace("\\", "\\\\"))
    
    print("\n# 2. 生成验证集 JSONL (日文Prompt)")
    cmd_val = f'python tools/scp2jsonl.py "++scp_file={val_wav_scp}" "++transcript_file={val_text_txt}" "++jsonl_file={val_jsonl}" "++prompt=\'语音转写成日文：\'"'
    print(cmd_val.replace("\\", "\\\\"))


if __name__ == "__main__":
    main()
