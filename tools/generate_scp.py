#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Galgame数据集格式转换工具
将解压后的音频文件转换为FunASR所需的scp和text格式
"""

import os
import glob
import random
from pathlib import Path
from tqdm import tqdm

# ====== 配置变量 ======
# 解压后的数据目录（包含 000000, 000001, ... 等子文件夹）
DATA_DIR = r"R:\datasets--litagin--Galgame_Speech_ASR_16kHz\data"

# 输出文件路径
OUTPUT_DIR = r"R:\datasets--litagin--Galgame_Speech_ASR_16kHz"

# 音频文件扩展名
AUDIO_EXTENSION = ".ogg"
TEXT_EXTENSION = ".txt"

# ====== 脚本开始 ======

def process_file_list(file_list, wav_scp_path, text_txt_path):
    success_count = 0
    missing_text_count = 0
    
    with open(wav_scp_path, 'w', encoding='utf-8') as f_wav, \
         open(text_txt_path, 'w', encoding='utf-8') as f_txt:
        
        for idx, audio_path in enumerate(tqdm(file_list, desc="Processing"), 1):
                 
            file_id = Path(audio_path).stem
            txt_path = audio_path.replace(AUDIO_EXTENSION, TEXT_EXTENSION)
            
            if not os.path.exists(txt_path):
                missing_text_count += 1
                continue
            
            try:
                with open(txt_path, 'r', encoding='utf-8') as t:
                    text_content = t.read().strip()
                
                f_wav.write(f"{file_id} {audio_path}\n")
                f_txt.write(f"{file_id} {text_content}\n")
                success_count += 1
                
            except Exception as e:
                print(f"警告：处理文件 {file_id} 时出错: {e}")
                continue
                
    return success_count, missing_text_count

def main():
    print("====== Galgame数据集格式转换工具 (含数据集划分) ======\n")
    
    if not os.path.exists(DATA_DIR):
        print(f"错误：数据目录不存在: {DATA_DIR}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"正在扫描目录: {DATA_DIR}")
    pattern = os.path.join(DATA_DIR, "**", f"*{AUDIO_EXTENSION}")
    audio_files = glob.glob(pattern, recursive=True)
    
    print(f"找到 {len(audio_files)} 个音频文件\n")
    
    if len(audio_files) == 0:
        print("错误：未找到任何音频文件")
        return

    # 随机打算并划分 8:2
    print("正在进行随机划分 (80% 训练, 20% 验证)...")
    random.seed(42) # 固定随机种子，保证可复现
    random.shuffle(audio_files)
    
    split_idx = int(len(audio_files) * 0.8)
    train_files = audio_files[:split_idx]
    val_files = audio_files[split_idx:]
    
    print(f"训练集数量: {len(train_files)}")
    print(f"验证集数量: {len(val_files)}\n")
    
    # Process Train
    print("生成训练集列表...")
    train_wav_scp = os.path.join(OUTPUT_DIR, "train_wav.scp")
    train_text_txt = os.path.join(OUTPUT_DIR, "train_text.txt")
    t_suc, t_miss = process_file_list(train_files, train_wav_scp, train_text_txt)
    
    # Process Val
    print("\n生成验证集列表...")
    val_wav_scp = os.path.join(OUTPUT_DIR, "val_wav.scp")
    val_text_txt = os.path.join(OUTPUT_DIR, "val_text.txt")
    v_suc, v_miss = process_file_list(val_files, val_wav_scp, val_text_txt)
    
    print("\n====== 转换完成 ======")
    print(f"训练集: 成功 {t_suc}, 缺失 {t_miss}")
    print(f"验证集: 成功 {v_suc}, 缺失 {v_miss}")
    
    print(f"\n输出文件位于: {OUTPUT_DIR}")
    print("\n下一步操作建议 (PowerShell):")
    print("# 1. 生成训练集 JSONL (日文Prompt)")
    cmd_train = f'python tools/scp2jsonl.py "++scp_file={train_wav_scp}" "++transcript_file={train_text_txt}" "++jsonl_file={os.path.join(OUTPUT_DIR, "train_example.jsonl")}" "++prompt=语音转写成日文："'
    print(cmd_train.replace("\\", "\\\\"))
    
    print("\n# 2. 生成验证集 JSONL (日文Prompt)")
    cmd_val = f'python tools/scp2jsonl.py "++scp_file={val_wav_scp}" "++transcript_file={val_text_txt}" "++jsonl_file={os.path.join(OUTPUT_DIR, "val_example.jsonl")}" "++prompt=语音转写成日文："'
    print(cmd_val.replace("\\", "\\\\"))

if __name__ == "__main__":
    main()
