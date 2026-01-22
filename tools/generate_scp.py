#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Galgame数据集格式转换工具
将解压后的音频文件转换为FunASR所需的scp和text格式
"""

import os
import glob
from pathlib import Path

# ====== 配置变量 ======
# 解压后的数据目录（包含 000000, 000001, ... 等子文件夹）
DATA_DIR = r"R:\datasets--litagin--Galgame_Speech_ASR_16kHz\data"

# 输出文件路径
OUTPUT_DIR = r"R:\datasets--litagin--Galgame_Speech_ASR_16kHz"
WAV_SCP_PATH = os.path.join(OUTPUT_DIR, "train_wav.scp")
TEXT_TXT_PATH = os.path.join(OUTPUT_DIR, "train_text.txt")

# 音频文件扩展名
AUDIO_EXTENSION = ".ogg"
TEXT_EXTENSION = ".txt"

# ====== 脚本开始 ======

def main():
    print("====== Galgame数据集格式转换工具 ======\n")
    
    # 检查数据目录是否存在
    if not os.path.exists(DATA_DIR):
        print(f"错误：数据目录不存在: {DATA_DIR}")
        return
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 查找所有音频文件
    print(f"正在扫描目录: {DATA_DIR}")
    
    # 使用glob递归查找所有ogg文件
    pattern = os.path.join(DATA_DIR, "**", f"*{AUDIO_EXTENSION}")
    audio_files = glob.glob(pattern, recursive=True)
    
    print(f"找到 {len(audio_files)} 个音频文件\n")
    
    if len(audio_files) == 0:
        print("错误：未找到任何音频文件")
        return
    
    # 统计变量
    success_count = 0
    missing_text_count = 0
    
    # 打开输出文件
    with open(WAV_SCP_PATH, 'w', encoding='utf-8') as f_wav, \
         open(TEXT_TXT_PATH, 'w', encoding='utf-8') as f_txt:
        
        for idx, audio_path in enumerate(audio_files, 1):
            # 显示进度
            if idx % 1000 == 0 or idx == len(audio_files):
                print(f"处理进度: {idx}/{len(audio_files)} ({idx*100//len(audio_files)}%)")
            
            # 提取文件ID（文件名不含扩展名）
            file_id = Path(audio_path).stem
            
            # 对应的文本文件路径
            txt_path = audio_path.replace(AUDIO_EXTENSION, TEXT_EXTENSION)
            
            if not os.path.exists(txt_path):
                missing_text_count += 1
                continue
            
            try:
                # 读取文本内容
                with open(txt_path, 'r', encoding='utf-8') as t:
                    text_content = t.read().strip()
                
                # 写入 train_wav.scp: ID /path/to/audio
                f_wav.write(f"{file_id} {audio_path}\n")
                
                # 写入 train_text.txt: ID text_content
                f_txt.write(f"{file_id} {text_content}\n")
                
                success_count += 1
                
            except Exception as e:
                print(f"警告：处理文件 {file_id} 时出错: {e}")
                continue
    
    # 显示统计信息
    print("\n====== 转换完成 ======")
    print(f"成功处理: {success_count} 个")
    print(f"缺少文本文件: {missing_text_count} 个")
    print(f"\n输出文件:")
    print(f"  - {WAV_SCP_PATH}")
    print(f"  - {TEXT_TXT_PATH}")
    print("\n现在可以运行以下命令生成JSONL文件:")
    print(f"python tools/scp2jsonl.py ^")
    print(f"  \"++scp_file={WAV_SCP_PATH}\" ^")
    print(f"  \"++transcript_file={TEXT_TXT_PATH}\" ^")
    print(f"  \"++jsonl_file={os.path.join(OUTPUT_DIR, 'train_example.jsonl')}\" ^")
    print(f"  ++limit=1000")
    print(f"或者 (Windows PowerShell):")
    # 使用双反斜杠 \\ 来转义反斜杠，或者使用 repr()
    cmd = f'python tools/scp2jsonl.py "++scp_file={WAV_SCP_PATH}" "++transcript_file={TEXT_TXT_PATH}" "++jsonl_file={os.path.join(OUTPUT_DIR, "train_example.jsonl")}" ++limit=1000'
    print(cmd.replace("\\", "\\\\"))

if __name__ == "__main__":
    main()
