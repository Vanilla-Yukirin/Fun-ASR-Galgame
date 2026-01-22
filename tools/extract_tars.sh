#!/bin/bash

# 下载数据集
# huggingface-cli download --repo-type dataset litagin/Galgame_Speech_ASR_16kHz --local-dir /mnt/o/datasets/huggingface/hub/datasets--litagin--Galgame_Speech_ASR_16kHz --local-dir-use-symlinks False --resume-download

# ====== 配置变量 ======
# 源目录（包含tar文件的目录）
SOURCE_DIR="/mnt/o/datasets/huggingface/hub/datasets--litagin--Galgame_Speech_ASR_16kHz/data"

# 目标目录（解压到的目录）
TARGET_DIR="/mnt/r/datasets--litagin--Galgame_Speech_ASR_16kHz/data"

# tar文件前缀
TAR_PREFIX="galgame-speech-asr-16kHz-train-"

# ====== 脚本开始 ======

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}====== Galgame数据集解压工具 ======${NC}"
echo ""

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}错误：源目录不存在: $SOURCE_DIR${NC}"
    exit 1
fi

# 创建目标目录
mkdir -p "$TARGET_DIR"

# 获取所有tar文件并排序
mapfile -t tar_files < <(ls "$SOURCE_DIR"/${TAR_PREFIX}*.tar 2>/dev/null | sort)

if [ ${#tar_files[@]} -eq 0 ]; then
    echo -e "${RED}错误：在 $SOURCE_DIR 中未找到任何tar文件${NC}"
    exit 1
fi

echo -e "找到 ${GREEN}${#tar_files[@]}${NC} 个tar文件"
echo ""

# 询问用户要解压多少个文件
read -p "请输入要解压的文件数量 (1-${#tar_files[@]}, 按Enter解压全部): " num_to_extract

# 如果用户没有输入，则解压全部
if [ -z "$num_to_extract" ]; then
    num_to_extract=${#tar_files[@]}
elif ! [[ "$num_to_extract" =~ ^[0-9]+$ ]] || [ "$num_to_extract" -lt 1 ] || [ "$num_to_extract" -gt ${#tar_files[@]} ]; then
    echo -e "${RED}无效的数量，将解压全部文件${NC}"
    num_to_extract=${#tar_files[@]}
fi

echo ""
echo -e "${GREEN}将解压前 $num_to_extract 个tar文件${NC}"
echo ""

# 统计变量
extracted=0
skipped=0
failed=0

# 开始解压
for i in $(seq 0 $((num_to_extract - 1))); do
    tar_file="${tar_files[$i]}"
    
    # 提取文件编号 (例如: 000000)
    filename=$(basename "$tar_file")
    # 从文件名中提取6位数字编号
    file_number=$(echo "$filename" | grep -oP '(?<='$TAR_PREFIX')[0-9]{6}')
    
    if [ -z "$file_number" ]; then
        echo -e "${RED}警告：无法从文件名提取编号: $filename${NC}"
        ((failed++))
        continue
    fi
    
    # 目标文件夹
    target_folder="$TARGET_DIR/$file_number"
    
    # 检查是否已经解压（断点续传）
    if [ -d "$target_folder" ] && [ "$(ls -A "$target_folder" 2>/dev/null)" ]; then
        echo -e "${YELLOW}[跳过]${NC} $file_number - 文件夹已存在且包含文件"
        ((skipped++))
        continue
    fi
    
    # 创建目标文件夹
    mkdir -p "$target_folder"
    
    # 解压
    echo -e "${GREEN}[解压]${NC} $file_number - $(basename "$tar_file")"
    
    if tar -xf "$tar_file" -C "$target_folder"; then
        ((extracted++))
        echo -e "  ${GREEN}✓${NC} 解压完成"
    else
        echo -e "  ${RED}✗${NC} 解压失败"
        ((failed++))
        # 删除可能部分解压的文件夹
        rm -rf "$target_folder"
    fi
    
    echo ""
done

# 显示统计信息
echo ""
echo -e "${GREEN}====== 解压完成 ======${NC}"
echo -e "成功解压: ${GREEN}$extracted${NC} 个"
echo -e "跳过（已存在）: ${YELLOW}$skipped${NC} 个"
echo -e "失败: ${RED}$failed${NC} 个"
echo -e "总计处理: $((extracted + skipped + failed)) 个"
echo ""
echo -e "解压目录: ${GREEN}$TARGET_DIR${NC}"
