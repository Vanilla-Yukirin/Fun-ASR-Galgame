#!/bin/bash

# 下载数据集
# huggingface-cli download --repo-type dataset litagin/Galgame_Speech_ASR_16kHz --local-dir /mnt/o/datasets/huggingface/hub/datasets--litagin--Galgame_Speech_ASR_16kHz --local-dir-use-symlinks False --resume-download

# ====== 配置变量 ======
# 源目录（包含tar文件的目录）
SOURCE_DIR="/mnt/d/ML/datasets--litagin--Galgame_Speech_ASR_16kHz/data"

# 目标目录（解压到的目录）
TARGET_DIR="/mnt/d/datasets--litagin--Galgame_Speech_ASR_16kHz/data"
# TARGET_DIR="/mnt/d/ML/datasets--litagin--Galgame_Speech_ASR_16kHz/data"

# tar文件前缀
TAR_PREFIX="galgame-speech-asr-16kHz-train-"

# ====== 脚本开始 ======

# 颜色定义
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export RED='\033[0;31m'
export PINK='\033[38;5;213m' # Pink color
export NC='\033[0m' # No Color

# 导出变量供子进程使用
export SOURCE_DIR
export TARGET_DIR
export TAR_PREFIX
export GREEN
export YELLOW
export RED
export NC

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

# 检测CPU核心数
if command -v nproc >/dev/null 2>&1; then
    THREADS=$(nproc)
else
    THREADS=4
fi
echo -e "使用线程数: ${GREEN}$THREADS${NC}"
echo ""

# 询问遇到已存在文件的处理策略
echo -e "${YELLOW}解压策略：当遇到已存在的文件夹时:${NC}"
echo -e "  [s] 跳过 (Skip) - 默认"
echo -e "  [o] 覆盖 (Overwrite) - 会删除原有文件夹重新解压"
read -p "请选择策略 [s/o]: " exist_policy

if [[ "$exist_policy" == "o" || "$exist_policy" == "O" ]]; then
    OVERWRITE_EXISTING="true"
    echo -e "策略: ${RED}覆盖模式${NC}"
else
    OVERWRITE_EXISTING="false"
    echo -e "策略: ${GREEN}跳过模式${NC}"
fi
echo ""

export OVERWRITE_EXISTING

# 创建临时目录用于统计
LOG_DIR=$(mktemp -d)
export LOG_DIR

# 定义解压函数
extract_task() {
    local tar_file="$1"
    
    # 提取文件编号 (例如: 000000)
    local filename=$(basename "$tar_file")
    # 从文件名中提取6位数字编号
    local file_number=$(echo "$filename" | grep -oP '(?<='$TAR_PREFIX')[0-9]{6}')
    
    if [ -z "$file_number" ]; then
        echo "FAIL:$filename"
        touch "$LOG_DIR/fail_${filename}"
        return
    fi
    
    # 目标文件夹
    local target_folder="$TARGET_DIR/$file_number"
    
    # 检查是否已经解压（断点续传）
    if [ -d "$target_folder" ]; then
        if [ "$OVERWRITE_EXISTING" == "true" ]; then
             # 覆盖模式：删除旧文件夹，继续解压
             rm -rf "$target_folder"
        elif [ "$(ls -A "$target_folder" 2>/dev/null)" ]; then
             # 跳过模式：如果文件夹非空，则跳过
             echo "SKIP:$file_number"
             touch "$LOG_DIR/skip_${file_number}"
             return
        fi
    fi
    
    # 创建目标文件夹
    mkdir -p "$target_folder"
    
    if tar -xf "$tar_file" -C "$target_folder"; then
        echo "OK:$file_number"
        touch "$LOG_DIR/ok_${file_number}"
    else
        echo "FAIL:$file_number"
        touch "$LOG_DIR/fail_${file_number}"
        # 删除可能部分解压的文件夹
        rm -rf "$target_folder"
    fi
}
export -f extract_task

# 准备要处理的文件列表
files_to_process=("${tar_files[@]:0:num_to_extract}")
total_files=${#files_to_process[@]}

# 进度条
draw_progress_bar() {
    local current=$1
    local total=$2
    local width=50
    # 防止除以零
    if [ "$total" -eq 0 ]; then total=1; fi
    local percent=$((current * 100 / total))
    local filled=$((width * percent / 100))
    local empty=$((width - filled))
    
    printf "\r["
    printf "${PINK}%0.s#${NC}" $(seq 1 $filled)
    printf "%0.s." $(seq 1 $empty)
    printf "] %d%% (%d/%d)" "$percent" "$current" "$total"
}

# 开始并行解压
# 使用 xargs -P 进行多线程处理
echo -e "正在解压..."
start_time=$(date +%s)
current_count=0

# 通过管道读取输出并更新进度条
# 初始化进度条
draw_progress_bar 0 "$total_files"

printf "%s\n" "${files_to_process[@]}" | xargs -P "$THREADS" -I {} bash -c 'extract_task "$@"' _ "{}" | \
while read -r line; do
    status=$(echo "$line" | cut -d':' -f1)
    id=$(echo "$line" | cut -d':' -f2)
    
    ((current_count++))
    draw_progress_bar "$current_count" "$total_files"
    
    if [ "$status" == "FAIL" ]; then
        printf "\r\033[K${RED}✗ 解压失败: $id${NC}\n"
        draw_progress_bar "$current_count" "$total_files"
    elif [ "$status" == "SKIP" ]; then
        :
    fi
done

echo "" # 换行

end_time=$(date +%s)
duration=$((end_time - start_time))

# 统计结果
extracted=$(find "$LOG_DIR" -name "ok_*" | wc -l)
skipped=$(find "$LOG_DIR" -name "skip_*" | wc -l)
failed=$(find "$LOG_DIR" -name "fail_*" | wc -l)

# Clean up temp dir
rm -rf "$LOG_DIR"

# 显示统计信息
echo ""
echo -e "${GREEN}====== 解压完成 ======${NC}"
echo -e "耗时: ${duration} 秒"
echo -e "成功解压: ${GREEN}$extracted${NC} 个"
echo -e "跳过（已存在）: ${YELLOW}$skipped${NC} 个"
echo -e "失败: ${RED}$failed${NC} 个"
echo -e "总计处理: $((extracted + skipped + failed)) 个"
echo ""
echo -e "解压目录: ${GREEN}$TARGET_DIR${NC}"
