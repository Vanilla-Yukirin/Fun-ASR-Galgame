#!/bin/bash

# ====== 配置区域 ======
# 你可以直接修改这里的路径，或者保持默认
SOURCE_DIR="/mnt/o/datasets/huggingface/hub/datasets--litagin--Galgame_Speech_ASR_16kHz/data"
# 损坏文件列表保存位置
BAD_FILES_LOG="bad_files.txt"

# ====== 颜色定义 ======
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}====== Galgame 数据集完整性体检工具 ======${NC}"
echo -e "检查目录: $SOURCE_DIR"
echo ""

# 检查目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}错误：找不到目录 $SOURCE_DIR${NC}"
    exit 1
fi

# 清空旧的日志
> "$BAD_FILES_LOG"

# 获取所有 tar 文件
echo "正在扫描文件列表..."
mapfile -t tar_files < <(ls "$SOURCE_DIR"/galgame-speech-asr-16kHz-train-*.tar 2>/dev/null | sort)
total_files=${#tar_files[@]}

if [ "$total_files" -eq 0 ]; then
    echo -e "${RED}未找到 tar 文件！${NC}"
    exit 1
fi

echo -e "共找到 ${YELLOW}$total_files${NC} 个文件，准备开始体检..."
echo "----------------------------------------"

good_count=0
bad_count=0

# 开始循环检查
for i in "${!tar_files[@]}"; do
    file="${tar_files[$i]}"
    filename=$(basename "$file")
    
    # 计算进度 (i+1)
    current=$((i + 1))
    
    # 显示正在检查（不换行）
    echo -ne "[${current}/${total_files}] 正在检查: $filename ... "
    
    # === 核心检查逻辑 ===
    # 尝试读取压缩包内的第一行文件名
    # 2>&1 把错误输出也捕获，防止刷屏
    first_content=$(tar -tf "$file" 2>&1 | head -n 1)
    exit_code=$?

    # 判断逻辑：
    # 1. 如果 tar 命令返回值非 0 (报错) -> 坏
    # 2. 如果 tar 命令返回 0 但输出为空 (僵尸文件) -> 坏
    if [ $exit_code -ne 0 ] || [ -z "$first_content" ]; then
        echo -e "${RED}✗ [损坏/无效]${NC}"
        # 记录到坏文件列表
        echo "$file" >> "$BAD_FILES_LOG"
        ((bad_count++))
    else
        echo -e "${GREEN}✓ [正常]${NC}"
        ((good_count++))
    fi
done

echo "----------------------------------------"
echo -e "${CYAN}====== 体检报告 ======${NC}"
echo -e "总计检查: $total_files"
echo -e "正常文件: ${GREEN}$good_count${NC}"
echo -e "损坏文件: ${RED}$bad_count${NC}"
echo ""

if [ $bad_count -gt 0 ]; then
    echo -e "${RED}注意：已发现 $bad_count 个损坏文件！${NC}"
    echo -e "损坏文件的完整路径已保存至: ${YELLOW}$BAD_FILES_LOG${NC}"
    echo ""
    echo -e "你可以使用以下命令一键删除它们："
    echo -e "${CYAN}cat $BAD_FILES_LOG | xargs rm${NC}"
    echo -e "(删除后请记得重新运行 huggingface download 命令进行补全)"
else
    echo -e "${GREEN}恭喜！所有文件看起来都很健康！${NC}"
fi