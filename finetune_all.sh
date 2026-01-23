#!/bin/bash

# FunASR WSL Finetune Script (Bash)
# 全量数据（前80%）训练，仅tail1000验证

# ====== 配置区域 ======

# 1. 获取当前脚本所在目录作为工作区根目录
# 假设脚本在 Fun-ASR-Galgame 根目录下运行
workspace=`pwd`

# 2. 显卡设置
export CUDA_VISIBLE_DEVICES="0"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

# 3. 模型路径 (会自动从 ModelScope 下载或使用本地缓存)
model_dir="FunAudioLLM/Fun-ASR-Nano-2512"

# 4. 数据路径 (都在 D 盘)
external_root="/mnt/d/ML/datasets--litagin--Galgame_Speech_ASR_16kHz"
train_data="${external_root}/train_0_3746131.jsonl"
val_data="${external_root}/tail1000.jsonl"

# 5. 输出路径
output_dir="${external_root}/outputs"
log_file="${output_dir}/log.txt"

# 6. DeepSpeed 配置
# 既然要用 DeepSpeed，需要指定配置文件
# 假设 deepspeed_conf 就在当前项目目录下
deepspeed_config="${workspace}/deepspeed_conf/ds_stage1.json"

# 创建输出目录
mkdir -p "${output_dir}"

echo "Log file: ${log_file}"
echo "Model: ${model_dir}"
echo "Train Data: ${train_data}"

# 7. 训练参数
batch_size=4000
lr=0.0002
max_epoch=5

# ====== 启动命令 ======

# 分布式/DeepSpeed 启动参数
DISTRIBUTED_ARGS="
    --nnodes ${WORLD_SIZE:-1} \
    --nproc_per_node $gpu_num \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-26669}
"

# 获取 funasr-train 命令 (DS版本)
# 如果是 conda 环境，确保已安装 funasr
train_tool="/home/vanilla0302/anaconda3/envs/Fun-ASR/bin/funasr-train"

echo "Running command with torchrun..."

# 使用 torchrun启动，这是启用 DeepSpeed 的推荐方式
torchrun $DISTRIBUTED_ARGS \
"${train_tool}" \
++model="${model_dir}" \
++trust_remote_code=true \
++train_data_set_list="${train_data}" \
++valid_data_set_list="${val_data}" \
++dataset_conf.data_split_num=1 \
++dataset_conf.batch_sampler="BatchSampler" \
++dataset_conf.batch_size=${batch_size} \
++dataset_conf.sort_size=1024 \
++dataset_conf.batch_type="token" \
++dataset_conf.num_workers=4 \
++train_conf.max_epoch=${max_epoch} \
++train_conf.log_interval=1 \
++train_conf.resume=true \
++train_conf.validate_interval=500 \
++train_conf.save_checkpoint_interval=500 \
++train_conf.keep_nbest_models=5 \
++train_conf.avg_nbest_model=5 \
++train_conf.use_deepspeed=true \
++train_conf.deepspeed_config="${deepspeed_config}" \
++optim_conf.lr=${lr} \
++audio_encoder_conf.freeze=false \
++audio_adaptor_conf.freeze=false \
++llm_conf.freeze=true \
# ++audio_encoder_conf.activation_checkpoint=true \
++output_dir="${output_dir}" 2>&1 | tee "${log_file}"

# tensorboard --logdir /mnt/d/ML/datasets--litagin--Galgame_Speech_ASR_16kHz/outputs/tensorboard

# 开启梯度检查点需要修改：
# 135c135
# <             inputs = inputs.clone() * mask
# ---
# >             inputs = inputs * mask
# 141c141
# <         x = x + inputs
# ---
# >         x += inputs
# 562c562
# <         xs_pad = xs_pad * (self.output_size() ** 0.5)
# ---
# >         xs_pad *= self.output_size() ** 0.5