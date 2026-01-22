# FunASR Windows Finetune Script (PowerShell)
# 专为 0-1000 条数据微调设计

# ====== 配置区域 ======

# 1. 显卡设置
$env:CUDA_VISIBLE_DEVICES = "0"
$gpu_num = 1

# 2. 模型路径 (会自动从 ModelScope 下载或使用本地缓存)
$model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"

# 3. 数据路径 (请确认文件名是否正确)
$workspace_dir = "R:\datasets--litagin--Galgame_Speech_ASR_16kHz"
$train_data = Join-Path $workspace_dir "train_0_1000.jsonl"
$val_data = Join-Path $workspace_dir "val_0_1000.jsonl"

# 4. 输出路径
$output_dir = Join-Path $workspace_dir "outputs_0_1000"
$log_file = Join-Path $output_dir "log.txt"

# 创建输出目录
if (-not (Test-Path -Path $output_dir)) {
    New-Item -ItemType Directory -Force -Path $output_dir | Out-Null
}

Write-Host "Log file: $log_file"
Write-Host "Model: $model_dir"
Write-Host "Train Data: $train_data"

# 5. 训练参数
# batch_size=6000 token (约占用 10-12G 显存)
$batch_size = 6000

# 学习率 (Adapter 微调建议稍微大一点，0.0002 是默认值)
$lr = 0.0002

# 训练轮数
$max_epoch = 50

# ====== 启动命令 ======

# 获取 funasr-train 命令的路径 (假设在当前环境下)
$train_tool = "funasr-train"

# 构建参数列表 (注意 Windows 下换行符 `)
$cmd_args = @(
    "++model='$model_dir'",
    "++trust_remote_code=true",
    "++train_data_set_list='$train_data'",
    "++valid_data_set_list='$val_data'",
    "++dataset_conf.data_split_num=1",
    "++dataset_conf.batch_sampler='BatchSampler'",
    "++dataset_conf.batch_size=$batch_size",
    "++dataset_conf.sort_size=1024",
    "++dataset_conf.batch_type='token'",
    "++dataset_conf.num_workers=0",  # Windows 下建议设为 0 以避免多进程报错，Linux 可设为 4
    "++train_conf.max_epoch=$max_epoch",
    "++train_conf.log_interval=1",
    "++train_conf.resume=true",
    "++train_conf.validate_interval=200", # 每200步验证一次
    "++train_conf.save_checkpoint_interval=200", # 每200步保存一次
    "++train_conf.keep_nbest_models=5",
    "++train_conf.avg_nbest_model=5",
    "++train_conf.use_deepspeed=false", # Windows 禁用 DeepSpeed
    "++optim_conf.lr=$lr",
    
    # === 微调策略 (Adapter 微调) ===
    # 数据量小 (<1000小时)，建议只训练 Adaptor
    "++audio_encoder_conf.freeze=true",   # 冻结 Encoder
    "++audio_adaptor_conf.freeze=false",  # 训练 Adaptor
    "++llm_conf.freeze=true",             # 冻结 LLM
    
    "++output_dir='$output_dir'"
)

# 打印完整命令供调试
Write-Host "Running command:"
Write-Host "python -m funasr.bin.train $cmd_args"

# 运行训练 (重定向日志)
# 注意：PowerShell 中重定向 stderr 需要用 2>&1
python -m funasr.bin.train $cmd_args 2>&1 | Tee-Object -FilePath $log_file


# tensorboard --logdir R:\datasets--litagin--Galgame_Speech_ASR_16kHz\outputs_0_1000\tensorboard