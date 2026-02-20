import argparse
import torch
import os
from funasr import AutoModel

def main():
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description="FunASR 演示脚本 - 自动适配微调权重")
    parser.add_argument("--base_model", type=str, default="FunAudioLLM/Fun-ASR-Nano-2512", 
                        help="基础模型 ID (例如 ModelScope ID)")
    parser.add_argument("--model_dir", type=str, default=None, 
                        help="微调后的输出文件夹 (包含 model.pt.best)")
    parser.add_argument("--init_param", type=str, default=None, 
                        help="特定权重文件路径 (例如训练出的 model.pt.best)")
    parser.add_argument("--audio_file", type=str, required=True, 
                        help="要识别的音频文件路径")
    parser.add_argument("--prompt", type=str, default="语音转写成日文：", 
                        help="推理提示词 (例如 '语音转写成日文：')")
    
    args = parser.parse_args()

    # 2. 逻辑处理：自动寻找权重文件
    actual_init_param = args.init_param
    if not actual_init_param and args.model_dir:
        # 尝试寻找最好的权重
        best_pt = os.path.join(args.model_dir, "model.pt.best")
        if os.path.exists(best_pt):
            actual_init_param = best_pt
        else:
            # 尝试找普通权重
            normal_pt = os.path.join(args.model_dir, "model.pt")
            if os.path.exists(normal_pt):
                actual_init_param = normal_pt

    # 3. 设备检测 (GPU/MPS/CPU)
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"使用设备: {device}")
    if actual_init_param:
        print(f"加载微调权重: {actual_init_param}")

    # 4. 加载模型
    # model: 设为官方 ID 以确保分词器加载正常 (避开本地 configuration.json 中的绝对路径坑)
    # init_param: 指向你的本地权重文件
    model = AutoModel(
        model=args.base_model,
        init_param=actual_init_param,
        trust_remote_code=True,
        remote_code="./model.py",
        device=device
    )

    # 5. 执行推理
    print(f"开始推理 (Prompt: {args.prompt})...")
    res = model.generate(
        input=[args.audio_file],
        cache={},
        batch_size=1,
        prompt=args.prompt
    )

    # 6. 输出结果
    if res and len(res) > 0:
        text = res[0].get("text", "")
        print("\n识别结果:")
        print("-" * 30)
        print(text)
        print("-" * 30)
    else:
        print("未检测到识别结果。")

if __name__ == "__main__":
    main()
