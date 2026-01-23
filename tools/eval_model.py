import argparse
import os
import subprocess
import sys

def run_command(cmd, shell=False):
    """Run a command in the shell and check for errors."""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        # result = subprocess.run(cmd, shell=shell, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # We want to see output in real time
        result = subprocess.run(cmd, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Automated Evaluation Script for FunASR Models")
    parser.add_argument("--model_dir", type=str, required=True, 
                        help="Path to the model directory. Use 'default' for the pretrained base model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for decoding (e.g., cuda, cpu)")
    parser.add_argument("--gpu_id", type=int, default=0, help="Accurate GPU ID")
    parser.add_argument("--output_name", type=str, required=True, 
                        help="Prefix for output files (e.g., 'epoch_1_step_5000')")
    parser.add_argument("--scp_file", type=str, default="/mnt/d/ML/datasets--litagin--Galgame_Speech_ASR_16kHz/tail1000.scp",
                        help="Path to the validation scp file")
    parser.add_argument("--ref_norm_text", type=str, default="/mnt/d/ML/datasets--litagin--Galgame_Speech_ASR_16kHz/tail1000_norm.txt",
                        help="Path to the normalized reference text file")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for decoding (e.g., '语音转写成日文：')")

    args = parser.parse_args()

    # Define output paths
    # If model_dir is default, save in current dir or a specific eval dir? 
    # Let's save in the same dir as ref_norm_text's parent or current dir if default
    
    if args.model_dir == "default":
        # For default model, save outputs in the current directory or a dedicated eval folder
        output_dir = "eval_results/default"
    else:
        output_dir = args.model_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    decode_out = os.path.join(output_dir, f"{args.output_name}_decode.txt")
    norm_out = os.path.join(output_dir, f"{args.output_name}_norm.txt")
    cer_out = os.path.join(output_dir, f"{args.output_name}_cer.txt")
    
    # 1. Decoding
    print("-" * 50)
    print("Step 1: Decoding...")
    
    decode_cmd = [
        "python", "decode.py",
        f"++scp_file={args.scp_file}",
        f"++output_file={decode_out}",
        f"++device={args.device}",
        f"++ngpu={1}",
        f"++gpuid_list=[{args.gpu_id}]"
    ]
    
    if args.model_dir != "default":
        # Check if model_dir needs to be passed, assuming decode.py supports it based on Hydra or arg parsing
        # If decode.py is hydra-based, usually it's ++param=value. 
        # But if it loads from a checkpoint, we need to know the specific parameter name.
        # Assuming ++model_dir or ++init_param is used for checkpoint loading.
        # Based on typical FunASR, it might be ++init_param=.../model.pt or ++model_path=...
        # Let's assume it points to the model_dir/model.pt or similar. 
        # USER INTENT: "传入model_dir" implies the directory containing checkpoints or config.
        # Let's try appending ++init_param={args.model_dir}/model.pt if it's a directory
        
        # Heuristic: if model_dir is a file (model.pt), use it directly. If dir, look for model.pt
        if os.path.isdir(args.model_dir):
             # Try to find the best model or specific one? 
             # Usually user points to a specific checkpoint folder or the training output dir.
             # Let's assume user passes the EXACT directory containing 'model.pt' OR user passes the full path to model.pt
             # But wait, standard FunASR inference usually takes `++model_path` or `++init_param`.
             # Let's use `++init_param` which is common for overriding weights.
             model_path = os.path.join(args.model_dir, "model.pt")
             if not os.path.exists(model_path):
                 # Fallback: maybe the user passed the path to the .pt file directly?
                 print(f"Warning: {model_path} not found. Assuming input is the checkpoint file itself.")
                 model_path = args.model_dir
        else:
             model_path = args.model_dir
             
        decode_cmd.append(f"++init_param={model_path}")
        
        # Also need to point to config.yaml if it exists in the model dir?
        config_path = os.path.join(os.path.dirname(model_path), "config.yaml")
        if os.path.exists(config_path):
            decode_cmd.append(f"++config={config_path}")

    if args.prompt:
         decode_cmd.append(f"++prompt='{args.prompt}'")

    run_command(decode_cmd)
    
    # 2. Normalization
    print("-" * 50)
    print("Step 2: Normalizing...")
    # python tools/whisper_mix_normalize.py <src> <dst>
    norm_cmd = ["python", "tools/whisper_mix_normalize.py", decode_out, norm_out]
    run_command(norm_cmd)
    
    # 3. Computing WER/CER
    print("-" * 50)
    print("Step 3: Computing CER...")
    # compute-wer --text --mode=present <ref> <hyp>
    # Note: Using shell=True for this one to handle redirection > nicely, or just use python to write file
    
    # We construct the command string for compute-wer
    # Assuming compute-wer is available in path.
    # The arguments "ark:..." usually required for Kaldi tools if input is not via pipe, but compute-wer often takes files directly if text mode.
    # User example: compute-wer "ref" "hyp" output.txt -> wait, compute-wer params are usually: compute-wer [options] <ref-rspecifier> <hyp-rspecifier>
    # User's example: compute-wer "..." "..." cer_out.txt
    # This looks like: compute-wer ref_file hyp_file > output
    
    # Let's try the standard way:
    cmd_str = f"compute-wer --text --mode=present ark:{args.ref_norm_text} ark:{norm_out} > {cer_out}"
    print(f"Running: {cmd_str}")
    subprocess.run(cmd_str, shell=True, check=True)
    
    # Print tail
    print("-" * 50)
    print(f"Results saved to: {cer_out}")
    print("Summary:")
    # Run tail
    subprocess.run(f"tail -n 8 {cer_out}", shell=True)

if __name__ == "__main__":
    main()
