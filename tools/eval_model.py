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


def check_skip(filepath, step_name):
    """Checks if file exists and asks user if they want to skip the step."""
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        while True:
            response = input(f"\nExample output '{os.path.basename(filepath)}' already exists.\nSkip {step_name}? [y/n]: ").lower().strip()
            if response in ['y', 'yes']:
                print(f"Skipping {step_name}...")
                return True
            elif response in ['n', 'no']:
                return False
    return False

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
    parser.add_argument("--yes", action="store_true", help="Automatically say yes to skip prompts if files exist")

    args = parser.parse_args()

    # Define output paths
    if args.model_dir == "default":
        output_dir = "eval_results/default"
    else:
        output_dir = args.model_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    decode_out = os.path.join(output_dir, f"{args.output_name}_decode.txt")
    norm_out = os.path.join(output_dir, f"{args.output_name}_norm.txt")
    cer_out = os.path.join(output_dir, f"{args.output_name}_cer.txt")
    
    # helper for auto-yes
    def ask_skip(fpath, step):
        if args.yes and os.path.exists(fpath) and os.path.getsize(fpath) > 0:
            print(f"Skipping {step} (auto-yes)...")
            return True
        return check_skip(fpath, step)

    # 1. Decoding
    print("-" * 50)
    if not ask_skip(decode_out, "Step 1: Decoding"):
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
             # Use input model_dir directly, similar to user's Windows script
             # This allows FunASR to handle config and model loading internally
             decode_cmd.append(f"++model_dir={args.model_dir}")

        if args.prompt:
             decode_cmd.append(f"++prompt='{args.prompt}'")

        run_command(decode_cmd)
    
    # 2. Normalization
    print("-" * 50)
    if not ask_skip(norm_out, "Step 2: Normalizing"):
        print("Step 2: Normalizing...")
        norm_cmd = ["python", "tools/whisper_mix_normalize.py", decode_out, norm_out]
        run_command(norm_cmd)
    
    # 3. Computing WER/CER
    print("-" * 50)
    if not ask_skip(cer_out, "Step 3: Computing CER"):
        print("Step 3: Computing CER...")
        # Updated command based on 'compute-wer --help'
        cmd_str = f"compute-wer -c {args.ref_norm_text} {norm_out} {cer_out}"
        print(f"Running: {cmd_str}")
        subprocess.run(cmd_str, shell=True, check=True)
    
    # Print tail
    print("-" * 50)
    print(f"Results saved to: {cer_out}")
    print("Summary:")
    if os.path.exists(cer_out):
        subprocess.run(f"tail -n 8 {cer_out}", shell=True)

if __name__ == "__main__":
    main()

