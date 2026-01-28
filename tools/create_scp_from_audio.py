import argparse
import os
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Generate wav.scp from a directory of audio files.")
    parser.add_argument("audio_dir", help="Path to the directory containing audio files")
    parser.add_argument("output_dir", help="Directory to save the generated wav.scp")
    parser.add_argument("--ext", default=".flac", help="Audio file extension (default: .flac)")
    args = parser.parse_args()

    audio_path = Path(args.audio_dir).expanduser().absolute()
    output_path = Path(args.output_dir).expanduser().absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    scp_file = output_path / "wav.scp"
    
    files = sorted(list(audio_path.glob(f"*{args.ext}")))
    
    if not files:
        print(f"No files found in {audio_path} with extension {args.ext}")
        return

    print(f"Found {len(files)} files in {audio_path}")
    print(f"Writing to {scp_file}...")

    with open(scp_file, "w", encoding="utf-8") as f:
        for file_path in tqdm(files):
            utt_id = file_path.stem  # Use filename without extension as utt_id
            f.write(f"{utt_id}\t{file_path}\n")

    print("Done.")

if __name__ == "__main__":
    main()
