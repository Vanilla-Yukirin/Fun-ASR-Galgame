import argparse
from pathlib import Path
from typing import Optional, Tuple


def parse_line(line: str) -> Optional[Tuple[str, str]]:
    line = line.rstrip("\n")
    if not line:
        return None
    if "\t" in line:
        parts = line.split("\t", 1)
    else:
        parts = line.split(maxsplit=1)
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild scp/text manifests using a new audio directory. "
            "Uses transcript (utt_id + text) and assumes audio files are named <utt_id><ext>."
        )
    )
    parser.add_argument("--audio-dir", required=True, help="Directory containing audio files")
    parser.add_argument("--transcript", required=True, help="Transcript file with utt_id and text")
    parser.add_argument("--output-scp", required=True, help="Output scp file path")
    parser.add_argument("--output-text", required=True, help="Output text file path")
    parser.add_argument("--ext", default=".flac", help="Audio file extension (default: .flac)")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip entries whose audio file is missing instead of failing.",
    )
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir).expanduser().absolute()
    transcript_path = Path(args.transcript).expanduser().absolute()
    out_scp = Path(args.output_scp).expanduser().absolute()
    out_txt = Path(args.output_text).expanduser().absolute()

    out_scp.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    missing = 0

    with transcript_path.open("r", encoding="utf-8") as fin, out_scp.open("w", encoding="utf-8") as f_scp, out_txt.open("w", encoding="utf-8") as f_txt:
        for line in fin:
            parsed = parse_line(line)
            if parsed is None:
                continue
            utt, text = parsed
            audio_path = audio_dir / f"{utt}{args.ext}"
            total += 1
            if not audio_path.exists():
                missing += 1
                if args.skip_missing:
                    continue
                raise FileNotFoundError(f"Audio not found for {utt}: {audio_path}")
            f_scp.write(f"{utt}\t{audio_path}\n")
            f_txt.write(f"{utt}\t{text}\n")
            written += 1

    print(f"Processed transcript: {total} entries")
    print(f"Written: {written}")
    if missing:
        print(f"Missing audio: {missing}{' (skipped)' if args.skip_missing else ''}")
    print(f"SCP: {out_scp}")
    print(f"TEXT: {out_txt}")


if __name__ == "__main__":
    main()
