import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple


def parse_line(line: str) -> Optional[Tuple[str, str]]:
    line = line.rstrip("\n")
    if not line:
        return None
    if "\t" in line:
        parts = line.split("\t", 1)
    else:
        parts = line.split(maxsplit=1)
    if len(parts) == 1:
        # Allow empty text; treat as ""
        return parts[0], ""
    return parts[0], parts[1]


def load_pair(text_path: Path, scp_path: Path, strict_utt_match: bool) -> List[Tuple[str, str, str]]:
    texts = text_path.read_text(encoding="utf-8").splitlines()
    scps = scp_path.read_text(encoding="utf-8").splitlines()
    if len(texts) != len(scps):
        raise ValueError(
            f"Line count mismatch: text {text_path} has {len(texts)}, scp {scp_path} has {len(scps)}"
        )

    merged: List[Tuple[str, str, str]] = []
    for idx, (t_line, s_line) in enumerate(zip(texts, scps)):
        t_parsed = parse_line(t_line)
        s_parsed = parse_line(s_line)
        if t_parsed is None or s_parsed is None:
            raise ValueError(f"Failed to parse line {idx} in text or scp: {t_line!r} / {s_line!r}")
        t_utt, t_text = t_parsed
        s_utt, s_path = s_parsed
        if strict_utt_match and t_utt != s_utt:
            raise ValueError(
                f"Utt mismatch at line {idx}: text utt={t_utt}, scp utt={s_utt}. Use --no-strict-utt to bypass."
            )
        utt = t_utt if strict_utt_match else t_utt or s_utt
        merged.append((utt, t_text, s_path))
    return merged


def write_outputs(entries: List[Tuple[str, str, str]], out_text: Path, out_scp: Path):
    out_text.parent.mkdir(parents=True, exist_ok=True)
    out_scp.parent.mkdir(parents=True, exist_ok=True)
    with out_text.open("w", encoding="utf-8") as f_txt, out_scp.open("w", encoding="utf-8") as f_scp:
        for utt, text, wav in entries:
            f_txt.write(f"{utt}\t{text}\n")
            f_scp.write(f"{utt}\t{wav}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle-mix multiple text/scp pairs while preserving line alignment within each pair."
    )
    parser.add_argument(
        "--text-files",
        nargs="+",
        required=True,
        help="List of input text files (utt_id<TAB>text).",
    )
    parser.add_argument(
        "--scp-files",
        nargs="+",
        required=True,
        help="List of input scp files (utt_id<TAB>path). Must align with text-files order.",
    )
    parser.add_argument("--output-text", required=True, help="Output mixed text file.")
    parser.add_argument("--output-scp", required=True, help="Output mixed scp file.")
    parser.add_argument(
        "--seed", type=int, default=1234, help="Shuffle seed to keep reproducibility."
    )
    parser.add_argument(
        "--no-strict-utt",
        action="store_true",
        help="Allow differing utt_ids between text/scp lines (still position-aligned).",
    )
    args = parser.parse_args()

    if len(args.text_files) != len(args.scp_files):
        raise SystemExit("text-files and scp-files counts must match")

    all_entries: List[Tuple[str, str, str]] = []
    for t_path, s_path in zip(args.text_files, args.scp_files):
        entries = load_pair(Path(t_path), Path(s_path), strict_utt_match=not args.no_strict_utt)
        all_entries.extend(entries)

    rnd = random.Random(args.seed)
    rnd.shuffle(all_entries)

    write_outputs(all_entries, Path(args.output_text), Path(args.output_scp))

    print(f"Mixed {len(all_entries)} entries from {len(args.text_files)} pairs")
    print(f"Text -> {args.output_text}")
    print(f"SCP  -> {args.output_scp}")


if __name__ == "__main__":
    main()
