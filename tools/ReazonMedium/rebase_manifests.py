import argparse
import random
from pathlib import Path
from typing import List, Optional, Set, Tuple


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


def load_avoid_set(path: Optional[str]) -> Set[str]:
    if not path:
        return set()
    avoid_path = Path(path).expanduser().absolute()
    avoid_set: Set[str] = set()
    with avoid_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt = line.split(maxsplit=1)[0]
            avoid_set.add(utt)
    return avoid_set


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild manifests and split into train/val with strict val size. "
            "Entries in --avoid-val-list are excluded from val and prioritized into train."
        )
    )
    parser.add_argument("--audio-dir", required=True, help="Directory containing audio files")
    parser.add_argument("--transcript", required=True, help="Transcript file with utt_id and text")

    parser.add_argument("--output-train-scp", required=True, help="Output train scp file path")
    parser.add_argument("--output-train-text", required=True, help="Output train text file path")
    parser.add_argument("--output-val-scp", required=True, help="Output val scp file path")
    parser.add_argument("--output-val-text", required=True, help="Output val text file path")

    parser.add_argument("--ext", default=".flac", help="Audio file extension (default: .flac)")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip entries whose audio file is missing instead of failing.",
    )
    parser.add_argument("--val-size", type=int, default=2000, help="Exact number of val entries to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--avoid-val-list",
        default=None,
        help=(
            "Path to list file. Each line is whitespace-separated text like transcript; "
            "the first token (utt_id) will be excluded from val."
        ),
    )

    args = parser.parse_args()

    if args.val_size <= 0:
        raise ValueError("--val-size must be > 0")

    audio_dir = Path(args.audio_dir).expanduser().absolute()
    transcript_path = Path(args.transcript).expanduser().absolute()

    out_train_scp = Path(args.output_train_scp).expanduser().absolute()
    out_train_txt = Path(args.output_train_text).expanduser().absolute()
    out_val_scp = Path(args.output_val_scp).expanduser().absolute()
    out_val_txt = Path(args.output_val_text).expanduser().absolute()

    out_train_scp.parent.mkdir(parents=True, exist_ok=True)
    out_train_txt.parent.mkdir(parents=True, exist_ok=True)
    out_val_scp.parent.mkdir(parents=True, exist_ok=True)
    out_val_txt.parent.mkdir(parents=True, exist_ok=True)

    avoid_set = load_avoid_set(args.avoid_val_list)

    total = 0
    valid = 0
    missing = 0

    entries: List[Tuple[str, str, str]] = []
    for line in transcript_path.open("r", encoding="utf-8"):
        parsed = parse_line(line)
        if parsed is None:
            continue
        utt, text = parsed
        total += 1
        audio_path = audio_dir / f"{utt}{args.ext}"
        if not audio_path.exists():
            missing += 1
            if args.skip_missing:
                continue
            raise FileNotFoundError(f"Audio not found for {utt}: {audio_path}")
        entries.append((utt, text, str(audio_path)))
        valid += 1

    val_candidates = [idx for idx, (utt, _, _) in enumerate(entries) if utt not in avoid_set]
    if len(val_candidates) < args.val_size:
        raise ValueError(
            f"Not enough val candidates after avoid filter: need {args.val_size}, got {len(val_candidates)}"
        )

    rng = random.Random(args.seed)
    selected_val_indices: List[int] = []
    candidate_pool = val_candidates[:]

    while len(selected_val_indices) < args.val_size:
        pick = rng.randrange(len(candidate_pool))
        selected_val_indices.append(candidate_pool.pop(pick))

    val_index_set = set(selected_val_indices)

    train_entries: List[Tuple[str, str, str]] = []
    val_entries: List[Tuple[str, str, str]] = []

    for idx, item in enumerate(entries):
        if idx in val_index_set:
            val_entries.append(item)
        else:
            train_entries.append(item)

    with out_train_scp.open("w", encoding="utf-8") as f_train_scp, \
        out_train_txt.open("w", encoding="utf-8") as f_train_txt, \
        out_val_scp.open("w", encoding="utf-8") as f_val_scp, \
        out_val_txt.open("w", encoding="utf-8") as f_val_txt:

        for utt, text, wav in train_entries:
            f_train_scp.write(f"{utt}\t{wav}\n")
            f_train_txt.write(f"{utt}\t{text}\n")

        for utt, text, wav in val_entries:
            f_val_scp.write(f"{utt}\t{wav}\n")
            f_val_txt.write(f"{utt}\t{text}\n")

    print(f"Processed transcript: {total} entries")
    print(f"Valid (audio exists): {valid}")
    if missing:
        print(f"Missing audio: {missing}{' (skipped)' if args.skip_missing else ''}")

    print(f"Avoid list size: {len(avoid_set)}")
    print(f"Val candidates after avoid filter: {len(val_candidates)}")
    print(f"Val selected: {len(val_entries)}")
    print(f"Train selected: {len(train_entries)}")

    print(f"TRAIN SCP : {out_train_scp}")
    print(f"TRAIN TEXT: {out_train_txt}")
    print(f"VAL SCP   : {out_val_scp}")
    print(f"VAL TEXT  : {out_val_txt}")


if __name__ == "__main__":
    main()
