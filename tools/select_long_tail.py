import argparse
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def parse_line(line: str) -> Optional[Tuple[str, str]]:
    line = line.strip()
    if not line:
        return None
    if "\t" in line:
        parts = line.split("\t", 1)
    else:
        parts = line.split(maxsplit=1)
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def is_valid_word(word: str, min_len: int, filter_alnum: bool, filter_kana: bool) -> bool:
    if len(word) < min_len:
        return False
    if filter_alnum and re.match(r"^[0-9A-Za-z]+$", word):
        return False
    if filter_kana and re.match(r"^[ぁ-んァ-ンー]+$", word):
        return False
    if re.match(r"^[\W_]+$", word):
        return False
    return True


def iter_text(path: Path) -> Iterable[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue
            yield parsed


def build_counts(text_path: Path, min_len: int, filter_alnum: bool, filter_kana: bool, verbose_every: int) -> Counter:
    import MeCab

    tagger = MeCab.Tagger()
    counter: Counter = Counter()
    for idx, (_utt_id, text) in enumerate(iter_text(text_path)):
        node = tagger.parseToNode(text)
        while node:
            word = node.surface
            features = node.feature.split(",") if node.feature else []
            pos = features[0] if features else ""
            if pos == "名詞" and is_valid_word(word, min_len, filter_alnum, filter_kana):
                counter[word] += 1
            node = node.next
        if verbose_every and idx and idx % verbose_every == 0:
            print(f"[pass1] processed {idx} lines")
    return counter


def collect_long_tail_lines(
    text_path: Path,
    deficits: dict,
    min_len: int,
    filter_alnum: bool,
    filter_kana: bool,
    verbose_every: int,
    max_dup_per_utt: Optional[int],
) -> List[Tuple[str, str]]:
    import MeCab

    tagger = MeCab.Tagger()
    selected: List[Tuple[str, str]] = []
    remaining = {w: d for w, d in deficits.items() if d > 0}

    for idx, (utt_id, text) in enumerate(iter_text(text_path)):
        hits = []
        node = tagger.parseToNode(text)
        while node:
            word = node.surface
            features = node.feature.split(",") if node.feature else []
            pos = features[0] if features else ""
            if pos == "名詞" and is_valid_word(word, min_len, filter_alnum, filter_kana):
                if word in remaining and remaining[word] > 0:
                    hits.append(word)
            node = node.next

        if hits:
            need = max(remaining[w] for w in hits)
            if max_dup_per_utt is not None:
                need = min(need, max_dup_per_utt)
            for i in range(need):
                selected.append((f"{utt_id}_dup{i+1}", text))
            for w in hits:
                remaining[w] = max(0, remaining[w] - need)

        if verbose_every and idx and idx % verbose_every == 0:
            print(f"[pass2] processed {idx} lines (selected {len(selected)})")

        if remaining and all(v == 0 for v in remaining.values()):
            break

    return selected


def load_scp(path: Path) -> dict:
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue
            utt, wav = parsed
            mapping[utt] = wav
    return mapping


def write_outputs(
    entries: List[Tuple[str, str]],
    scp_map: dict,
    out_scp: Path,
    out_txt: Path,
):
    missing = 0
    with out_scp.open("w", encoding="utf-8") as f_scp, out_txt.open("w", encoding="utf-8") as f_txt:
        for utt, text in entries:
            base_utt = utt.rsplit("_dup", 1)[0]
            wav = scp_map.get(base_utt)
            if wav is None:
                missing += 1
                continue
            f_scp.write(f"{utt}\t{wav}\n")
            f_txt.write(f"{utt}\t{text}\n")
    if missing:
        print(f"WARN: {missing} entries skipped because utt_id not found in SCP")


def main():
    parser = argparse.ArgumentParser(
        description="Select sentences containing long-tail Japanese words and duplicate them, shuffling outputs."
    )
    parser.add_argument("--input-text", required=True, help="Input text file (utt_id\ttext)")
    parser.add_argument("--input-scp", required=True, help="Input scp file (utt_id\tpath)")
    parser.add_argument("--output-text", required=True, help="Output text file path")
    parser.add_argument("--output-scp", required=True, help="Output scp file path")
    parser.add_argument("--threshold", type=int, default=200, help="Word frequency <= threshold is considered long-tail")
    parser.add_argument(
        "--target-count",
        type=int,
        default=200,
        help="Ensure each long-tail word appears at least this many times; duplication is computed from deficits.",
    )
    parser.add_argument(
        "--max-dup-per-utt",
        type=int,
        default=None,
        help="Optional cap on how many times a single sentence can be duplicated.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument("--min-len", type=int, default=2, help="Minimum word length to consider")
    parser.add_argument("--no-filter-alnum", action="store_true", help="Keep pure alnum tokens")
    parser.add_argument("--no-filter-kana", action="store_true", help="Keep pure kana tokens")
    parser.add_argument("--verbose-every", type=int, default=100000, help="Print progress every N lines (0 to disable)")
    args = parser.parse_args()

    text_path = Path(args.input_text)
    scp_path = Path(args.input_scp)
    out_txt = Path(args.output_text)
    out_scp = Path(args.output_scp)

    print("Pass 1: counting words...")
    counts = build_counts(
        text_path,
        min_len=args.min_len,
        filter_alnum=not args.no_filter_alnum,
        filter_kana=not args.no_filter_kana,
        verbose_every=args.verbose_every,
    )

    long_tail_words = {w for w, c in counts.items() if c <= args.threshold}
    print(f"Identified {len(long_tail_words)} long-tail words (<= {args.threshold})")

    deficits = {w: max(0, args.target_count - counts[w]) for w in long_tail_words}
    total_deficit = sum(deficits.values())
    print(f"Total remaining occurrences needed: {total_deficit}")

    print("Pass 2: selecting lines and duplicating...")
    selected = collect_long_tail_lines(
        text_path,
        deficits,
        min_len=args.min_len,
        filter_alnum=not args.no_filter_alnum,
        filter_kana=not args.no_filter_kana,
        verbose_every=args.verbose_every,
        max_dup_per_utt=args.max_dup_per_utt,
    )

    print(f"Selected {len(selected)} duplicated entries before shuffling")
    rnd = random.Random(args.seed)
    rnd.shuffle(selected)
    print("Shuffled output entries")

    print("Loading SCP map...")
    scp_map = load_scp(scp_path)

    print("Writing outputs...")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_scp.parent.mkdir(parents=True, exist_ok=True)
    write_outputs(selected, scp_map, out_scp, out_txt)

    print("Done.")
    print(f"Text: {out_txt}")
    print(f"SCP : {out_scp}")


if __name__ == "__main__":
    main()
