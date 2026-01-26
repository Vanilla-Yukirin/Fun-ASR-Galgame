import argparse
import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple

from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description=
        "Download ReazonSpeech via Hugging Face datasets, copy FLAC to target, and emit scp/text/jsonl manifests."
    )
    p.add_argument(
        "--subset",
        default="tiny",
        choices=[
            "tiny",
            "small",
            "medium",
            "large",
            "all",
            "small-v1",
            "medium-v1",
            "all-v1",
        ],
        help="Dataset config to load (size/version).",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Dataset split to load (default: train).",
    )
    p.add_argument(
        "--output_dir",
        default=str(Path("/root/autodl-tmp").expanduser() / "reazonspeech"),
        help="Directory to place copied audio and manifests.",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit the number of samples for quick tests.",
    )
    p.add_argument(
        "--num_threads",
        type=int,
        default=8,
        help="Number of threads for file copying.",
    )
    p.add_argument(
        "--agree_terms",
        action="store_true",
        help=(
            "You must agree to use the dataset solely for the purpose of Japanese Copyright Act Article 30-4."
        ),
    )
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to load_dataset.",
    )
    p.add_argument(
        "--streaming",
        action="store_true",
        help="Use datasets streaming mode to iterate without pre-downloading entire split.",
    )
    return p.parse_args()


def ensure_dirs(base: Path) -> Dict[str, Path]:
    audio_dir = base / "audio"
    manifest_dir = base / "manifests"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return {"audio": audio_dir, "manifests": manifest_dir}


def load_reazonspeech(subset: str, split: str, trust_remote_code: bool, streaming: bool):
    from datasets import load_dataset, Audio

    ds = load_dataset(
        "reazon-research/reazonspeech",
        subset,
        trust_remote_code=trust_remote_code,
        split=split,
        streaming=streaming,
    )
    ds = ds.cast_column("audio", Audio(decode=False))
    return ds


def item_to_paths(item: Dict, idx: int) -> Tuple[str, str, str]:
    # Expected schema with decode=False:
    # {
    #   'name': '000/0000000000000.flac',
    #   'audio': {'path': '/path/to/extracted/flac', 'bytes': ...},
    #   'transcription': '...'
    # }
    name = item.get("name") or f"utt_{idx:09d}.flac"
    base = os.path.splitext(os.path.basename(name))[0]
    utt_id = base
    src_path = item["audio"]["path"]
    text = item.get("transcription", "")
    return utt_id, src_path, text


def main():
    args = parse_args()

    if not args.agree_terms:
        raise SystemExit(
            "You must pass --agree_terms to confirm use under Japanese Copyright Act Article 30-4."
        )

    base_out = Path(args.output_dir).expanduser() / args.subset / args.split
    dirs = ensure_dirs(base_out)
    audio_out = dirs["audio"]
    manifest_out = dirs["manifests"]

    # Load dataset
    ds = load_reazonspeech(
        subset=args.subset,
        split=args.split,
        trust_remote_code=args.trust_remote_code,
        streaming=args.streaming,
    )

    # Prepare iteration
    if args.streaming:
        iterator = enumerate(ds)
        length = args.max_samples if args.max_samples is not None else None
        pbar = tqdm(iterator, total=length, desc="Iterating (streaming)")
    else:
        if args.max_samples is not None:
            ds = ds.select(range(min(args.max_samples, len(ds))))
        pbar = tqdm(range(len(ds)), desc="Copying")

    # Collect copy jobs and manifests
    copy_jobs = []
    scp_lines = []
    text_lines = []
    jsonl_lines = []

    def plan_one(idx: int, item: Dict):
        utt_id, src_path, text = item_to_paths(item, idx)
        # Keep original filename; flatten to utt_id.flac
        dst = audio_out / f"{utt_id}.flac"
        return (utt_id, src_path, text, dst)

    if args.streaming:
        planned = []
        for idx, item in pbar:
            planned.append(plan_one(idx, item))
            if args.max_samples is not None and len(planned) >= args.max_samples:
                break
    else:
        planned = [plan_one(i, ds[i]) for i in pbar]

    # Copy with threads
    with ThreadPoolExecutor(max_workers=args.num_threads) as ex:
        future_to_job = {
            ex.submit(shutil.copy2, src, dst): (utt, src, text, dst)
            for (utt, src, text, dst) in planned
        }
        for fut in tqdm(as_completed(future_to_job), total=len(future_to_job), desc="Writing"):
            utt, src, text, dst = future_to_job[fut]
            try:
                fut.result()
            except Exception as e:
                # Skip failed copies but log manifest for visibility
                print(f"WARN: failed to copy {src} -> {dst}: {e}")
            rel_audio = dst.as_posix()
            scp_lines.append(f"{utt}\t{rel_audio}\n")
            text_lines.append(f"{utt}\t{text}\n")
            jsonl_lines.append(
                {
                    "audio": rel_audio,
                    "transcription": text,
                    "utt_id": utt,
                }
            )

    # Write manifests
    scp_path = manifest_out / f"{args.split}.scp"
    text_path = manifest_out / f"{args.split}.txt"
    jsonl_path = manifest_out / f"{args.split}.jsonl"

    with open(scp_path, "w", encoding="utf-8") as f:
        f.writelines(scp_lines)
    with open(text_path, "w", encoding="utf-8") as f:
        f.writelines(text_lines)
    # Write jsonl
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in jsonl_lines:
            # Manual JSONL to avoid extra dependencies
            import json

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nDone.")
    print(f"Audio: {audio_out}")
    print(f"SCP: {scp_path}")
    print(f"TEXT: {text_path}")
    print(f"JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
