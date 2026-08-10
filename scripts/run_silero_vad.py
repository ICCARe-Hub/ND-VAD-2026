import os
import json
import argparse
from pathlib import Path

import torch


def iter_audio_files(audio_dir: Path, ext: str, recursive: bool):
    if recursive:
        yield from audio_dir.rglob(f"*.{ext}")
    else:
        yield from audio_dir.glob(f"*.{ext}")


def main():
    ap = argparse.ArgumentParser(description="Run Silero VAD and save speech segments as JSON per file.")
    ap.add_argument("--audio_dir", required=True, help="Directory containing WAV files")
    ap.add_argument("--label_dir", required=True, help="Directory to write JSON label files")
    ap.add_argument("--sr", type=int, default=16000, help="Sampling rate to read audio at (default: 16000)")
    ap.add_argument("--ext", default="wav", help="Audio extension without dot (default: wav)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--limit", type=int, default=0, help="Process only N files (0 = all)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON label files")
    ap.add_argument("--progress_every", type=int, default=100, help="Print progress every N files (default: 100)")
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    label_dir = Path(args.label_dir)
    sr = args.sr

    if not audio_dir.exists():
        raise SystemExit(f"audio_dir does not exist: {audio_dir}")

    label_dir.mkdir(parents=True, exist_ok=True)

    # Load Silero VAD
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True
    )

    (get_speech_timestamps, _, read_audio, _, _) = utils

    model.eval()

    files = sorted(iter_audio_files(audio_dir, args.ext, args.recursive))
    if args.limit and args.limit > 0:
        files = files[:args.limit]

    if not files:
        raise SystemExit(f"No *.{args.ext} files found in {audio_dir} (recursive={args.recursive})")
    print(f"Found {len(files)} *.{args.ext} files in {audio_dir}")

    for i, wav_path in enumerate(files, 1):
        wav_path = Path(wav_path)
        out_path = label_dir/(wav_path.stem + ".json")

        if out_path.exists() and not args.overwrite:
            # if already labeled, continue
            continue

        audio = read_audio(str(wav_path), sampling_rate=sr)

        speech = get_speech_timestamps(
            audio,
            model,
            sampling_rate=sr
        )

        out = {
            "sampling_rate": sr,
            "speech_segments": speech
        }

        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

        if i % args.progress_every == 0 or i == len(files):
            speech_ratio = (
                sum(seg["end"] - seg["start"] for seg in speech) / len(audio)
                if speech else 0.0
            )
            print(f"[{i}/{len(files)}] {wav_path.name} | speech_ratio={speech_ratio:.2f}")


if __name__ == "__main__":
    main()