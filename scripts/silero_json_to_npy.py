import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf


def iter_audio_files(audio_dir: Path, ext: str, recursive: bool):
    if recursive:
        yield from audio_dir.rglob(f"*.{ext}")
    else:
        yield from audio_dir.glob(f"*.{ext}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True, type=Path)
    ap.add_argument("--json_dir", required=True, type=Path, help="Dir containing silero JSON labels")
    ap.add_argument("--out_dir", required=True, type=Path, help="Dir to write .npy labels")
    ap.add_argument("--ext", default="wav")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--require_sr", type=int, default=16000, help="Fail if wav sr != this")
    args = ap.parse_args()

    audio_dir: Path = args.audio_dir
    json_dir: Path = args.json_dir
    out_dir: Path = args.out_dir

    if not audio_dir.exists():
        raise SystemExit(f"audio_dir not found: {audio_dir}")
    if not json_dir.exists():
        raise SystemExit(f"json_dir not found: {json_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(iter_audio_files(audio_dir, args.ext, args.recursive))
    if not wavs:
        raise SystemExit(f"No .{args.ext} files found in {audio_dir}")

    for wav_path in wavs:
        rel = wav_path.relative_to(audio_dir)
        json_path = json_dir / rel.with_suffix(".json")
        npy_path = out_dir / rel.with_suffix(".npy")

        if not json_path.exists():
            raise SystemExit(f"Missing JSON for {rel}: expected {json_path}")

        if npy_path.exists() and not args.overwrite:
            continue

        # Read wav length and SR
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if sr != args.require_sr:
            raise SystemExit(f"{rel}: wav sr={sr}, expected {args.require_sr}")

        n = len(audio)

        # Load silero segments (sample indices)
        obj = json.loads(json_path.read_text())
        segs = obj.get("speech_segments", [])
        json_sr = obj.get("sampling_rate", None)
        if json_sr is not None and json_sr != sr:
            raise SystemExit(f"{rel}: json sampling_rate={json_sr} != wav sr={sr}")

        raw = np.zeros(n, dtype=np.int64)

        for seg in segs:
            s = int(seg["start"])
            e = int(seg["end"])

            s = max(0, min(s, n))
            e = max(0, min(e, n))
            if e > s:
                raw[s:e] = 1

        npy_path.parent.mkdir(parents=True, exist_ok=True)
        win = int(sr * 0.025)   # 25 ms = 400 samples
        hop = int(sr * 0.010)   # 10 ms = 160 samples

        frames = []

        for i in range(0, len(raw) - win, hop):
            chunk = raw[i:i+win]
            frames.append(1 if chunk.mean() > 0.5 else 0)

        frame_labels = np.array(frames, dtype=np.int64)

        np.save(npy_path, frame_labels)

    print(f"Done. Wrote .npy labels under: {out_dir}")


if __name__ == "__main__":
    main()