import argparse
from pathlib import Path
import json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True, help="Directory containing WAV files")
    ap.add_argument("--label_dir", required=True, help="Directory containing NPY labels")
    ap.add_argument("--out_jsonl", required=True, help="Output manifest JSONL path")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--append", action="store_true", help="Append to existing JSONL file")
    args = ap.parse_args()

    wav_dir = Path(args.wav_dir)
    label_dir = Path(args.label_dir)
    out_jsonl = Path(args.out_jsonl)

    if args.recursive:
        wavs = sorted(wav_dir.rglob("*.wav"))
    else:
        wavs = sorted(wav_dir.glob("*.wav"))

    print("Found wavs:", len(wavs))

    rows = []
    missing = 0

    for w in wavs:
        npy = label_dir / (w.stem + ".npy")
        if not npy.exists():
            missing += 1
            continue

        # paths relative to dataset root (parent of wav_dir)
        base = wav_dir.parent
        wav_rel = w.relative_to(base)
        npy_rel = npy.relative_to(base)

        rows.append({
            "audio_path": str(wav_rel),
            "voice_activity_path": str(npy_rel)
        })

    print("Pairs:", len(rows))
    print("Missing labels:", missing)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with open(out_jsonl, mode) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print("Wrote:", out_jsonl)

if __name__ == "__main__":
    main()