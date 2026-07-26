import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import re
from scipy.signal import fftconvolve as convolve

def open_textgrid_safe(path):
    for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin1"):
        try:
            with open(path, "r", encoding=enc) as f:
                f.read(1024)
            return open(path, "r", encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode TextGrid: {path}")

def parse_textgrid_intervals(textgrid_path, target_tier_index=2):
    intervals = []

    current_tier_index = 0
    xmin = None
    xmax = None
    in_target_tier = False

    with open_textgrid_safe(textgrid_path) as f:
        for line in f:
            line = line.strip()

            # Count only real numbered tiers: item [1]:, item [2]:, ...
            if re.match(r"item \[\d+\]:", line):
                current_tier_index += 1
                in_target_tier = (current_tier_index == target_tier_index)
                xmin = None
                xmax = None
                continue

            if not in_target_tier:
                continue

            if line.startswith("xmin ="):
                xmin = float(line.split("=", 1)[1].strip())

            elif line.startswith("xmax ="):
                xmax = float(line.split("=", 1)[1].strip())

            elif line.startswith("text ="):
                m = re.search(r'"(.*)"', line)
                text = m.group(1) if m else ""

                is_speech = (text.strip() != "")

                if xmin is not None and xmax is not None:
                    intervals.append((xmin, xmax, is_speech, text))

                xmin = None
                xmax = None

    return intervals


def iter_audio_files(audio_dir: Path, ext: str, recursive: bool):
    if recursive:
        yield from audio_dir.rglob(f"*.{ext}")
    else:
        yield from audio_dir.glob(f"*.{ext}")


def build_sample_labels(intervals, sr, n_samples):
    raw = np.zeros(n_samples, dtype=np.int64)

    for start_sec, end_sec, is_speech, _text in intervals:
        if not is_speech:
            continue

        s = int(start_sec * sr)
        e = int(end_sec * sr)

        s = max(0, min(s, n_samples))
        e = max(0, min(e, n_samples))

        if e > s:
            raw[s:e] = 1

    return raw

"""
def build_frame_labels(raw, sr, win_sec=0.025, hop_sec=0.010, threshold=0.5):
    win = int(sr * win_sec)
    hop = int(sr * hop_sec)

    if len(raw) < win:
        return np.zeros(0, dtype=np.int64), win, hop

    frames = []
    for i in range(0, len(raw) - win + 1, hop):
        chunk = raw[i:i + win]
        frames.append(1 if chunk.mean() > threshold else 0)


    frame_labels = np.array(frames, dtype=np.int64)
    return frame_labels, win, hop
"""

def build_frame_labels(raw, sr, win_sec=0.025, hop_sec=0.010, threshold=0.5):
    win = int(sr * win_sec)
    hop = int(sr * hop_sec)

    raw = np.asarray(raw).reshape(-1).astype(np.int64)

    if len(raw) < win:
        return np.zeros(0, dtype=np.int64), win, hop

    frames = []
    for i in range(0, len(raw) - win, hop):
        chunk = raw[i:i + win]
        frames.append(1 if chunk.mean() > threshold else 0)

    frame_labels = np.array(frames, dtype=np.int64)
    return frame_labels, win, hop


def sanity_check_file(rel, intervals, raw, frame_labels, n_samples, sr, win, hop, verbose=False):
    if len(intervals) == 0:
        print(f"[WARN] {rel}: no intervals parsed from target tier")

    unique_vals = np.unique(frame_labels)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(f"{rel}: frame labels are not binary: {unique_vals}")

    if n_samples < win:
      expected_n_frames = 0
    else:
        expected_n_frames = len(range(0, n_samples - win, hop))
    if len(frame_labels) != expected_n_frames:
        raise ValueError(
            f"{rel}: frame count mismatch, got {len(frame_labels)}, expected {expected_n_frames}"
        )

    speech_sample_ratio = raw.mean() if len(raw) > 0 else 0.0
    speech_frame_ratio = frame_labels.mean() if len(frame_labels) > 0 else 0.0

    if verbose:
        print(
            f"[OK] {rel} | intervals={len(intervals)} | "
            f"samples={n_samples} | frames={len(frame_labels)} | "
            f"sample_speech_ratio={speech_sample_ratio:.4f} | "
            f"frame_speech_ratio={speech_frame_ratio:.4f}"
        )

    return {
        "interval_count": len(intervals),
        "sample_speech_ratio": speech_sample_ratio,
        "frame_speech_ratio": speech_frame_ratio,
        "n_frames": len(frame_labels),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True, type=Path)
    ap.add_argument("--textgrid_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--ext", default="wav")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--require_sr", type=int, default=16000)
    ap.add_argument("--target_tier_index", type=int, default=2)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    audio_dir = args.audio_dir
    textgrid_dir = args.textgrid_dir
    out_dir = args.out_dir

    if not audio_dir.exists():
        raise SystemExit(f"audio_dir not found: {audio_dir}")
    if not textgrid_dir.exists():
        raise SystemExit(f"textgrid_dir not found: {textgrid_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(iter_audio_files(audio_dir, args.ext, args.recursive))
    if not wavs:
        raise SystemExit(f"No .{args.ext} files found in {audio_dir}")

    total_files = 0
    written_files = 0
    skipped_missing_tg = 0
    aggregate_frame_ratios = []

    for wav_path in wavs:
        total_files += 1

        rel = wav_path.relative_to(audio_dir)
        tg_candidates = [
            textgrid_dir/rel.with_suffix(".TextGrid"),
            textgrid_dir/rel.with_suffix(".textgrid"),
        ]
        tg_path = next(
            (candidate for candidate in tg_candidates if candidate.exists()),
            None,
        )
        npy_path = out_dir/rel.with_suffix(".npy")

        if tg_path is None:
            print(f"[WARN] Missing TextGrid, skipping: {rel}")
            skipped_missing_tg += 1
            continue

        if npy_path.exists() and not args.overwrite:
            if args.verbose:
                print(f"[SKIP] Exists already: {rel}")
            continue

        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if sr != args.require_sr:
            raise SystemExit(f"{rel}: wav sr={sr}, expected {args.require_sr}")

        n_samples = len(audio)

        intervals = parse_textgrid_intervals(
            tg_path,
            target_tier_index=args.target_tier_index
        )

        raw = build_sample_labels(intervals, sr, n_samples)
        frame_labels, win, hop = build_frame_labels(raw, sr)

        stats = sanity_check_file(
            rel=rel,
            intervals=intervals,
            raw=raw,
            frame_labels=frame_labels,
            n_samples=n_samples,
            sr=sr,
            win=win,
            hop=hop,
            verbose=args.verbose
        )

        aggregate_frame_ratios.append(stats["frame_speech_ratio"])

        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, frame_labels)
        written_files += 1

    print("\nDone.")
    print(f"Total wav files found: {total_files}")
    print(f"Files written: {written_files}")
    print(f"Files skipped due to missing TextGrid: {skipped_missing_tg}")

    if aggregate_frame_ratios:
        arr = np.array(aggregate_frame_ratios, dtype=np.float32)
        print(
            f"Frame speech ratio summary | min={arr.min():.4f} | "
            f"mean={arr.mean():.4f} | max={arr.max():.4f}"
        )

    print(f"Wrote .npy labels under: {out_dir}")


if __name__ == "__main__":
    main()