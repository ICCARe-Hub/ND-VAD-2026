import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def iter_audio_files(audio_dir: Path, ext: str, recursive: bool):
    if recursive:
        yield from audio_dir.rglob(f"*.{ext}")
    else:
        yield from audio_dir.glob(f"*.{ext}")


def load_audio_mono_16k(wav_path: Path, require_sr: int) -> torch.Tensor:
    """
    Load audio as mono float32 tensor of shape [num_samples].
    """
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)

    if sr != require_sr:
        raise ValueError(f"{wav_path}: sr={sr}, expected {require_sr}")

    # If stereo/multi-channel, average to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    return torch.from_numpy(audio)


def align_spec_to_label(
    spec: torch.Tensor,
    label_len: int,
    wav_path: Path,
    max_diff: int = 1,
) -> torch.Tensor:
    """
    Align spectrogram time frames to label length.

    spec shape: [1, freq, time]
    label_len: number of frame labels

    Strategy:
    - if exact match: keep as-is
    - if spec is longer by <= max_diff: trim right side
    - if spec is shorter by <= max_diff: pad by repeating last frame
    - otherwise: raise error
    """
    spec_len = spec.shape[-1]
    diff = spec_len - label_len

    if diff == 0:
        return spec

    if abs(diff) > max_diff:
        raise ValueError(
            f"{wav_path.name}: spectrogram frames ({spec_len}) vs label frames ({label_len}) "
            f"mismatch by {abs(diff)} > allowed max_diff={max_diff}"
        )

    if diff > 0:
        # spec too long -> trim
        return spec[..., :label_len]

    # spec too short -> pad by repeating last frame
    pad_frames = label_len - spec_len
    last_frame = spec[..., -1:].repeat(1, 1, pad_frames)
    return torch.cat([spec, last_frame], dim=-1)


def main():
    ap = argparse.ArgumentParser(
        description="Generate NAS_VAD-compatible spectrogram .npy files from wavs."
    )
    ap.add_argument("--audio_dir", required=True, type=Path, help="Directory containing source WAV files")
    ap.add_argument("--label_dir", required=True, type=Path, help="Directory containing frame-level .npy labels")
    ap.add_argument("--out_dir", required=True, type=Path, help="Directory to write *_spec.npy files")
    ap.add_argument("--ext", default="wav", help="Audio extension without dot (default: wav)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing spectrogram files")
    ap.add_argument("--require_sr", type=int, default=16000, help="Expected sampling rate (default: 16000)")
    ap.add_argument("--n_fft", type=int, default=400, help="FFT size (default: 400)")
    ap.add_argument("--hop_length", type=int, default=160, help="Hop length (default: 160)")
    ap.add_argument(
        "--max_frame_diff",
        type=int,
        default=1,
        help="Allowed frame mismatch between spectrogram and label before failing (default: 1)",
    )
    ap.add_argument(
        "--progress_every",
        type=int,
        default=250,
        help="Print progress every N files (default: 250)",
    )
    args = ap.parse_args()

    audio_dir: Path = args.audio_dir
    label_dir: Path = args.label_dir
    out_dir: Path = args.out_dir

    if not audio_dir.exists():
        raise SystemExit(f"audio_dir not found: {audio_dir}")
    if not label_dir.exists():
        raise SystemExit(f"label_dir not found: {label_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Match repo-style preprocessing as closely as possible:
    # - n_fft=400
    # - hop_length=160
    # - default Spectrogram behavior (power=2, center=True)
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )

    spec_transform = torchaudio.transforms.Spectrogram(
    n_fft=args.n_fft,
    win_length=args.n_fft,
    hop_length=args.hop_length,
    center=False,
)

    wavs = sorted(iter_audio_files(audio_dir, args.ext, args.recursive))
    if not wavs:
        raise SystemExit(f"No .{args.ext} files found in {audio_dir}")

    total = len(wavs)
    written = 0
    skipped = 0

    for idx, wav_path in enumerate(wavs, 1):
        wav_path = Path(wav_path)
        rel = wav_path.relative_to(audio_dir)

        label_path = label_dir / rel.with_suffix(".npy")
        out_path = out_dir / rel.with_suffix("")
        out_path = out_path.parent / f"{out_path.name}_spec.npy"

        if not label_path.exists():
            raise SystemExit(f"Missing label for {rel}: expected {label_path}")

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        # Load wav
        audio = load_audio_mono_16k(wav_path, args.require_sr)

        # Load frame labels
        label = np.load(label_path)
        if label.ndim != 1:
            raise ValueError(f"{label_path}: expected 1D frame labels, got shape {label.shape}")
        label_len = int(label.shape[0])

        # Compute spectrogram
        # Input audio: [num_samples]
        # Output spec: [freq, time]
        spec = spec_transform(audio)

        # Save as [1, freq, time] to match repo README expectation
        spec = spec.unsqueeze(0)

        # Align to label frame count
        spec = align_spec_to_label(
            spec=spec,
            label_len=label_len,
            wav_path=wav_path,
            max_diff=args.max_frame_diff,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, spec.cpu().numpy().astype(np.float32))
        written += 1

        if idx % args.progress_every == 0 or idx == total:
            print(
                f"[{idx}/{total}] wrote {out_path.name} | "
                f"spec_shape={tuple(spec.shape)} | label_len={label_len}"
            )

    print()
    print("Done.")
    print(f"Audio dir : {audio_dir}")
    print(f"Label dir : {label_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Written   : {written}")
    print(f"Skipped   : {skipped}")


if __name__ == "__main__":
    main()