import random
import numpy as np
import soundfile as sf
from pathlib import Path

TRAIN_CLEAN = Path("datasets/MS-SNSD/train/clean")
VAL_CLEAN   = Path("datasets/MS-SNSD/val/clean")
NOISE_DIR   = Path("external/MS-SNSD/noise_train")

TRAIN_OUT = Path("datasets/MS-SNSD/train/noisy")
VAL_OUT   = Path("datasets/MS-SNSD/val/noisy")

SNR_LEVELS = [15, 20, 25]
TARGET_RMS_DB = -25

def rms(x):
    return np.sqrt(np.mean(x ** 2))


def normalize_to_db(x, target_db):
    scalar = 10 ** (target_db / 20) / (rms(x) + 1e-12)
    return x * scalar


def snr_mixer(clean, noise, snr_db):
    clean = normalize_to_db(clean, TARGET_RMS_DB)
    noise = normalize_to_db(noise, TARGET_RMS_DB)

    noise_scalar = np.sqrt(
        rms(clean) / (10 ** (snr_db / 20)) / (rms(noise) + 1e-12)
    )
    noise = noise * noise_scalar
    return clean + noise


def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def mix_split(clean_dir, out_dir, noise_files):
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_files = sorted(clean_dir.glob("*.wav"))

    for clean_path in clean_files:
        clean, sr = load_audio(clean_path)

        noise_path = random.choice(noise_files)
        noise, _ = load_audio(noise_path)

        if len(noise) < len(clean):
            repeats = int(np.ceil(len(clean) / len(noise)))
            noise = np.tile(noise, repeats)
        noise = noise[:len(clean)]

        for snr in SNR_LEVELS:
            noisy = snr_mixer(clean, noise, snr)

            out_name = (
                f"{clean_path.stem}"
                f"_SNR{snr}"
                f"_{noise_path.stem}.wav"
            )
            sf.write(out_dir / out_name, noisy, sr)


def main():
    assert NOISE_DIR.exists()
    noise_files = [
        p for p in NOISE_DIR.glob("*.wav")
        if p.name.startswith("Babble") or p.name.startswith("AirConditioner")
    ]
    assert noise_files, "No valid noise files found"

    print(f"Using {len(noise_files)} noise files")

    print("Mixing TRAIN set...")
    mix_split(TRAIN_CLEAN, TRAIN_OUT, noise_files)

    print("Mixing VAL set...")
    mix_split(VAL_CLEAN, VAL_OUT, noise_files)

    print("Done.")


if __name__ == "__main__":
    main()