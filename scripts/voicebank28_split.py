import os
import shutil
from pathlib import Path
from collections import defaultdict

SRC_DIR = Path("external/noisy_trainset_28spk_wav")

TRAIN_OUT = Path("datasets/Voicebank28/train/noisy")
VAL_OUT   = Path("datasets/Voicebank28/val/noisy")

N_TRAIN_SPK = 26
N_VAL_SPK = 2

MODE = "copy"   # copy or symlink


def get_speaker_id(wav_path: Path) -> str:
    # ex: p226_001.wav -> p226
    return wav_path.stem.split("_")[0]


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if MODE == "symlink":
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def main():
    wavs = list(SRC_DIR.glob("*.wav"))
    assert wavs, "No wav files found"

    speaker_to_files = defaultdict(list)
    for wav in wavs:
        spk = get_speaker_id(wav)
        speaker_to_files[spk].append(wav)

    speakers = sorted(speaker_to_files.keys())
    assert len(speakers) == 28, f"Expected 28 speakers, found {len(speakers)}"

    train_speakers = set(speakers[:N_TRAIN_SPK])
    val_speakers = set(speakers[N_TRAIN_SPK:N_TRAIN_SPK + N_VAL_SPK])

    print(f"Train speakers ({len(train_speakers)}): {sorted(train_speakers)}")
    print(f"Val speakers ({len(val_speakers)}): {sorted(val_speakers)}")

    train_count = 0
    val_count = 0

    for spk, files in speaker_to_files.items():
        if spk in train_speakers:
            for wav in files:
                link_or_copy(wav, TRAIN_OUT / wav.name)
                train_count += 1
        elif spk in val_speakers:
            for wav in files:
                link_or_copy(wav, VAL_OUT / wav.name)
                val_count += 1

    print(f"Train noisy files: {train_count}")
    print(f"Val noisy files: {val_count}")


if __name__ == "__main__":
    main()