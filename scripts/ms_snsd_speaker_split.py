import os
import shutil
from pathlib import Path
from collections import defaultdict

CLEAN_SRC = Path("external/MS-SNSD/clean_train")

TRAIN_OUT = Path("datasets/MS-SNSD/train/clean")
VAL_OUT   = Path("datasets/MS-SNSD/val/clean")

N_TOTAL_SPK = 28
N_TRAIN_SPK = 26
MODE = "copy"  # "copy" or "symlink"


def get_speaker_id(wav_path: Path) -> str:
    # Extract speaker ID from VoiceBank-style filename; ex: p226_001.wav -> p226
    return wav_path.stem.split("_")[0]


def link_or_copy(src, dst):
    if dst.exists():
        return

    if MODE == "symlink":
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:
            pass  # fall back to copy

    shutil.copy2(src, dst)


def main():
    assert CLEAN_SRC.exists(), f"{CLEAN_SRC} not found"

    wavs = list(CLEAN_SRC.glob("*.wav"))
    assert wavs, "No wav files found"

    speaker_to_wavs = defaultdict(list)
    for wav in wavs:
        spk = get_speaker_id(wav)
        speaker_to_wavs[spk].append(wav)

    speakers = sorted(speaker_to_wavs.keys())

    if len(speakers) < N_TOTAL_SPK:
        raise RuntimeError(
            f"Expected at least {N_TOTAL_SPK} speakers, found {len(speakers)}"
        )

    selected_speakers = speakers[:N_TOTAL_SPK]
    train_speakers = set(selected_speakers[:N_TRAIN_SPK])
    val_speakers = set(selected_speakers[N_TRAIN_SPK:])

    print(f"Total speakers selected: {len(selected_speakers)}")
    print(f"Train speakers: {len(train_speakers)}")
    print(f"Val speakers: {len(val_speakers)}")

    train_count = 0
    val_count = 0

    for spk, files in speaker_to_wavs.items():
        if spk in train_speakers:
            for wav in files:
                link_or_copy(wav, TRAIN_OUT / wav.name)
                train_count += 1
        elif spk in val_speakers:
            for wav in files:
                link_or_copy(wav, VAL_OUT / wav.name)
                val_count += 1

    print(f"Train clean files: {train_count}")
    print(f"Val clean files: {val_count}")


if __name__ == "__main__":
    main()