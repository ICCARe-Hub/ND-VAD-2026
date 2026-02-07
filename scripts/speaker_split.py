from pathlib import Path
import csv

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_TRAIN_DIR = REPO_ROOT / "external" / "MS-SNSD" / "clean_train"
OUT_CSV = REPO_ROOT / "datasets" / "MS-SNSD" / "speaker_split.csv"

def main():
    wavs = CLEAN_TRAIN_DIR.glob("*.wav")

    speakers = set()
    for wav in wavs:
        speaker = wav.stem.split("_")[0]
        speakers.add(speaker)

    speakers = sorted(speakers)

    if len(speakers) < 28:
        raise RuntimeError(f"Expected at least 28 speakers, found {len(speakers)}")

    selected = speakers[:28]
    train_speakers = selected[:26]
    val_speakers = selected[26:]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["speaker", "split"])

        for spk in train_speakers:
            writer.writerow([spk, "train"])

        for spk in val_speakers:
            writer.writerow([spk, "val"])

    print("MS-SNSD speaker split created:")
    print(f"  Train speakers: {len(train_speakers)}")
    print(f"  Val speakers: {len(val_speakers)}")
    print(f"Wrote: {OUT_CSV}")

if __name__ == "__main__":
    main()