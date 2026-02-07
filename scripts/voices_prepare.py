import os
import csv
import random
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Set, Tuple
from collections import defaultdict
import yaml
import soundfile as sf


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str):
    if dst.exists():
        return
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "copy":
        import shutil
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def load_excluded_speakers(manifest_path: Optional[str]) -> Set[str]:
    if not manifest_path:
        return set()
    mp = Path(manifest_path)
    if not mp.exists():
        return set()

    speakers: Set[str] = set()
    with open(mp, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            spk = (row.get("speaker") or "").strip()
            if spk:
                speakers.add(spk)
    return speakers


def parse_voices_tokens(stem: str) -> Dict[str, Optional[object]]:
    parts = stem.split("-")
    out: Dict[str, Optional[object]] = {
        "mic_id": None,
        "mic_type": None,
        "mic_location": None,
        "deg": None,
    }

    mc_i = None
    for i, p in enumerate(parts):
        if p.startswith("mc") and len(p) >= 4 and p[2:].isdigit():
            mc_i = i
            break

    if mc_i is None:
        return out

    # mic_id
    try:
        out["mic_id"] = int(parts[mc_i][2:])
    except Exception:
        pass

    # mic_type + location
    try:
        out["mic_type"] = parts[mc_i + 1]
    except Exception:
        pass

    try:
        out["mic_location"] = parts[mc_i + 2]
    except Exception:
        pass

    # degree
    try:
        dg = parts[mc_i + 3]
        if dg.startswith("dg") and dg[2:].isdigit():
            out["deg"] = int(dg[2:])
    except Exception:
        pass

    return out


def parse_path_metadata(wav: Path, split: str) -> Tuple[str, str, str]:
    try:
        parts = wav.parts
        idx = parts.index(split)
        room = parts[idx + 1]
        cond = parts[idx + 2]
        speaker = parts[idx + 3]
        return room, cond, speaker
    except Exception:
        return "", "", ""


def list_wavs(devkit_root: Path, distant_root: Path, split: str, rooms, conditions, speakers) -> List[Path]:
    base = devkit_root / distant_root / split
    wavs: List[Path] = []

    for room in rooms:
        for cond in conditions:
            room_cond = base / room / cond
            if not room_cond.exists():
                continue

            if speakers:
                spk_dirs = [room_cond / spk for spk in speakers]
            else:
                spk_dirs = [p for p in room_cond.iterdir() if p.is_dir() and p.name.startswith("sp")]

            for spk_dir in spk_dirs:
                if not spk_dir.exists():
                    continue
                wavs.extend(spk_dir.rglob("*.wav"))

    return wavs


def normalize_str_list(x) -> List[str]:
    if not x:
        return []
    return [str(v).strip() for v in x if str(v).strip()]

def write_split(selected, out_root, split, mode):
    noisy_out = out_root / "noisy"
    ensure_dir(noisy_out)

    manifest_path = out_root / "manifest.csv"
    total_sec = 0.0

    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dst_file",
            "src_path",
            "duration_sec",
            "samplerate",
            "split",
            "room",
            "condition",
            "speaker",
            "mic_id",
            "mic_type",
            "mic_location",
            "deg",
        ])

        for src, dur, sr in selected:
            room, cond, speaker = parse_path_metadata(src, split)
            tokens = parse_voices_tokens(src.stem)

            dst = noisy_out / src.name
            link_or_copy(src, dst, mode)

            w.writerow([
                src.name,
                str(src),
                "" if dur is None else f"{dur:.3f}",
                "" if sr is None else sr,
                split,
                room,
                cond,
                speaker,
                tokens["mic_id"],
                tokens["mic_type"],
                tokens["mic_location"],
                tokens["deg"],
            ])

            #print(f"{out_root.name}: {len(selected)} files")

    print(f"{out_root.name}: {len(selected)} files, {total_sec/3600:.3f} hours")


def main(cfg_path: Path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Required
    devkit_root = Path(cfg["devkit_root"])
    distant_root = Path(cfg["distant_root"])
    split = str(cfg["split"]).strip()
    rooms = cfg["rooms"]
    conditions = cfg["conditions"]
    speakers = cfg.get("speakers", []) or []
    
    max_hours = cfg.get("max_hours", None)
    if max_hours is not None:
        max_hours = float(max_hours)
    
    seed = int(cfg["seed"])
    mode = str(cfg["mode"]).strip()
    out_root = Path(cfg["out_root"])
    speaker_count = cfg.get("speaker_count", None)
    if speaker_count is not None:
        speaker_count = int(speaker_count)

    val_out_root = cfg.get("val_out_root", None)
    if val_out_root is not None:
        val_out_root = Path(val_out_root)


    # Optional speaker exclusion
    exclude_manifest = cfg.get("exclude_speakers_from_manifest")
    excluded_speakers = load_excluded_speakers(exclude_manifest)

    # Optional mic filtering
    mic_types = set(normalize_str_list(cfg.get("mic_types")))
    mic_locations = set(normalize_str_list(cfg.get("mic_locations")))

    print(f"Config: split={split} out_root={out_root}")
    print(f"Excluded speakers: {len(excluded_speakers)}")
    print(f"Mic type filter: {sorted(mic_types) if mic_types else 'ALL'}")
    print(f"Mic location filter: {sorted(mic_locations) if mic_locations else 'ALL'}")

    noisy_out = out_root / "noisy"
    ensure_dir(noisy_out)

    # Collect candidates
    candidates = list_wavs(devkit_root, distant_root, split, rooms, conditions, speakers)
    if not candidates:
        raise RuntimeError("No WAV files matched rooms/conditions/speakers. Check config and dataset paths.")

    total_candidates = len(candidates)

    # Filter by speaker exclusion + mic filters
    filtered: List[Path] = []
    for wav in candidates:
        room, cond, speaker = parse_path_metadata(wav, split)
        if speaker and speaker in excluded_speakers:
            continue

        tokens = parse_voices_tokens(wav.stem)

        if mic_types:
            if tokens["mic_type"] is None or tokens["mic_type"] not in mic_types:
                continue

        if mic_locations:
            if tokens["mic_location"] is None or tokens["mic_location"] not in mic_locations:
                continue

        filtered.append(wav)

    candidates = filtered
    print(f"Matched candidates: {total_candidates}")
    print(f"Matched candidates (after speaker+mic filters): {len(candidates)}")

    if not candidates:
        raise RuntimeError("No WAV files remain after speaker and/or mic filtering.")

    random.seed(seed)
    random.shuffle(candidates)

    speaker_to_wavs = defaultdict(list)

    for wav in candidates:
        _, _, speaker = parse_path_metadata(wav, split)
        if speaker:
            speaker_to_wavs[speaker].append(wav)

    all_speakers = sorted(speaker_to_wavs.keys())

    random.seed(seed)
    random.shuffle(all_speakers)

    total_speakers = len(all_speakers)
    print(f"Total speakers after filtering: {total_speakers}")

    if speaker_count is not None:
        if speaker_count >= total_speakers:
            raise ValueError(
                f"speaker_count ({speaker_count}) must be < total speakers ({total_speakers})"
            )

        train_speakers = set(all_speakers[:speaker_count])
        val_speakers = set(all_speakers[speaker_count:])

        print(f"Train speakers: {len(train_speakers)}")
        print(f"Val speakers: {len(val_speakers)}")
    else:
        train_speakers = set(all_speakers)
        val_speakers = set()

    train_selected: List[Tuple[Path, float, int]] = []
    val_selected: List[Tuple[Path, float, int]] = []

    train_sec = 0.0
    val_sec = 0.0

    for speaker, wavs in speaker_to_wavs.items():
        for wav in wavs:
            dur = None
            sr = None

            if speaker in train_speakers:
                train_selected.append((wav, dur, sr))
            elif speaker in val_speakers:
                val_selected.append((wav, dur, sr))

    write_split(train_selected, out_root, split, mode)

    if val_out_root is not None and val_selected:
        write_split(val_selected, val_out_root, split, mode)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to voices_select.yaml")
    args = ap.parse_args()
    main(Path(args.cfg))