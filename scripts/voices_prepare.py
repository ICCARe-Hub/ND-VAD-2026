import os
import csv
import random
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Set, Tuple

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

    speaker_percentage = cfg.get("speaker_percentage", None)
    if speaker_percentage is not None:
        speaker_percentage = float(speaker_percentage)
        if not (0.0 < speaker_percentage <= 1.0):
            raise ValueError("speaker_percentage must be in (0, 1].")
    
    seed = int(cfg["seed"])
    mode = str(cfg["mode"]).strip()
    out_root = Path(cfg["out_root"])


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

    speaker_limit = None
    if speaker_percentage is not None:
        speakers_in_candidates = []
        for wav in candidates:
            _, _, speaker = parse_path_metadata(wav, split)
            if speaker:
                speakers_in_candidates.append(speaker)

        unique_speakers = sorted(set(speakers_in_candidates))
        total_speakers = len(unique_speakers)

        speaker_limit = int(total_speakers * speaker_percentage)
        speaker_limit = max(1, speaker_limit)

        print(f"Total speakers after filtering: {total_speakers}")
        print(f"Speaker percentage: {speaker_percentage}")
        print(f"Speaker cutoff: {speaker_limit}")

    # Select until max_hours
    selected: List[Tuple[Path, float, int]] = []
    total_sec = 0.0
    seen_speakers: Set[str] = set()

    for wav in candidates:
        room, cond, speaker = parse_path_metadata(wav, split)

        # Speaker percentage stopping condition
        if speaker_limit is not None:
            if speaker not in seen_speakers and len(seen_speakers) >= speaker_limit:
                break

        info = sf.info(str(wav))
        dur = info.frames / info.samplerate

        # Max-hours stopping condition (optional)
        if max_hours is not None:
            if total_sec + dur > max_hours * 3600:
                continue

        selected.append((wav, dur, info.samplerate))
        total_sec += dur
        if speaker:
            seen_speakers.add(speaker)

        if max_hours is not None:
            if total_sec >= max_hours * 3600:
                break


    # Write manifest and create links/copies
    manifest_path = out_root / "manifest.csv"
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

            dst_name = src.name
            dst = noisy_out / dst_name
            link_or_copy(src, dst, mode)

            w.writerow([
                dst_name,
                str(src),
                f"{dur:.3f}",
                sr,
                split,
                room,
                cond,
                speaker,
                tokens["mic_id"],
                tokens["mic_type"],
                tokens["mic_location"],
                tokens["deg"],
            ])

    print(f"Selected files: {len(selected)}")
    print(f"Total duration: {total_sec/3600:.3f} hours")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Output noisy directory: {noisy_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to voices_select.yaml")
    args = ap.parse_args()
    main(Path(args.cfg))