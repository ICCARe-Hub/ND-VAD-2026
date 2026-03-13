#!/usr/bin/env python3
import argparse
import json
import math
import random
import re
from pathlib import Path
from collections import defaultdict

# Patterns / parsing
VOICEBANK_ID_RE = re.compile(r"(?:^|/)(p\d+_\d+)(?:\b|_)")
MS_SNSD_SNR_RE = re.compile(r"_SNR(-?\d+)_")
MS_SNSD_NOISE_RE = re.compile(r"_SNR-?\d+_([A-Za-z]+)_")
VOICES_TAG_RE = re.compile(r"(?:^|/).*VOiCES.*\b(babb|tele)\b", re.IGNORECASE)

def detect_dataset(audio_path: str) -> str:
    if MS_SNSD_SNR_RE.search(audio_path):
        return "MS-SNSD"
    if "VOiCES" in audio_path:
        return "VOiCES"
    if VOICEBANK_ID_RE.search(audio_path):
        return "VoiceBank+DEMAND"
    return "Other"

def extract_voicebank_id(audio_path: str):
    m = VOICEBANK_ID_RE.search(audio_path)
    return m.group(1) if m else None

def load_voicebank(map_path: Path):
    """
    Expected per line (whitespace-separated):
      p234_001 AirConditioner 15
      p236_002 babble 10
    Returns { "p234_001": ("AirConditioner", 15), ... }
    """
    mapping = {}
    with map_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            key = parts[0]
            noise = parts[1]
            try:
                snr = int(parts[2])
            except ValueError:
                continue
            mapping[key] = (noise, snr)
    return mapping

def parse_ms_snsd_noise_snr(audio_path: str):
    snr = None
    m = MS_SNSD_SNR_RE.search(audio_path)
    if m:
        snr = int(m.group(1))
    noise = "Unknown"
    m2 = MS_SNSD_NOISE_RE.search(audio_path)
    if m2:
        noise = m2.group(1)
    return noise, snr

def parse_voices_tag(audio_path: str):
    m = VOICES_TAG_RE.search(audio_path)
    if m:
        return m.group(1).lower()
    low = audio_path.lower()
    if "babb" in low:
        return "babb"
    if "tele" in low:
        return "tele"
    return "unknown"

def combo_key(audio_path: str, vb_map: dict):
    """
    Returns (dataset, noise_type, snr_str) used for grouping.
    """
    ds = detect_dataset(audio_path)

    if ds == "VoiceBank+DEMAND":
        vid = extract_voicebank_id(audio_path)
        if vid and vid in vb_map:
            noise, snr = vb_map[vid]
            return (ds, str(noise), str(snr))
        return (ds, "Unknown", "Unknown")

    if ds == "MS-SNSD":
        noise, snr = parse_ms_snsd_noise_snr(audio_path)
        return (ds, str(noise), str(snr) if snr is not None else "Unknown")

    if ds == "VOiCES":
        tag = parse_voices_tag(audio_path)
        return (ds, tag, "NA")

    return (ds, "NA", "NA")

# JSONL IO
def read_jsonl(path: Path):
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def parse_fractions(s: str):
    """
    Accepts:
      - Percent style: "1,2,5,10"  -> [0.01, 0.02, 0.05, 0.10]
      - Explicit percent signs: "1%,2%" -> [0.01, 0.02]
      - True fractions: "0.01,0.1" -> [0.01, 0.1]
    Rule:
      - If the token has a '%' OR the numeric value is >= 1, treat it as a percent and divide by 100.
      - Otherwise treat it as an actual fraction.
    """
    out = []
    for part in s.split(","):
        raw = part.strip()
        if not raw:
            continue

        is_percent = raw.endswith("%")
        raw_num = raw.replace("%", "").strip()

        v = float(raw_num)

        # Key fix: >= 1 means percent (so "1" => 0.01)
        if is_percent or v >= 1.0:
            v /= 100.0

        # sanity clamp
        if v <= 0:
            continue
        out.append(v)

    return out

def parse_dataset_weights(s: str):
    """
    Optional override:
      "VoiceBank+DEMAND=0.4,MS-SNSD=0.4,VOiCES=0.2"
    If empty => use original proportions from input JSONL.
    """
    if not s:
        return {}
    w = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=")
        w[k.strip()] = float(v.strip())
    total = sum(w.values())
    if total <= 0:
        raise ValueError("Dataset weights must sum to > 0")
    return {k: v / total for k, v in w.items()}

def round_robin_sample(combos_to_rows: dict, quota: int, rng: random.Random):
    """
    combos_to_rows: { combo: [rows...] }
    Picks rows by cycling through combos until quota met.
    Randomizes within each combo.
    """
    combos = list(combos_to_rows.keys())
    if not combos or quota <= 0:
        return []

    # shuffle order of combos for fairness
    rng.shuffle(combos)

    # shuffle within each combo
    pools = {}
    for c in combos:
        pool = combos_to_rows[c][:]
        rng.shuffle(pool)
        pools[c] = pool

    # pointers per combo
    idx = {c: 0 for c in combos}

    chosen = []
    used = set()

    ci = 0
    while len(chosen) < quota:
        c = combos[ci % len(combos)]
        ci += 1

        pool = pools[c]
        if not pool:
            continue

        # If we ran out in this combo, reshuffle and restart (still random)
        if idx[c] >= len(pool):
            rng.shuffle(pool)
            idx[c] = 0

        r = pool[idx[c]]
        idx[c] += 1

        # avoid duplicates (in case of weird repeats)
        key = (r.get("audio_path"), r.get("voice_activity_path"))
        if key in used:
            continue

        chosen.append(r)
        used.add(key)

        # hard stop if quota exceeds total unique available
        if len(used) >= sum(len(v) for v in combos_to_rows.values()):
            break

    return chosen

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--voicebank", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", default="train")
    ap.add_argument("--fractions", default="1,2,5,10,20,50")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dataset_weights", default="",
                    help="Optional override weights across datasets")
    ap.add_argument("--drop_other", action="store_true",
                    help="Drop rows whose dataset is 'Other'")
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    vb_map = load_voicebank(Path(args.voicebank))
    out_dir = Path(args.out_dir)

    rows = read_jsonl(in_path)

    if args.drop_other:
        rows = [r for r in rows if detect_dataset(r["audio_path"]) != "Other"]

    # Organize rows: dataset -> combo -> rows
    ds_combo_rows = defaultdict(lambda: defaultdict(list))
    ds_counts = defaultdict(int)

    for r in rows:
        apath = r["audio_path"]
        ds = detect_dataset(apath)
        ck = combo_key(apath, vb_map)  # (dataset, noise, snr)
        ds_combo_rows[ds][ck].append(r)
        ds_counts[ds] += 1

    # Determine dataset weights
    user_w = parse_dataset_weights(args.dataset_weights)
    if user_w:
        ds_w = {ds: user_w[ds] for ds in user_w if ds in ds_counts}
        # renormalize over present datasets
        s = sum(ds_w.values())
        ds_w = {k: v / s for k, v in ds_w.items()}
    else:
        total = sum(ds_counts.values())
        ds_w = {ds: ds_counts[ds] / total for ds in ds_counts}

    fracs = parse_fractions(args.fractions)
    total_all = sum(ds_counts.values())

    base_rng = random.Random(args.seed)

    for frac in fracs:
        pct = int(round(frac * 100))
        target_n = int(round(frac * total_all))

        rng = random.Random(args.seed + pct)

        # allocate per dataset
        alloc = {ds: int(round(ds_w.get(ds, 0.0) * target_n)) for ds in ds_w}

        # fix rounding drift to hit target_n exactly
        cur = sum(alloc.values())
        delta = target_n - cur
        if delta != 0:
            order = sorted(alloc.keys(), key=lambda d: ds_counts[d], reverse=True)
            i = 0
            while delta != 0 and i < 100000:
                ds = order[i % len(order)]
                if delta > 0:
                    alloc[ds] += 1
                    delta -= 1
                else:
                    if alloc[ds] > 0:
                        alloc[ds] -= 1
                        delta += 1
                i += 1

        subset = []
        for ds, quota in alloc.items():
            subset.extend(round_robin_sample(ds_combo_rows[ds], quota, rng))

        rng.shuffle(subset)

        out_path = out_dir / f"{args.prefix}_{pct}pct.jsonl"
        write_jsonl(out_path, subset)
        print(f"Wrote {out_path} n={len(subset)} alloc={alloc}")

if __name__ == "__main__":
    main()