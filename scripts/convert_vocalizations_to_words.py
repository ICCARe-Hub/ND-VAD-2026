from pathlib import Path
import argparse
import re

KEEP = {
    "$non-word",
    "$vocal-gesture",
    "$audible-breath",
}

def convert_file(src, dst):
    text = src.read_text()

    # quick sanity checks
    if text.count('class = "IntervalTier"') != 1:
        raise ValueError("expected exactly one IntervalTier")

    if text.count('name = "vocalizations"') != 1:
        raise ValueError('expected one tier named "vocalizations"')

    counts = {
        "word": 0,
        "empty": 0,
        "non-word": 0,
        "vocal-gesture": 0,
        "audible-breath": 0,
    }

    def replace(match):
        label = match.group(1).strip()

        if label == "":
            counts["empty"] += 1
            return match.group(0)

        if label in KEEP:
            counts[label[1:]] += 1
            return match.group(0)

        counts["word"] += 1
        return match.group(0).replace(match.group(1), "$word")

    converted = re.sub(
        r'(?m)^\s*text = "(.*?)"\s*$',
        replace,
        text,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(converted)

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.TextGrid"))

    if not files:
        raise SystemExit("no TextGrid files found")

    totals = {
        "word": 0,
        "empty": 0,
        "non-word": 0,
        "vocal-gesture": 0,
        "audible-breath": 0,
    }

    errors = []

    for src in files:
        try:
            counts = convert_file(
                src,
                args.output_dir / src.name,
            )

            for key in totals:
                totals[key] += counts[key]

        except Exception as e:
            errors.append((src.name, str(e)))

    print(f"files found: {len(files)}")
    print(f"files converted: {len(files) - len(errors)}")
    print(f"errors: {len(errors)}")
    print()
    print(f"$word: {totals['word']}")
    print(f"$non-word: {totals['non-word']}")
    print(f"$vocal-gesture: {totals['vocal-gesture']}")
    print(f"$audible-breath: {totals['audible-breath']}")
    print(f"empty: {totals['empty']}")

    if errors:
        print("\nerrors:")
        for name, error in errors:
            print(f"{name}: {error}")


if __name__ == "__main__":
    main()