#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import tarfile
from pathlib import Path


CH_ID_RE = re.compile(r"^CH-(\d+)-")


def load_existing_ids(pdf_dir: Path) -> set[str]:
    existing = set()
    for path in pdf_dir.glob("CH-*.pdf"):
        existing.add(re.sub(r"_full\.pdf$", "", path.name))
    return existing


def load_catalog_ids(csv_paths: list[Path]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for csv_path in csv_paths:
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                norm_id = row["norm_id"].strip()
                if not norm_id.startswith("CH-") or norm_id in seen:
                    continue
                if CH_ID_RE.match(norm_id) is None:
                    continue
                seen.add(norm_id)
                ids.append(norm_id)
    ids.sort(key=patent_sort_key)
    return ids


def patent_sort_key(norm_id: str) -> tuple[int, str]:
    match = CH_ID_RE.match(norm_id)
    if match is None:
        raise ValueError(f"Unsupported CH id: {norm_id}")
    return int(match.group(1)), norm_id


def select_evenly_spaced(ids: list[str], count: int) -> list[str]:
    if count <= 0:
        return []
    if count > len(ids):
        raise ValueError(f"Requested {count} ids, but only {len(ids)} are available")

    selected: list[str] = []
    used_indexes: set[int] = set()
    total = len(ids)
    for i in range(count):
        raw_index = round((i + 0.5) * total / count - 0.5)
        index = min(max(raw_index, 0), total - 1)
        while index in used_indexes and index + 1 < total:
            index += 1
        while index in used_indexes and index - 1 >= 0:
            index -= 1
        if index in used_indexes:
            raise RuntimeError("Unable to build a unique evenly spaced sample")
        used_indexes.add(index)
        selected.append(ids[index])
    return selected


def write_manifest(ids: list[str], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "norm_id", "png_name"])
        for idx, norm_id in enumerate(ids, start=1):
            writer.writerow([idx, norm_id, f"{norm_id}.png"])


def extract_pngs(tar_path: Path, ids: list[str], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = {f"./data/{norm_id}.png": norm_id for norm_id in ids}
    extracted = 0

    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive:
            norm_id = wanted.get(member.name)
            if norm_id is None:
                continue
            extracted_file = archive.extractfile(member)
            if extracted_file is None:
                continue
            out_path = output_dir / f"{norm_id}.png"
            out_path.write_bytes(extracted_file.read())
            extracted += 1
            if extracted == len(ids):
                break

    return extracted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and extract an evenly spaced CH sample from the external tarball."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/Volumes/Crucial_X10/patentcity_dev/CH"),
        help="Directory containing CH_present.csv, CH_extra.csv and png.tar.gz",
    )
    parser.add_argument(
        "--existing-pdf-dir",
        type=Path,
        default=Path("data/raw_pdf"),
        help="Directory used to exclude already selected CH PDFs",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=401,
        help="Number of additional CH files to select",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/gold_standard_CH/PNGs_extracted"),
        help="Where to extract PNG files",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/gold_standard_CH/ch_additional_401_manifest.csv"),
        help="Where to write the selected ids manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir
    catalog_paths = [source_dir / "CH_present.csv", source_dir / "CH_extra.csv"]
    tar_path = source_dir / "png.tar.gz"

    existing_ids = load_existing_ids(args.existing_pdf_dir)
    all_ids = load_catalog_ids(catalog_paths)
    remaining_ids = [norm_id for norm_id in all_ids if norm_id not in existing_ids]
    selected_ids = select_evenly_spaced(remaining_ids, args.count)

    write_manifest(selected_ids, args.manifest_path)
    extracted = extract_pngs(tar_path, selected_ids, args.output_dir)
    if extracted != len(selected_ids):
        raise RuntimeError(
            f"Extracted {extracted} files, expected {len(selected_ids)}. "
            "Some selected ids were not found in the tar archive."
        )

    print(f"existing_ids={len(existing_ids)}")
    print(f"available_ids={len(all_ids)}")
    print(f"selected_ids={len(selected_ids)}")
    print(f"output_dir={args.output_dir}")
    print(f"manifest_path={args.manifest_path}")


if __name__ == "__main__":
    main()
