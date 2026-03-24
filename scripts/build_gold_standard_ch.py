#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path


PDF_ID_RE = re.compile(r"^(CH-\d+-[A-Z0-9]+)_full\.pdf$")
TXT_ID_RE = re.compile(r"^(CH-\d+-[A-Z0-9]+)_full\.txt$")


def load_existing_pdf_ids(pdf_dir: Path) -> list[str]:
    ids = []
    for path in sorted(pdf_dir.glob("CH-*.pdf")):
        match = PDF_ID_RE.match(path.name)
        if match:
            ids.append(match.group(1))
    return ids


def load_manifest_ids(manifest_path: Path) -> list[str]:
    ids = []
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ids.append(row["norm_id"].strip())
    return ids


def write_manifest(ids: list[str], manifest_path: Path, source_label: str) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "norm_id", "png_name", "source"])
        for idx, norm_id in enumerate(ids, start=1):
            writer.writerow([idx, norm_id, f"{norm_id}.png", source_label])


def write_combined_manifest(existing_ids: list[str], additional_ids: list[str], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "norm_id", "png_name", "source"])
        rank = 1
        for source, ids in (("existing_seed_99", existing_ids), ("additional_401", additional_ids)):
            for norm_id in ids:
                writer.writerow([rank, norm_id, f"{norm_id}.png", source])
                rank += 1


def extract_missing_pngs(tar_path: Path, ids: list[str], output_dir: Path) -> tuple[int, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = {norm_id for norm_id in ids if not (output_dir / f"{norm_id}.png").exists()}
    if not missing:
        return 0, []

    wanted = {f"./data/{norm_id}.png": norm_id for norm_id in missing}
    extracted = 0
    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive:
            norm_id = wanted.get(member.name)
            if norm_id is None:
                continue
            extracted_file = archive.extractfile(member)
            if extracted_file is None:
                continue
            (output_dir / f"{norm_id}.png").write_bytes(extracted_file.read())
            extracted += 1
            if extracted == len(missing):
                break

    unresolved = sorted(norm_id for norm_id in missing if not (output_dir / f"{norm_id}.png").exists())
    return extracted, unresolved


def copy_seed_files(src_dir: Path, dest_dir: Path, suffix: str) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(src_dir.glob(f"CH-*{suffix}")):
        shutil.copy2(path, dest_dir / path.name)
        copied += 1
    return copied


def write_list_manifest(ids: list[str], manifest_path: Path, header: str) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([header])
        for norm_id in ids:
            writer.writerow([norm_id])


def _find_pdf_renderer() -> list[str] | None:
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        return [pdftoppm]
    pdftocairo = shutil.which("pdftocairo")
    if pdftocairo:
        return [pdftocairo]
    return None


def render_seed_pdfs_to_png(
    *,
    existing_pdf_dir: Path,
    png_dir: Path,
    norm_ids: list[str],
) -> tuple[int, list[str]]:
    renderer = _find_pdf_renderer()
    if renderer is None:
        return 0, norm_ids[:]

    png_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    failed: list[str] = []
    use_pdftoppm = Path(renderer[0]).name == "pdftoppm"

    for norm_id in norm_ids:
        pdf_path = existing_pdf_dir / f"{norm_id}_full.pdf"
        out_png = png_dir / f"{norm_id}.png"
        if out_png.exists():
            continue
        if not pdf_path.exists():
            failed.append(norm_id)
            continue

        if use_pdftoppm:
            cmd = [
                renderer[0],
                "-f",
                "1",
                "-singlefile",
                "-png",
                str(pdf_path),
                str(out_png.with_suffix("")),
            ]
        else:
            cmd = [
                renderer[0],
                "-f",
                "1",
                "-singlefile",
                "-png",
                str(pdf_path),
                str(out_png.with_suffix("")),
            ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out_png.exists():
            failed.append(norm_id)
            continue
        rendered += 1

    return rendered, failed


def symlink_force(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(os.path.relpath(src, dst.parent), dst)


def materialize_docs_dir(
    *,
    docs_dir: Path,
    png_dir: Path,
    existing_ids: list[str],
    additional_ids: list[str],
    existing_pdf_dir: Path,
) -> tuple[int, int]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    for entry in docs_dir.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    png_links = 0
    pdf_links = 0

    for norm_id in additional_ids:
        src = png_dir / f"{norm_id}.png"
        if not src.exists():
            raise FileNotFoundError(f"Missing additional PNG: {src}")
        symlink_force(src, docs_dir / src.name)
        png_links += 1

    for norm_id in existing_ids:
        png_src = png_dir / f"{norm_id}.png"
        if png_src.exists():
            symlink_force(png_src, docs_dir / png_src.name)
            png_links += 1
            continue

        pdf_src = existing_pdf_dir / f"{norm_id}_full.pdf"
        if not pdf_src.exists():
            raise FileNotFoundError(f"Missing fallback PDF: {pdf_src}")
        symlink_force(pdf_src, docs_dir / pdf_src.name)
        pdf_links += 1

    return png_links, pdf_links


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified Swiss gold-standard workspace with 500 PNG docs.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/Volumes/Crucial_X10/patentcity_dev/CH"),
        help="Directory containing png.tar.gz",
    )
    parser.add_argument(
        "--existing-pdf-dir",
        type=Path,
        default=Path("data/raw_pdf"),
        help="Directory containing the initial 99 Swiss PDFs",
    )
    parser.add_argument(
        "--existing-txt-dir",
        type=Path,
        default=Path("data/ocr_text"),
        help="Directory containing the initial 99 Swiss OCR txt files",
    )
    parser.add_argument(
        "--additional-manifest",
        type=Path,
        default=Path("data/gold_standard_CH/ch_additional_401_manifest.csv"),
        help="CSV manifest listing the 401 additional ids",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=Path("data/gold_standard_CH"),
        help="Target directory for the unified CH corpus",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_dir = args.gold_dir
    png_dir = gold_dir / "PNGs_extracted"
    docs_dir = gold_dir / "docs_500"
    seed_pdf_dir = gold_dir / "seed_raw_pdf"
    seed_txt_dir = gold_dir / "seed_ocr_text"

    existing_ids = load_existing_pdf_ids(args.existing_pdf_dir)
    additional_ids = load_manifest_ids(args.additional_manifest)

    existing_manifest = gold_dir / "ch_existing_seed_99_manifest.csv"
    combined_manifest = gold_dir / "ch_gold_500_manifest.csv"
    write_manifest(existing_ids, existing_manifest, "existing_seed_99")
    write_combined_manifest(existing_ids, additional_ids, combined_manifest)

    extracted, unresolved_existing_png_ids = extract_missing_pngs(args.source_dir / "png.tar.gz", existing_ids, png_dir)
    rendered_from_pdf, still_missing_png_ids = render_seed_pdfs_to_png(
        existing_pdf_dir=args.existing_pdf_dir,
        png_dir=png_dir,
        norm_ids=unresolved_existing_png_ids,
    )
    copied_pdfs = copy_seed_files(args.existing_pdf_dir, seed_pdf_dir, "_full.pdf")
    copied_txts = copy_seed_files(args.existing_txt_dir, seed_txt_dir, "_full.txt")
    write_list_manifest(unresolved_existing_png_ids, gold_dir / "ch_existing_seed_png_missing_from_tar.csv", "norm_id")
    write_list_manifest(still_missing_png_ids, gold_dir / "ch_existing_seed_png_still_missing.csv", "norm_id")
    png_links, pdf_links = materialize_docs_dir(
        docs_dir=docs_dir,
        png_dir=png_dir,
        existing_ids=existing_ids,
        additional_ids=additional_ids,
        existing_pdf_dir=args.existing_pdf_dir,
    )

    print(f"existing_ids={len(existing_ids)}")
    print(f"additional_ids={len(additional_ids)}")
    print(f"combined_ids={len(existing_ids) + len(additional_ids)}")
    print(f"new_pngs_extracted_for_existing_ids={extracted}")
    print(f"existing_ids_without_png_in_tar={len(unresolved_existing_png_ids)}")
    print(f"rendered_existing_pdfs_to_png={rendered_from_pdf}")
    print(f"existing_ids_still_without_png={len(still_missing_png_ids)}")
    print(f"copied_seed_pdfs={copied_pdfs}")
    print(f"copied_seed_txts={copied_txts}")
    print(f"docs_dir_png_links={png_links}")
    print(f"docs_dir_pdf_links={pdf_links}")
    print(f"png_dir={png_dir}")
    print(f"docs_dir={docs_dir}")
    print(f"combined_manifest={combined_manifest}")


if __name__ == "__main__":
    main()
