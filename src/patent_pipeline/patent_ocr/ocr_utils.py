import csv
import time
import argparse
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image, ImageOps
import cv2
import pytesseract
import regexp as re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from typing import Optional, Tuple
from io import BytesIO
from collections import Counter

# 📚 Fréquences moyennes par langue (sources combinées : Norvig, Lewand, Lexique3)
REF_FREQ = {
    "fr": {"e":14.7, "a":8.4, "i":7.5, "s":7.9, "t":7.2, "r":6.5, "n":7.0, "c":3.3, "o":5.2, "u":6.3},
    "de": {"e":17.4, "a":6.5, "i":7.6, "s":7.3, "t":6.2, "r":7.0, "n":9.8, "c":3.1, "o":2.5, "u":4.0},
    "en": {"e":12.7, "a":8.2, "i":6.9, "s":6.3, "t":9.1, "r":6.0, "n":6.7, "c":2.8, "o":7.5, "u":2.3},
    "it": {"e":11.8, "a":11.5, "i":10.1, "s":4.9, "t":5.6, "r":6.4, "n":6.8, "c":4.5, "o":9.8, "u":3.2},
    "mix": {"e":14.0, "a":8.0, "i":7.0, "s":7.0, "t":7.0, "r":6.5, "n":7.5, "c":3.5, "o":6.0, "u":4.5},
}

TESSERACT_LANGS = "frk+deu+eng+fra+ita"


# ----------------------------------------------------------------------
# 🧹 Image preprocessing utils
# ----------------------------------------------------------------------
def deskew_image(pil_img, max_angle: float = 10.0):
    gray = np.array(pil_img.convert("L"))
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.bitwise_not(bw)

    coords = np.column_stack(np.where(bw > 0))
    if coords.size == 0:
        return pil_img

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    angle = -(90 + angle) if angle < -45 else -angle

    if abs(angle) < 0.1 or abs(angle) > max_angle:
        return pil_img

    # rotation sur l’original en RGB pour éviter les surprises dtype
    img_cv = np.array(pil_img.convert("RGB"))  # <-- convert en RGB
    (h, w) = img_cv.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img_cv, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated)


def autocrop_image(pil_img, threshold=245):
    """
    Supprime les marges quasi-blanches autour du texte.
    threshold : 0-255 au-dessus duquel on considère la zone blanche.
    """
    np_img = np.array(pil_img.convert("L"))
    mask = np_img < threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return pil_img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # slices non inclusives
    cropped = pil_img.crop((x0, y0, x1, y1))
    return cropped

def clahe(gray, clip=2.0, grid=8):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)

def unsharp(gray, sigma=1.0, strength=0.6):
    # léger (évite les halos qui ferment les boucles du “2”)
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1 + strength, blur, -strength, 0)

def otsu_with_bias(gray, bias=8):
    # Otsu puis on relève un peu le seuil pour ne PAS boucher les ouvertures
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    t = max(0, min(255, t + bias))
    _, bw = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    return bw

def preprocess_image(pil_img):
    # 0) légère montée d’échelle pour stabiliser les chiffres fins
    up = pil_img.resize(
        (int(pil_img.width * 1.5), int(pil_img.height * 1.5)),
        Image.LANCZOS
    )

    # 1) deskew très léger AVANT tout (réutilise ta fonction)
    up = deskew_image(up)

    # 2) gris → CLAHE (local, mieux qu’equalizeHist pour éviter d’amplifier le bruit)
    gray = np.array(up.convert("L"))
    gray = clahe(gray, clip=2.0, grid=8)

    # 3) unsharp léger (sigma 1.0, strength 0.6) : rend les bords nets sans halo
    gray = unsharp(gray, sigma=1.0, strength=0.6)

    # 4) binarisation douce : Otsu + petit biais vers le blanc
    bw = otsu_with_bias(gray, bias=5)

    # 5) sortie PIL + autocontrast + autocrop
    pil_out = Image.fromarray(bw)
    pil_out = ImageOps.autocontrast(pil_out)
    pil_out = autocrop_image(pil_out, threshold=245)
    return pil_out


# ----------------------------------------------------------------------
# 🌍 Helpers
# ----------------------------------------------------------------------
def detect_doc_country(path: Path) -> str:
    """
    Extrait le code pays (2 lettres) du nom de fichier, ex: CH-18890-A_full.pdf → ch
    """
    match = re.match(r"([A-Z]{2})-", path.stem)
    return match.group(1).lower() if match else "unknown"


def map_country(country_code: str) -> str:
    """
    Mappe un pays vers les langues Tesseract à utiliser.
    """
    country_code = country_code.lower()

    mapping = {
        "ch": "fra+deu+ita",  # 🇨🇭 Suisse
        "fr": "fra+eng",           # 🇫🇷 France
        "de": "frk+deu+eng",       # 🇩🇪 Allemagne
        "it": "ita+eng",           # 🇮🇹 Italie
        "gb": "eng",               # 🇬🇧 Royaume-Uni
        "us": "eng",               # 🇺🇸 États-Unis
        "be": "fra+deu+eng",       # 🇧🇪 Belgique
        "ca": "fra+eng",           # 🇨🇦 Canada
    }
    return mapping.get(country_code, TESSERACT_LANGS)

# ----------------------------------------------------------------------
# 🚀 Unified dispatcher
# ----------------------------------------------------------------------
def process_all_docs(raw_dir, out_dir, report_file,
                     backend="tesseract", force=False, country_hint=None,
                     limit=None, threads=1, batch_size=8):
    """
    Single entrypoint for all OCR pipelines.
    Chooses CPU or GPU path depending on backend.
    """
    raw_dir, out_dir, report_file = Path(raw_dir), Path(out_dir), Path(report_file)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(raw_dir.glob("*.pdf"))
    pngs = sorted(raw_dir.glob("*.png"))
    docs = (pdfs + pngs)[:limit] if limit else (pdfs + pngs)

    print(f"➡️  {len(docs)} documents found (backend={backend})")

    if backend in ("tesseract", "paddleocr"):
        process_all_CPU(docs, out_dir, report_file, backend=backend,
                        threads=threads, force=force, country_hint=country_hint)
    elif backend in ("doctr", "easyocr"):
        process_all_GPU(docs, out_dir, report_file, backend=backend,
                        batch_size=batch_size,force =force)
    else:
        raise ValueError(f"❌ Unsupported backend: {backend}")


# ----------------------------------------------------------------------
# 🧠 CPU PIPELINE — (Tesseract)
# ----------------------------------------------------------------------
def process_one_CPU(args):
    """Worker executed in subprocesses."""
    doc, out_dir, backend, force, country_hint = args
    out_file = out_dir / f"{doc.stem}.txt"
    if out_file.exists() and not force:
        return {"file_name": doc.name, "status": "skipped"}

    country = country_hint or detect_doc_country(doc)
    lang = map_country(country)

    if backend == "tesseract":
        if doc.suffix.lower() == ".pdf":
            images = convert_from_path(doc, dpi=300, first_page=1, last_page=1)
            img = preprocess_image(images[0]) if images else None
        else:
            img = preprocess_image(Image.open(doc))
        text = pytesseract.image_to_string(img, lang=lang) if img else ""

    else:
        raise ValueError(f"Unknown CPU backend: {backend}")

    out_file.write_text(text, encoding="utf-8")

    # 💡 Compte uniquement les lettres pertinentes
    key_letters = list("ecoagpnrvthkslyq")
    text_lower = text.lower()
    total_letters = sum(c.isalpha() for c in text_lower) or 1
    cnt = Counter(c for c in text_lower if c in key_letters)
    counts = {l: cnt.get(l, 0) / total_letters * 100 for l in key_letters}


    # ✅ Retourne uniquement les champs connus
    res = {"file_name": doc.name, "status": "ok"}
    for l in key_letters:
        res[f"freq_{l}"] = round(counts[l], 2)

    return res


def process_all_CPU(
    docs, out_dir, report_file,
    backend="tesseract", threads=4, force=False, country_hint=None
):
    """Multi-process CPU OCR pipeline."""
    start = time.time()

    # 🧩 Prépare les arguments complets pour chaque worker
    args = [(doc, out_dir, backend, force, country_hint) for doc in docs]

    # 🧠 Lettres les plus sensibles aux erreurs de boucle / fermeture
    key_letters = list("ecoagpnrvthkslyq")

    fieldnames = ["file_name", "status"] + [f"freq_{l}" for l in key_letters]
    # ---- Run multiprocessing ----
    with report_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=threads) as pool:
            for res in tqdm(pool.map(process_one_CPU, args), total=len(docs),
                            desc=f"🧩 OCR CPU ({backend})"):

                # ✅ garde uniquement les champs prévus
                filtered = {k: v for k, v in res.items() if k in fieldnames or not k.startswith("freq_")}
                for l in key_letters:
                    filtered.setdefault(f"freq_{l}", 0.0)

                writer.writerow(res)

    elapsed = time.time() - start
    avg = elapsed / max(1, len(docs))
    print(f"✅ CPU {backend} done: {elapsed:.1f}s total ({avg:.2f}s/doc)")


# ----------------------------------------------------------------------
# ⚡ GPU PIPELINE — (docTR / EasyOCR)
# ----------------------------------------------------------------------
def preprocess_pdf_for_doctr(path_str: str):
    """TOP-LEVEL: returns (path_str, PIL.Image RGB)."""
    try:
        p = Path(path_str)
        imgs = convert_from_path(p, dpi=300, first_page=1, last_page=1)
        if not imgs:
            print(f"[WARN] No page extracted from {p.name}")
            return None
        pil_rgb = imgs[0].convert("RGB")
        print(f"[INFO preprocess-pdf] {p.name}: PIL RGB {pil_rgb.size}")
        return (path_str, pil_rgb)
    except Exception as e:
        print(f"[ERROR preprocess-pdf] {Path(path_str).name}: {e}")
        return None

def process_all_GPU(docs, out_dir, report_file,
                    backend="doctr", batch_size=8, force=False):
    """
    Batch GPU pipeline.
    Each backend (doctr, easyocr, etc.) handled via if/elif.
    """
    import torch, gc
    from tqdm import tqdm

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"🚀 Using {backend.upper()} on {device} (batch={batch_size})")

    # 🧹 Filtrage des fichiers (skip existants si pas --force)
    todo = []
    skipped = []
    for p in docs:
        out_txt = out_dir / f"{p.stem}.txt"
        if out_txt.exists() and not force:
            skipped.append(p)
        else:
            todo.append(p)

    print(f"📂 {len(todo)} to process, {len(skipped)} skipped (existing files)")

    if not todo:
        print("🎉 Nothing to process, all files already OCR’d.")
        return

    # Backend = docTR
    if backend == "doctr":
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import queue, threading, gc

        predictor = ocr_predictor(
            det_arch="linknet_resnet18",
            reco_arch="sar_resnet31",
            pretrained=True
        ).to(device)



        def process_all_async(todo, out_dir, writer):
            q = queue.Queue(maxsize=batch_size)
            total = len(todo)
            pbar_pre = tqdm(total=total, desc="🧱 CPU preprocess", position=0, leave=True)
            pbar_ocr = tqdm(total=total, desc="🤖 GPU OCR (docTR)", position=1, leave=True)

            def producer():
                pdfs = [d for d in todo if d.suffix.lower() == ".pdf"]
                imgs = [d for d in todo if d.suffix.lower() != ".pdf"]

                # PDFs via process pool -> returns PIL.Image
                with ProcessPoolExecutor(max_workers=4) as pool:
                    futs = [pool.submit(preprocess_pdf_for_doctr, str(d)) for d in pdfs]
                    for fut in as_completed(futs):
                        res = fut.result()
                        if res is not None:
                            q.put(res)   # (path_str, PIL.Image)
                        pbar_pre.update(1)

                # PNG/JPG inline -> also push PIL.Image
                for d in imgs:
                    try:
                        pil_rgb = Image.open(d).convert("RGB")
                        q.put((str(d), pil_rgb))
                    except Exception as e:
                        print(f"[ERROR preprocess-img] {d.name}: {e}")
                    finally:
                        pbar_pre.update(1)

                q.put(None)

            def consumer(batch_size=4):
                batch = []
                metas = []
                while True:
                    item = q.get()
                    if item is None:
                        # Traiter le dernier batch
                        if batch:
                            process_batch(batch, metas)
                        break

                    path_str, pil_img = item
                    batch.append(pil_img)
                    metas.append(path_str)

                    if len(batch) >= batch_size:
                        process_batch(batch, metas)
                        batch, metas = [], []

            def process_batch(batch, metas):
                try:
                    bufs = []
                    for img in batch:
                        buf = BytesIO()
                        img.save(buf, format="PNG")
                        bufs.append(buf.getvalue())

                    docfile = DocumentFile.from_images(bufs)
                    result = predictor(docfile)
                    rendered = result.render()
                    if isinstance(rendered, list):
                        for meta, text in zip(metas, rendered):
                            p = Path(meta)
                            (out_dir / f"{p.stem}.txt").write_text(text, encoding="utf-8")
                            writer.writerow({"file_name": p.name, "status": "ok"})
                    else:
                        # fallback si docTR ne renvoie pas une liste
                        for meta in metas:
                            p = Path(meta)
                            (out_dir / f"{p.stem}.txt").write_text(rendered, encoding="utf-8")
                            writer.writerow({"file_name": p.name, "status": "ok"})
                except Exception as e:
                    print(f"❌ Error batch {[Path(m).name for m in metas]}: {e}")
                    for meta in metas:
                        writer.writerow({"file_name": Path(meta).name, "status": "error"})
                finally:
                    pbar_ocr.update(len(batch))
                    gc.collect()

            t = threading.Thread(target=producer, daemon=True)
            t.start()
            consumer()
            t.join()
            pbar_pre.close(); pbar_ocr.close()


    # ---- Execution adaptative (async pour docTR) ----
    start = time.time()
    with report_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "status"])
        writer.writeheader()

        if backend == "doctr":
            print("🧠 Running async docTR pipeline (CPU→GPU overlap)...")
            process_all_async(todo, out_dir, writer)
        else:
            raise ValueError(f"❌ Unsupported GPU backend: {backend}")

    total = time.time() - start
    avg = total / max(1, len(todo))
    print(f"✅ GPU {backend} done: {total:.1f}s total ({avg:.2f}s/doc)")



# ----------------------------------------------------------------------
# 🎛️ CLI Entrypoint
# ----------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--report_file", required=True)
    p.add_argument("--backend", type=str, default="tesseract",
                   choices=["tesseract", "doctr", "easyocr"])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    process_all_docs(
        args.raw_dir, args.out_dir, args.report_file,
        backend=args.backend, force=args.force,
        limit=args.limit, threads=args.threads,
        batch_size=args.batch_size
    )

    print(f"📊 Report generated: {args.report_file}")
