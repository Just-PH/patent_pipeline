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
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from io import BytesIO
from collections import Counter
from skimage import filters, img_as_ubyte
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
filepath = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt"
)
model = YOLOv10(filepath)




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

def otsu_with_bias(gray, bias=5):
    # Otsu puis on relève un peu le seuil pour ne PAS boucher les ouvertures
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    t = max(0, min(255, t + bias))
    _, bw = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    return bw

def sauvola_threshold(gray, window=25, k=0.2):
    thresh_sauvola = filters.threshold_sauvola(gray, window_size=window, k=k)
    binary = gray > thresh_sauvola
    return img_as_ubyte(binary)

def niblack_threshold(gray, window=25, k=0.8):
    thresh = filters.threshold_niblack(gray, window_size=window, k=k)
    binary = gray > thresh
    return img_as_ubyte(binary)

def preprocess_image(pil_img,method="otsu"):
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

    # 3) Sélection du mode de binarisation
    if method == "otsu":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "sauvola":
        bw = sauvola_threshold(gray, window=25, k=0.3)
    elif method == "niblack":
        bw = niblack_threshold(gray, window=25, k=0.8)
    else:
        raise ValueError(f"❌ Unknown preprocessing method: {method}")

    # 5) sortie PIL + autocontrast + autocrop
    pil_out = Image.fromarray(bw)
    pil_out = ImageOps.autocontrast(pil_out)
    pil_out = autocrop_image(pil_out, threshold=245)
    return pil_out

def yolo_segment_and_ocr(img, lang, config):
    """
    Segmente l'image en blocs (nombre auto) grâce à YOLO DocLayNet.
    OCRise chaque bloc séparément.
    Recompose en un texte propre.
    """
    # Convertir en RGB (YOLO attend RGB)
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- 1) YOLO segmentation ---
    results = _yolo_layout(rgb)[0]   # une seule image

    blocks = []
    for b in results.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        label = _yolo_layout.names[int(b.cls[0])]
        conf = float(b.conf[0])
        blocks.append({
            "bbox": (x1, y1, x2, y2),
            "label": label,
            "conf": conf
        })

    # --- 2) Tri des blocs dans l'ordre de lecture ---
    blocks = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))

    # --- 3) OCR de chaque bloc ---
    texts = []

    for blk in blocks:
        x1, y1, x2, y2 = map(int, blk["bbox"])
        crop = rgb[y1:y2, x1:x2]

        # OCR Tesseract sur le crop
        t = pytesseract.image_to_string(crop, lang=lang, config=config)
        t = t.strip()
        if t:
            texts.append(t)

    # --- 4) Reconstruction propre ---
    final_text = "\n\n".join(texts)
    return final_text

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

KEY_LETTERS = list("ecoagpnrvthkslyq")

def compute_letter_freq(text: str):
    text_lower = text.lower()
    total_letters = sum(c.isalpha() for c in text_lower) or 1
    cnt = Counter(c for c in text_lower if c in KEY_LETTERS)
    return {f"freq_{l}": round(cnt.get(l, 0) / total_letters * 100, 2) for l in KEY_LETTERS}

# ----------------------------------------------------------------------
# 🚀 Unified dispatcher
# ----------------------------------------------------------------------
def process_all_docs(raw_dir, out_dir, report_file,
                     backend="tesseract", force=False, country_hint=None,
                     limit_ocr=None, threads=1, batch_size=4, preproc_method="sauvola"):
    """
    Single entrypoint for all OCR pipelines.
    Chooses CPU or GPU path depending on backend.
    """
    raw_dir, out_dir, report_file = Path(raw_dir), Path(out_dir), Path(report_file)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(raw_dir.glob("*.pdf"))
    pngs = sorted(raw_dir.glob("*.png"))
    docs = (pdfs + pngs)[:limit_ocr] if limit_ocr else (pdfs + pngs)

    print(f"➡️  {len(docs)} documents found (backend={backend})")
    print(f"➡️ Preprocessing method: {preproc_method}")
    if backend in ("tesseract", "paddleocr"):
        process_all_CPU(docs, out_dir, report_file, backend=backend,
                        threads=threads, force=force, country_hint=country_hint,preproc_method=preproc_method)
    elif backend in ("doctr", "easyocr"):
        process_all_GPU(docs, out_dir, report_file, backend=backend,
                        batch_size=batch_size,force =force,preproc_method=preproc_method)
    else:
        raise ValueError(f"❌ Unsupported backend: {backend}")


# ----------------------------------------------------------------------
# 🧠 CPU PIPELINE — (Tesseract)
# ----------------------------------------------------------------------
def process_one_CPU(args):
    """Worker executed in subprocesses."""
    doc, out_dir, backend, force, country_hint, preproc_method = args
    out_file = out_dir / f"{doc.stem}.txt"
    if out_file.exists() and not force:
        return {"file_name": doc.name, "status": "skipped"}

    country = country_hint or detect_doc_country(doc)
    lang = map_country(country)

    if backend == "tesseract":
        if doc.suffix.lower() == ".pdf":
            images = convert_from_path(doc, dpi=300, first_page=1, last_page=1)
            img = preprocess_image(pil_img=images[0], method=preproc_method)
        else:
            img = preprocess_image(pil_img=Image.open(doc), method=preproc_method)

        # config = (
        #     "--oem 1 "
        #     "--psm 4 "
        #     "-c textord_noise_rej=0 "
        #     "-c textord_min_xheight=8"
        #     "-c preserve_interword_spaces=1"
        # )

        # config = (
        #     # "--oem 1 "
        #     # "--psm 12 "
        #     "-c textord_noise_rej=0 "
        #     "-c textord_min_xheight=8"
        # ) #Basic config --- IGNORE ---

        # text = pytesseract.image_to_string(img, lang=lang, config = config) if img else ""
        config = (
            "--oem 1"
            "--psm 6"
            "-c preserve_interword_spaces=1 "
            "-c textord_min_xheight=10 "
            )
        text = yolo_segment_and_ocr(img, lang, config)

    else:
        raise ValueError(f"Unknown CPU backend: {backend}")

    out_file.write_text(text, encoding="utf-8")

    freqs = compute_letter_freq(text)

    return {
        "file_name": doc.name,
        "status": "ok",
        **freqs
    }


def process_all_CPU(
    docs, out_dir, report_file,
    backend="tesseract", threads=4, force=False, country_hint=None,preproc_method="sauvola"
):
    """Multi-process CPU OCR pipeline."""
    start = time.time()

    # 🧩 Prépare les arguments complets pour chaque worker
    args = [(doc, out_dir, backend, force, country_hint, preproc_method) for doc in docs]

    fieldnames = ["file_name", "status"] + [f"freq_{l}" for l in KEY_LETTERS]
    # ---- Run multiprocessing ----
    with report_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=threads) as pool:
            for res in tqdm(pool.map(process_one_CPU, args), total=len(docs),
                            desc=f"🧩 OCR CPU ({backend})"):


                writer.writerow(res)

    elapsed = time.time() - start
    avg = elapsed / max(1, len(docs))
    print(f"✅ CPU {backend} done: {elapsed:.1f}s total ({avg:.2f}s/doc)")


# ----------------------------------------------------------------------
# GPU PIPELINE — (docTR) — MPS / CUDA / CPU compatible
# ----------------------------------------------------------------------
def preprocess_pdf_for_doctr(path: Path, preproc_method: str = "sauvola"):
    try:
        imgs = convert_from_path(path, dpi=300, first_page=1, last_page=1)
        if not imgs:
            return None
        return preprocess_image(imgs[0], method=preproc_method)
    except Exception as e:
        print(f"[ERROR preprocess-pdf] {path.name}: {e}")
        return None


def process_all_GPU(docs, out_dir, report_file,
                    backend="doctr", batch_size=4, force=False, preproc_method="sauvola"):
    import torch
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    from io import BytesIO

    # --- Device ---
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"GPU OCR ({backend.upper()}) on {device} | batch_size={batch_size}")

    # --- Filtrer les fichiers ---
    todo = [doc for doc in docs if not (out_dir / f"{doc.stem}.txt").exists() or force]
    if not todo:
        print("Nothing to process.")
        return

    # --- Modèle ---
    predictor = ocr_predictor(
        det_arch="linknet_resnet18",
        reco_arch="sar_resnet31",
        pretrained=True
    ).to(device)

    # --- CSV ---
    fieldnames = ["file_name", "status"] + [f"freq_{l}" for l in KEY_LETTERS]
    with report_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # --- Chargement + préproc ---
        def load_and_preprocess(doc_path):
            try:
                if doc_path.suffix.lower() == ".pdf":
                    pil_img = preprocess_pdf_for_doctr(doc_path, preproc_method)
                else:
                    pil_img = Image.open(doc_path).convert("RGB")
                    pil_img = preprocess_image(pil_img, method=preproc_method)
                if pil_img is None:
                    return doc_path, None

                # → Convertir en bytes PNG (nécessaire pour MPS)
                buf = BytesIO()
                pil_img.save(buf, format="PNG")
                return doc_path, buf.getvalue()
            except Exception as e:
                print(f"[ERROR load] {doc_path.name}: {e}")
                return doc_path, None

        # --- Batch processing ---
        batch_bytes = []
        batch_metas = []
        pbar = tqdm(total=len(todo), desc="OCR GPU (docTR)", leave=True)

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_doc = {executor.submit(load_and_preprocess, doc): doc for doc in todo}

            for future in as_completed(future_to_doc):
                doc_path, img_bytes = future.result()

                if img_bytes is None:
                    writer.writerow({"file_name": doc_path.name, "status": "error"})
                    pbar.update(1)
                    continue

                batch_bytes.append(img_bytes)
                batch_metas.append(doc_path)

                # --- Batch plein ---
                if len(batch_bytes) >= batch_size:
                    texts = []  # ← Toujours défini
                    try:
                        # → from_images accepte list[bytes]
                        docfile = DocumentFile.from_images(batch_bytes)
                        result = predictor(docfile)
                        texts = [page.render() for page in result.pages]

                        for meta_doc, text in zip(batch_metas, texts):
                            out_path = out_dir / f"{meta_doc.stem}.txt"
                            out_path.write_text(text, encoding="utf-8")
                            freqs = compute_letter_freq(text)
                            writer.writerow({
                                "file_name": meta_doc.name,
                                "status": "ok",
                                **freqs
                            })
                    except Exception as e:
                        print(f"Batch error: {e}")
                        for meta in batch_metas:
                            writer.writerow({"file_name": meta.name, "status": "error"})
                        texts = []  # même en erreur
                    finally:
                        batch_bytes.clear()
                        batch_metas.clear()
                        pbar.update(len(texts))
                        if device == "cuda":
                            torch.cuda.empty_cache()

            # --- Dernier batch ---
            if batch_bytes:
                texts = []
                try:
                    docfile = DocumentFile.from_images(batch_bytes)
                    result = predictor(docfile)
                    texts = [page.render() for page in result.pages]

                    for meta_doc, text in zip(batch_metas, texts):
                        out_path = out_dir / f"{meta_doc.stem}.txt"
                        out_path.write_text(text, encoding="utf-8")
                        freqs = compute_letter_freq(text)
                        writer.writerow({
                            "file_name": meta_doc.name,
                            "status": "ok",
                            **freqs
                        })
                except Exception as e:
                    print(f"Final batch error: {e}")
                    for meta in batch_metas:
                        writer.writerow({"file_name": meta.name, "status": "error"})
                finally:
                    pbar.update(len(texts))

        pbar.close()

    total_time = pbar.format_dict["elapsed"]
    avg = total_time / max(1, len(todo))
    print(f"GPU {backend} done: {total_time:.1f}s total ({avg:.2f}s/doc)")

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
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit_ocr", type=int)
    p.add_argument("--country_hint", type=str, default=None,
               help="Force country code for all docs (e.g., 'ch', 'fr')")
    p.add_argument("--preproc_method", type=str, default="sauvola",
               choices=["otsu", "sauvola", "niblack"],
               help="OCR preprocessing method to use")

    args = p.parse_args()

    process_all_docs(
        args.raw_dir, args.out_dir, args.report_file,
        backend=args.backend, force=args.force,
        limit_ocr=args.limit_ocr, threads=args.threads,
        batch_size=args.batch_size, country_hint=args.country_hint, preproc_method=args.preproc
    )

    print(f"📊 Report generated: {args.report_file}")
