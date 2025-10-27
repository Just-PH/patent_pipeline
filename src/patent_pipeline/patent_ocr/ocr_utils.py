import csv
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

TESSERACT_LANGS = "frk+deu+eng+fra+ita"


# ---------- Prétraitement ----------
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

def preprocess_image(img):
    img = deskew_image(img)
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = img.point(lambda x: 0 if x < 140 else 255, "1")
    img = autocrop_image(img, threshold=245)
    return img

# ---------- Détection du pays ----------
def detect_doc_country(path: Path) -> str:
    """
    Extrait le code pays (2 lettres) du nom de fichier, ex: CH-18890-A_full.pdf → ch
    """
    match = re.match(r"([A-Z]{2})-", path.stem)
    return match.group(1).lower() if match else "unknown"

# ---------- Mapping pays → langues Tesseract ----------
def map_country(country_code: str) -> str:
    """
    Mappe un pays vers les langues Tesseract à utiliser.
    """
    country_code = country_code.lower()

    mapping = {
        "ch": "fra+deu+ita+eng",  # 🇨🇭 Suisse
        "fr": "fra+eng",           # 🇫🇷 France
        "de": "frk+deu+eng",       # 🇩🇪 Allemagne
        "it": "ita+eng",           # 🇮🇹 Italie
        "gb": "eng",               # 🇬🇧 Royaume-Uni
        "us": "eng",               # 🇺🇸 États-Unis
        "be": "fra+deu+eng",       # 🇧🇪 Belgique
        "ca": "fra+eng",           # 🇨🇦 Canada
    }
    return mapping.get(country_code, TESSERACT_LANGS)

# ---------- OCR ----------
def doc_to_text(path: Path, lang: str = TESSERACT_LANGS, is_pdf=True, backend='tesseract') -> str:
    """
    Effectue l'OCR sur un fichier PDF ou PNG avec Tesseract ou docTR.
    Retourne le texte brut reconnu.
    """
    try:
        if backend == "tesseract":
            # 🔹 OCR classique avec Tesseract
            if is_pdf:
                images = convert_from_path(path, dpi=300, first_page=1, last_page=1)
                if not images:
                    return ""
                img = preprocess_image(images[0])
            else:
                img = preprocess_image(Image.open(path))
            text = pytesseract.image_to_string(img, lang=lang)
            return text

        elif backend == "doctr":
            # 🔹 OCR deep learning avec docTR
            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor
            import torch

            device = (
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )

            predictor = ocr_predictor(
                det_arch="linknet_resnet18",
                reco_arch="sar_resnet31",
                pretrained=True
            ).to(device)


            if is_pdf:
                doc = DocumentFile.from_pdf(path)
            else:
                doc = DocumentFile.from_images(path)

            result = predictor(doc)
            text = result.render()
            return text
        else:
            raise ValueError(f"Unknown backend: {backend}")

    except Exception as e:
        print(f"⚠️ Erreur OCR sur {path.name} ({backend}): {e}")
        return ""


# ---------- Process pour un document ----------
def process_one(doc, out_dir, force=False, country_hint=None, backend='tesseract'):
    is_pdf = doc.suffix.lower() == ".pdf"
    out_file = out_dir / (doc.stem + ".txt")

    if out_file.exists() and not force:
        return {
            "file_name": doc.name,
            "file_type": "pdf" if is_pdf else "png",
            "country_detected": "skipped",
            "tesseract_lang": "skipped",
            "txt_path": str(out_file)
        }

    country = country_hint or detect_doc_country(doc)
    tesseract_lang = map_country(country)
    text = doc_to_text(doc, lang=tesseract_lang, is_pdf=is_pdf, backend=backend)
    out_file.write_text(text, encoding="utf-8")

    return {
        "file_name": doc.name,
        "file_type": "pdf" if is_pdf else "png",
        "country_detected": country,
        "tesseract_lang": tesseract_lang,
        "txt_path": str(out_file)
    }


# ---------- Pipeline principal ----------
def process_all_docs(raw_dir, out_dir, report_file, force=False, limit=None, threads=1, country_hint=None, backend='tesseract'):
    raw_dir, out_dir, report_file = Path(raw_dir), Path(out_dir), Path(report_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(raw_dir.glob("*.pdf"))
    pngs = sorted(raw_dir.glob("*.png"))
    docs = pdfs + pngs
    if limit:
        docs = docs[:limit]

    print(f"➡️  {len(docs)} documents à traiter (dans {raw_dir})")
    print(f"🧠 Exécution parallèle sur {threads} thread(s)")

    with report_file.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["file_name", "file_type", \
            "country_detected", "tesseract_lang","backend", "txt_path"])
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=threads) as executor:
            futures = {executor.submit(process_one, doc, out_dir, force, country_hint,backend): doc for doc in docs}

            for fut in tqdm(as_completed(futures), total=len(futures), desc="🧩 OCR parallèle", unit="doc"):
                try:
                    res = fut.result()
                    writer.writerow(res)
                except Exception as e:
                    doc = futures[fut]
                    writer.writerow({
                        "file_name": doc.name,
                        "file_type": "pdf" if doc.suffix.lower() == ".pdf" else "png",
                        "country_detected": "error",
                        "tesseract_lang": "error",
                        "backend": backend,
                        "txt_path": ""
                    })

# ---------- Entrée CLI ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--report_file", required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--country_hint", type=str, help="Code pays (ch, fr, de, it...)")
    p.add_argument("--backend", type=str, default="tesseract",
               choices=["tesseract", "doctr"],
               help="OCR backend à utiliser")
    args = p.parse_args()

    process_all_docs(
        args.raw_dir, args.out_dir, args.report_file,
        force=args.force, limit=args.limit, threads=args.threads,
        country_hint=args.country_hint, backend=args.backend
    )

    print(f"📊 Rapport généré: {args.report_file}")
