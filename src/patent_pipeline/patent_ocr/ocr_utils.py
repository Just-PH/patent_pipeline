import csv
import argparse
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image, ImageOps
import pytesseract
from langdetect import detect
import regexp as re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
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

def preprocess_image(img):
    img = deskew_image(img)
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = img.point(lambda x: 0 if x < 140 else 255, "1")
    return img

def fix_dates(text: str) -> str:
    text = re.sub(r"107\s+mai", "1er mai", text, flags=re.IGNORECASE)
    text = re.sub(r"l9(\d{2})", r"19\1", text)
    text = re.sub(r"I9(\d{2})", r"19\1", text)
    return text

# ---------- Détection langue ----------
def detect_doc_lang(path: Path, is_pdf=True) -> str:
    try:
        if is_pdf:
            img = convert_from_path(path, dpi=200, first_page=1, last_page=1)[0]
        else:
            img = Image.open(path)
        img = preprocess_image(img)
        text = pytesseract.image_to_string(img, lang=TESSERACT_LANGS)
        if not text.strip():
            return "unknown"
        return detect(text)
    except Exception:
        return "unknown"

# ---------- OCR ----------
def doc_to_text(path: Path, lang: str = TESSERACT_LANGS, is_pdf=True) -> str:
    try:
        if is_pdf:
            images = convert_from_path(path, dpi=300, first_page=1, last_page=1)
            if not images:
                return ""
            img = preprocess_image(images[0])
        else:
            img = preprocess_image(Image.open(path))
        text = pytesseract.image_to_string(img, lang=lang)
        return fix_dates(text)
    except Exception:
        return ""
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


# ---------- OCR pour un document ----------
def process_one(doc, out_dir, force=False, country_hint=None):
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
    text = doc_to_text(doc, lang=tesseract_lang, is_pdf=is_pdf)
    out_file.write_text(text, encoding="utf-8")

    return {
        "file_name": doc.name,
        "file_type": "pdf" if is_pdf else "png",
        "country_detected": country,
        "tesseract_lang": tesseract_lang,
        "txt_path": str(out_file)
    }


# ---------- Pipeline principal ----------
def process_all_docs(raw_dir, out_dir, report_file, force=False, limit=None, threads=1, country_hint=None):
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
        writer = csv.DictWriter(csvfile, fieldnames=["file_name", "file_type", "country_detected", "tesseract_lang", "txt_path"])
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=threads) as executor:
            futures = {executor.submit(process_one, doc, out_dir, force, country_hint): doc for doc in docs}

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
                        "txt_path": "",
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
    args = p.parse_args()

    process_all_docs(
        args.raw_dir, args.out_dir, args.report_file,
        force=args.force, limit=args.limit, threads=args.threads,
        country_hint=args.country_hint
    )

    print(f"📊 Rapport généré: {args.report_file}")
