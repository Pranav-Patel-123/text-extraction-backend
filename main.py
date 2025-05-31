import os
import shutil
import mimetypes
import zipfile
import tempfile
from pathlib import Path
import logging

import pdfplumber
import PyPDF2
from docx import Document
from PIL import Image
import pytesseract
import textract
import pandas as pd
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text

import google.generativeai as genai
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import easyocr

# ─── LOGGING CONFIGURATION ────────────────────────────────────────────────────
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(console_handler)

# ─── CONFIGURE ENVIRONMENT & TESSERACT ─────────────────────────────────────────
load_dotenv()

# 1) Gemini API key (must be set in your .env)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env")
genai.configure(api_key=GEMINI_API_KEY)

# 2) Tesseract executable (optional, only needed if you want OCR fallback)
#    If you set TESSERACT_CMD in your environment (e.g. "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"),
#    pytesseract will use that path. Otherwise, OCR will be disabled.
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    logger.debug(f"[DEBUG] Tesseract command set to: {TESSERACT_CMD}")
else:
    logger.warning(
        "[WARNING] TESSERACT_CMD not set. OCR fallback will be disabled. "
        "To enable OCR on image-based uploads, install Tesseract and set TESSERACT_CMD."
    )

# 3) Check once whether Tesseract is truly available
try:
    pytesseract.get_tesseract_version()
    OCR_AVAILABLE = True
    logger.debug("[DEBUG] Tesseract OCR found. OCR_AVAILABLE = True")
except (pytesseract.TesseractNotFoundError, Exception):
    OCR_AVAILABLE = False
    logger.warning(
        "[WARNING] Tesseract OCR is not installed or not found at the configured path. "
        "OCR will be disabled for images and image-based PDFs."
    )

app = FastAPI(title="Universal File Text-Extractor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── UTILITIES ────────────────────────────────────────────────────────────────
def format_with_gemini(text: str) -> str:
    """
    Send the raw extracted text to Gemini for nice formatting.
    """
    logger.debug("[DEBUG] Sending text to Gemini for formatting (length=%d chars)", len(text))
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    resp = model.generate_content(
        f"Please format the following extracted text nicely for display:\n\n{text}"
    )
    logger.debug("[DEBUG] Received formatted text from Gemini (length=%d chars)", len(resp.text))
    return resp.text


def extract_pdf(path: str) -> str:
    """
    Extract text from a PDF using pdfplumber. If pdfplumber.extract_text() fails
    on every page and OCR is unavailable (or yields no text), try PyPDF2 as a fallback.
    Raises ValueError if, after both strategies, no text was obtained.
    """
    logger.debug(f"[DEBUG] Starting PDF extraction for: {path}")
    text_parts = []

    try:
        with pdfplumber.open(path) as pdf:
            num_pages = len(pdf.pages)
            logger.debug(f"[DEBUG] PDF opened successfully, page count: {num_pages}")

            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = None

                # 1) Try standard extract_text()
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        snippet = repr(page_text[:100]).replace("\n", "\\n")
                        logger.debug(
                            f"[DEBUG] Page {page_number} extract_text success "
                            f"(first 100 chars): {snippet}"
                        )
                        text_parts.append(page_text)
                        continue
                    else:
                        logger.debug(f"[DEBUG] Page {page_number} extract_text returned no text.")
                except Exception as e:
                    # If extract_text fails (e.g. 'LTLine' object has no attribute 'original_path'),
                    # we skip directly to OCR (if available).
                    logger.warning(f"[WARNING] extract_text failed on page {page_number}: {e}")

                # 2) OCR fallback for this page (only if Tesseract is configured)
                if OCR_AVAILABLE:
                    try:
                        page_image = page.to_image(resolution=150)
                        pil_img = page_image.original  # PIL Image
                        ocr_text = pytesseract.image_to_string(pil_img)
                        if ocr_text and ocr_text.strip():
                            snippet = repr(ocr_text[:100]).replace("\n", "\\n")
                            logger.debug(
                                f"[DEBUG] Page {page_number} OCR fallback success "
                                f"(first 100 chars): {snippet}"
                            )
                            text_parts.append(ocr_text)
                        else:
                            logger.debug(f"[DEBUG] Page {page_number} OCR fallback yielded no text.")
                    except Exception as ocre:
                        logger.error(f"[ERROR] OCR fallback failed on page {page_number}: {ocre}")
                else:
                    logger.debug(f"[DEBUG] Skipping OCR on page {page_number} (OCR unavailable).")

    except Exception as exc:
        # If pdfplumber cannot open or parse the PDF at all, we log and move to fallback section
        logger.error(f"[ERROR] Unable to open or parse PDF with pdfplumber: {exc}")
        # We do NOT raise here, because we want to try PyPDF2 as a fallback below

    # 3) Combine whatever pdfplumber (and OCR) gave us
    combined = "\n".join(text_parts).strip()

    # 4) If pdfplumber/OCR gave nothing, attempt PyPDF2 fallback
    if not combined:
        logger.debug("[DEBUG] No text retrieved via pdfplumber/OCR. Trying PyPDF2 fallback.")
        try:
            reader = PyPDF2.PdfReader(path)
            fallback_text = ""
            for p_index, p in enumerate(reader.pages):
                try:
                    p_text = p.extract_text()
                    if p_text:
                        fallback_text += p_text + "\n"
                except Exception as e:
                    logger.warning(f"[WARNING] PyPDF2 failed on page {p_index + 1}: {e}")
            fallback_text = fallback_text.strip()
            if fallback_text:
                logger.debug(f"[DEBUG] PyPDF2 fallback succeeded, total length: {len(fallback_text)} chars")
                return fallback_text
            else:
                logger.debug("[DEBUG] PyPDF2 fallback also returned no text.")
        except Exception as pe:
            logger.error(f"[ERROR] PyPDF2 fallback itself failed: {pe}")

        # 5) If we reach here, neither pdfplumber/OCR nor PyPDF2 produced any text
        if not OCR_AVAILABLE:
            # OCR was never available (or disabled), so inform the user accordingly
            logger.debug("[DEBUG] PDF had no extractable text and OCR is unavailable.")
            raise ValueError(
                "No text could be extracted from this PDF. "
                "If it is image-only, please install/configure Tesseract (set TESSERACT_CMD) to enable OCR."
            )
        else:
            # OCR was available but produced no text (or pdfplumber truly failed on every page)
            logger.debug("[DEBUG] PDF had no extractable text even after OCR fallback.")
            raise ValueError("No text found in PDF (it may be image-only or corrupted).")

    # 6) If we got here, pdfplumber (and possibly OCR) gave us something
    logger.debug(f"[DEBUG] PDF extraction complete, total text length: {len(combined)} chars")
    return combined


def extract_docx(path: str) -> str:
    logger.debug(f"[DEBUG] Starting DOCX extraction for: {path}")
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    combined = "\n".join(paragraphs).strip()
    logger.debug(f"[DEBUG] DOCX extraction complete, paragraph count: {len(paragraphs)}")
    return combined


def extract_doc(path: str) -> str:
    logger.debug(f"[DEBUG] Starting DOC extraction for: {path}")
    try:
        raw = textract.process(path)
        text = raw.decode(errors="ignore")
        logger.debug(f"[DEBUG] DOC extraction complete, length: {len(text)} chars")
        return text
    except Exception as e:
        logger.error(f"[ERROR] Error extracting .doc: {e}")
        raise ValueError(f"Error extracting .doc: {e}")


def extract_txt(path: str) -> str:
    logger.debug(f"[DEBUG] Starting TXT extraction for: {path}")
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        logger.debug(f"[DEBUG] TXT extraction complete, length: {len(content)} chars")
        return content
    except Exception as e:
        logger.error(f"[ERROR] Error reading .txt file: {e}")
        raise ValueError(f"Error reading .txt file: {e}")


def extract_rtf(path: str) -> str:
    logger.debug(f"[DEBUG] Starting RTF extraction for: {path}")
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        text = rtf_to_text(raw)
        logger.debug(f"[DEBUG] RTF extraction complete, length: {len(text)} chars")
        return text
    except Exception as e:
        logger.error(f"[ERROR] Error extracting RTF: {e}")
        raise ValueError(f"Error extracting RTF: {e}")


def extract_html(path: str) -> str:
    logger.debug(f"[DEBUG] Starting HTML extraction for: {path}")
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n")
        logger.debug(f"[DEBUG] HTML extraction complete, length: {len(text)} chars")
        return text
    except Exception as e:
        logger.error(f"[ERROR] Error extracting HTML: {e}")
        raise ValueError(f"Error extracting HTML: {e}")


def extract_excel(path: str) -> str:
    logger.debug(f"[DEBUG] Starting Excel extraction for: {path}")
    try:
        xls = pd.read_excel(path, sheet_name=None, dtype=str)
    except Exception as e:
        logger.error(f"[ERROR] Error opening Excel file: {e}")
        raise ValueError(f"Error opening Excel file: {e}")

    pieces = []
    for sheet_name, df in xls.items():
        rows_combined = df.fillna("").astype(str).agg(" | ".join, axis=1).str.cat(sep="\n")
        pieces.append(f"--- Sheet: {sheet_name} ---\n{rows_combined}")
        logger.debug(f"[DEBUG] Extracted sheet '{sheet_name}', row count: {len(df)}")

    combined = "\n\n".join(pieces).strip()
    logger.debug(f"[DEBUG] Excel extraction complete, total length: {len(combined)} chars")
    return combined


def extract_csv(path: str) -> str:
    logger.debug(f"[DEBUG] Starting CSV extraction for: {path}")
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.error(f"[ERROR] Error reading CSV: {e}")
        raise ValueError(f"Error reading CSV: {e}")

    combined = df.fillna("").astype(str).agg(" | ".join, axis=1).str.cat(sep="\n")
    logger.debug(f"[DEBUG] CSV extraction complete, row count: {len(df)}")
    return combined


def extract_image(path: str) -> str:
    """
    Extract text from an image using EasyOCR (no external Tesseract binary required).
    Returns all recognized text lines joined with newline separators.
    Raises ValueError if OCR fails or the image is unreadable.
    """

    logger.debug(f"[DEBUG] Starting EasyOCR on image: {path}")

    try:
        # 1) Optionally verify that PIL can open it (optional, but catches invalid files early).
        img = Image.open(path)
        img.verify()  # Only verifies format, does not load full image into memory
    except Exception as open_err:
        logger.error(f"[ERROR] Cannot open/verify image file '{path}': {open_err}")
        raise ValueError(f"Cannot open image for OCR: {open_err}")

    try:
        # 2) Instantiate an EasyOCR reader for English. 
        #    You can pass a list of languages, e.g. ['en', 'fr'] if you expect mixed text.
        reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if you have CUDA available

        # 3) Let EasyOCR run on the image path. detail=0 returns text strings only.
        ocr_results = reader.readtext(path, detail=0)

        if not ocr_results:
            logger.debug("[DEBUG] EasyOCR ran successfully but found no text.")
            return ""

        # 4) Join all detected text lines into a single string, separated by newlines.
        combined_text = "\n".join(ocr_results)
        logger.debug(f"[DEBUG] EasyOCR complete, total length: {len(combined_text)} chars")
        return combined_text

    except Exception as ocr_err:
        logger.error(f"[ERROR] EasyOCR failed on image '{path}': {ocr_err}")
        raise ValueError(f"EasyOCR failed on image: {ocr_err}")



def extract_zip(path: str) -> str:
    logger.debug(f"[DEBUG] Starting ZIP extraction for: {path}")
    text_parts = []
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(tmpdir)
            logger.debug(f"[DEBUG] ZIP extracted to temporary dir: {tmpdir}")

            for root, _, files in os.walk(tmpdir):
                for fn in files:
                    fp = os.path.join(root, fn)
                    logger.debug(f"[DEBUG] Found file in ZIP: {fp}")
                    try:
                        part_text = handle_file(fp)
                        if part_text:
                            text_parts.append(part_text)
                            logger.debug(
                                f"[DEBUG] Extracted text from {fp}, length: {len(part_text)} chars"
                            )
                        else:
                            logger.debug(f"[DEBUG] No text extracted from {fp} (empty).")
                    except ValueError as ve:
                        logger.debug(
                            f"[DEBUG] Skipping unsupported file in ZIP: {fp} -> {ve}"
                        )

        combined = "\n\n".join(text_parts).strip()
        logger.debug(f"[DEBUG] ZIP extraction complete, total text length: {len(combined)} chars")
        return combined
    except Exception as e:
        logger.error(f"[ERROR] Error extracting ZIP: {e}")
        raise ValueError(f"Error extracting ZIP: {e}")


def handle_file(file_path: str) -> str:
    """
    Dispatch based on file extension (case-insensitive). Raises ValueError if unsupported.
    """
    ext = Path(file_path).suffix.lower().lstrip(".")
    logger.debug(f"[DEBUG] Handling file '{file_path}' with detected extension: '{ext}'")

    if ext == "pdf":
        return extract_pdf(file_path)
    elif ext == "docx":
        return extract_docx(file_path)
    elif ext == "doc":
        return extract_doc(file_path)
    elif ext in ("txt", "md"):
        return extract_txt(file_path)
    elif ext == "rtf":
        return extract_rtf(file_path)
    elif ext in ("html", "htm"):
        return extract_html(file_path)
    elif ext in ("xlsx", "xls"):
        return extract_excel(file_path)
    elif ext == "csv":
        return extract_csv(file_path)
    elif ext in ("png", "jpg", "jpeg", "tiff", "bmp", "gif", "webp"):
        return extract_image(file_path)
    elif ext == "zip":
        return extract_zip(file_path)
    else:
        mime_type, _ = mimetypes.guess_type(file_path)
        logger.debug(f"[DEBUG] MIME type guess for '{file_path}': {mime_type}")
        if mime_type == "application/pdf":
            return extract_pdf(file_path)
        logger.error(f"[ERROR] Unsupported extension for file '{file_path}': .{ext}")
        raise ValueError(f"Unsupported extension: .{ext}")


# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.post("/upload/", summary="Upload a file and get extracted + formatted text")
async def upload_file(file: UploadFile = File(...)):
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    dest_path = os.path.join(temp_dir, file.filename)

    logger.debug(f"[DEBUG] Received upload request for filename: {file.filename}")

    try:
        # 1) Save uploaded file to disk
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.debug(f"[DEBUG] Saved uploaded file to: {dest_path}")

        # 2) Extract raw text
        raw_text = handle_file(dest_path)
        logger.debug(f"[DEBUG] Raw text extracted (length={len(raw_text)} chars)")

        # 3) If no text, return 204
        if not raw_text.strip():
            logger.debug("[DEBUG] Extracted text is empty after stripping whitespace.")
            raise HTTPException(status_code=204, detail="No text found in uploaded file.")

        # 4) Format with Gemini
        formatted_text = format_with_gemini(raw_text)
        logger.debug(
            f"[DEBUG] Sending formatted text back to client (length={len(formatted_text)} chars)"
        )
        return JSONResponse({"formatted_text": formatted_text})

    except ValueError as ve:
        logger.error(f"[DEBUG] ValueError during extraction: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    except HTTPException as he:
        logger.debug(f"[DEBUG] HTTPException raised: {he.detail}")
        raise

    except Exception as exc:
        logger.exception("[DEBUG] Unexpected error during upload")
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}")

    finally:
        # Cleanup: delete the temp file if it exists
        try:
            if os.path.exists(dest_path):
                os.remove(dest_path)
                logger.debug(f"[DEBUG] Deleted temporary file: {dest_path}")
        except OSError as cleanup_err:
            logger.error(f"[ERROR] Failed to delete temporary file '{dest_path}': {cleanup_err}")


@app.get("/", summary="Health check")
async def root():
    return {
        "status": "ok",
        "supported": [
            "pdf",
            "docx",
            "doc",
            "txt",
            "rtf",
            "html",
            "xls",
            "xlsx",
            "csv",
            "png",
            "jpg",
            "jpeg",
            "tiff",
            "bmp",
            "gif",
            "webp",
            "zip",
        ],
    }


# ─── ENTRYPOINT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
