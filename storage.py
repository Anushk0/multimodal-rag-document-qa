"""Shared per-session file storage, used by both app.py (Streamlit) and
api.py (FastAPI) so upload/listing logic isn't duplicated between them."""

from pathlib import Path
from typing import List

UPLOADED_ROOT = Path("uploaded_docs")


def session_upload_dir(session_id: str) -> Path:
    return UPLOADED_ROOT / session_id


def safe_filename(name: str) -> str:
    cleaned = "".join(ch for ch in name if ch.isalnum() or ch in (" ", ".", "_", "-")).strip()
    return cleaned or "document.pdf"


def save_uploaded_pdf_bytes(filename: str, data: bytes, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / safe_filename(filename)

    # Avoid overwriting existing files by adding a numeric suffix.
    if dest.exists():
        stem, suffix = dest.stem, dest.suffix
        i = 1
        while True:
            candidate = upload_dir / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                dest = candidate
                break
            i += 1

    dest.write_bytes(data)
    return dest


def list_uploaded_pdfs(upload_dir: Path) -> List[Path]:
    if not upload_dir.exists():
        return []
    return sorted(p for p in upload_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
