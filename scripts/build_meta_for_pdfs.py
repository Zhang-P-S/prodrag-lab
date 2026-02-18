#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def guess_title_from_filename(stem: str) -> str:
    # stem is filename without suffix
    t = stem.replace("_", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t

def quick_lang_guess(text: str) -> str:
    # very rough: count CJK vs ascii letters
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    if cjk > latin * 2 and cjk > 50:
        return "zh"
    if latin > cjk * 2 and latin > 50:
        return "en"
    return "mixed_or_unknown"

def extract_pdf_firstpage_text(pdf_path: Path, max_chars: int = 4000) -> str:
    try:
        with fitz.open(pdf_path) as doc:
            if doc.page_count == 0:
                return ""
            text = doc.load_page(0).get_text("text") or ""
            text = text.strip()
            return text[:max_chars]
    except Exception:
        return ""

def extract_basic_pdf_info(pdf_path: Path):
    try:
        with fitz.open(pdf_path) as doc:
            return {
                "page_count": doc.page_count,
                "metadata": doc.metadata or {},
            }
    except Exception as e:
        return {
            "page_count": None,
            "metadata": {},
            "open_error": repr(e),
        }

def best_title(pdf_path: Path, filename_title: str, pdf_meta: dict, first_text: str) -> str:
    # Prefer PDF embedded title if it looks real
    meta_title = (pdf_meta.get("metadata") or {}).get("title") or ""
    meta_title = meta_title.strip()
    if 3 <= len(meta_title) <= 200 and not meta_title.lower().startswith("microsoft word"):
        return meta_title

    # Try first non-empty line from first page as title candidate (common for guidelines)
    if first_text:
        for line in first_text.splitlines():
            line = line.strip()
            if 6 <= len(line) <= 80:
                # avoid lines that look like dates/page numbers
                if not re.fullmatch(r"[\d\W_]+", line):
                    return line

    return filename_title

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="data/raw/zh/pdf", help="directory containing PDFs")
    ap.add_argument("--out_dir", default="data/raw/zh", help="base dir to write meta/ and manifest.jsonl")
    ap.add_argument("--run_id", default=None, help="e.g. 2026-01-28_meta_run001")
    ap.add_argument("--glob", default="*.pdf", help="glob pattern inside input_dir")
    args = ap.parse_args()

    run_id = args.run_id or (datetime.now().strftime("%Y-%m-%d") + "_meta_run001")
    input_dir = Path(args.input_dir)
    base = Path(args.out_dir) 
    meta_dir = base / "meta"
    base.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = base / "manifest.jsonl"

    pdfs = sorted(input_dir.glob(args.glob))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {input_dir} with pattern {args.glob}")

    n_ok = 0
    with manifest_path.open("w", encoding="utf-8") as mf:
        for pdf in pdfs:
            file_title = guess_title_from_filename(pdf.stem)
            sha = sha256_file(pdf)

            basic = extract_basic_pdf_info(pdf)
            first_text = extract_pdf_firstpage_text(pdf)
            lang = quick_lang_guess(first_text)

            title = best_title(pdf, file_title, basic, first_text)

            rec = {
                "doc_id": f"pdf:{sha[:16]}",
                "source": "local_pdf",
                "title": title,
                "title_from": "pdf_meta|first_page|filename",
                "language": lang,
                "sha256": sha,
                "file_name": pdf.name,
                "download_path": str(pdf.as_posix()),
                "page_count": basic.get("page_count"),
                "pdf_metadata": basic.get("metadata", {}),
                "created_at": utc_now_iso(),
                "status": "ok" if sha else "failed",
            }
            if "open_error" in basic:
                rec["status"] = "failed"
                rec["error"] = basic["open_error"]

            # write per-doc meta
            meta_path = meta_dir / f"{sha[:16]}.json"
            meta_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

            # write manifest line
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if rec["status"] == "ok":
                n_ok += 1

    print(f"Done. PDFs: {len(pdfs)}, OK: {n_ok}")
    print(f"Meta dir: {meta_dir}")
    print(f"Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
