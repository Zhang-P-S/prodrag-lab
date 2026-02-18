#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import feedparser
import httpx
from tqdm import tqdm

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}.pdf"

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_filename(s: str) -> str:
    s = re.sub(r"[^\w\-\.]+", "_", s, flags=re.UNICODE)
    return s[:150].strip("_")

def arxiv_search(query: str, start: int, max_results: int, sort_by="submittedDate", sort_order="descending"):
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    url = f"{ARXIV_API}?{urlencode(params)}"
    feed = feedparser.parse(url)
    return feed

def extract_arxiv_id(entry) -> str:
    # entry.id looks like: http://arxiv.org/abs/2401.01234v2
    m = re.search(r"arxiv\.org/abs/([^/]+)$", entry.id)
    if not m:
        return ""
    return m.group(1)

def normalize_id(arxiv_id: str) -> str:
    # keep versionless id for file name consistency
    # 2401.01234v2 -> 2401.01234
    return re.sub(r"v\d+$", "", arxiv_id)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default=None, help="e.g. 2026-01-27_run001")
    ap.add_argument("--out_dir", default="data/raw/arxiv", help="base output dir")
    ap.add_argument("--max_papers", type=int, default=400, help="target count (after de-dupe)")
    ap.add_argument("--page_size", type=int, default=100, help="arXiv API page size (<=300 recommended)")
    ap.add_argument("--sleep", type=float, default=1.0, help="sleep seconds between API pages")
    ap.add_argument("--timeout", type=float, default=30.0)

    # You can pass multiple queries; they'll be combined and de-duped.
    ap.add_argument("--query", action="append", required=True,
                    help='arXiv API query string, e.g. \'(cat:cs.CL OR cat:cs.LG) AND all:"multimodal"\'')
    args = ap.parse_args()

    run_id = args.run_id or (datetime.now().strftime("%Y-%m-%d") + "_run001")
    base = Path(args.out_dir)
    # base = Path(args.out_dir) / run_id
    pdf_dir = base / "pdf"
    meta_dir = base / "meta"
    base.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = base / "manifest.jsonl"

    seen = set()
    records = []

    client = httpx.Client(timeout=args.timeout, headers={"User-Agent": "zps-rag-crawler/1.0 (arXiv API)"} )

    try:
        pbar = tqdm(total=args.max_papers, desc="Collected")
        for q in args.query:
            start = 0
            while len(seen) < args.max_papers:
                feed = arxiv_search(q, start=start, max_results=args.page_size)
                entries = getattr(feed, "entries", []) or []
                if not entries:
                    break

                for e in entries:
                    if len(seen) >= args.max_papers:
                        break

                    arxiv_id_full = extract_arxiv_id(e)
                    if not arxiv_id_full:
                        continue
                    arxiv_id = normalize_id(arxiv_id_full)

                    if arxiv_id in seen:
                        continue
                    seen.add(arxiv_id)

                    title = (e.title or "").replace("\n", " ").strip()
                    summary = (e.summary or "").replace("\n", " ").strip()
                    authors = [a.name for a in getattr(e, "authors", [])] if getattr(e, "authors", None) else []
                    published = getattr(e, "published", None)
                    updated = getattr(e, "updated", None)

                    pdf_url = ARXIV_PDF.format(arxiv_id=arxiv_id)
                    pdf_path = pdf_dir / f"{arxiv_id}.pdf"
                    meta_path = meta_dir / f"{arxiv_id}.json"

                    rec = {
                        "doc_id": f"arxiv:{arxiv_id}",
                        "source": "arxiv",
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "authors": authors,
                        "published": published,
                        "updated": updated,
                        "summary": summary,
                        "query": q,
                        "url_abs": e.id,
                        "url_pdf": pdf_url,
                        "download_path": str(pdf_path.as_posix()),
                        "meta_path": str(meta_path.as_posix()),
                        "sha256": None,
                        "status": "pending",
                        "fetched_at": utc_now_iso(),
                    }

                    # download pdf
                    try:
                        if not pdf_path.exists():
                            r = client.get(pdf_url, follow_redirects=True)

                            r.raise_for_status()
                            pdf_path.write_bytes(r.content)
                        rec["sha256"] = sha256_file(pdf_path)
                        rec["status"] = "ok"
                    except Exception as ex:
                        rec["status"] = "failed"
                        rec["error"] = repr(ex)

                    # write meta
                    meta_payload = {
                        "doc_id": rec["doc_id"],
                        "title": title,
                        "authors": authors,
                        "published": published,
                        "updated": updated,
                        "summary": summary,
                        "url_abs": e.id,
                        "url_pdf": pdf_url,
                        "query": q,
                    }
                    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

                    # append manifest
                    with manifest_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    records.append(rec)
                    pbar.update(1)

                start += args.page_size
                time.sleep(args.sleep)

        pbar.close()
    finally:
        client.close()

    print(f"Done. Saved to: {base}")
    print(f"Manifest: {manifest_path}")
    print(f"Total unique papers: {len(seen)}")

if __name__ == "__main__":
    main()
