import os, json, hashlib
from pathlib import Path
import fitz  # PyMuPDF
import yaml
from tqdm import tqdm

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def extract_pdf_text(pdf_path: Path) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc.load_page(i).get_text("text")
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages

def iter_manifest(manifest_path: Path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    out_dir = Path(cfg["interim_text_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for run in cfg["runs"]:
        manifest = Path(run["manifest"])
        pdf_root = Path(run["pdf_root"])

        for item in tqdm(list(iter_manifest(manifest)), desc=f"extract {run['name']}"):
            # 你 manifest 字段可能不同：这里做一个兼容写法
            rel_pdf = item.get("pdf") or item.get("pdf_path") or item.get("path")
            if not rel_pdf:
                raise ValueError(f"manifest item missing pdf path keys: {item.keys()}")

            pdf_path = (pdf_root / rel_pdf) if not str(rel_pdf).endswith(".pdf") else (pdf_root / rel_pdf)
            if not pdf_path.exists():
                # 也可能 manifest 里已经是相对 raw 的路径
                pdf_path = Path(rel_pdf)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            doc_id = item.get("doc_id") or sha1(str(pdf_path))
            pages = extract_pdf_text(pdf_path)

            out = {
                "doc_id": doc_id,
                "source": run["name"],
                "pdf_path": str(pdf_path),
                "meta": {k: v for k, v in item.items() if k not in ["pdf", "pdf_path", "path"]},
                "pages": pages,
            }

            out_path = out_dir / f"{doc_id}.json"
            with open(out_path, "w", encoding="utf-8") as w:
                json.dump(out, w, ensure_ascii=False)

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/pipeline.yaml")
