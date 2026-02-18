import json, re
from pathlib import Path
import yaml
from tqdm import tqdm

def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def split_with_overlap(text: str, chunk_size: int, overlap: int):
    # 基于字符的稳定切分：先按段落合并，再滑窗切
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    merged = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 2 <= chunk_size:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                merged.append(cur)
            cur = p
    if cur:
        merged.append(cur)

    # 对超长段落做兜底切分
    final = []
    for m in merged:
        if len(m) <= chunk_size:
            final.append(m)
        else:
            start = 0
            while start < len(m):
                end = min(len(m), start + chunk_size)
                final.append(m[start:end])
                start = max(end - overlap, start + 1)
    return final

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    in_dir = Path(cfg["interim_text_dir"])
    out_path = Path(cfg["processed_chunks_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_size = cfg["chunking"]["chunk_size"]
    overlap = cfg["chunking"]["chunk_overlap"]
    min_chars = cfg["chunking"]["min_chunk_chars"]

    files = sorted(in_dir.glob("*.json"))
    with open(out_path, "w", encoding="utf-8") as w:
        for fp in tqdm(files, desc="chunk"):
            doc = json.loads(fp.read_text(encoding="utf-8"))
            doc_id = doc["doc_id"]
            meta = doc.get("meta", {})
            source = doc.get("source", "")
            pdf_path = doc.get("pdf_path", "")

            for page_obj in doc["pages"]:
                page = page_obj["page"]
                text = clean_text(page_obj["text"])
                if len(text) < min_chars:
                    continue

                chunks = split_with_overlap(text, chunk_size=chunk_size, overlap=overlap)
                for i, ch in enumerate(chunks):
                    if len(ch) < min_chars:
                        continue
                    rec = {
                        "chunk_id": f"{doc_id}_p{page}_c{i}",
                        "doc_id": doc_id,
                        "source": source,
                        "pdf_path": pdf_path,
                        "page": page,
                        "chunk_index": i,
                        "text": ch,
                        "meta": meta,
                    }
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/pipeline.yaml")
