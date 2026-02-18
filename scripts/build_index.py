import json, os, pickle
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer

def load_chunks(chunks_path: Path):
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    chunks_path = Path(cfg["processed_chunks_path"])
    out_dir = Path(cfg["index"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["embedding"]["model_name"]
    batch_size = cfg["embedding"]["batch_size"]

    chunks = load_chunks(chunks_path)
    texts = [c["text"] for c in chunks]

    model = SentenceTransformer(model_name)
    emb = []
    for i in tqdm(range(0, len(texts), batch_size), desc="embed"):
        batch = texts[i:i+batch_size]
        vec = model.encode(batch, normalize_embeddings=True)  # cosine friendly
        emb.append(vec)
    X = np.vstack(emb).astype("float32")

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)  # 内积 + normalize => cosine
    index.add(X)

    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "docstore.pkl", "wb") as f:
        pickle.dump(chunks, f)

    meta = {
        "model_name": model_name,
        "num_chunks": len(chunks),
        "dim": dim,
        "index_type": "IndexFlatIP",
        "normalized": True,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/pipeline.yaml")
