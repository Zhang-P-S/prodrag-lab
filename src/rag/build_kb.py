"""
构建知识库流水线（按数据源强制区分中英文）：
1) 抽取PDF文字层 -> data/interim/text/<doc_id>.json
2) 清洗并分块 -> data/processed/chunks/chunks.jsonl（含 lang 字段）
3) 建索引并持久化：
   - FAISS 英文库：data/index/faiss_en/
   - FAISS 中文库：data/index/faiss_zh/
   - BM25 单库：data/index/bm25/

核心原则：
- 不做“内容语言检测”
- lang 由 build.yaml 的 runs[*].lang 决定（例如 arxiv=en，zh_techreports=zh）
"""

import json
import re
import pickle
import hashlib
from pathlib import Path
from datetime import datetime

import fitz  # PyMuPDF：抽取PDF文字层（非OCR）
import yaml
from tqdm import tqdm

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# BM25依赖：pip install rank_bm25
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

# 中文分词可选：pip install jieba
try:
    import jieba
except Exception:
    jieba = None


# =============================
# 通用工具
# =============================

def sha1(s: str) -> str:
    """当manifest没有doc_id时，用路径sha1生成稳定doc_id"""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def now_iso() -> str:
    """统一写UTC时间到meta，便于版本追踪"""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def read_yaml(path: str) -> dict:
    """读取YAML配置"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_mkdir(p: Path):
    """确保目录存在"""
    p.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: Path):
    """流式读取JSONL，避免一次性加载占内存"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)


def write_json(path: Path, obj: dict):
    """写JSON（带缩进）"""
    with open(path, "w", encoding="utf-8") as w:
        json.dump(obj, w, ensure_ascii=False, indent=2)


def normalize_whitespace(s: str) -> str:
    """基础清洗：去掉空字符、规范空行与换行"""
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# =============================
# Stage 1：抽取PDF文本
# =============================

def extract_pdf_text(pdf_path: Path, max_pages: int | None = None) -> list[dict]:
    """
    从PDF抽取每页文本：
    - 只对有文字层的PDF有效
    - 扫描版（图片PDF）会抽不到，需要OCR（此脚本不做）
    """
    doc = fitz.open(pdf_path)
    pages = []

    n = len(doc)
    if max_pages is not None:
        n = min(n, max_pages)

    for i in range(n):
        text = doc.load_page(i).get_text("text")
        pages.append({"page": i + 1, "text": text})

    doc.close()
    return pages


def resolve_pdf_path(pdf_root: Path, item: dict) -> Path | None:
    """
    兼容不同manifest字段：
    - pdf / pdf_path / download_path / path
    支持绝对路径与相对路径
    """
    rel_pdf = (
        item.get("pdf")
        or item.get("pdf_path")
        or item.get("download_path")
        or item.get("file_path")
        or item.get("path")
    )
    if not rel_pdf:
        return None

    p = Path(rel_pdf)

    # manifest给了绝对路径且存在
    if p.is_absolute() and p.exists():
        return p

    # 相对路径：默认相对pdf_root
    cand = pdf_root / rel_pdf
    if cand.exists():
        return cand

    # 兜底：相对当前工作目录
    if p.exists():
        return p

    return None


def main_extract_text(cfg: dict):
    """
    遍历runs：
    - 读取manifest
    - 定位pdf
    - 抽取文字
    - 写入 interim_text_dir
    重点：把 run.lang 写进 doc（后续chunk阶段直接继承）
    """
    out_dir = Path(cfg["interim_text_dir"])
    safe_mkdir(out_dir)

    max_pages = cfg.get("extract", {}).get("max_pages", None)

    for run in cfg["runs"]:
        run_name = run["name"]
        run_lang = run.get("lang")  # ⭐由配置指定语言
        if run_lang not in ("en", "zh"):
            raise ValueError(f"runs[{run_name}] 缺少或错误的 lang（只能 en/zh）：{run_lang}")

        manifest = Path(run["manifest"])
        pdf_root = Path(run["pdf_root"])

        for item in tqdm(iter_jsonl(manifest), desc=f"抽取文本 {run_name}"):
            pdf_path = resolve_pdf_path(pdf_root, item)
            if pdf_path is None:
                print(f"[WARN] manifest缺少pdf字段：keys={list(item.keys())}")
                continue
            if not pdf_path.exists():
                print(f"[WARN] 缺失PDF：{pdf_path} doc_id={item.get('doc_id')}")
                continue

            doc_id = item.get("doc_id") or sha1(str(pdf_path))
            pages = extract_pdf_text(pdf_path, max_pages=max_pages)

            out = {
                "doc_id": doc_id,
                "source": run_name,
                "lang": run_lang,      # ⭐关键：语言来自run配置
                "pdf_path": str(pdf_path),
                "meta": {k: v for k, v in item.items()
                         if k not in ["pdf", "pdf_path", "download_path", "path"]},
                "pages": pages,
            }

            out_path = out_dir / f"{doc_id}.json"
            with open(out_path, "w", encoding="utf-8") as w:
                json.dump(out, w, ensure_ascii=False)


# =============================
# Stage 2：分块（继承lang）
# =============================

def split_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    按字符分块（简单稳）：
    1) 先按段落(\n\n)合并到chunk_size附近
    2) 对超长块用滑窗切分（带overlap）
    """
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

    final = []
    for m in merged:
        if len(m) <= chunk_size:
            final.append(m)
        else:
            start = 0
            while start < len(m):
                end = min(len(m), start + chunk_size)
                final.append(m[start:end])
                # 避免overlap太大导致停滞
                start = max(end - overlap, start + 1)

    return final


def main_chunk(cfg: dict):
    """
    读取 interim_text_dir 的每篇文档：
    - 清洗每页文本
    - 分块
    - 写 chunks.jsonl
    重点：chunk.lang 直接继承 doc.lang（不做内容检测）
    """
    in_dir = Path(cfg["interim_text_dir"])
    out_path = Path(cfg["processed_chunks_path"])
    safe_mkdir(out_path.parent)

    chunk_size = int(cfg["chunking"]["chunk_size"])
    overlap = int(cfg["chunking"]["chunk_overlap"])
    min_chars = int(cfg["chunking"]["min_chunk_chars"])

    files = sorted(in_dir.glob("*.json"))
    with open(out_path, "w", encoding="utf-8") as w:
        for fp in tqdm(files, desc="分块 chunk"):
            doc = json.loads(fp.read_text(encoding="utf-8"))
            doc_id = doc["doc_id"]
            source = doc.get("source", "")
            pdf_path = doc.get("pdf_path", "")
            meta = doc.get("meta", {})

            # ⭐关键：语言由run配置写入doc，这里直接拿来用
            doc_lang = doc.get("lang")
            if doc_lang not in ("en", "zh"):
                raise ValueError(f"doc缺少lang或非法：doc_id={doc_id}, lang={doc_lang}")

            for page_obj in doc.get("pages", []):
                page = page_obj.get("page")
                text = normalize_whitespace(page_obj.get("text", ""))

                # 太短通常是目录/页眉页脚噪声，直接跳过
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
                        "lang": doc_lang,     # ⭐继承，不检测
                        "pdf_path": pdf_path,
                        "page": page,
                        "chunk_index": i,
                        "content": ch,
                        "meta": meta,
                    }
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_chunks(chunks_path: Path) -> list[dict]:
    """加载chunks.jsonl（建索引会用到）"""
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                chunks.append(json.loads(s))
    return chunks


# =============================
# Stage 3：BM25（单库）
# =============================

def tokenize_for_bm25(text: str, mode: str = "auto") -> list[str]:
    """
    BM25分词策略：
    - auto：如果装了jieba且文本含中文，jieba会更适合；否则走英文token
    说明：你现在“按run分中英”，但BM25单库仍可用auto分词兼容
    """
    t = normalize_whitespace(text).lower()

    if mode == "zh":
        if jieba is None:
            # 没装jieba就退化为bigram（能用但不如jieba）
            t2 = re.sub(r"\s+", "", t)
            if len(t2) <= 2:
                return [t2] if t2 else []
            return [t2[i:i + 2] for i in range(len(t2) - 1)]
        return [w.strip() for w in jieba.lcut(t) if w.strip()]

    if mode == "en":
        return re.findall(r"[a-z0-9]+", t)

    # auto：简单策略——如果有jieba且含中文就按zh，否则按en
    if jieba is not None and re.search(r"[\u4e00-\u9fff]", t):
        return [w.strip() for w in jieba.lcut(t) if w.strip()]
    return re.findall(r"[a-z0-9]+", t)


def build_bm25(cfg: dict, chunks: list[dict]):
    """构建BM25单库并持久化"""
    if not cfg["index"]["bm25"].get("enabled", True):
        return

    if BM25Okapi is None:
        raise RuntimeError("你开启了BM25，但未安装rank_bm25：pip install rank_bm25")

    out_dir = Path(cfg["index"]["bm25"]["out_dir"])
    safe_mkdir(out_dir)

    mode = cfg.get("bm25", {}).get("tokenizer", "auto")
    k1 = float(cfg.get("bm25", {}).get("k1", 1.5))
    b = float(cfg.get("bm25", {}).get("b", 0.75))

    tokenized = []
    for c in tqdm(chunks, desc="BM25分词 tokenize(bm25)"):
        tokenized.append(tokenize_for_bm25(c["content"], mode=mode))

    bm25 = BM25Okapi(tokenized, k1=k1, b=b)

    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    # docstore用于把检索到的doc_id/页码/内容返回给RAG
    with open(out_dir / "docstore.pkl", "wb") as f:
        pickle.dump(chunks, f)

    meta = {
        "built_at": now_iso(),
        "num_chunks": len(chunks),
        "bm25": {"tokenizer": mode, "k1": k1, "b": b},
        "chunking": cfg["chunking"],
    }
    write_json(out_dir / "meta.json", meta)


# =============================
# Stage 3：FAISS（中英双库）
# =============================

def build_faiss_one(cfg: dict, chunks: list[dict], lang: str):
    """
    构建某一种语言的FAISS索引：
    - lang=en 使用 embedding.en
    - lang=zh 使用 embedding.zh
    """
    emb_cfg = cfg["embedding"][lang]
    model_name = emb_cfg["model_name"]
    batch_size = int(emb_cfg["batch_size"])
    normalize = bool(emb_cfg.get("normalize_embeddings", True))

    out_dir = Path(cfg["index"]["faiss"][lang]["out_dir"])
    safe_mkdir(out_dir)

    texts = [c["content"] for c in chunks]
    if len(texts) == 0:
        print(f"[INFO] {lang} chunks为空，跳过FAISS构建")
        return

    # 注意：首次运行会下载模型（需要网络）；建议在环境里先cache
    model = SentenceTransformer(model_name)

    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"向量化 embed(faiss_{lang})"):
        batch = texts[i:i + batch_size]
        vec = model.encode(batch, normalize_embeddings=normalize)
        embs.append(vec)

    X = np.vstack(embs).astype("float32")
    dim = int(X.shape[1])

    # normalize=True时：内积≈余弦相似度
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    faiss.write_index(index, str(out_dir / "index.faiss"))

    # 为每个语言库保存自己的docstore（便于检索后定位对应chunks）
    with open(out_dir / "docstore.pkl", "wb") as f:
        pickle.dump(chunks, f)

    meta = {
        "built_at": now_iso(),
        "lang": lang,
        "model_name": model_name,
        "num_chunks": len(chunks),
        "dim": dim,
        "index_type": "IndexFlatIP",
        "normalize_embeddings": normalize,
        "chunking": cfg["chunking"],
    }
    write_json(out_dir / "meta.json", meta)


def build_faiss_bilingual(cfg: dict, chunks: list[dict]):
    """按chunk.lang把数据切成en/zh两份，然后分别建FAISS索引"""
    if not cfg["index"]["faiss"].get("enabled", True):
        return

    chunks_en = [c for c in chunks if c.get("lang") == "en"]
    chunks_zh = [c for c in chunks if c.get("lang") == "zh"]

    if cfg["index"]["faiss"]["en"].get("enabled", True):
        build_faiss_one(cfg, chunks_en, "en")
    else:
        print("[INFO] FAISS(en) disabled")

    if cfg["index"]["faiss"]["zh"].get("enabled", True):
        build_faiss_one(cfg, chunks_zh, "zh")
    else:
        print("[INFO] FAISS(zh) disabled")


def main_index(cfg: dict):
    """读取chunks后：先FAISS双库，再BM25单库"""
    chunks_path = Path(cfg["processed_chunks_path"])
    chunks = load_chunks(chunks_path)

    build_faiss_bilingual(cfg, chunks)
    build_bm25(cfg, chunks)


# =============================
# pipeline入口
# =============================

def main(cfg_path: str):
    cfg = read_yaml(cfg_path)

    steps = cfg.get("pipeline", {})
    if steps.get("extract", True):
        main_extract_text(cfg)
    if steps.get("chunk", True):
        main_chunk(cfg)
    if steps.get("index", True):
        main_index(cfg)


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/build.yaml"
    main(cfg_path)
