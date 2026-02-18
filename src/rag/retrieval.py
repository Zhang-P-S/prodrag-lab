# rag/retrieval.py
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import jieba
import torch
import faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -------------------------
# 工具：语言判断（仅用于路由优先级）
# -------------------------
def is_likely_english(text: str) -> bool:
    if not text:
        return False
    # 英文字母占比（非常粗但够用来路由）
    en = sum(ch.isascii() and ch.isalpha() for ch in text)
    return en / max(len(text), 1) > 0.25


def tokenize_en(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def tokenize_zh(text: str) -> List[str]:
    return [w for w in jieba.lcut(text) if w.strip()]


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    # ✅ 用 sigmoid 把 logit 映射到 0~1，阈值才有意义
    return 1.0 / (1.0 + np.exp(-x))


# -------------------------
# Reranker：精排（cross-encoder）
# -------------------------
class Reranker:
    """精排模型：对 (query, doc) 输出相关性分数（logit）"""

    def __init__(self, model_name: str, device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    @torch.no_grad()
    def score(
        self,
        query: str,
        docs: List[str],
        batch_size: int = 64,
        max_length: int = 512,
    ) -> np.ndarray:
        # ✅ 输出是 logit（未做 sigmoid/softmax），保持通用性
        pairs = [[query, d] for d in docs]
        out = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.model(**inputs).logits.view(-1).float()
            out.append(logits.detach().cpu().numpy())
        return np.concatenate(out, axis=0)


# -------------------------
# 双库 Hybrid Retriever：FAISS(en/zh) + BM25(单库) + rerank
# -------------------------
class DualIndexHybridRetriever:
    """
    适配你的 build_kb 产物：
    - FAISS：en/zh 两套各自的 index + docstore
    - BM25：单库（从 bm25.pkl 直接加载）
    """

    def __init__(
        self,
        index_root: str,
        embed_model_zh: str = "BAAI/bge-small-zh-v1.5",
        embed_model_en: str = "BAAI/bge-small-en-v1.5",
        reranker_name: str = "BAAI/bge-reranker-base",
    ):
        self.root = Path(index_root)

        # 1) 加载英文 FAISS
        self.faiss_en_dir = self.root / "faiss_en"
        self.faiss_en = faiss.read_index(str(self.faiss_en_dir / "index.faiss"))
        self.en_chunks = self._load_docstore(self.faiss_en_dir / "docstore.pkl")
        self.en_meta = json.loads((self.faiss_en_dir / "meta.json").read_text(encoding="utf-8"))

        # 2) 加载中文 FAISS
        self.faiss_zh_dir = self.root / "faiss_zh"
        self.faiss_zh = faiss.read_index(str(self.faiss_zh_dir / "index.faiss"))
        self.zh_chunks = self._load_docstore(self.faiss_zh_dir / "docstore.pkl")
        self.zh_meta = json.loads((self.faiss_zh_dir / "meta.json").read_text(encoding="utf-8"))

        # 3) 加载 BM25（单库：bm25.pkl + docstore.pkl）
        self.bm25_dir = self.root / "bm25"
        with open(self.bm25_dir / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        self.bm25_chunks = self._load_docstore(self.bm25_dir / "docstore.pkl")

        # 4) 向量模型（需与建库一致）
        self.embed_zh = SentenceTransformer(embed_model_zh)
        self.embed_en = SentenceTransformer(embed_model_en)

        # 5) reranker
        self.reranker = Reranker(reranker_name)

        # 6) doc_id -> idx 映射（用于把 BM25 idx 映射回 faiss 库）
        self._docid2en = {c.get("chunk_id"): i for i, c in enumerate(self.en_chunks)}
        self._docid2zh = {c.get("chunk_id"): i for i, c in enumerate(self.zh_chunks)}

    def _load_docstore(self, pkl_path: Path) -> List[dict]:
        with open(pkl_path, "rb") as f:
            ds = pickle.load(f)
        if isinstance(ds, list):
            return ds
        if isinstance(ds, dict):
            return list(ds.values())
        raise TypeError(f"Unsupported docstore type: {type(ds)}")

    # -------------------------
    # Dense recall：在指定库上做向量检索
    # -------------------------
    def _dense_recall_on(
        self,
        query: str,
        lang: str,
        topk: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if lang == "en":
            q_emb = self.embed_en.encode([query], normalize_embeddings=True).astype("float32")
            scores, idxs = self.faiss_en.search(q_emb, topk)
            return scores[0], idxs[0]
        else:
            q_emb = self.embed_zh.encode([query], normalize_embeddings=True).astype("float32")
            scores, idxs = self.faiss_zh.search(q_emb, topk)
            return scores[0], idxs[0]

    # -------------------------
    # BM25 recall（单库）
    # -------------------------
    def _bm25_recall(self, query: str, topk: int) -> List[int]:
        # ✅ 与 build 时尽量一致：英文正则分词、中文jieba
        if is_likely_english(query):
            toks = tokenize_en(query)
        else:
            toks = tokenize_zh(query)

        scores = self.bm25.get_scores(toks)
        idxs = np.argsort(scores)[-topk:][::-1]
        return [int(i) for i in idxs]

    def _map_bm25_candidates_to_faiss(self, bm25_idxs: List[int]) -> Tuple[List[int], List[int]]:
        """
        把 BM25 单库召回的 chunk 映射到 en/zh 两个库的索引号：
        - 通过 chunk_id 在 en_chunks / zh_chunks 中定位
        """
        en_ids, zh_ids = [], []
        for i in bm25_idxs:
            ck = self.bm25_chunks[i]
            cid = ck.get("chunk_id")
            if not cid:
                continue
            if cid in self._docid2en:
                en_ids.append(self._docid2en[cid])
            elif cid in self._docid2zh:
                zh_ids.append(self._docid2zh[cid])
        return en_ids, zh_ids

    # -------------------------
    # 统一候选融合：dense(en/zh) + bm25(映射回en/zh)
    # -------------------------
    def _hybrid_candidates(
        self,
        query: str,
        dense_topk: int,
        bm25_topk: int,
        merge_topk: int,
        prefer_lang: str,
    ) -> Tuple[List[int], List[int]]:
        """
        返回： (en_candidate_ids, zh_candidate_ids)
        prefer_lang：优先用哪一库做 dense（另一库也会在双路时补充）
        """
        # 1) dense：优先库
        en_dense, zh_dense = [], []
        if prefer_lang == "en":
            _, en_idxs = self._dense_recall_on(query, "en", dense_topk)
            en_dense = [int(i) for i in en_idxs if int(i) >= 0]
        else:
            _, zh_idxs = self._dense_recall_on(query, "zh", dense_topk)
            zh_dense = [int(i) for i in zh_idxs if int(i) >= 0]

        # 2) bm25：单库召回 -> 映射到两个库
        bm25_idxs = self._bm25_recall(query, bm25_topk)
        en_from_bm25, zh_from_bm25 = self._map_bm25_candidates_to_faiss(bm25_idxs)

        # 3) 去重合并 + 截断（⚠️ 单库上限分别控制，避免某库爆炸）
        def merge(a: List[int], b: List[int], limit: int) -> List[int]:
            out, seen = [], set()
            for x in a + b:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
                if len(out) >= limit:
                    break
            return out

        en_cands = merge(en_dense, en_from_bm25, merge_topk)
        zh_cands = merge(zh_dense, zh_from_bm25, merge_topk)
        return en_cands, zh_cands

    # -------------------------
    # ✅ 改进点核心：候选“配额制”+ 双语 rerank
    # -------------------------
    def retrieve(
        self,
        query: str,
        translated_query: Optional[str] = None,
        dense_topk: int = 30,
        bm25_topk: int = 30,
        merge_topk: int = 60,
        fusion_topk: int = 120,
        rerank_topk: int = 8,
        enable_rerank: bool = True,
        return_meta: bool = False,

        # ✅ 新增：候选配额（防止“中文候选把英文/arxiv挤出rerank候选池”）
        fusion_en_ratio: float = 0.5,   # 候选池里英文占比（0~1）
        rerank_en_ratio: float = 0.5,   # 最终topK里英文占比（0~1）
        min_en_candidates: int = 10,    # 候选池英文最少保底
        min_zh_candidates: int = 10,    # 候选池中文最少保底
    ):
        """
        返回：
          - return_meta=False：List[Dict]
          - return_meta=True：Tuple[List[Dict], Dict]

        关键行为（你现在最需要的）：
          1) en/zh 候选池“配额制”，避免某一库把另一库挤没
          2) 双语 rerank：英文候选优先用 translated_query 打分（跨语更稳）
          3) meta 里提供 top1_score(top1_logit) 与 top1_prob(sigmoid) 方便做拒答阈值
        """
        prefer = "en" if is_likely_english(query) else "zh"

        # A路：原 query 候选
        en_a, zh_a = self._hybrid_candidates(
            query, dense_topk, bm25_topk, merge_topk, prefer_lang=prefer
        )

        # B路：翻译 query 候选（如果有）
        en_b, zh_b = [], []
        if translated_query and translated_query.strip():
            prefer_b = "en" if is_likely_english(translated_query) else "zh"
            en_b, zh_b = self._hybrid_candidates(
                translated_query, dense_topk, bm25_topk, merge_topk, prefer_lang=prefer_b
            )

        # ✅ 去重融合
        def uniq(xs: List[int]) -> List[int]:
            out, seen = [], set()
            for x in xs:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        en_merged = uniq(en_a + en_b)
        zh_merged = uniq(zh_a + zh_b)

        # -------------------------
        # ✅ 候选“配额制”构建（核心修复点）
        # -------------------------
        # 先算总配额：英文多少、中文多少
        en_cap = int(round(fusion_topk * float(fusion_en_ratio)))
        zh_cap = fusion_topk - en_cap

        # 保底：避免 en 或 zh 被挤没
        en_cap = max(en_cap, int(min_en_candidates))
        zh_cap = max(zh_cap, int(min_zh_candidates))

        # 但也要避免超过 fusion_topk 太多（最多允许 +min 的微小膨胀）
        # 这里采用“再收缩一次”的方式，把总长度控制回 fusion_topk
        # （工程上避免 candidates 过大导致 rerank 慢）
        total_cap = en_cap + zh_cap
        if total_cap > fusion_topk:
            # 按比例缩回去（保持相对占比）
            scale = fusion_topk / float(total_cap)
            en_cap = max(1, int(en_cap * scale))
            zh_cap = max(1, fusion_topk - en_cap)

        # 取候选（⚠️ 这里不再“先拼再截断”，而是分别截断）
        en_candidates = [self.en_chunks[i] for i in en_merged[:en_cap]]
        zh_candidates = [self.zh_chunks[i] for i in zh_merged[:zh_cap]]

        # 拼成总候选（顺序：prefer 先，但两边都一定在）
        if prefer == "en":
            candidates = en_candidates + zh_candidates
        else:
            candidates = zh_candidates + en_candidates

        meta = {
            "prefer_lang": prefer,
            "candidate_size": len(candidates),
            "enable_rerank": bool(enable_rerank),
            "top1_score": None,   # logit
            "top1_prob": None,    # sigmoid(logit)
            "en_cap": en_cap,
            "zh_cap": zh_cap,
        }

        if not candidates:
            return ([], meta) if return_meta else []

        # -------------------------
        # ✅ 关闭 rerank：直接返回前 rerank_topk
        # -------------------------
        if not enable_rerank:
            results = []
            for c in candidates[:rerank_topk]:
                item = c.copy()
                item["rerank_score"] = None
                item["rerank_prob"] = None
                results.append(item)
            return (results, meta) if return_meta else results

        # -------------------------
        # ✅ 双语 rerank（核心修复点）
        #   - 英文候选优先用 translated_query（如果 query 是中文）
        #   - 中文候选优先用原 query（如果 query 是中文）
        # -------------------------
        # 选择“中文query/英文query”
        if is_likely_english(query):
            q_en = query
            q_zh = translated_query.strip() if (translated_query and translated_query.strip()) else query
        else:
            q_zh = query
            q_en = translated_query.strip() if (translated_query and translated_query.strip()) else query

        # 分别打分，避免“中文query打英文chunk”导致相关性偏低
        en_scores = None
        zh_scores = None

        if en_candidates:
            en_texts = [c.get("content", "") for c in en_candidates]
            en_scores = self.reranker.score(q_en, en_texts)  # logit
        if zh_candidates:
            zh_texts = [c.get("content", "") for c in zh_candidates]
            zh_scores = self.reranker.score(q_zh, zh_texts)  # logit

        # -------------------------
        # ✅ 在最终 topK 也做“配额制”（防止 top8 全被 pdf 占满）
        # -------------------------
        en_take = int(round(rerank_topk * float(rerank_en_ratio)))
        zh_take = rerank_topk - en_take
        en_take = max(0, min(en_take, rerank_topk))
        zh_take = max(0, min(zh_take, rerank_topk))

        picked: List[dict] = []

        # 先各自取 top
        if en_candidates and en_scores is not None and en_take > 0:
            en_order = np.argsort(en_scores)[::-1][:en_take]
            for j in en_order:
                item = en_candidates[int(j)].copy()
                logit = float(en_scores[int(j)])
                item["rerank_score"] = logit
                item["rerank_prob"] = float(sigmoid(logit))
                picked.append(item)

        if zh_candidates and zh_scores is not None and zh_take > 0:
            zh_order = np.argsort(zh_scores)[::-1][:zh_take]
            for j in zh_order:
                item = zh_candidates[int(j)].copy()
                logit = float(zh_scores[int(j)])
                item["rerank_score"] = logit
                item["rerank_prob"] = float(sigmoid(logit))
                picked.append(item)

        # 如果某一边不够，再用另一边补齐
        if len(picked) < rerank_topk:
            remain = rerank_topk - len(picked)

            # 先把“未被选中”的剩余候选合并起来补
            leftovers: List[tuple[dict, float]] = []

            if en_candidates and en_scores is not None:
                used_ids = {it.get("chunk_id") for it in picked}
                for i, c in enumerate(en_candidates):
                    cid = c.get("chunk_id")
                    if cid in used_ids:
                        continue
                    leftovers.append((c, float(en_scores[i])))

            if zh_candidates and zh_scores is not None:
                used_ids = {it.get("chunk_id") for it in picked}
                for i, c in enumerate(zh_candidates):
                    cid = c.get("chunk_id")
                    if cid in used_ids:
                        continue
                    leftovers.append((c, float(zh_scores[i])))

            # 按 logit 统一排序补齐
            leftovers.sort(key=lambda x: x[1], reverse=True)
            for c, logit in leftovers[:remain]:
                item = c.copy()
                item["rerank_score"] = float(logit)
                item["rerank_prob"] = float(sigmoid(logit))
                picked.append(item)

        # 最终再按分数排序一次（保证输出 topK 的次序正确）
        picked.sort(key=lambda x: float(x.get("rerank_score", -1e9)), reverse=True)
        results = picked[:rerank_topk]

        if results:
            meta["top1_score"] = float(results[0]["rerank_score"])
            meta["top1_prob"] = float(results[0]["rerank_prob"])
        return (results, meta) if return_meta else results
