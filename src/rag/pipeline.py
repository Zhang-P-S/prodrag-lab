# rag/pipeline.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

import time
import jieba

from llm.base import build_llm
from llm.schemas import ChatMessage, GenerateConfig, LLMResponse

from .retrieval import DualIndexHybridRetriever, is_likely_english, sigmoid

_JIEBA_INITED = False


def translate_query(query: str, llm, target_lang: str) -> str:
    """双路检索的翻译器：同一个可插拔 LLM 后端即可（API / 本地都行）"""
    if target_lang == "en":
        instr = "请把下面问题翻译成英文，只输出英文，不要解释："
    else:
        instr = "请把下面问题翻译成中文，只输出中文，不要解释："

    messages = [
        ChatMessage(role="system", content="你是一个专业翻译器。"),
        ChatMessage(role="user", content=f"{instr}\n{query}"),
    ]
    resp: LLMResponse = llm.generate(messages, GenerateConfig(max_tokens=128, temperature=0.0))
    return resp.text.strip()


def build_prompt(query: str, chunks: List[dict], max_chars: int = 4500) -> str:
    """强制 citations + 允许拒答（严格 JSON 协议）"""
    used = 0
    ctx_parts = []
    for i, ck in enumerate(chunks, 1):
        piece = f"[{i}] {ck.get('chunk_id','')} ({ck.get('page','')})\n{ck.get('content','')}\n"
        if used + len(piece) > max_chars:
            break
        ctx_parts.append(piece)
        used += len(piece)

    ctx = "\n".join(ctx_parts).strip()
    return f"""
你是一个严格的证据问答助手。你只能使用【资料】中的信息回答，禁止使用常识或猜测。

【输出硬性规则】
- 你必须只输出一个 JSON 对象，不要输出任何额外文本、不要 markdown、不要代码块、不要 <think>。
- citations 只能从【资料】里出现的 chunk_id 中选择；不得编造或引入未出现的 chunk_id。
- 若资料不足以回答：refusal=true，answer=""，citations=[]（不要解释原因）。
- 若资料足以回答：refusal=false，answer 用中文简洁回答（1-3句），citations 列出支持该结论的 chunk_id（1-3个即可）。

【资料】
{ctx}

【问题】
{query}

现在只输出 JSON：
{{"refusal": true/false, "answer": "...", "citations": ["chunk_id", "..."]}}
""".strip()


def run_rag_once(
    query: str,
    llm_cfg: Dict[str, Any],
    index_dir: str,
    refusal_threshold: float = 0.5,
    dual_lang: bool = True,
    dense_topk: int = 30,
    bm25_topk: int = 30,
    merge_topk: int = 60,
    fusion_topk: int = 120,
    rerank_topk: int = 8,
):
    """
    ✅ 完整 RAG：双路检索 +（配额制）候选融合 +（双语）rerank + 生成
    - 强制：检索阶段保证 en/zh 都进候选池，避免 arxiv 被 pdf 挤掉
    - 强制：forced_refusal 用 sigmoid 概率阈值判断（阈值 0.5 才有意义）
    """
    global _JIEBA_INITED
    if not _JIEBA_INITED:
        jieba.initialize()
        _JIEBA_INITED = True

    t0 = time.time()

    # 1) 构建可插拔 LLM（API / Local / LoRA）
    t1 = time.time()
    llm = build_llm(llm_cfg)
    print(f"[B] build_llm: {time.time()-t1:.3f}s")

    # 2) 初始化检索器（FastAPI 里建议启动时 new 一次，这里 demo 就先这样）
    t2 = time.time()
    retriever = DualIndexHybridRetriever(index_root=index_dir)
    print(f"[C] init retriever: {time.time()-t2:.3f}s")

    # 3) 双路：翻译 query（让中文问题也能搜英文 arxiv）
    t3 = time.time()
    translated = None
    if dual_lang:
        if is_likely_english(query):
            translated = translate_query(query, llm, target_lang="zh")
        else:
            translated = translate_query(query, llm, target_lang="en")
    print(f"[D] translate_query: {time.time()-t3:.3f}s")

    # 4) retrieve（dense+bm25 融合 + 候选配额 + 双语 rerank）
    t4 = time.time()
    top_chunks, rmeta = retriever.retrieve(
        query=query,
        translated_query=translated,
        dense_topk=dense_topk,
        bm25_topk=bm25_topk,
        merge_topk=merge_topk,
        fusion_topk=fusion_topk,
        rerank_topk=rerank_topk,
        enable_rerank=True,
        return_meta=True,

        # ✅ 你可以按需求微调：
        # 候选池 en/zh 各一半，最终 top8 里也尽量保证一半英文（防止全是 pdf）
        fusion_en_ratio=0.5,
        rerank_en_ratio=0.5,
        min_en_candidates=10,
        min_zh_candidates=10,
    )
    print(f"[E] retrieve total: {time.time()-t4:.3f}s")

    # 取 top1 分数（logit）与概率（0~1）
    top1_logit = float(rmeta.get("top1_score") or -1e9)
    top1_prob = float(rmeta.get("top1_prob") or sigmoid(top1_logit))

    # ✅ 阈值判断应该在 0~1 概率空间做
    forced_refusal = top1_prob < float(refusal_threshold)

    citations = [c.get("chunk_id", "") for c in top_chunks if c.get("chunk_id")]

    # 5) prompt + generate
    t5 = time.time()
    prompt = build_prompt(query, top_chunks)
    messages = [
        ChatMessage(role="system", content="你是一个严谨的问答助手，只根据证据回答。"),
        ChatMessage(role="user", content=prompt),
    ]
    resp = llm.generate(
        messages,
        GenerateConfig(max_tokens=512, temperature=0.2, return_meta=True),
    )
    print(f"[F] llm.generate: {time.time()-t5:.3f}s")

    print(f"[Z] total: {time.time()-t0:.3f}s")

    return {
        "query": query,
        "translated_query": translated,

        # 检索/拒答诊断信息（建议写入 runs 便于分析）
        "top1_score": top1_logit,          # logit
        "top1_prob": top1_prob,            # 0~1
        "refusal_threshold": float(refusal_threshold),
        "forced_refusal": bool(forced_refusal),
        "retrieve_meta": rmeta,

        # 证据与输出
        "citations": citations,
        "top_chunks": [
            {"chunk_id": c.get("chunk_id"), "page": c.get("page"), "rerank_score": c.get("rerank_score")}
            for c in top_chunks
        ],
        "llm_raw": resp.text,
        "llm_meta": resp.meta,
    }
# 流式输出
def run_rag_stream(
    query: str,
    llm,
    retriever,
    dual_lang: bool = True,
    dense_topk: int = 30,
    bm25_topk: int = 30,
    merge_topk: int = 60,
    rerank_topk: int = 8,
):
    """
    ✅ 真·流式 RAG（generator）
    - 对外：yield 文本 token
    - 内部：同时收集 answer / citations / top_chunks
    """

    # ---------- 1) 双语 query ----------
    translated = None
    if dual_lang:
        if is_likely_english(query):
            translated = translate_query(query, llm, target_lang="zh")
        else:
            translated = translate_query(query, llm, target_lang="en")

    # ---------- 2) retrieve ----------
    top_chunks = retriever.retrieve(
        query=query,
        translated_query=translated,
        dense_topk=dense_topk,
        bm25_topk=bm25_topk,
        merge_topk=merge_topk,
        rerank_topk=rerank_topk,
    )

    # ---------- 3) prompt ----------
    # prompt = build_prompt(query, top_chunks)

    # messages = [
    #     ChatMessage(role="system", content="你是一个严谨的医学问答助手，只输出最终答案。"),
    #     ChatMessage(role="user", content=prompt),
    # ]
    messages = build_rag_messages_strict(query, top_chunks)

    # ---------- 4) 真正的流式输出 ----------
    answer_chunks = []
    # yield "[DEBUG] start streaming\n"
    for delta in llm.stream_generate(
        messages,
        GenerateConfig(max_tokens=512, temperature=0.2),
    ):
        answer_chunks.append(delta)
        yield delta   # 🔥 关键：对外 yield

    # ---------- 5) 结束标记 ----------
    yield "\n"  # 给 CLI 一个自然的结束换行

    # ---------- 6) 把结构化结果“塞”在 generator 的属性里 ----------
    final_answer = "".join(answer_chunks).strip()

    run_rag_stream.last_result = {
        "answer": final_answer,
        "citations": [c["chunk_id"] for c in top_chunks if c.get("chunk_id")],
        "top_chunks": [
            {
                "chunk_id": c.get("chunk_id"),
                "page": c.get("page"),
                "rerank_score": float(c.get("rerank_score", 0.0)),
            }
            for c in top_chunks
        ],
    }
