# rag/pipeline.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import os
import time
import jieba

from llm.base import build_llm
from llm.schemas import ChatMessage, GenerateConfig, LLMResponse

from .retrieval import DualIndexHybridRetriever, is_likely_english, sigmoid

_JIEBA_INITED = False


# ---------------------------
# 1) 翻译器：永远使用 API deepseek-chat
# ---------------------------

def build_translator_llm(llm_cfg: Dict[str, Any]):
    """
    构建“翻译专用 LLM”：强制走 API deepseek-chat
    - 不影响你原来的生成器（本地7B / API）
    - API key 建议走 env：DEEPSEEK_API_KEY（或你 configs 里已有）
    """
    # 从原 cfg 尽量复用 api_key/base_url/provider
    api_cfg = (llm_cfg.get("api") or {}) if isinstance(llm_cfg.get("api"), dict) else {}
    api_key = api_cfg.get("api_key") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError(
            "[translator_llm] 未找到 API Key。请设置环境变量 DEEPSEEK_API_KEY，或在 llm_cfg.api.api_key 配置。"
        )

    translator_cfg = {
        "backend": "api",
        "api": {
            "provider": api_cfg.get("provider", "deepseek"),
            "api_key": api_key,
            "model": "deepseek-chat",  # ✅ 永远用 deepseek-chat
            "base_url": api_cfg.get("base_url", "https://api.deepseek.com"),
        },
        "local": llm_cfg.get("local", {}) if isinstance(llm_cfg.get("local"), dict) else {},
    }
    return build_llm(translator_cfg)


def clean_translation(text: str, target_lang: str) -> str:
    """
    清洗翻译输出，避免 7B/LLM 夹带解释导致检索 query 变脏：
    - 只取第一行
    - 去引号
    - 粗略过滤（英文目标时，若非 ASCII 比例太高则回退）
    """
    if not text:
        return ""
    t = text.strip().replace("“", "").replace("”", "").replace('"', "").strip()
    t = t.splitlines()[0].strip()

    if target_lang == "en":
        # 简单判断：非 ASCII 太多说明“没翻干净”
        non_ascii = sum(1 for ch in t if ord(ch) > 127)
        if len(t) > 0 and non_ascii / max(1, len(t)) > 0.25:
            return ""
    return t


def translate_query(query: str, translator_llm, target_lang: str) -> str:
    """双路检索翻译器：永远用 translator_llm（API deepseek-chat）"""
    if target_lang == "en":
        instr = "请把下面问题翻译成英文，只输出英文，不要解释："
    else:
        instr = "请把下面问题翻译成中文，只输出中文，不要解释："

    messages = [
        ChatMessage(role="system", content="你是一个专业翻译器。"),
        ChatMessage(role="user", content=f"{instr}\n{query}"),
    ]
    resp: LLMResponse = translator_llm.generate(messages, GenerateConfig(max_tokens=128, temperature=0.0))
    return clean_translation(resp.text, target_lang=target_lang)


# ---------------------------
# 2) 证据语言检测：英文占比高 -> answer 允许英文（训练/评测更稳）
# ---------------------------

def detect_evidence_lang(top_chunks: List[Dict[str, Any]]) -> Tuple[str, float]:
    """
    返回 (dominant_lang, en_ratio)
    - dominant_lang: "en" | "zh" | "mix"
    - en_ratio: 0~1
    """
    if not top_chunks:
        return "mix", 0.5

    en_votes = 0
    total = 0
    for ch in top_chunks:
        txt = (ch.get("text") or ch.get("content") or "").strip()
        if not txt:
            continue
        total += 1
        if is_likely_english(txt):
            en_votes += 1

    if total == 0:
        return "mix", 0.5

    en_ratio = en_votes / total
    if en_ratio >= 0.65:
        return "en", en_ratio
    if en_ratio <= 0.35:
        return "zh", en_ratio
    return "mix", en_ratio


# ---------------------------
# 3) 动态 en/zh 配额策略：根据 query 语言自动调
# ---------------------------

def compute_lang_quota(query: str) -> Dict[str, Any]:
    """
    经验策略（你可以后续按评测再调）：
    - 中文 query：仍需要英文候选（搜 arxiv），但别硬塞太多中文噪声
    - 英文 query：主要英文，中文候选可以为 0（除非你中文索引非常强）
    """
    q_is_en = is_likely_english(query)

    if q_is_en:
        return {
            "fusion_en_ratio": 0.8,
            "rerank_en_ratio": 0.8,
            "min_en_candidates": 12,
            "min_zh_candidates": 0,
        }
    else:
        return {
            "fusion_en_ratio": 0.6,
            "rerank_en_ratio": 0.6,
            "min_en_candidates": 12,
            "min_zh_candidates": 4,
        }


# ---------------------------
# 原 strict prompt（不改也行）
# ---------------------------

def build_rag_messages_strict(query: str, top_chunks: List[Dict[str, Any]]) -> List[ChatMessage]:
    ctx_lines = []
    for i, ch in enumerate(top_chunks, 1):
        cid = ch.get("chunk_id") or ch.get("id") or ch.get("chunk") or f"chunk_{i}"
        text = ch.get("text") or ch.get("content") or ""
        ctx_lines.append(f"[{cid}]\n{text}".strip())

    context_block = "\n\n".join(ctx_lines).strip()

    system = (
        "你是一个企业知识库问答助手（RAG）。\n"
        "规则：\n"
        "1) 只能使用给定的【检索上下文】作答，不允许使用常识补全或编造。\n"
        "2) 如果上下文不足以回答，必须明确拒答，并说明缺少哪些信息。\n"
        "3) 只要回答中出现事实性陈述，必须在句末用 [chunk_id] 形式给出引用；可多个。\n"
        "4) 不要输出与问题无关的内容。\n"
    )

    user = (
        f"【检索上下文】\n{context_block}\n\n"
        f"【用户问题】\n{query}\n\n"
        "请严格按规则回答。"
    )

    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ]


# ---------------------------
# JSON 协议 prompt：新增“证据英文为主可英文回答” + “术语/数字优先沿用原文”
# ---------------------------

def build_prompt(
    query: str,
    chunks: List[dict],
    max_chars: int = 4500,
    answer_lang: str = "zh",  # "zh" | "en" | "auto"
) -> str:
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

    # ✅ 输出语言规则（训练/评测建议用 auto）
    if answer_lang == "en":
        lang_rule = "- 若资料足以回答：refusal=false，answer 用英文简洁回答（1-3句）。\n"
    elif answer_lang == "auto":
        lang_rule = (
            "- 若资料足以回答：refusal=false，answer 允许使用【与证据一致的语言】回答（优先英文证据则可英文）。\n"
        )
    else:
        lang_rule = "- 若资料足以回答：refusal=false，answer 用中文简洁回答（1-3句）。\n"

    return f"""
你是一个严格的证据问答助手。你只能使用【资料】中的信息回答，禁止使用常识或猜测。

【输出硬性规则】
- 你必须只输出一个 JSON 对象，不要输出任何额外文本、不要 markdown、不要代码块、不要 <think>。
- citations 只能从【资料】里出现的 chunk_id 中选择；不得编造或引入未出现的 chunk_id。
- 若资料不足以回答：refusal=true，answer=""，citations=[]（不要解释原因）。
{lang_rule}- citations 列出支持该结论的 chunk_id（1-3个即可）。
- 若【资料】为英文：优先沿用原文术语、专有名词与数字/单位，不强制翻译这些内容（避免翻译导致事实错误）。

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

    # ✅ 新增：训练/评测建议 True（证据英文为主则允许英文回答）
    prefer_evidence_lang: bool = True,
):
    global _JIEBA_INITED
    if not _JIEBA_INITED:
        jieba.initialize()
        _JIEBA_INITED = True

    t0 = time.time()

    # 1) 生成器 LLM：按 llm_cfg（本地7B / API均可）
    t1 = time.time()
    llm = build_llm(llm_cfg)
    print(f"[B] build_llm(generator): {time.time()-t1:.3f}s")

    # 1.5) 翻译器 LLM：强制 API deepseek-chat
    t1b = time.time()
    translator_llm = build_translator_llm(llm_cfg)
    print(f"[B2] build_llm(translator): {time.time()-t1b:.3f}s")

    # 2) 初始化检索器
    t2 = time.time()
    retriever = DualIndexHybridRetriever(index_root=index_dir)
    print(f"[C] init retriever: {time.time()-t2:.3f}s")

    # 3) 双路：翻译 query（永远用 translator_llm）
    t3 = time.time()
    translated = None
    if dual_lang:
        if is_likely_english(query):
            translated = translate_query(query, translator_llm, target_lang="zh")
        else:
            translated = translate_query(query, translator_llm, target_lang="en")

        # 翻译失败回退
        if not translated:
            translated = None
    print(f"[D] translate_query: {time.time()-t3:.3f}s")

    # 4) retrieve：动态配额策略
    quota = compute_lang_quota(query)

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

        # ✅ 动态配额（替换你原来写死的 0.5/10/10）
        fusion_en_ratio=quota["fusion_en_ratio"],
        rerank_en_ratio=quota["rerank_en_ratio"],
        min_en_candidates=quota["min_en_candidates"],
        min_zh_candidates=quota["min_zh_candidates"],
    )
    print(f"[E] retrieve total: {time.time()-t4:.3f}s")

    # 取 top1 分数（logit）与概率（0~1）
    top1_logit = float(rmeta.get("top1_score") or -1e9)
    top1_prob = float(rmeta.get("top1_prob") or sigmoid(top1_logit))

    forced_refusal = top1_prob < float(refusal_threshold)

    citations = [c.get("chunk_id", "") for c in top_chunks if c.get("chunk_id")]

    # 5) prompt + generate（根据证据语言自动调整 answer_lang）
    t5 = time.time()

    dominant_lang, en_ratio = detect_evidence_lang(top_chunks)
    if prefer_evidence_lang and dominant_lang == "en":
        answer_lang = "auto"  # ✅ 允许英文回答
    else:
        answer_lang = "zh"

    prompt = build_prompt(query, top_chunks, answer_lang=answer_lang)
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

        "top1_score": top1_logit,
        "top1_prob": top1_prob,
        "refusal_threshold": float(refusal_threshold),
        "forced_refusal": bool(forced_refusal),
        "retrieve_meta": rmeta,

        "evidence_lang": dominant_lang,
        "evidence_en_ratio": en_ratio,
        "answer_lang_mode": answer_lang,

        "citations": citations,
        "top_chunks": [
            {"chunk_id": c.get("chunk_id"), "page": c.get("page"), "rerank_score": c.get("rerank_score")}
            for c in top_chunks
        ],
        "llm_raw": resp.text,
        "llm_meta": resp.meta,
    }


# ---------------------------
# 流式输出：同样使用 translator_llm + 动态配额 + 证据语言策略（可选）
# ---------------------------

def run_rag_stream(
    query: str,
    llm,
    retriever,
    dual_lang: bool = True,
    dense_topk: int = 30,
    bm25_topk: int = 30,
    merge_topk: int = 60,
    rerank_topk: int = 8,

    # ✅ 新增：传入翻译器（API deepseek-chat）
    translator_llm=None,
):
    # ---------- 1) 双语 query（永远用 translator_llm） ----------
    translated = None
    if dual_lang:
        if translator_llm is None:
            raise RuntimeError("[run_rag_stream] dual_lang=True 但未传 translator_llm（建议用 build_translator_llm 构造）。")

        if is_likely_english(query):
            translated = translate_query(query, translator_llm, target_lang="zh")
        else:
            translated = translate_query(query, translator_llm, target_lang="en")
        if not translated:
            translated = None

    # ---------- 2) retrieve（动态配额，兼容老接口） ----------
    quota = compute_lang_quota(query)
    try:
        top_chunks = retriever.retrieve(
            query=query,
            translated_query=translated,
            dense_topk=dense_topk,
            bm25_topk=bm25_topk,
            merge_topk=merge_topk,
            rerank_topk=rerank_topk,

            fusion_en_ratio=quota["fusion_en_ratio"],
            rerank_en_ratio=quota["rerank_en_ratio"],
            min_en_candidates=quota["min_en_candidates"],
            min_zh_candidates=quota["min_zh_candidates"],
            enable_rerank=True,
        )
    except TypeError:
        # 回退：如果 retrieve 不支持这些参数，就按你原来的方式调用
        top_chunks = retriever.retrieve(
            query=query,
            translated_query=translated,
            dense_topk=dense_topk,
            bm25_topk=bm25_topk,
            merge_topk=merge_topk,
            rerank_topk=rerank_topk,
        )

    # ---------- 3) prompt ----------
    # ⚠️ 流式聊天你现在用的是 build_rag_messages_strict（句末 [chunk_id] 协议）
    # 离线评测强烈建议跑 run_rag_once 的 JSON 协议（更稳更好解析）
    messages = build_rag_messages_strict(query, top_chunks)

    # ---------- 4) 真正的流式输出 ----------
    answer_chunks = []
    for delta in llm.stream_generate(
        messages,
        GenerateConfig(max_tokens=512, temperature=0.2),
    ):
        answer_chunks.append(delta)
        yield delta

    yield "\n"

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