# -*- coding: utf-8 -*-
"""
对比评测：base vs sft vs dpo
- 同一份 test set
- 同一套 retriever（含双语翻译 + 动态配额 + rerank）
- 重要改动：不再要求 7B 输出 JSON/citations（避免格式不稳定）
  - 模型只输出答案正文（或固定拒答短语）
  - citations 由系统根据 rerank top chunks 自动生成（更稳定、可审计、可复现）
输出：
- runs/base.jsonl
- runs/sft.jsonl
- runs/dpo.jsonl
"""

from __future__ import annotations
import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm

from llm.base import build_llm
from llm.schemas import ChatMessage, GenerateConfig
from rag.retrieval import DualIndexHybridRetriever, is_likely_english

_JSON_RE = re.compile(r"\{.*\}", re.S)  # legacy: kept for backward-compat/debug


def read_jsonl(path: Path) -> List[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def append_jsonl(path: Path, items: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def load_llm_cfg_for_build_llm(yaml_path: Path) -> Dict[str, Any]:
    """
    兼容你项目里常见的 configs/rag.yaml 结构：顶层有 llm:
    """
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if isinstance(cfg, dict) and "llm" in cfg and isinstance(cfg["llm"], dict):
        return cfg["llm"]
    if isinstance(cfg, dict):
        return cfg
    raise ValueError(f"Bad llm cfg: {yaml_path}")


def build_translator_llm(llm_cfg: Dict[str, Any]):
    """
    翻译器固定用 deepseek-chat（API）
    - 优先 llm_cfg.api.api_key，否则读 env: DEEPSEEK_API_KEY / OPENAI_API_KEY / API_KEY
    """
    api = llm_cfg.get("api") or {}
    api_key = api.get("api_key") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("No API key for translator. Set DEEPSEEK_API_KEY or llm.api.api_key.")

    translator_cfg = {
        "backend": "api",
        "api": {
            "provider": api.get("provider", "deepseek"),
            "api_key": api_key,
            "model": "deepseek-chat",
            "base_url": api.get("base_url", "https://api.deepseek.com"),
        },
        "local": llm_cfg.get("local", {}) if isinstance(llm_cfg.get("local"), dict) else {},
    }
    return build_llm(translator_cfg)


def clean_translation(text: str, target_lang: str) -> str:
    if not text:
        return ""
    t = text.strip().replace("“", "").replace("”", "").replace('"', "").strip()
    t = t.splitlines()[0].strip()
    if target_lang == "en":
        non_ascii = sum(1 for ch in t if ord(ch) > 127)
        if len(t) > 0 and non_ascii / max(1, len(t)) > 0.25:
            return ""
    return t


def translate_query(query: str, translator_llm, target_lang: str) -> Optional[str]:
    instr = "请把下面问题翻译成英文，只输出英文，不要解释：" if target_lang == "en" else "请把下面问题翻译成中文，只输出中文，不要解释："
    messages = [
        ChatMessage(role="system", content="你是一个专业翻译器。"),
        ChatMessage(role="user", content=f"{instr}\n{query}"),
    ]
    resp = translator_llm.generate(messages, GenerateConfig(max_tokens=128, temperature=0.0))
    out = clean_translation(resp.text or "", target_lang)
    return out if out else None


def detect_evidence_is_english(top_chunks: List[Dict[str, Any]]) -> bool:
    # 只用 chunk 文本做启发式判断：多数为英文则认为英文证据为主
    if not top_chunks:
        return False
    votes, total = 0, 0
    for ch in top_chunks:
        txt = (ch.get("content") or ch.get("text") or "").strip()
        if not txt:
            continue
        total += 1
        if is_likely_english(txt):
            votes += 1
    if total == 0:
        return False
    return (votes / total) >= 0.65


def compute_lang_quota(query: str) -> Dict[str, Any]:
    # 动态配额（避免硬塞噪声）
    if is_likely_english(query):
        return dict(fusion_en_ratio=0.8, rerank_en_ratio=0.8, min_en_candidates=12, min_zh_candidates=0)
    return dict(fusion_en_ratio=0.6, rerank_en_ratio=0.6, min_en_candidates=12, min_zh_candidates=4)

_CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)

def clean_model_answer(raw: str) -> str:
    """
    把模型输出清洗成“答案正文”：
    - 去掉 <think>...</think> 或 </think> 之前的内容
    - 去掉 ```...``` 代码块（含 ```json）
    - 去掉明显的 json 行
    - 去掉 The answer is/答案是 等前缀
    - 最多保留 2 句 + 截断
    """
    s = (raw or "").strip()

    # 1) 处理 think：如果有 </think>，只取其后；如果有 <think>...</think>，删掉整个块
    s = _THINK_BLOCK_RE.sub("", s)
    if "</think>" in s:
        s = s.split("</think>", 1)[1].strip()

    # 2) 去掉代码块（```json ... ```）
    s = _CODE_FENCE_RE.sub("", s).strip()

    # 3) 去掉常见前缀
    for prefix in ["The answer is:", "Answer:", "答案是：", "答案:", "Final Answer:"]:
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()

    # 4) 去掉看起来像 JSON 的行
    lines = []
    for line in s.splitlines():
        t = line.strip()
        if not t:
            continue
        if t.startswith("{") or t.startswith("}") or t.startswith("[") or t.startswith("]"):
            continue
        if t.startswith('"') and t.endswith('",'):
            continue
        if '"status"' in t or '"answers"' in t or '"text"' in t:
            continue
        lines.append(t)
    s = " ".join(lines).strip()

    # 5) 压缩空白
    s = re.sub(r"\s+", " ", s).strip()

    # 6) 最多保留 2 句（防止啰嗦/跑题）
    parts = re.split(r"(?<=[\.\!\?。！？])\s+", s)
    if len(parts) > 2:
        s = " ".join(parts[:2]).strip()

    # 7) 最终截断
    return s[:500]

def build_prompt_answer(query: str, chunks: List[dict], max_chars: int = 4500, allow_english_answer: bool = True) -> str:
    used = 0
    ctx_parts = []
    for i, ck in enumerate(chunks, 1):
        piece = f"[{i}] {ck.get('chunk_id','')} ({ck.get('page','')})\n{ck.get('content','')}\n"
        if used + len(piece) > max_chars:
            break
        ctx_parts.append(piece)
        used += len(piece)
    ctx = "\n".join(ctx_parts).strip()

    lang_rule = (
        "- 若资料足以回答：优先使用与【资料】一致的语言作答（英文证据为主可英文）。\n"
        if allow_english_answer
        else "- 若资料足以回答：请用中文简洁回答（1-2句）。\n"
    )

    return f"""
你是一个严格的证据问答助手。你只能使用【资料】中的信息作答，禁止使用常识、猜测或引入资料之外的内容。

【输出硬性规则】
1) 只输出最终答案正文（最多 2 句），不要输出任何思考过程/推理过程/解释过程。
2) 禁止输出：引用列表、chunk_id、JSON、markdown、代码块、反引号 ```、标签（如 <think> </think>）。
3) 不要输出多余前缀（例如 “The answer is:” / “答案是：” 都不要），直接给出答案内容本身。
4) 若资料不足以回答，必须只输出这句固定短语（不要加任何别的字）：
资料不足以回答该问题。
{lang_rule}
5) 若【资料】为英文：优先沿用原文术语、专有名词、数字与单位，不强制翻译。

【覆盖关键点要求（非常重要）】
- 如果【资料】中出现“枚举/列表/多个并列要点”（例如若干任务、组件、步骤、数字构成），答案必须尽量覆盖这些要点（可以用分号/逗号串起来），但仍然保持最多 2 句。
- 如果问题询问“architecture designed to do / designed for / is designed to … / consists of …”，答案必须包含证据里提到的关键机制/组件/流程（例如 controller、loop、memory、tools、tasks、outputs 等，只要证据中出现就要尽量提到）。
- 不要泛化成一句空话（例如只说“integrate tasks into workflow”而漏掉证据里明确列出的关键点）。

【资料】
{ctx}

【问题】
{query}
""".strip()

# def build_prompt_answer(query: str, chunks: List[dict], max_chars: int = 4500, allow_english_answer: bool = True) -> str:
#     used = 0
#     ctx_parts = []
#     for i, ck in enumerate(chunks, 1):
#         piece = f"[{i}] {ck.get('chunk_id','')} ({ck.get('page','')})\n{ck.get('content','')}\n"
#         if used + len(piece) > max_chars:
#             break
#         ctx_parts.append(piece)
#         used += len(piece)
#     ctx = "\n".join(ctx_parts).strip()

#     # 这里不再要求模型输出 citations/JSON（避免 7B 自由文本导致的格式不稳定）。
#     # citations 由系统侧根据 rerank 的 top chunks 自动产出（更稳定、可审计、可复现）。
#     lang_rule = (
#         "- 若资料足以回答：请用与证据一致的语言回答（英文证据为主时可英文）。\n"
#         if allow_english_answer
#         else "- 若资料足以回答：请用中文简洁回答（1-3句）。\n"
#     )


#     return f"""
#     你是一个严格的证据问答助手。你只能使用【资料】中的信息作答，禁止使用常识、猜测或引入资料之外的内容。

#     【输出硬性规则】
#     1) 只输出“最终答案”一段文本，不要输出任何思考过程/推理过程/解释过程。
#     2) 不要输出：引用列表、chunk_id、JSON、markdown、代码块、标签（如 <think> </think>）。
#     3) 不要输出多余前缀（例如 “The answer is:” 或 “答案是：” 也不要）。直接给出答案内容本身。
#     4) 若资料不足以回答，必须只输出这句固定短语（不要加任何别的字）：
#     资料不足以回答该问题。
#     {lang_rule}
#     5) 若【资料】为英文：优先沿用原文术语、专有名词、数字与单位，不强制翻译。

#     【作答要求】
#     - 尽量简洁：1~2 句，优先给出结论；不要背景介绍、不要复述问题。
#     - 数字/单位/专有名词必须与资料一致。

#     【资料】
#     {ctx}

#     【问题】
#     {query}
#     """.strip()


def choose_system_citations(top_chunks: List[Dict[str, Any]], retrieve_meta: Dict[str, Any], max_k: int = 3) -> List[str]:
    """
    系统侧生成 citations（不依赖模型输出）。

    新策略（更抗 top1 偶发不准）：
    - 至少引用 top1
    - 若 top2/top3 的 rerank_score 与 top1 接近（差值 <= delta），则一并引用（最多 max_k）
    - 去重保序
    """
    if not top_chunks:
        return []

    # 默认容忍阈值：top1 - delta 以内都算“接近”
    delta = 1.6  # 你也可以调 1.0~2.5
    top1_score = None
    try:
        top1_score = float(top_chunks[0].get("rerank_score"))
    except Exception:
        top1_score = None

    k = 1
    if top1_score is not None:
        for i in range(1, min(max_k, len(top_chunks))):
            s = top_chunks[i].get("rerank_score")
            try:
                s = float(s)
            except Exception:
                break
            if top1_score - s <= delta:
                k = i + 1
            else:
                break
    else:
        k = min(max_k, len(top_chunks))

    cits: List[str] = []
    for ch in top_chunks[:k]:
        cid = ch.get("chunk_id")
        if cid:
            cits.append(str(cid))

    # 去重保序
    seen = set()
    out = []
    for c in cits:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out

def run_one_question(
    *,
    question: str,
    llm,
    translator_llm,
    retriever: DualIndexHybridRetriever,
    index_params: dict,
) -> Dict[str, Any]:
    t0 = time.time()

    # 1) translate (for dual-lang retrieve)
    translated = None
    if index_params.get("dual_lang", True):
        translated = translate_query(question, translator_llm, target_lang=("zh" if is_likely_english(question) else "en"))

    # 2) retrieve (with rerank + dynamic quota)
    quota = compute_lang_quota(question)
    top_chunks, rmeta = retriever.retrieve(
        query=question,
        translated_query=translated,
        dense_topk=index_params["dense_topk"],
        bm25_topk=index_params["bm25_topk"],
        merge_topk=index_params["merge_topk"],
        fusion_topk=index_params["fusion_topk"],
        rerank_topk=index_params["rerank_topk"],
        enable_rerank=True,
        return_meta=True,
        **quota,
    )

    # 3) build prompt (answer-only)
    allow_en = detect_evidence_is_english(top_chunks)
    prompt = build_prompt_answer(question, top_chunks, allow_english_answer=allow_en)

    # 4) generate (temperature=0 for stability)
    messages = [
        ChatMessage(role="system", content="你是一个严谨的问答助手，只根据证据回答。"),
        ChatMessage(role="user", content=prompt),
    ]
    resp = llm.generate(
        messages,
        GenerateConfig(max_tokens=index_params["max_tokens"], temperature=0.0, return_meta=True),
    )

    # 5) system-side parse
    # refusal 由固定短语触发；citations 由系统根据 rerank top chunks 生成。
    raw0 = (resp.text or "").strip()
    cleaned = clean_model_answer(raw0)

    # refusal 由固定短语触发（允许模型输出里带点空白/换行）
    refusal = (cleaned == "资料不足以回答该问题。")
    answer = "" if refusal else cleaned

    citations = choose_system_citations(top_chunks, rmeta, max_k=3) if not refusal else []

    parsed = {
        "refusal": refusal,
        "answer": answer,
        "citations": citations,
        # parse_ok 恒 True：因为我们不依赖模型 JSON 解析
        "parse_ok": True,
        "parse_err": None,
        "citation_source": "system_rerank_topk",
    }

    latency_ms = (time.time() - t0) * 1000.0

    return {
        "question": question,
        "translated_query": translated,
        "top_chunks": [{"chunk_id": c.get("chunk_id"), "page": c.get("page"), "rerank_score": c.get("rerank_score")} for c in top_chunks],
        "retrieve_meta": rmeta,
        "llm_raw": resp.text,
        "llm_meta": resp.meta,
        "latency_ms": latency_ms,
        "parsed": parsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=str, required=True, help="data/dpo/qa_raw_test.jsonl")
    ap.add_argument("--llm_cfg", type=str, required=True, help="configs/rag.yaml (contains api/local)")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--lora_sft", type=str, default="")
    ap.add_argument("--lora_dpo", type=str, default="")
    ap.add_argument("--index_dir", type=str, default="data/index")
    ap.add_argument("--out_dir", type=str, default="runs")

    # retriever params
    ap.add_argument("--dense_topk", type=int, default=30)
    ap.add_argument("--bm25_topk", type=int, default=30)
    ap.add_argument("--merge_topk", type=int, default=60)
    ap.add_argument("--fusion_topk", type=int, default=120)
    ap.add_argument("--rerank_topk", type=int, default=8)

    # 由于不再输出 JSON，max_tokens 更偏向“答案完整性”
    ap.add_argument("--max_tokens", type=int, default=512)
    args = ap.parse_args()

    test_items = read_jsonl(Path(args.test))
    llm_cfg = load_llm_cfg_for_build_llm(Path(args.llm_cfg))

    # 统一：translator 永远 api deepseek-chat
    translator_llm = build_translator_llm(llm_cfg)

    # retriever 只初始化一次
    retriever = DualIndexHybridRetriever(index_root=args.index_dir)

    def build_local_llm(lora_path: str) -> Any:
        cfg = dict(llm_cfg)
        cfg["backend"] = "local"
        cfg.setdefault("local", {})
        cfg["local"]["model_path"] = args.model_path
        cfg["local"]["lora_path"] = (lora_path if lora_path else None)
        return build_llm(cfg)

    settings = [
        ("dpo", args.lora_dpo),
    ]
    # settings = [
    #     ("baselocal", ""),
    #     ("sft", args.lora_sft),
    #     ("dpo", args.lora_dpo),
    # ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_params = {
        "dual_lang": True,
        "dense_topk": args.dense_topk,
        "bm25_topk": args.bm25_topk,
        "merge_topk": args.merge_topk,
        "fusion_topk": args.fusion_topk,
        "rerank_topk": args.rerank_topk,
        "max_tokens": args.max_tokens,
    }

    for name, lora in settings:
        llm = build_local_llm(lora)
        out_path = out_dir / f"{name}.jsonl"
        if out_path.exists():
            out_path.unlink()  # 每次重跑清空，避免混杂

        batch = []
        for it in tqdm(test_items, desc=f"run[{name}]"):
            qid = it.get("qid") or it.get("id") or ""
            question = it["question"]
            result = run_one_question(
                question=question,
                llm=llm,
                translator_llm=translator_llm,
                retriever=retriever,
                index_params=index_params,
            )
            result["qid"] = qid
            batch.append(result)

            if len(batch) >=1:
                append_jsonl(out_path, batch)
                batch = []
        if batch:
            append_jsonl(out_path, batch)

        print(f"[ok] wrote: {out_path}")


if __name__ == "__main__":
    main()