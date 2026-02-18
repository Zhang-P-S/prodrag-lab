#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT / Base 评测入口（鲁棒版 + 不覆盖输出文件 + 适配你的 configs/sft/*.yaml 格式）

"""
from __future__ import annotations

import argparse
import json
import os
import random
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import yaml

# 复用你项目里的组件
from llm.base import build_llm
from llm.schemas import ChatMessage, GenerateConfig
from rag.retrieval import DualIndexHybridRetriever
def is_likely_english(text: str) -> bool:
    """粗粒度英文判断：用于决定是否需要翻译、以及翻译方向。"""
    if not text:
        return False
    en = sum(ch.isascii() and ch.isalpha() for ch in text)
    return en / max(len(text), 1) > 0.25


def _sigmoid(x: float) -> float:
    # 把 reranker logit 映射到 0~1，阈值（0.5等）才有意义
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def translate_query(llm, query: str, target_lang: str) -> str:
    """用同一个 llm 做轻量翻译（评测用，temperature=0）。"""
    if target_lang == "en":
        instr = "把下面问题翻译成英文，只输出英文，不要解释："
    else:
        instr = "把下面问题翻译成中文，只输出中文，不要解释："

    messages = [
        ChatMessage(role="system", content="你是一个专业翻译器。"),
        ChatMessage(role="user", content=f"{instr}\n{query}"),
    ]
    resp = safe_generate(
        llm,
        messages,
        GenerateConfig(max_tokens=128, temperature=0.0),
    )
    return (resp.text or "").strip()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping/dict, got: {type(data)}")
    return data


def now_ms() -> float:
    return time.time() * 1000.0


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data
# ---------------------------
# 工具：环境变量展开（支持 ${VAR}）
# ---------------------------

_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

def expand_env_vars(obj: Any) -> Any:
    """
    将 YAML 中的 ${DEEPSEEK_API_KEY} 这种占位符替换为 os.environ 里的值。
    这是“工程化习惯”：避免把 key 明文写进仓库。
    """
    if isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env_vars(x) for x in obj]
    if isinstance(obj, str):
        def repl(m):
            name = m.group(1)
            return os.getenv(name, "")
        return _VAR_RE.sub(repl, obj)
    return obj


def write_jsonl_line(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fp.flush()
def write_jsonl_append(p: Path, obj: dict) -> None:
    """评测跑到一半也能看到产出：逐条 append 落盘，避免中途崩了全没了"""
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def strip_think(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.I)
    text = text.replace("```", "")
    return text.strip()


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = strip_think(text)

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start_positions = [m.start() for m in re.finditer(r"\{", s)]
    for st in start_positions:
        depth = 0
        for ed in range(st, len(s)):
            ch = s[ed]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[st : ed + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
    return None


def coerce_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        t = x.strip().lower()
        if t in ("true", "yes", "y", "1"):
            return True
        if t in ("false", "no", "n", "0"):
            return False
    return None


def normalize_citations(cits: Any) -> List[str]:
    if cits is None:
        return []
    if isinstance(cits, str):
        cits = [cits]
    if not isinstance(cits, list):
        return []
    out: List[str] = []
    for it in cits:
        if not it:
            continue
        if isinstance(it, str):
            out.append(it.strip())
    return [c for c in out if c]


def safe_generate(llm, messages, gen_cfg, max_retry=3, sleep_sec=5):
    last_err = None
    for i in range(max_retry):
        try:
            return llm.generate(messages, gen_cfg)
        except Exception as e:
            last_err = e
            print(f"[WARN] LLM timeout, retry {i+1}/{max_retry}: {e}")
            time.sleep(sleep_sec)
    raise last_err


def build_prompt(ctx: str, query: str) -> str:
    return f"""你将基于给定资料回答问题。

要求：
1) 若资料不足以支撑结论，请输出 refusal=true，并用一句话说明缺少什么证据。
2) 无论是否拒答，必须输出 citations（chunk_id 列表），只从资料中选择。
3) 避免编造，不要输出推理过程或多余解释。
4) 只输出一个 JSON 对象，不要使用 Markdown 代码块。

资料：
{ctx}

问题：{query}

输出 JSON 格式：
{{
  "refusal": true/false,
  "answer": "...",
  "citations": ["chunk_id1", "chunk_id2"]
}}
"""


@dataclass
class EvalControls:
    refusal_threshold: float = 0.5
    margin_threshold: float = 0.0
    enable_refusal: bool = True
    autofill_citations: bool = True
    autofill_topk: int = 4


def retrieve_context(
    retriever: DualIndexHybridRetriever,
    question: str,
    *,
    translated_query: Optional[str],
    retrieve_topk: int,
    retrieve_kwargs: Dict[str, Any],
) -> Tuple[str, List[str], Optional[float], Optional[float], Dict[str, Any]]:
    kw = dict(retrieve_kwargs or {})
    kw["rerank_topk"] = int(retrieve_topk)
    kw["return_meta"] = True

    results, meta = retriever.retrieve(question, translated_query=translated_query, **kw)

    retrieved_chunk_ids: List[str] = []
    parts: List[str] = []
    for c in results:
        cid = c.get("chunk_id") or c.get("id") or ""
        txt = c.get("content") or c.get("text") or ""
        if cid:
            retrieved_chunk_ids.append(str(cid))
        parts.append(f"[{cid}]\n{txt}\n")

    ctx = "\n".join(parts).strip()
    top1_score = meta.get("top1_score") if isinstance(meta, dict) else None
    top2_score = meta.get("top2_score") if isinstance(meta, dict) else None
    return ctx, retrieved_chunk_ids, top1_score, top2_score, (meta if isinstance(meta, dict) else {})


def parse_model_output(raw_text: str) -> Tuple[bool, str, List[str], Dict[str, Any]]:
    meta: Dict[str, Any] = {"parsed_json": False, "fallback": False}
    cleaned = strip_think(raw_text)

    obj = extract_first_json_object(cleaned)
    if isinstance(obj, dict):
        meta["parsed_json"] = True
        refusal = coerce_bool(obj.get("refusal"))
        answer = obj.get("answer")
        citations = normalize_citations(obj.get("citations"))

        if not isinstance(answer, str):
            answer = "" if answer is None else str(answer)

        if refusal is None:
            refusal = (answer.strip() == "")
            meta["refusal_inferred"] = True

        return bool(refusal), answer.strip(), citations, meta

    meta["fallback"] = True
    return False, cleaned.strip(), [], meta


def resolve_out_path(cfg: dict) -> str:
    """
    适配你的 paths 配置，并默认避免覆盖：
    - paths.run_out: 可以是文件路径(推荐) 或者目录
    - paths.append_run_id: 默认 True；文件已存在时自动加 _<run_id>
    - paths.overwrite: 默认 False；只有 True 才允许覆盖
    - paths.run_id: 可指定；否则用时间戳
    """
    paths_cfg = cfg.get("paths", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}

    run_out = paths_cfg.get("run_out") or eval_cfg.get("run_out") or cfg.get("run_out")
    out_dir = eval_cfg.get("out_dir", "runs/eval")
    setting = cfg.get("setting") or eval_cfg.get("setting") or "baseline"

    append_run_id = bool(paths_cfg.get("append_run_id", True))
    overwrite = bool(paths_cfg.get("overwrite", False))
    run_id = paths_cfg.get("run_id") or cfg.get("run_id") or time.strftime("%Y-%m-%d_%H-%M-%S")

    # 1) 用户明确给文件路径
    if isinstance(run_out, str) and run_out.strip():
        out_path = run_out.strip()

        # 如果用户传的是目录（不以 .jsonl 结尾），则在目录下自动命名
        if not out_path.lower().endswith(".jsonl"):
            ensure_dir(out_path)
            return os.path.join(out_path, f"{run_id}_{setting}.jsonl")

        # 传的是具体文件：默认不覆盖
        ensure_dir(os.path.dirname(out_path) or ".")
        if os.path.exists(out_path) and (not overwrite):
            if append_run_id:
                base, ext = os.path.splitext(out_path)
                return f"{base}_{run_id}{ext}"
            else:
                raise FileExistsError(
                    f"Output file exists: {out_path}. "
                    f"Set paths.overwrite=true or paths.append_run_id=true."
                )
        return out_path

    # 2) 没给 run_out：落到 out_dir
    ensure_dir(out_dir)
    return os.path.join(out_dir, f"{run_id}_{setting}.jsonl")


def run_one(
    llm,
    retriever: DualIndexHybridRetriever,
    controls: EvalControls,
    q: str,
    *,
    dual_lang: bool,
    retrieve_topk: int,
    gen_cfg: Dict[str, Any],
    retrieve_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    t0 = now_ms()

    t_retrieve0 = now_ms()

    translated_q = None
    if dual_lang:
        # 中文问题 -> 翻成英文去搜 arxiv/en；英文问题 -> 翻成中文补充 zh/pdf
        try:
            if is_likely_english(q):
                translated_q = translate_query(llm, q, target_lang="zh")
            else:
                translated_q = translate_query(llm, q, target_lang="en")
        except Exception:
            translated_q = None

    ctx, retrieved_chunk_ids, top1_score, top2_score, retrieve_meta = retrieve_context(

        retriever, q, translated_query=translated_q, retrieve_topk=retrieve_topk, retrieve_kwargs=retrieve_kwargs
    )
    t_retrieve = now_ms() - t_retrieve0

    prompt = build_prompt(ctx=ctx, query=q)
    messages = [
        ChatMessage(role="system", content="你是一个严谨的问答助手，只根据证据回答。"),
        ChatMessage(role="user", content=prompt),
    ]

    # 生成参数：优先使用 configs/sft/*.yaml 里的 gen
    max_tokens = int(gen_cfg.get("max_tokens", 256))
    temperature = float(gen_cfg.get("temperature", 0.2))

    t_llm0 = now_ms()
    resp = safe_generate(
        llm,
        messages,
        GenerateConfig(max_tokens=max_tokens, temperature=temperature, return_meta=True),
    )
    t_llm = now_ms() - t_llm0

    raw_text = getattr(resp, "text", None) or str(resp)
    model_refusal, answer, used_citations, parse_meta = parse_model_output(raw_text)

    forced_refusal = False
    top1_prob = None
    margin = None

    if controls.enable_refusal:
        if top1_score is None:
            forced_refusal = True
        else:
            top1_score = float(top1_score)
            top1_prob = _sigmoid(top1_score)

            # 1️⃣ 概率阈值
            low_conf = top1_prob < float(controls.refusal_threshold)

            # 2️⃣ margin 判定（需要你从 meta 里拿 top2_score）
            low_margin = False
            if top2_score is not None and float(controls.margin_threshold) > 0:
                margin = top1_score - float(top2_score)
                low_margin = margin < float(controls.margin_threshold)

            forced_refusal = low_conf or low_margin


    final_refusal = (model_refusal or forced_refusal) if controls.enable_refusal else model_refusal

    if controls.autofill_citations and not used_citations and retrieved_chunk_ids:
        used_citations = retrieved_chunk_ids[: max(1, controls.autofill_topk)]
        parse_meta["citations_autofilled"] = True

    if final_refusal and not answer:
        answer = "资料不足以支撑结论。"

    wall_ms = now_ms() - t0

    return {
        "answer": answer,
        "model_refusal": bool(model_refusal),
        "forced_refusal": bool(forced_refusal),
        "final_refusal": bool(final_refusal),
        "used_citations": used_citations,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "top1_score": top1_score,
        "top1_prob": top1_prob,
        "top2_score": top2_score,
        "margin": margin,
        "retrieve_meta": retrieve_meta,
        "translated_query": translated_q,
        "refusal_threshold": controls.refusal_threshold,
        "timing_ms": {"retrieve": t_retrieve, "llm": t_llm, "total": t_retrieve + t_llm},
        "wall_ms": wall_ms,
        "llm_raw": raw_text,
        "llm_parse_meta": parse_meta,
        "llm_meta": getattr(resp, "meta", None),
    }


def main() -> None:
    # 建立参数读取器
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML config path")
    args = ap.parse_args()
    # 读取参数
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = expand_env_vars(cfg)
    # 实例化参数
    setting = cfg.get("setting", "unknown")
    qa_path = Path(cfg["paths"]["qa_jsonl"])
    # 优先支持断点续跑：如果用户指定的 run_out 已存在，就继续写它
    # 否则再使用 resolve_out_path() 生成一个“默认不覆盖”的新路径
    out_path_cfg = Path(cfg["paths"]["run_out"])
    out_path = out_path_cfg if out_path_cfg.exists() else Path(resolve_out_path(cfg))
    index_dir = cfg["paths"]["index_dir"]

    eval_cfg = cfg.get("eval", {}) or {}
    paths_cfg = cfg.get("paths", {}) or {}
    retrieval_cfg = cfg.get("retrieval", {}) or {}
    controls_cfg = cfg.get("controls", {}) or {}
    dual_lang = bool(eval_cfg.get("dual_lang", True))

    seed = int(eval_cfg.get("seed", 42))
    limit = eval_cfg.get("max_items", eval_cfg.get("limit", None))

    retrieve_topk = int(eval_cfg.get("retrieve_topk", retrieval_cfg.get("rerank_topk", 8)))

    controls = EvalControls(
        refusal_threshold=float(controls_cfg.get("refusal_threshold", 0.5)),
        margin_threshold=float(controls_cfg.get("margin_threshold", 0.0)),
        enable_refusal=bool(controls_cfg.get("enable_refusal", True)),
        autofill_citations=bool(controls_cfg.get("autofill_citations", True)),
        autofill_topk=int(controls_cfg.get("autofill_topk", 4)),
    )

    llm_cfg = cfg.get("llm", {}) or {}
    gen_cfg = cfg.get("gen", {}) or {}

    index_root = (
        retrieval_cfg.get("index_root")
        or retrieval_cfg.get("index_dir")
        or paths_cfg.get("index_dir")
        or cfg.get("index_dir")
    )
    if not index_root:
        raise ValueError("Missing config: paths.index_dir (or retrieval.index_dir/index_root)")

    retriever = DualIndexHybridRetriever(
        index_root=index_root,
        embed_model_zh=retrieval_cfg.get("embed_model_zh", "BAAI/bge-small-zh-v1.5"),
        embed_model_en=retrieval_cfg.get("embed_model_en", "BAAI/bge-small-en-v1.5"),
        reranker_name=retrieval_cfg.get("reranker_name", "BAAI/bge-reranker-base"),
    )

    retrieve_kwargs = dict(
        dense_topk=int(retrieval_cfg.get("dense_topk", 30)),
        bm25_topk=int(retrieval_cfg.get("bm25_topk", 30)),
        merge_topk=int(retrieval_cfg.get("merge_topk", 60)),
        fusion_topk=int(retrieval_cfg.get("fusion_topk", 120)),
        rerank_topk=int(retrieval_cfg.get("rerank_topk", 8)),
        enable_rerank=bool(controls_cfg.get("enable_rerank", True)),

        # ✅ 新增：en/zh 候选池与最终 topK 的“保底配额”参数（防止 arxiv 被 pdf 挤出 topK）
        fusion_en_ratio=float(retrieval_cfg.get("fusion_en_ratio", 0.5)),
        rerank_en_ratio=float(retrieval_cfg.get("rerank_en_ratio", 0.5)),
        min_en_candidates=int(retrieval_cfg.get("min_en_candidates", 10)),
        min_zh_candidates=int(retrieval_cfg.get("min_zh_candidates", 10)),
    )

    random.seed(seed)

    qa_items = read_jsonl(qa_path)
    if limit:
        qa_items = qa_items[: int(limit)]

    llm = build_llm(llm_cfg)
    # 清空旧输出（保证可复现）
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_qids = set()
    if out_path.exists():
        for it in read_jsonl(out_path):
            done_qids.add(it["qid"])
        print(f"[eval_run] resume from existing run, done={len(done_qids)}")

    print(f"[eval_run] setting={setting} qa={qa_path} out={out_path} items={len(qa_items)}")

    # 注意：这里仍然是写模式；但 resolve_out_path 已经默认避免覆盖
    for i, item in enumerate(qa_items, 1):
        qid = item.get("qid")
        if qid in done_qids:
            continue
        question = item.get("question")

        gold_citations = item.get("gold_citations", [])

        one = run_one(
            llm,
            retriever,
            controls,
            question,
            dual_lang=bool(eval_cfg.get("dual_lang", True)),
            retrieve_topk=retrieve_topk,
            gen_cfg=gen_cfg,
            retrieve_kwargs=retrieve_kwargs,
        )

        row = {
            "qid": qid,
            "setting": setting,
            "question": question,
            "gold_citations": gold_citations,
            "final_refusal": one["final_refusal"],
            "model_refusal": one["model_refusal"],
            "forced_refusal": one["forced_refusal"],
            "answer": one["answer"],
            "used_citations": one["used_citations"],
            "retrieved_chunk_ids": one["retrieved_chunk_ids"],
            "top1_score": one["top1_score"],
            "refusal_threshold": one["refusal_threshold"],
            "timing_ms": one["timing_ms"],
            "wall_ms": one["wall_ms"],
            "llm_parse_meta": one["llm_parse_meta"],
            "llm_raw": one["llm_raw"],
        }
        write_jsonl_append(out_path, row)

        if i % 10 == 0:
            print(f"[eval_run] {i}/{len(qa_items)} done")
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
