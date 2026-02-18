#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG / SFT / Base 评测入口（鲁棒版）
- 兼容你给的 configs/eval/*.yaml（不需要改 YAML）
- 修复 retrieve_context 内部引用未定义变量导致的缩进/运行错误
- 默认不覆盖输出文件（若已存在则自动加 run_id 后缀），并支持断点续跑

用法：
  python src/eval_rag/eval_run_fixed.py --config configs/eval/baseline.yaml

你也可以把本文件内容合并回你原来的 eval_run.py（看下面“关键改动点”）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# 复用你项目里的组件
from llm.base import build_llm
from llm.schemas import ChatMessage, GenerateConfig
from rag.retrieval import DualIndexHybridRetriever


# ---------------------------
# 基础 IO
# ---------------------------
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


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = str(path)
    data: List[Dict[str, Any]] = []
    if not os.path.exists(p):
        return data
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def write_jsonl_append(p: Path, obj: dict) -> None:
    """逐条 append 落盘，避免中途崩了全没了"""
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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


# ---------------------------
# LLM 输出解析
# ---------------------------
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

    # 粗暴但鲁棒：从第一个 '{' 起找最先闭合的 JSON 对象
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

        # refusal 字段缺失时，用“是否空回答”做一个兜底推断
        if refusal is None:
            refusal = (answer.strip() == "")
            meta["refusal_inferred"] = True

        return bool(refusal), answer.strip(), citations, meta

    meta["fallback"] = True
    return False, cleaned.strip(), [], meta


# ---------------------------
# 生成 / 重试
# ---------------------------
def safe_generate(llm, messages, gen_cfg, max_retry=3, sleep_sec=5):
    last_err = None
    for i in range(max_retry):
        try:
            return llm.generate(messages, gen_cfg)
        except Exception as e:
            last_err = e
            print(f"[WARN] LLM error, retry {i+1}/{max_retry}: {e}")
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


# ---------------------------
# 控制项
# ---------------------------
@dataclass
class EvalControls:
    refusal_threshold: float = 0.5
    enable_refusal: bool = True
    autofill_citations: bool = True
    autofill_topk: int = 4


# ---------------------------
# 关键修复点：retrieve_context 不再引用未定义变量
# ---------------------------
def retrieve_context(
    retriever: DualIndexHybridRetriever,
    question: str,
    *,
    prompt_topk: int,
    retrieve_kwargs: Dict[str, Any],
) -> Tuple[str, List[str], Optional[float]]:
    """
    只做一件事：调用 retriever.retrieve 拿 top_chunks + meta，并把 top_chunks 拼成 ctx。
    修复前的问题：函数里直接用了 retrieval_cfg/q/translated/controls 等未定义变量，导致缩进/运行错误。
    """
    kw = dict(retrieve_kwargs or {})
    kw["rerank_topk"] = int(prompt_topk)
    kw["return_meta"] = True

    # translated_query：如果你后续要做“中英双路”，可以在这里接入翻译；
    # 现在先传 None，保持最小可用。
    translated_query = kw.pop("translated_query", None)

    top_chunks, meta = retriever.retrieve(
        query=question,
        translated_query=translated_query,
        dense_topk=int(kw.get("dense_topk", 30)),
        bm25_topk=int(kw.get("bm25_topk", 30)),
        merge_topk=int(kw.get("merge_topk", 60)),
        fusion_topk=int(kw.get("fusion_topk", 120)),
        rerank_topk=int(kw.get("rerank_topk", prompt_topk)),
        enable_rerank=bool(kw.get("enable_rerank", True)),
        return_meta=True,
    )

    retrieved_chunk_ids: List[str] = []
    parts: List[str] = []
    for c in top_chunks:
        cid = c.get("chunk_id") or c.get("id") or ""
        txt = c.get("content") or c.get("text") or ""
        if cid:
            retrieved_chunk_ids.append(str(cid))
        parts.append(f"[{cid}]\n{txt}\n")

    ctx = "\n".join(parts).strip()
    top1_score = meta.get("top1_score") if isinstance(meta, dict) else None
    return ctx, retrieved_chunk_ids, top1_score


def resolve_out_path(cfg: dict) -> str:
    """
    默认避免覆盖：
    - paths.run_out 是具体文件：若存在则加 _<run_id>
    - paths.run_out 是目录：自动命名 <run_id>_<setting>.jsonl
    - paths.overwrite=true 才允许覆盖
    """
    paths_cfg = cfg.get("paths", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}

    run_out = paths_cfg.get("run_out") or eval_cfg.get("run_out") or cfg.get("run_out")
    out_dir = eval_cfg.get("out_dir", "runs/eval")
    setting = cfg.get("setting") or eval_cfg.get("setting") or "baseline"

    append_run_id = bool(paths_cfg.get("append_run_id", True))
    overwrite = bool(paths_cfg.get("overwrite", False))
    run_id = paths_cfg.get("run_id") or cfg.get("run_id") or time.strftime("%Y-%m-%d_%H-%M-%S")

    # 1) 用户明确给 run_out
    if isinstance(run_out, str) and run_out.strip():
        out_path = run_out.strip()

        # 目录：在目录下自动命名
        if not out_path.lower().endswith(".jsonl"):
            ensure_dir(out_path)
            return os.path.join(out_path, f"{run_id}_{setting}.jsonl")

        # 文件：默认不覆盖
        ensure_dir(os.path.dirname(out_path) or ".")
        if os.path.exists(out_path) and (not overwrite):
            if append_run_id:
                base, ext = os.path.splitext(out_path)
                return f"{base}_{run_id}{ext}"
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
    prompt_topk: int,
    gen_cfg: Dict[str, Any],
    retrieve_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    t0 = now_ms()

    # 1) 检索
    t_retrieve0 = now_ms()
    ctx, retrieved_chunk_ids, top1_score = retrieve_context(
        retriever, q, prompt_topk=prompt_topk, retrieve_kwargs=retrieve_kwargs
    )
    t_retrieve = now_ms() - t_retrieve0

    # 2) 生成
    prompt = build_prompt(ctx=ctx, query=q)
    messages = [
        ChatMessage(role="system", content="你是一个严谨的问答助手，只根据证据回答。"),
        ChatMessage(role="user", content=prompt),
    ]

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

    # 3) 强制拒答（可选）
    forced_refusal = False
    if controls.enable_refusal:
        if top1_score is None:
            forced_refusal = True
        else:
            forced_refusal = (top1_score < controls.refusal_threshold)

    final_refusal = (model_refusal or forced_refusal) if controls.enable_refusal else model_refusal

    # 4) citations 兜底（可选）
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
        "refusal_threshold": controls.refusal_threshold,
        "timing_ms": {"retrieve": t_retrieve, "llm": t_llm, "total": t_retrieve + t_llm},
        "wall_ms": wall_ms,
        "llm_raw": raw_text,
        "llm_parse_meta": parse_meta,
        "llm_meta": getattr(resp, "meta", None),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML config path")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = expand_env_vars(cfg)

    setting = cfg.get("setting", "unknown")
    paths_cfg = cfg.get("paths", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}
    retrieval_cfg = cfg.get("retrieval", {}) or {}
    controls_cfg = cfg.get("controls", {}) or {}

    qa_path = Path(paths_cfg["qa_jsonl"])
    index_root = paths_cfg.get("index_dir") or retrieval_cfg.get("index_root") or retrieval_cfg.get("index_dir")
    if not index_root:
        raise ValueError("Missing config: paths.index_dir (or retrieval.index_root/index_dir)")

    out_path = Path(resolve_out_path(cfg))

    seed = int(eval_cfg.get("seed", 42))
    limit = eval_cfg.get("max_items", eval_cfg.get("limit", None))

    # 这个值对应你 YAML 里的 retrieval.prompt_topk（最终拼 prompt 的 chunk 数）
    prompt_topk = int(retrieval_cfg.get("prompt_topk", 8))

    controls = EvalControls(
        refusal_threshold=float(controls_cfg.get("refusal_threshold", 0.5)),
        enable_refusal=bool(controls_cfg.get("enable_refusal", True)),
        autofill_citations=bool(controls_cfg.get("autofill_citations", True)),
        autofill_topk=int(controls_cfg.get("autofill_topk", 4)),
    )

    llm_cfg = cfg.get("llm", {}) or {}
    gen_cfg = cfg.get("gen", {}) or {}

    retriever = DualIndexHybridRetriever(
        index_root=index_root,
        embed_model_zh=retrieval_cfg.get("embed_model_zh", "BAAI/bge-small-zh-v1.5"),
        embed_model_en=retrieval_cfg.get("embed_model_en", "BAAI/bge-small-en-v1.5"),
        reranker_name=retrieval_cfg.get("reranker_name", "BAAI/bge-reranker-base"),
    )

    # 统一把检索参数放在这里，retrieve_context 只负责消费它们
    retrieve_kwargs = dict(
        dense_topk=int(retrieval_cfg.get("dense_topk", 30)),
        bm25_topk=int(retrieval_cfg.get("bm25_topk", 30)),
        merge_topk=int(retrieval_cfg.get("merge_topk", 60)),
        fusion_topk=int(retrieval_cfg.get("fusion_topk", 120)),
        rerank_topk=int(retrieval_cfg.get("rerank_topk", 8)),  # 即使不 rerank，也给一个默认
        enable_rerank=bool(controls_cfg.get("enable_rerank", True)),
    )

    random.seed(seed)

    qa_items = read_jsonl(qa_path)
    if limit:
        qa_items = qa_items[: int(limit)]

    llm = build_llm(llm_cfg)

    # 断点续跑：如果 out_path 已存在，则跳过已完成的 qid
    done_qids = set()
    if out_path.exists():
        for it in read_jsonl(out_path):
            if "qid" in it:
                done_qids.add(it["qid"])
        print(f"[eval_run] resume from existing run, done={len(done_qids)}")

    print(f"[eval_run] setting={setting} qa={qa_path} out={out_path} items={len(qa_items)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(qa_items, 1):
        qid = item.get("qid")
        if qid in done_qids:
            continue

        question = item.get("question", "")
        gold_citations = item.get("gold_citations", []) or []

        one = run_one(
            llm,
            retriever,
            controls,
            question,
            prompt_topk=prompt_topk,
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
