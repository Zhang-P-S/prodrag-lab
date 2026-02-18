# scripts/gen_qa.py
# -*- coding: utf-8 -*-
"""
生成离线评测 QA 数据集：
- 从 data/processed/chunks/chunks.jsonl 抽 chunk
- 调你项目现有 LLM（build_llm）生成问答
- 自动过滤低质量样本，最终保留 200 条

重点：本脚本会把 configs/rag.yaml 里的 llm 配置「整理成 build_llm 期望的结构」
避免你一直遇到的 KeyError: 'api_key'
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# 这两个 import 依赖你项目现有实现（你线上已经跑通了，所以这里也能用）
from llm.base import build_llm
from llm.schemas import ChatMessage, GenerateConfig


# ---------------------------
# 基础 IO：读写 jsonl
# ---------------------------

def read_jsonl(path: Path) -> List[dict]:
    """读取 jsonl：每行一个 JSON"""
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: List[dict]) -> None:
    """写出 jsonl：每行一个 JSON"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


# ---------------------------
# LLM 输出 JSON 抽取（稳健）
# ---------------------------

_JSON_RE = re.compile(r"\{.*\}", re.S)

def extract_json_obj(text: str) -> Optional[dict]:
    """
    有些模型会在 JSON 前后加解释/代码块，这里做一个稳健抽取：
    - 先用正则抓最外层 {...}
    - 再 json.loads
    """
    if not text:
        return None
    s = text.strip()
    m = _JSON_RE.search(s)
    if not m:
        return None
    blob = m.group(0).strip()
    try:
        return json.loads(blob)
    except Exception:
        # 再尝试去掉 ```json ``` 等包裹
        blob2 = blob.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(blob2)
        except Exception:
            return None


# ---------------------------
# 关键：读取并适配 llm 配置
# ---------------------------

def _get_env_api_key() -> Optional[str]:
    """
    推荐你用环境变量注入 key（更安全）：
      export DEEPSEEK_API_KEY="sk-xxx"
    也兼容你可能设置过的 OPENAI_API_KEY
    """
    return (
        os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("API_KEY")
    )


def load_llm_cfg_for_build_llm(cfg_path: Path) -> Dict[str, Any]:
    """
    读取 configs/rag.yaml 或纯 llm.yaml，
    并整理为 build_llm() 一定能吃的结构：

    API 模式必须满足：
      cfg["backend"] == "api"
      cfg["api"]["api_key"] 存在（可以是 env 注入）
    本地模式必须满足：
      cfg["backend"] == "local"
      cfg["local"]["model_path"] 等存在
    """
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"[gen_qa] YAML root must be dict: {cfg_path}")

    # 如果是 rag.yaml：顶层有 llm / rag
    llm = data.get("llm", data)
    if not isinstance(llm, dict):
        raise ValueError(f"[gen_qa] 'llm' must be dict in {cfg_path}")

    backend = (llm.get("backend") or "api").strip()

    # -------- API 模式：严格构造 nested api dict --------
    if backend == "api":
        api = llm.get("api") or {}
        if not isinstance(api, dict):
            api = {}

        # 先从 yaml 取；如果没有就用 env 兜底
        api_key = api.get("api_key") or _get_env_api_key()

        # 这里“强制”让 api_key 变成一个存在的 key（避免 KeyError）
        # 如果最终还是 None/空字符串，我们会给出清晰提示，让你设置 env。
        provider = api.get("provider", "")
        model = api.get("model", "")
        base_url = api.get("base_url", "")

        cfg = {
            "backend": "api",
            "api": {
                "provider": provider,
                "api_key": api_key if api_key is not None else "",
                "model": model,
                "base_url": base_url,
            },
            # 给 local 也留一个（build_llm 可能会访问，但通常不会）
            "local": llm.get("local", {}) if isinstance(llm.get("local", {}), dict) else {},
        }

        # 如果 key 为空，直接报错（比 KeyError 友好一万倍）
        if not cfg["api"]["api_key"]:
            raise RuntimeError(
                "[gen_qa] 没有拿到 API Key。\n"
                "解决方案二选一：\n"
                "1) 在 configs/rag.yaml 的 llm.api.api_key 填入 key（不推荐提交到仓库）；\n"
                "2) 推荐：在 shell 里设置环境变量：\n"
                '   export DEEPSEEK_API_KEY="sk-xxxxxxxx"\n'
                "然后重新运行脚本。"
            )

        return cfg

    # -------- 本地模式：构造 nested local dict --------
    if backend == "local":
        local = llm.get("local") or {}
        if not isinstance(local, dict):
            local = {}

        cfg = {
            "backend": "local",
            "api": llm.get("api", {}) if isinstance(llm.get("api", {}), dict) else {},
            "local": {
                "model_path": local.get("model_path", ""),
                "lora_path": local.get("lora_path", None),
                "dtype": local.get("dtype", "float16"),
                "device": local.get("device", "cuda"),
            },
        }

        if not cfg["local"]["model_path"]:
            raise RuntimeError(
                "[gen_qa] backend=local 但没有 local.model_path。\n"
                "请检查 configs/rag.yaml 的 llm.local.model_path 是否填写正确。"
            )

        return cfg

    raise ValueError(f"[gen_qa] Unknown llm backend: {backend} (in {cfg_path})")


# ---------------------------
# Chunk 规范化与抽样
# ---------------------------

def normalize_chunk(raw: dict) -> Optional[dict]:
    """
    兼容 chunks.jsonl 的字段差异：
    - chunk_id / id
    - content / text
    - meta 可选
    """
    cid = raw.get("chunk_id") or raw.get("id")
    if not cid:
        return None

    content = raw.get("content") or raw.get("text") or ""
    content = str(content).strip()

    # chunk 太短 → 生成 QA 质量会非常差（问题空泛、答案不成句）
    if len(content) < 200:
        return None

    # 可选过滤：明显参考文献/DOI 堆积的 chunk（影响 QA 质量）
    bad_patterns = [r"\bdoi\b", r"参考文献", r"\breferences\b", r"et al\."]
    hit_bad = sum(1 for p in bad_patterns if re.search(p, content, re.I))
    if hit_bad >= 2:
        return None

    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    doc_id = raw.get("doc_id") or meta.get("doc_id") or ""
    page = raw.get("page") or meta.get("page")

    return {
        "chunk_id": str(cid),
        "doc_id": str(doc_id),
        "page": page,
        "content": content,
        "meta": meta,
    }


def sample_pool(chunks: List[dict], pool_size: int, seed: int) -> List[dict]:
    """抽一个候选池，后面轮询生成 QA，避免全量 3W+ chunk 太慢"""
    rnd = random.Random(seed)
    rnd.shuffle(chunks)
    return chunks[:pool_size]


# ---------------------------
# 生成 prompt 与质量过滤
# ---------------------------

def make_prompt(chunk_id: str, content: str) -> str:
    """
    让 LLM 生成 1 条 QA，并强制 citations 只能是当前 chunk_id
    （这样 gold_citations 就是确定的，评测 Recall/MRR 才能算）
    """
    return f"""
你是一个严谨的数据集构建助手。请仅根据给定资料生成 1 条问答样本，要求：
- 问题必须可以被资料直接回答（不需要外部知识）。
- 答案必须简洁、客观、可由资料逐句支撑，避免泛泛而谈。
- citations 必须且只能包含下面给定的 chunk_id（不允许编造其他 id）。
- 输出必须是严格 JSON（不要代码块、不要额外解释）。

给定 chunk_id: {chunk_id}

资料：
{content}

请输出 JSON：
{{
  "question": "...",
  "answer": "...",
  "citations": ["{chunk_id}"],
  "difficulty": "easy|mid|hard"
}}
""".strip()


def is_good_qa(obj: dict, expected_chunk_id: str) -> Tuple[bool, str]:
    """规则过滤：长度、difficulty、引用必须精确匹配 chunk_id、拒答倾向等"""
    if not isinstance(obj, dict):
        return False, "not_dict"

    q = str(obj.get("question", "")).strip()
    a = str(obj.get("answer", "")).strip()
    cits = obj.get("citations", [])
    diff = str(obj.get("difficulty", "")).strip().lower()

    if len(q) < 8 or len(a) < 20:
        return False, "too_short"
    if len(q) > 200 or len(a) > 900:
        return False, "too_long"
    if diff not in {"easy", "mid", "hard"}:
        return False, "bad_difficulty"
    if not isinstance(cits, list) or len(cits) != 1:
        return False, "bad_citations_len"
    if str(cits[0]).strip() != expected_chunk_id:
        return False, "cit_not_match"

    # 过滤掉“拒答式答案”，否则评测时会污染数据集质量
    bad_ans = ["无法确定", "资料不足", "无法回答", "无法从资料中得出", "cannot determine", "insufficient"]
    low = a.lower()
    if any(x.lower() in low for x in bad_ans):
        return False, "refusal_like"

    return True, "ok"


def generate_one(llm, chunk: dict, max_tokens: int = 256) -> Optional[dict]:
    """
    单条生成：
    - 调 LLM
    - 抽 JSON
    - 过滤
    - 返回标准化样本
    """
    prompt = make_prompt(chunk["chunk_id"], chunk["content"])
    messages = [
        ChatMessage(role="system", content="你是一个严谨的问答数据集构建助手。"),
        ChatMessage(role="user", content=prompt),
    ]

    # 温度低一点 → 更稳定输出 JSON
    resp = llm.generate(messages, GenerateConfig(max_tokens=max_tokens, temperature=0.2))
    obj = extract_json_obj(resp.text)
    if obj is None:
        return None

    ok, _ = is_good_qa(obj, chunk["chunk_id"])
    if not ok:
        return None

    # 统一字段：后面 eval_run / eval_metrics 会直接读取
    return {
        "qid": "",  # 之后统一编号
        "question": str(obj["question"]).strip(),
        "gold_answer": str(obj["answer"]).strip(),
        "gold_citations": [chunk["chunk_id"]],
        "difficulty": str(obj["difficulty"]).strip().lower(),
        "source_doc_id": chunk.get("doc_id", ""),
        "source_page": chunk.get("page", None),
        "source_chunk_id": chunk["chunk_id"],
    }


# ---------------------------
# main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, default="data/processed/chunks/chunks.jsonl")
    ap.add_argument("--out_raw", type=str, default="data/sft/qa_raw.jsonl")
    ap.add_argument("--out_keep", type=str, default="data/sft/qa_v1.jsonl")
    ap.add_argument("--llm_cfg", type=str, required=True, help="configs/rag.yaml (or pure llm yaml)")
    ap.add_argument("--n_raw", type=int, default=250)
    ap.add_argument("--n_keep", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool_size", type=int, default=800, help="候选 chunk 池大小，越大越多样，但会慢")
    ap.add_argument("--max_tries", type=int, default=2000, help="最多尝试生成次数（防止质量过滤后不够）")
    ap.add_argument("--max_tokens", type=int, default=256)
    args = ap.parse_args()

    chunks_path = Path(args.chunks)
    raw_out = Path(args.out_raw)
    keep_out = Path(args.out_keep)

    print(f"[gen_qa] loading chunks: {chunks_path}")
    raw_chunks = read_jsonl(chunks_path)

    # 先把 chunk 规范化并做一次粗过滤
    chunks: List[dict] = []
    for r in raw_chunks:
        ck = normalize_chunk(r)
        if ck:
            chunks.append(ck)

    print(f"[gen_qa] valid chunks: {len(chunks)}")
    if not chunks:
        raise RuntimeError("[gen_qa] No valid chunks after filtering. Check chunks.jsonl schema.")

    # 读取 llm 配置并构建模型（关键：适配 build_llm 的结构）
    llm_cfg = load_llm_cfg_for_build_llm(Path(args.llm_cfg))
    llm = build_llm(llm_cfg)

    # 候选池：避免全量 3-4 万 chunk 逐条请求导致时间爆炸
    pool_size = min(len(chunks), args.pool_size)
    pool = sample_pool(chunks[:], pool_size=pool_size, seed=args.seed)

    rnd = random.Random(args.seed)
    results: List[dict] = []
    seen_questions = set()

    tries = 0
    idx = 0
    while len(results) < args.n_raw and tries < args.max_tries:
        tries += 1
        chunk = pool[idx % len(pool)]
        idx += 1

        item = generate_one(llm, chunk, max_tokens=args.max_tokens)
        if not item:
            continue

        # 去重：相同问题不要重复
        qkey = item["question"].strip().lower()
        if qkey in seen_questions:
            continue
        seen_questions.add(qkey)

        results.append(item)

        # 打印进度（让你知道脚本在持续产出）
        if len(results) % 20 == 0:
            print(f"[gen_qa] got {len(results)}/{args.n_raw} (tries={tries})")

        # 如果池子太小导致重复，可以偶尔随机跳一下
        if tries % 200 == 0:
            idx = rnd.randint(0, len(pool) - 1)

    if len(results) < args.n_raw:
        print(f"[gen_qa] WARNING: only generated {len(results)} (<{args.n_raw}). Will proceed anyway.")

    # 原始集编号并落盘
    for i, it in enumerate(results, 1):
        it["qid"] = f"q{i:06d}"
    write_jsonl(raw_out, results)
    print(f"[gen_qa] wrote raw: {raw_out} ({len(results)})")

    # 最终保留：打乱后取前 n_keep
    rnd.shuffle(results)
    kept = results[: min(args.n_keep, len(results))]

    # keep 集合重新编号（保证连续，方便后续 eval）
    for i, it in enumerate(kept, 1):
        it["qid"] = f"q{i:06d}"
    write_jsonl(keep_out, kept)
    print(f"[gen_qa] wrote keep: {keep_out} ({len(kept)})")

    # 最后提示一下：建议你抽检 20-30 条
    print("[gen_qa] 建议：随机抽检 20-30 条 qa_v1_200.jsonl，确保问题/答案确实可由 chunk 支撑。")


if __name__ == "__main__":
    main()
