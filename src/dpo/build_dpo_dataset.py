# -*- coding: utf-8 -*-
"""
一键生成 3 组数据：
1) data/dpo/qa_raw.jsonl      (标准 raw：question / gold_answer / gold_citations / ... )
2) data/dpo/qa_sft.jsonl      (ShareGPT SFT：conversations + meta)
3) data/dpo/qa_dpo.jsonl      (DPO：conversations + chosen/rejected)

并额外生成 100 条测试集：
- qa_raw_test.jsonl / qa_sft_test.jsonl / qa_dpo_test.jsonl

设计目标：
- 单文件实现，少折腾
- 支持断点续跑（文件存在自动续写）
- 进度条可见，网络波动不白跑
- 生成“可控坏样本”用于训练引用准确性 & 拒答策略
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
from tqdm import tqdm

# 依赖你项目现有实现（与你 build_evalset.py 一致）
from llm.base import build_llm
from llm.schemas import ChatMessage, GenerateConfig


# ---------------------------
# 基础 IO：jsonl 读写 + 断点续跑
# ---------------------------

def read_jsonl(path: Path) -> List[dict]:
    """读取 jsonl：每行一个 JSON"""
    if not path.exists():
        return []
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            items.append(json.loads(s))
    return items


def append_jsonl(path: Path, items: List[dict]) -> None:
    """追加写 jsonl（断点续跑关键：只追加，不覆盖）"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def count_jsonl_lines(path: Path) -> int:
    """快速统计 jsonl 行数（用于续跑进度）"""
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def qid_from_index(i: int) -> str:
    """q000001 这种格式"""
    return f"q{i:06d}"


# ---------------------------
# JSON 抽取（稳健）
# ---------------------------

_JSON_RE = re.compile(r"\{.*\}", re.S)

def extract_json_obj(text: str) -> Optional[dict]:
    """
    模型可能在 JSON 前后加解释/代码块，这里稳健抽：
    - 正则抓最外层 {...}
    - json.loads
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
        blob2 = blob.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(blob2)
        except Exception:
            return None


# ---------------------------
# 读取并适配 llm_cfg（与 build_evalset.py 同风格）
# ---------------------------

def _get_env_api_key() -> Optional[str]:
    """推荐 env 注入 key：DEEPSEEK_API_KEY / OPENAI_API_KEY / API_KEY"""
    return os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")


def load_llm_cfg_for_build_llm(cfg_path: Path) -> Dict[str, Any]:
    """
    读取 configs/rag.yaml 或纯 llm.yaml，
    并整理为 build_llm() 期望结构。
    """
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"[build_dpo_dataset] YAML root must be dict: {cfg_path}")

    llm = data.get("llm", data)
    if not isinstance(llm, dict):
        raise ValueError(f"[build_dpo_dataset] 'llm' must be dict in {cfg_path}")

    backend = (llm.get("backend") or "api").strip()

    if backend == "api":
        api = llm.get("api") or {}
        if not isinstance(api, dict):
            api = {}

        api_key = api.get("api_key") or _get_env_api_key()
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
            "local": llm.get("local", {}) if isinstance(llm.get("local", {}), dict) else {},
        }

        if not cfg["api"]["api_key"]:
            raise RuntimeError(
                "[build_dpo_dataset] 没有拿到 API Key。\n"
                "解决方案二选一：\n"
                "1) 在 configs/rag.yaml 的 llm.api.api_key 填入 key（不推荐提交到仓库）；\n"
                "2) 推荐：export DEEPSEEK_API_KEY=\"sk-xxxxxxxx\" 后重试。"
            )

        return cfg

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
                "[build_dpo_dataset] backend=local 但没有 local.model_path。请检查 configs/rag.yaml。"
            )
        return cfg

    raise ValueError(f"[build_dpo_dataset] Unknown llm backend: {backend}")


# ---------------------------
# Chunk 规范化与抽样（同 build_evalset.py 思路）
# ---------------------------

def normalize_chunk(raw: dict) -> Optional[dict]:
    """兼容 chunks.jsonl schema 差异"""
    cid = raw.get("chunk_id") or raw.get("id")
    if not cid:
        return None

    content = raw.get("content") or raw.get("text") or ""
    content = str(content).strip()
    if len(content) < 200:
        return None

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
    rnd = random.Random(seed)
    rnd.shuffle(chunks)
    return chunks[:pool_size]


# ---------------------------
# 生成 QA（raw）用 prompt / 过滤
# ---------------------------

def make_qa_prompt(chunk_id: str, content: str) -> str:
    """
    生成 1 条可回答 QA，并强制 citations == [chunk_id]
    """
    return f"""
你是一个严谨的数据集构建助手。请仅根据给定资料生成 1 条问答样本，要求：
- 问题必须可以被资料直接回答（不需要外部知识）。
- 答案必须简洁、客观、可由资料逐句支撑。若资料主要为英文：允许 answer 用英文；专有名词/数字/单位优先沿用原文。
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
  "difficulty": "easy|medium|hard"
}}
""".strip()


def is_good_qa(obj: dict, expected_chunk_id: str) -> Tuple[bool, str]:
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
    if diff not in {"easy", "medium", "hard"}:
        return False, "bad_difficulty"
    if not isinstance(cits, list) or len(cits) != 1:
        return False, "bad_citations_len"
    if str(cits[0]).strip() != expected_chunk_id:
        return False, "cit_not_match"

    # 避免“拒答式答案”混入可回答集
    bad_ans = ["资料不足", "无法回答", "无法确定", "cannot determine", "insufficient"]
    low = a.lower()
    if any(x.lower() in low for x in bad_ans):
        return False, "refusal_like"

    return True, "ok"


def llm_generate_text(llm, messages: List[ChatMessage], max_tokens: int, temperature: float, retries: int, sleep_s: float) -> str:
    """带重试的 LLM 调用（网络抖动不白跑）"""
    last_err = None
    for _ in range(retries):
        try:
            resp = llm.generate(messages, GenerateConfig(max_tokens=max_tokens, temperature=temperature))
            return resp.text or ""
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    raise RuntimeError(f"LLM request failed after retries: {last_err}")


def generate_answerable_raw(llm, chunk: dict, max_tokens: int, retries: int) -> Optional[dict]:
    prompt = make_qa_prompt(chunk["chunk_id"], chunk["content"])
    messages = [
        ChatMessage(role="system", content="你是一个严谨的问答数据集构建助手。"),
        ChatMessage(role="user", content=prompt),
    ]

    text = llm_generate_text(llm, messages, max_tokens=max_tokens, temperature=0.2, retries=retries, sleep_s=1.0)
    obj = extract_json_obj(text)
    if obj is None:
        return None

    ok, _ = is_good_qa(obj, chunk["chunk_id"])
    if not ok:
        return None

    return {
        # qid 后面统一填
        "qid": "",
        "question": str(obj["question"]).strip(),
        "gold_answer": str(obj["answer"]).strip(),
        "gold_citations": [chunk["chunk_id"]],
        "difficulty": str(obj["difficulty"]).strip().lower(),
        "source_doc_id": chunk.get("doc_id", ""),
        "source_page": chunk.get("page", None),
        "source_chunk_id": chunk["chunk_id"],
    }


def make_refusal_raw_from_question(question: str, evidence_chunk: dict) -> dict:
    """
    可控“必须拒答”样本：
    - question 是“别的 chunk”能回答的问题
    - 证据给错 chunk → 正确行为应拒答
    """
    return {
        "qid": "",
        "question": question,
        "gold_answer": "资料不足以回答该问题。缺失点：未检索到能够支持该问题答案的有效 chunk。",
        "gold_citations": [],
        "difficulty": "hard",
        "source_doc_id": evidence_chunk.get("doc_id", ""),
        "source_page": evidence_chunk.get("page", None),
        "source_chunk_id": evidence_chunk["chunk_id"],
    }


# ---------------------------
# 构建 SFT / DPO prompt & answers
# ---------------------------

def build_chunk_index(chunks: List[dict]) -> Dict[str, dict]:
    idx = {}
    for c in chunks:
        idx[c["chunk_id"]] = c
    return idx


def build_human_prompt(question: str, evidence_items: List[Tuple[str, str]]) -> str:
    """
    统一 human prompt（SFT/DPO 共用）
    evidence_items: [(chunk_id, content_trunc), ...]
    """
    ev_lines = []
    for i, (cid, text) in enumerate(evidence_items, 1):
        ev_lines.append(f"({i}) chunk_id={cid}\n{text}")
    ev_block = "\n\n".join(ev_lines)

    return (
        "你是一个严格遵循证据的学术助手。\n\n"
        "[问题]\n"
        f"{question}\n\n"
        "[检索证据]\n"
        f"{ev_block}\n\n"
        "[规则]\n"
        "- 只能基于【检索证据】回答\n"
        "- 若证据不足以回答，请输出：资料不足以回答该问题。\n"
        "- 不要编造、不要引入常识推测\n\n"
        "[输出要求]\n"
        "- 先给答案正文\n"
        "- 末尾给出引用列表（每行一个 chunk_id），格式：\n"
        "  [引用]\n"
        "  - chunk_id"
    )


def sft_answer_text(gold_answer: str, gold_citations: List[str]) -> str:
    """
    SFT 输出：
    - 可回答：答案 + [引用] 列表
    - 拒答：包含缺失点说明 + citations: []
    """
    if not gold_citations:
        # 你要求“拒答并说明缺失点”
        return f"{gold_answer}\n\ncitations: []"
    cite_lines = "\n".join([f"- {c}" for c in gold_citations])
    return f"{gold_answer}\n\n[引用]\n{cite_lines}"


def dpo_answer_text_citations(gold_answer: str, citations: List[str]) -> str:
    """DPO 的 chosen/rejected 里按你示例用 citations: [...]"""
    return f'{gold_answer}\n\ncitations: {json.dumps(citations, ensure_ascii=False)}'


def make_rejected_variants(answer: str, gold_citations: List[str], wrong_citation_pool: List[str], mode: str, rnd: random.Random) -> Tuple[str, List[str]]:
    """
    构造“可控坏” rejected：
    - no_citations：答案对但 citations 空
    - wrong_citation：答案对但 citations 指错 chunk
    - incomplete：答案不完整 + citations 空
    """
    if mode == "no_citations":
        return answer, []
    if mode == "wrong_citation":
        if wrong_citation_pool:
            wrong = rnd.choice(wrong_citation_pool)
            if gold_citations and wrong == gold_citations[0] and len(wrong_citation_pool) > 1:
                wrong = rnd.choice([x for x in wrong_citation_pool if x != gold_citations[0]])
            return answer, [wrong]
        return answer, []
    if mode == "incomplete":
        # 截断到 30% 长度，且去掉引用
        cut = max(10, int(len(answer) * 0.3))
        return answer[:cut].rstrip("，。;； ") + "……", []
    return answer, []


def hallucinated_rejected_for_refusal(question: str) -> str:
    """
    对“必须拒答”的样本，rejected = 硬答/编造（可控坏）
    """
    return f"我认为该问题的答案可以确定：{question} 的答案是 OpenAI 等人。\n\ncitations: []"


# ---------------------------
# 主流程：生成 raw -> 同步写 sft/dpo
# ---------------------------

@dataclass
class GenConfig:
    out_dir: Path
    n_train: int
    n_test: int
    seed: int
    pool_size: int
    max_tokens: int
    max_tries: int
    retries: int
    evidence_max_chars: int
    refuse_ratio: float
    rejected_mode_weights: Dict[str, float]


def weighted_choice(rnd: random.Random, items: List[Tuple[str, float]]) -> str:
    total = sum(w for _, w in items)
    r = rnd.random() * total
    upto = 0.0
    for k, w in items:
        upto += w
        if upto >= r:
            return k
    return items[-1][0]


def generate_split(
    llm,
    chunks: List[dict],
    cfg: GenConfig,
    split_name: str,
    n_target: int,
) -> None:
    """
    split_name: "" or "_test"
    生成并写入：
    - qa_raw{split}.jsonl
    - qa_sft{split}.jsonl
    - qa_dpo{split}.jsonl
    """
    out_raw = cfg.out_dir / f"qa_raw{split_name}.jsonl"
    out_sft = cfg.out_dir / f"qa_sft{split_name}.jsonl"
    out_dpo = cfg.out_dir / f"qa_dpo{split_name}.jsonl"

    # 断点续跑：按 raw 已有行数作为“已完成”
    done = count_jsonl_lines(out_raw)
    if done >= n_target:
        print(f"[build_dpo_dataset] {out_raw} already has {done} >= {n_target}, skip.")
        return

    rnd = random.Random(cfg.seed + (999 if split_name else 0))

    # 候选池：避免对 3W+ chunk 全量遍历
    pool = sample_pool(chunks[:], pool_size=min(len(chunks), cfg.pool_size), seed=cfg.seed + 7)
    chunk_index = build_chunk_index(chunks)
    all_chunk_ids = list(chunk_index.keys())

    # 为了构造“拒答样本”，我们先准备一些“可回答问题缓存”
    answerable_questions_cache: List[str] = []

    # 去重：避免生成完全重复 question（只对本次进程内去重；续跑时不回读全量去重，省内存）
    seen_q = set()

    # 已完成的 qid 起点
    start_idx = done + 1

    pbar = tqdm(total=n_target, initial=done, desc=f"gen{split_name or '_train'}", ncols=90)

    tries = 0
    idx = 0
    batch_raw: List[dict] = []
    batch_sft: List[dict] = []
    batch_dpo: List[dict] = []

    # rejected 模式分布
    modes = list(cfg.rejected_mode_weights.items())

    while (done + len(batch_raw)) < n_target and tries < cfg.max_tries:
        tries += 1
        chunk = pool[idx % len(pool)]
        idx += 1

        # 先生成一个可回答样本（用于：正常样本 or 作为拒答样本的 question 来源）
        raw_item = generate_answerable_raw(llm, chunk, max_tokens=cfg.max_tokens, retries=cfg.retries)
        if not raw_item:
            continue

        qkey = raw_item["question"].strip().lower()
        if qkey in seen_q:
            continue
        seen_q.add(qkey)

        answerable_questions_cache.append(raw_item["question"])

        # 决定是否做成“必须拒答”
        make_refuse = (rnd.random() < cfg.refuse_ratio) and (len(answerable_questions_cache) >= 3)

        if make_refuse:
            # 用缓存里某个“别的可回答问题”，配上“错误证据 chunk”
            q_ref = rnd.choice(answerable_questions_cache)
            wrong_ev = chunk  # 当前 chunk 作为错误证据（与 q_ref 来源不同的概率很高）
            raw_item2 = make_refusal_raw_from_question(q_ref, evidence_chunk=wrong_ev)
            raw_item = raw_item2

        # 填 qid
        cur_i = start_idx + len(batch_raw)
        raw_item["qid"] = qid_from_index(cur_i)

        # 取证据内容（只给 1 个 chunk）
        ev_chunk_id = raw_item["source_chunk_id"]
        ev = chunk_index.get(ev_chunk_id)
        if not ev:
            # 极少数：source_chunk_id 不在 index（数据问题）→ 跳过
            continue
        ev_text = ev["content"][: cfg.evidence_max_chars]

        human = build_human_prompt(raw_item["question"], [(ev_chunk_id, ev_text)])

        # ----- SFT -----
        sft_item = {
            "id": raw_item["qid"],
            "conversations": [
                {"from": "human", "value": human},
                {"from": "gpt", "value": sft_answer_text(raw_item["gold_answer"], raw_item["gold_citations"])},
            ],
            "meta": {
                "source_doc_id": raw_item["source_doc_id"],
                "source_page": raw_item["source_page"],
                "source_chunk_id": raw_item["source_chunk_id"],
                "difficulty": raw_item["difficulty"],
            },
        }

        # ----- DPO -----
        if raw_item["gold_citations"]:
            # 可回答：chosen 正确引用；rejected 可控坏
            chosen_val = dpo_answer_text_citations(raw_item["gold_answer"], raw_item["gold_citations"])

            mode = weighted_choice(rnd, modes)
            bad_ans, bad_cits = make_rejected_variants(
                answer=raw_item["gold_answer"],
                gold_citations=raw_item["gold_citations"],
                wrong_citation_pool=all_chunk_ids,
                mode=mode,
                rnd=rnd,
            )
            rejected_val = dpo_answer_text_citations(bad_ans, bad_cits)
        else:
            # 必须拒答：chosen=拒答+缺失点；rejected=硬答/编造
            chosen_val = dpo_answer_text_citations(raw_item["gold_answer"], [])
            rejected_val = hallucinated_rejected_for_refusal(raw_item["question"])

        dpo_item = {
            "conversations": [{"from": "human", "value": human}],
            "chosen": {"from": "gpt", "value": chosen_val},
            "rejected": {"from": "gpt", "value": rejected_val},
        }

        batch_raw.append(raw_item)
        batch_sft.append(sft_item)
        batch_dpo.append(dpo_item)

        # 批量落盘：避免一次性崩溃丢失进度
        if len(batch_raw) >= 20:
            append_jsonl(out_raw, batch_raw)
            append_jsonl(out_sft, batch_sft)
            append_jsonl(out_dpo, batch_dpo)
            done += len(batch_raw)
            pbar.update(len(batch_raw))
            batch_raw, batch_sft, batch_dpo = [], [], []

    # flush
    if batch_raw:
        append_jsonl(out_raw, batch_raw)
        append_jsonl(out_sft, batch_sft)
        append_jsonl(out_dpo, batch_dpo)
        done += len(batch_raw)
        pbar.update(len(batch_raw))

    pbar.close()

    if done < n_target:
        print(f"[build_dpo_dataset] WARNING: only wrote {done}/{n_target} to {out_raw} (tries={tries}).")
    else:
        print(f"[build_dpo_dataset] wrote {done}/{n_target} to {out_raw} (+ sft/dpo).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_cfg", type=str, required=True, help="configs/rag.yaml (or llm yaml)")
    ap.add_argument("--chunks", type=str, default="data/processed/chunks/chunks.jsonl")
    ap.add_argument("--out_dir", type=str, default="data/dpo")

    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=100)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool_size", type=int, default=1200)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--max_tries", type=int, default=200000)
    ap.add_argument("--retries", type=int, default=5)

    ap.add_argument("--evidence_max_chars", type=int, default=1800)
    ap.add_argument("--refuse_ratio", type=float, default=0.15, help="必须拒答样本占比（0~1）")

    # rejected 的坏样本模式权重（可按你偏好调）
    ap.add_argument("--w_no_citations", type=float, default=0.55)
    ap.add_argument("--w_wrong_citation", type=float, default=0.30)
    ap.add_argument("--w_incomplete", type=float, default=0.15)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[build_dpo_dataset] loading chunks...")
    raw_chunks = read_jsonl(Path(args.chunks))
    chunks: List[dict] = []
    for r in raw_chunks:
        ck = normalize_chunk(r)
        if ck:
            chunks.append(ck)
    print(f"[build_dpo_dataset] valid chunks: {len(chunks)}")
    if not chunks:
        raise RuntimeError("[build_dpo_dataset] No valid chunks after filtering.")

    llm_cfg = load_llm_cfg_for_build_llm(Path(args.llm_cfg))
    llm = build_llm(llm_cfg)

    cfg = GenConfig(
        out_dir=out_dir,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
        pool_size=args.pool_size,
        max_tokens=args.max_tokens,
        max_tries=args.max_tries,
        retries=args.retries,
        evidence_max_chars=args.evidence_max_chars,
        refuse_ratio=args.refuse_ratio,
        rejected_mode_weights={
            "no_citations": args.w_no_citations,
            "wrong_citation": args.w_wrong_citation,
            "incomplete": args.w_incomplete,
        },
    )

    # 生成训练集
    generate_split(llm, chunks, cfg, split_name="", n_target=cfg.n_train)

    # 生成测试集
    generate_split(llm, chunks, cfg, split_name="_test", n_target=cfg.n_test)

    print("[build_dpo_dataset] DONE.")
    print(f"  - {out_dir/'qa_raw.jsonl'} / {out_dir/'qa_sft.jsonl'} / {out_dir/'qa_dpo.jsonl'}")
    print(f"  - {out_dir/'qa_raw_test.jsonl'} / {out_dir/'qa_sft_test.jsonl'} / {out_dir/'qa_dpo_test.jsonl'}")


if __name__ == "__main__":
    main()