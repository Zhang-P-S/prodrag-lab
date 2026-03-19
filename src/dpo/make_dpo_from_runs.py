#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从多个 runs/*.jsonl 自动构造 DPO 偏好数据（格式1：conversations + chosen + rejected）。

输出每行一个样本：
{
  "conversations": [{"from":"human","value":"..."}],
  "chosen": {"from":"gpt","value":"..."},
  "rejected": {"from":"gpt","value":"..."},
  "meta": {...}   # 可选，不影响训练
}

说明：
- 你的 LLaMA-Factory 在某些 stage（比如 rm / sharegpt 相关模板）会强依赖 "conversations" 字段
- 所以这里按你给的格式1来输出，避免 KeyError: 'conversations'
"""

from __future__ import annotations
import argparse
import json
import os
import re
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple, Optional


# -----------------------------
# 1) IO
# -----------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"[read_jsonl] JSON parse error at {path}:{line_no}: {e}")
    return items


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


# -----------------------------
# 2) chunks: chunk_id -> text
# -----------------------------
def load_chunks(chunks_jsonl: str) -> Dict[str, str]:
    """
    兼容字段：
    - {"chunk_id": "...", "text": "..."}
    - {"id": "...", "content": "..."}
    - {"chunk_id": "...", "content": "..."}
    """
    m: Dict[str, str] = {}
    for obj in read_jsonl(chunks_jsonl):
        cid = obj.get("chunk_id") or obj.get("id") or obj.get("cid")
        txt = obj.get("text") or obj.get("content") or obj.get("chunk_text")
        if cid and isinstance(txt, str) and txt.strip():
            m[str(cid)] = txt.strip()
    if not m:
        raise ValueError(f"[load_chunks] No chunks loaded from: {chunks_jsonl} (check keys: chunk_id/text)")
    return m


# -----------------------------
# 3) 解析 answer 里的 citations
# -----------------------------
_CIT_RE_1 = re.compile(r"citations?\s*:\s*(\[[^\]]*\])", re.IGNORECASE)
_CIT_RE_2 = re.compile(r"\"citations?\"\s*:\s*(\[[^\]]*\])", re.IGNORECASE)


def _safe_parse_list_literal(s: str) -> List[str]:
    s = s.strip()
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            return [str(x) for x in arr]
    except Exception:
        pass
    s2 = s.strip().lstrip("[").rstrip("]")
    parts = [p.strip().strip("\"'") for p in s2.split(",") if p.strip()]
    return [p for p in parts if p]


def extract_citations(item: Dict[str, Any]) -> List[str]:
    if isinstance(item.get("citations"), list):
        return [str(x) for x in item["citations"]]
    if isinstance(item.get("pred_citations"), list):
        return [str(x) for x in item["pred_citations"]]

    ans = item.get("answer") or item.get("output") or ""
    if not isinstance(ans, str):
        return []

    m = _CIT_RE_1.search(ans) or _CIT_RE_2.search(ans)
    if not m:
        return []
    return _safe_parse_list_literal(m.group(1))


# -----------------------------
# 4) refusal 判断（用于评分）
# -----------------------------
_REFUSAL_HINTS = [
    "资料不足", "无法回答", "无法确定", "没有提供", "未提供", "证据不足", "无法从资料中",
    "无法根据资料", "无法基于证据", "不支持", "无法支持"
]


def detect_refusal_from_text(ans: str) -> bool:
    if not ans:
        return False
    a = ans.strip()
    return any(h in a for h in _REFUSAL_HINTS)


def get_should_refuse(item: Dict[str, Any]) -> Optional[bool]:
    # 优先用标注字段
    for k in ("gold_refusal", "final_refusal", "should_refuse", "need_refusal"):
        if k in item and isinstance(item[k], bool):
            return item[k]
    # 兜底：gold_citations 为空 -> 应该拒答
    gold = item.get("gold_citations")
    if isinstance(gold, list):
        return (len(gold) == 0)
    return None


def get_pred_refuse(item: Dict[str, Any]) -> bool:
    for k in ("model_refusal", "pred_refusal", "refusal"):
        if k in item and isinstance(item[k], bool):
            return item[k]
    ans = item.get("answer") or ""
    return detect_refusal_from_text(ans if isinstance(ans, str) else "")


# -----------------------------
# 5) retrieved ids
# -----------------------------
def get_retrieved_ids(item: Dict[str, Any]) -> List[str]:
    for k in ("retrieved", "top_chunks", "ctx_ids", "retrieved_ids"):
        v = item.get(k)
        if isinstance(v, list) and v:
            if all(isinstance(x, str) for x in v):
                return [str(x) for x in v]
            if all(isinstance(x, dict) for x in v):
                out = []
                for d in v:
                    cid = d.get("chunk_id") or d.get("id")
                    if cid:
                        out.append(str(cid))
                return out

    v = item.get("contexts")
    if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
        out = []
        for d in v:
            cid = d.get("chunk_id") or d.get("id")
            if cid:
                out.append(str(cid))
        return out
    return []


# -----------------------------
# 6) 评分：选 best / worst
# -----------------------------
def score_candidate(
    cand: Dict[str, Any],
    gold_citations: List[str],
    retrieved_ids: List[str],
) -> Dict[str, float]:
    pred_cits = extract_citations(cand)
    pred_refuse = get_pred_refuse(cand)

    # (1) cite precision: pred 中命中 gold 的比例
    if pred_cits:
        inter = set(pred_cits) & set(gold_citations)
        cite_prec = len(inter) / max(1, len(set(pred_cits)))
    else:
        cite_prec = 0.0

    # (2) cite in prompt: pred 是否来自检索证据（防漂）
    if pred_cits:
        cite_in_prompt = sum(1 for c in pred_cits if c in set(retrieved_ids)) / len(pred_cits)
    else:
        cite_in_prompt = 0.0

    # (3) refusal correctness
    should_refuse = get_should_refuse(cand)
    if should_refuse is None:
        refusal_correct = 0.5
    else:
        refusal_correct = 1.0 if (pred_refuse == should_refuse) else 0.0

    # (4) overcite penalty
    overcite = 0.0
    if len(pred_cits) > 3:
        overcite = min(1.0, (len(pred_cits) - 3) / 7.0)

    # (5) missing-citation penalty（非拒答但长回答却没引用）
    ans = cand.get("answer") or ""
    ans_len = len(ans) if isinstance(ans, str) else 0
    missing_cit = 0.0
    if (not pred_refuse) and (ans_len > 120) and (len(pred_cits) == 0):
        missing_cit = 1.0

    # 权重（可调）
    overall = (
        1.0 * cite_prec +
        1.0 * refusal_correct +
        0.5 * cite_in_prompt -
        0.3 * overcite -
        0.7 * missing_cit
    )

    return {
        "overall": float(overall),
        "cite_prec": float(cite_prec),
        "cite_in_prompt": float(cite_in_prompt),
        "refusal_correct": float(refusal_correct),
        "overcite": float(overcite),
        "missing_cit": float(missing_cit),
    }


# -----------------------------
# 7) 构造 prompt（作为 conversations[0].value）
# -----------------------------
DEFAULT_SYSTEM = (
    "你是一个严格的RAG助手。只能使用【证据chunks】中的信息回答。\n"
    "- 如果证据不足以支持问题：必须拒答，并说明缺失点。\n"
    "- 不允许编造，不允许使用常识补全。\n"
    "- 输出末尾必须给出 citations: [\"chunk_id\", ...]，且只能引用证据chunks中出现过的chunk_id。\n"
)

def build_user_prompt(question: str, retrieved_ids: List[str], chunk_map: Dict[str, str], topk: int = 8) -> str:
    ids = [cid for cid in retrieved_ids if cid in chunk_map][:topk]
    lines = [DEFAULT_SYSTEM.strip(), "", f"问题：{question}", "", "【证据chunks】"]
    for cid in ids:
        txt = chunk_map[cid].replace("\n", " ").strip()
        if len(txt) > 700:
            txt = txt[:700] + " ..."
        lines.append(f"[{cid}] {txt}")
    lines.append("")
    lines.append("请回答，并在末尾输出 citations: [\"chunk_id\", ...]")
    return "\n".join(lines)


# -----------------------------
# 8) 主逻辑
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="多个 runs/*.jsonl，作为候选来源")
    ap.add_argument("--chunks", required=True, help="chunks.jsonl (chunk_id -> text)")
    ap.add_argument("--out", required=True, help="输出 DPO jsonl（格式1）")
    ap.add_argument("--topk", type=int, default=8, help="prompt 中拼接的证据 chunks 数量")
    ap.add_argument("--margin", type=float, default=0.35, help="best-worst 分差阈值，太小就丢弃")
    args = ap.parse_args()

    chunk_map = load_chunks(args.chunks)

    # 读取 runs 并按 qid 聚合候选
    by_qid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for path in args.runs:
        items = read_jsonl(path)
        for it in items:
            qid = it.get("qid") or it.get("id") or it.get("question_id")
            if not qid:
                continue
            it["_src_run"] = os.path.basename(path)
            by_qid[str(qid)].append(it)

    stats = Counter()
    out_rows: List[Dict[str, Any]] = []

    for qid, cands in by_qid.items():
        if len(cands) < 2:
            stats["skip_not_enough_candidates"] += 1
            continue

        question = None
        gold_citations: List[str] = []
        retrieved_ids: List[str] = []

        for it in cands:
            if question is None and isinstance(it.get("question"), str):
                question = it["question"]
            if not gold_citations and isinstance(it.get("gold_citations"), list):
                gold_citations = [str(x) for x in it["gold_citations"]]
            if not retrieved_ids:
                retrieved_ids = get_retrieved_ids(it)

        if not question:
            stats["skip_missing_question"] += 1
            continue

        # 如果 retrieved 缺失：退化用 gold 作为证据（至少保证引用不会漂）
        if not retrieved_ids and gold_citations:
            retrieved_ids = gold_citations.copy()

        user_prompt = build_user_prompt(question, retrieved_ids, chunk_map, topk=args.topk)

        # 打分
        scored: List[Tuple[Dict[str, Any], Dict[str, float]]] = []
        for it in cands:
            if not isinstance(it.get("answer"), str):
                continue
            s = score_candidate(it, gold_citations=gold_citations, retrieved_ids=retrieved_ids)
            scored.append((it, s))

        if len(scored) < 2:
            stats["skip_not_enough_scored"] += 1
            continue

        scored.sort(key=lambda x: x[1]["overall"], reverse=True)
        best_it, best_s = scored[0]
        worst_it, worst_s = scored[-1]

        gap = best_s["overall"] - worst_s["overall"]
        if gap < args.margin:
            stats["skip_small_margin"] += 1
            continue

        chosen_ans = best_it["answer"]
        rejected_ans = worst_it["answer"]

        # ✅ 按你要求的格式1输出
        out_rows.append({
            "conversations": [
                {"from": "human", "value": user_prompt}
            ],
            "chosen": {"from": "gpt", "value": chosen_ans},
            "rejected": {"from": "gpt", "value": rejected_ans},
            "meta": {
                "qid": qid,
                "question": question,
                "gold_citations": gold_citations,
                "retrieved_ids": retrieved_ids[:args.topk],
                "chosen_src": best_it.get("_src_run"),
                "rejected_src": worst_it.get("_src_run"),
                "scores": {"chosen": best_s, "rejected": worst_s},
                "gap": gap,
                "chosen_pred_citations": extract_citations(best_it),
                "rejected_pred_citations": extract_citations(worst_it),
                "chosen_pred_refusal": get_pred_refuse(best_it),
                "rejected_pred_refusal": get_pred_refuse(worst_it),
            }
        })
        stats["kept"] += 1

    write_jsonl(args.out, out_rows)

    print("[make_dpo_from_runs] wrote:", args.out)
    print("[make_dpo_from_runs] stats:")
    for k, v in stats.most_common():
        print(f"  - {k}: {v}")
    print(f"[make_dpo_from_runs] total_qid: {len(by_qid)}  kept: {stats['kept']}  out_lines: {len(out_rows)}")


if __name__ == "__main__":
    main()