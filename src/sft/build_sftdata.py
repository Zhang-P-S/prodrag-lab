from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_chunk_index(chunks: List[dict]) -> Dict[str, dict]:
    idx = {}
    for c in chunks:
        cid = c.get("chunk_id")
        if cid:
            idx[cid] = c
    return idx


def truncate_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if max_chars <= 0:
        return s
    return s[:max_chars].strip()


def to_sharegpt_item(
    qa: dict,
    chunk: dict,
    evidence_max_chars: int,
    add_refusal_rule: bool = True,
) -> Optional[dict]:
    q = (qa.get("question") or "").strip()
    a = (qa.get("gold_answer") or "").strip()
    cites = qa.get("gold_citations") or []
    if not q or not a or not cites:
        return None

    cid = cites[0]
    evidence = truncate_text(chunk.get("content", ""), evidence_max_chars)
    if not evidence:
        return None

    # 你要训练的是“RAG 助手行为”：严格依证据 + 能拒答 + 给引用
    rules = ""
    if add_refusal_rule:
        rules = (
            "\n[规则]\n"
            "- 只能基于【检索证据】回答\n"
            "- 若证据不足以回答，请输出：资料不足以回答该问题。\n"
            "- 不要编造、不要引入常识推测\n"
        )

    user_prompt = (
        "你是一个严格遵循证据的学术助手。\n\n"
        f"[问题]\n{q}\n\n"
        "[检索证据]\n"
        f"(1) chunk_id={cid}\n"
        f"{evidence}\n"
        f"{rules}\n"
        "[输出要求]\n"
        "- 先给答案正文\n"
        "- 末尾给出引用列表（每行一个 chunk_id），格式：\n"
        "  [引用]\n"
        "  - chunk_id\n"
    ).strip()

    assistant = (a.strip() + "\n\n[引用]\n- " + cid).strip()

    return {
        "id": qa.get("qid", ""),
        "conversations": [
            {"from": "human", "value": user_prompt},
            {"from": "gpt", "value": assistant},
        ],
        "meta": {
            "source_doc_id": qa.get("source_doc_id", chunk.get("doc_id", "")),
            "source_page": qa.get("source_page", chunk.get("page", None)),
            "source_chunk_id": qa.get("source_chunk_id", cid),
            "difficulty": qa.get("difficulty", "medium"),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, required=True, help="chunks.jsonl path")
    ap.add_argument("--qa", type=str, required=True, help="qa_500.jsonl path")
    ap.add_argument("--out_dir", type=str, default="data/sft_rag_sharegpt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--evidence_max_chars", type=int, default=1800)
    ap.add_argument("--limit_per_chunk", type=int, default=2, help="avoid overusing one chunk")
    args = ap.parse_args()

    random.seed(args.seed)

    chunks = read_jsonl(Path(args.chunks))
    qa_rows = read_jsonl(Path(args.qa))
    print(f"[make_sft] chunks={len(chunks)} qa={len(qa_rows)}")

    cidx = build_chunk_index(chunks)

    # 限流：同一 chunk 最多生成 N 条（防止模型只学会某一页）
    used_per_chunk: Dict[str, int] = {}

    sft: List[dict] = []
    miss = 0
    skipped = 0

    for qa in qa_rows:
        cites = qa.get("gold_citations") or []
        if not cites:
            skipped += 1
            continue
        cid = cites[0]
        chunk = cidx.get(cid)
        if not chunk:
            miss += 1
            continue

        if args.limit_per_chunk > 0:
            used = used_per_chunk.get(cid, 0)
            if used >= args.limit_per_chunk:
                skipped += 1
                continue
            used_per_chunk[cid] = used + 1

        item = to_sharegpt_item(
            qa=qa,
            chunk=chunk,
            evidence_max_chars=args.evidence_max_chars,
            add_refusal_rule=True,
        )
        if item:
            sft.append(item)
        else:
            skipped += 1

    print(f"[make_sft] built={len(sft)} miss_chunk={miss} skipped={skipped}")

    random.shuffle(sft)
    n_val = max(1, int(len(sft) * args.val_ratio))
    val = sft[:n_val]
    train = sft[n_val:]

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)

    print(f"[make_sft] wrote train={len(train)} val={len(val)} to {out_dir}")


if __name__ == "__main__":
    main()
