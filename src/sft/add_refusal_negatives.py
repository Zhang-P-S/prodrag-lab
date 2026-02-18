from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple


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


def build_chunk_pool(chunks: List[dict]) -> List[Tuple[str, str, str, Any]]:
    """
    Return list of (chunk_id, content, doc_id, page)
    """
    pool = []
    for c in chunks:
        cid = c.get("chunk_id")
        content = (c.get("content") or "").strip()
        if cid and content:
            pool.append((cid, content, c.get("doc_id", ""), c.get("page", None)))
    return pool


_EVID_RE = re.compile(
    r"(\[检索证据\]\s*\n\(1\)\s*chunk_id=)([^\n]+)\n(.*?)(\n(?:\[规则\]|\[输出要求\]|\Z))",
    flags=re.S,
)

def replace_evidence(human_text: str, new_chunk_id: str, new_content: str, max_chars: int) -> str:
    """
    Replace the evidence chunk_id and its content in the human prompt.
    """
    new_content = (new_content or "").strip()
    if max_chars > 0:
        new_content = new_content[:max_chars].strip()

    m = _EVID_RE.search(human_text)
    if not m:
        # If format doesn't match, return original (safer than corrupting)
        return human_text

    prefix, _old_id, _old_content, suffix = m.group(1), m.group(2), m.group(3), m.group(4)
    replaced = human_text[:m.start()] + prefix + new_chunk_id + "\n" + new_content + suffix + human_text[m.end():]
    return replaced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True, help="original train.jsonl (ShareGPT)")
    ap.add_argument("--chunks", type=str, required=True, help="chunks.jsonl used to sample unrelated evidence")
    ap.add_argument("--out", type=str, required=True, help="output enhanced train jsonl")
    ap.add_argument("--neg_ratio", type=float, default=0.15, help="fraction to convert into refusal negatives (0.1~0.2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--evidence_max_chars", type=int, default=1800)
    ap.add_argument("--keep_originals", action="store_true",
                    help="if set, keep originals AND add new negatives (dataset size increases). "
                         "otherwise, modify selected samples in-place (dataset size unchanged).")
    args = ap.parse_args()

    random.seed(args.seed)

    train_path = Path(args.train)
    chunks_path = Path(args.chunks)
    out_path = Path(args.out)

    data = read_jsonl(train_path)
    chunks = read_jsonl(chunks_path)
    pool = build_chunk_pool(chunks)

    if not pool:
        raise RuntimeError("No valid chunks found (need chunk_id + content).")

    n = len(data)
    k = max(1, int(n * args.neg_ratio))
    idxs = list(range(n))
    random.shuffle(idxs)
    chosen = set(idxs[:k])

    # quick map for current citation in assistant (optional)
    def get_current_citation(item: dict) -> str:
        try:
            gpt_text = item["conversations"][1]["value"]
        except Exception:
            return ""
        m = re.search(r"^\s*-\s*(\S+)\s*$", gpt_text, flags=re.M)
        return m.group(1).strip() if m else ""

    new_items = []
    converted = 0
    skipped = 0

    for i, item in enumerate(data):
        if i not in chosen:
            continue

        # basic checks
        if not isinstance(item, dict) or "conversations" not in item or len(item["conversations"]) < 2:
            skipped += 1
            continue

        human = item["conversations"][0].get("value", "")
        old_cid = get_current_citation(item)

        # sample a new chunk id different from old
        tries = 0
        while True:
            tries += 1
            cid, content, doc_id, page = random.choice(pool)
            if cid != old_cid or tries > 10:
                break

        new_human = replace_evidence(human, cid, content, args.evidence_max_chars)

        # If evidence replacement failed (format mismatch), skip to avoid corrupt samples
        if new_human == human:
            skipped += 1
            continue

        refusal_answer = "资料不足以回答该问题。\n\n[引用]\n- " + cid

        neg = json.loads(json.dumps(item, ensure_ascii=False))  # deep-ish copy
        # assign new id
        old_id = neg.get("id", f"sample_{i}")
        neg["id"] = f"{old_id}__neg"
        neg["conversations"][0]["value"] = new_human
        neg["conversations"][1]["value"] = refusal_answer

        # update meta
        meta = neg.get("meta", {})
        meta["is_negative"] = True
        meta["source_chunk_id"] = cid
        meta["source_doc_id"] = doc_id
        meta["source_page"] = page
        neg["meta"] = meta

        new_items.append(neg)
        converted += 1

    if args.keep_originals:
        out_rows = data + new_items
    else:
        # modify in-place for chosen items: replace those items with the negative version
        out_rows = []
        neg_map = {it["id"].replace("__neg", ""): it for it in new_items}
        # We keyed by old id; but old id might be empty. We'll fall back to index-based replacement.
        neg_by_index = {idx: it for idx, it in zip(sorted(chosen), new_items)}
        for i, item in enumerate(data):
            if i in neg_by_index:
                out_rows.append(neg_by_index[i])
            else:
                out_rows.append(item)

    random.shuffle(out_rows)
    write_jsonl(out_path, out_rows)

    print(f"[add_neg] loaded train={n}, target_neg={k}")
    print(f"[add_neg] converted={converted}, skipped={skipped}")
    print(f"[add_neg] keep_originals={args.keep_originals} -> out={len(out_rows)} saved to {out_path}")


if __name__ == "__main__":
    main()
# 在原 450 条基础上“新增”负样本，数据集变大（更稳）
# python scripts/add_refusal_negatives.py \
#   --train data/sft_rag_sharegpt/train.jsonl \
#   --chunks data/processed/chunks/chunks.jsonl \
#   --out data/sft_rag_sharegpt/train_with_neg.jsonl \
#   --neg_ratio 0.15 \
#   --keep_originals
# 把其中 15% 的样本直接改成负样本（数据量不变）
# python scripts/add_refusal_negatives.py \
#   --train data/sft_rag_sharegpt/train.jsonl \
#   --chunks data/processed/chunks/chunks.jsonl \
#   --out data/sft_rag_sharegpt/train_neg_inplace.jsonl \
#   --neg_ratio 0.15
