# -*- coding: utf-8 -*-
"""
eval_matrics_answer_only_v9.py

你跑 v8 发现 SimSentMax 仍然很低：这很正常——v8 仍然是“词面相似度”(lexical)。
当模型答案是同义改写/扩写（比如加了 program value proposition 等），词面 overlap 可能很小 => lexical 指标会低。

v9 改成“语义相似度”为主（Embedding Cosine），不依赖 sentence-transformers，直接用 transformers 做 mean pooling：
- EmbSimFull: pred整段 vs gold 的 embedding cosine
- EmbSimSentMax: pred句子与 gold embedding cosine 的最大值（推荐主指标）
- EmbAccSentMax@T: EmbSimSentMax >= T 的比例（建议阈值 0.65~0.80，先用 0.70）

同时保留 v8 的 TokF1 作为辅助（便于解释）。
不看引用、不看证据、不看拒答。

依赖：
pip install -U transformers torch

推荐 embedding 模型：
- 英文：/root/autodl-tmp/models/bge-small-en-v1.5  (或 BAAI/bge-small-en-v1.5)
- 中文：/root/autodl-tmp/models/bge-small-zh-v1.5
- 多语混合可用 bge-m3（如果你有）

用法示例：
python src/dpo/eval_matrics_answer_only_v9.py \
  --test data/dpo/qa_raw_test.jsonl \
  --runs_dir runs --settings sft \
  --use_clean \
  --emb_model_path /root/autodl-tmp/models/bge-small-en-v1.5 \
  --device cuda \
  --acc_thres 0.70 \
  --out_csv tables/answer_only_v9.csv --out_md tables/answer_only_v9.md
"""

from __future__ import annotations
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd

_RE_WS = re.compile(r"\s+")
_RE_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_RE_SENT = re.compile(r"(?<=[\.\!\?。！？])\s+")


def read_jsonl(path: Path) -> List[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    ascii_cnt = sum(1 for ch in s if ord(ch) < 128)
    return ascii_cnt / max(1, len(s))


def clean_answer(ans: str, max_chars: int = 1200) -> str:
    ans = (ans or "").strip()
    if "</think>" in ans:
        ans = ans.split("</think>", 1)[1].strip()

    prefixes = [
        "资料足以回答该问题。答案：",
        "资料足以回答该问题。",
        "最终答案：",
        "答案：",
        "Final Answer:",
        "The answer is:",
        "Answer:",
    ]
    for p in prefixes:
        if ans.startswith(p):
            ans = ans[len(p):].strip()

    ans = re.sub(r"\s+", " ", ans).strip()

    parts = _RE_SENT.split(ans)
    uniq = []
    seen = set()
    for s in parts:
        s = s.strip()
        if not s:
            continue
        key = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s.lower())[:160]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    ans = " ".join(uniq).strip()

    return ans[:max_chars]


def split_sentences(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    parts = _RE_SENT.split(s)
    out = [p.strip() for p in parts if p.strip()]
    return out if out else [s]


def tokenize_for_f1(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    if _ascii_ratio(s) >= 0.75:
        return _RE_WORD.findall(s.lower())
    toks = []
    buf = []
    for ch in s:
        if '\u4e00' <= ch <= '\u9fff':
            if buf:
                toks.extend(_RE_WORD.findall("".join(buf).lower()))
                buf = []
            toks.append(ch)
        else:
            buf.append(ch)
    if buf:
        toks.extend(_RE_WORD.findall("".join(buf).lower()))
    return [t for t in toks if t]


def token_f1(pred: str, gold: str) -> float:
    p = tokenize_for_f1(pred)
    g = tokenize_for_f1(gold)
    if not p or not g:
        return 0.0
    cp = Counter(p); cg = Counter(g)
    overlap = sum(min(cp[t], cg[t]) for t in cp.keys())
    prec = overlap / max(1, len(p))
    rec = overlap / max(1, len(g))
    return 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))


def extract_pred_answer(run_item: dict) -> str:
    parsed = run_item.get("parsed") or {}
    if isinstance(parsed, dict):
        ans = parsed.get("answer")
        if isinstance(ans, str) and ans.strip():
            return ans
    llm_raw = run_item.get("llm_raw")
    if isinstance(llm_raw, str) and llm_raw.strip():
        return llm_raw
    ans = run_item.get("answer")
    if isinstance(ans, str) and ans.strip():
        return ans
    return ""


# ===== Embedding via transformers mean pooling =====
class HFEmbedder:
    def __init__(self, model_path: str, device: str = "cpu", max_length: int = 512):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = max_length

        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def _mean_pool(last_hidden, attn_mask, torch):
        mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def encode(self, texts: List[str], batch_size: int = 32):
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                enc = self.tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                res = self.model(**enc)
                last_hidden = res.last_hidden_state
                emb = self._mean_pool(last_hidden, enc["attention_mask"], torch)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                out.append(emb.detach().cpu())
        return torch.cat(out, dim=0)


def cosine_from_normed(a_vec, b_vec) -> float:
    # vectors are normalized => dot = cosine
    return float((a_vec * b_vec).sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--settings", default="sft")
    ap.add_argument("--use_clean", action="store_true")

    ap.add_argument("--emb_model_path", required=True, help="embedding模型路径（本地或HF repo）")
    ap.add_argument("--device", default="cuda", help="cuda / cpu")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--acc_thres", type=float, default=0.70)

    ap.add_argument("--out_csv", default="tables/answer_only_v9.csv")
    ap.add_argument("--out_md", default="tables/answer_only_v9.md")

    args = ap.parse_args()

    test = read_jsonl(Path(args.test))
    gold_map: Dict[str, dict] = {str(it.get("qid")): it for it in test if it.get("qid") is not None}

    embedder = HFEmbedder(args.emb_model_path, device=args.device, max_length=args.max_length)

    rows = []
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]

    for setting in settings:
        run_path = Path(args.runs_dir) / f"{setting}.jsonl"
        runs = read_jsonl(run_path)

        n = 0
        valid = 0
        lat_sum = 0.0

        emb_full_sum = 0.0
        emb_sentmax_sum = 0.0
        emb_acc = 0.0
        tokf1_sum = 0.0

        # 为了效率：先收集本 setting 的所有 (gold, pred, pred_sents)
        samples = []
        for r in runs:
            qid = str(r.get("qid", ""))
            g = gold_map.get(qid)
            if not g:
                continue
            n += 1
            gold = str(g.get("gold_answer") or "")
            pred = extract_pred_answer(r)

            if args.use_clean:
                gold = clean_answer(gold)
                pred = clean_answer(pred)

            if gold.strip() and pred.strip():
                valid += 1
                sents = split_sentences(pred)[:12]
                samples.append((gold, pred, sents))

            lat_sum += float(r.get("latency_ms") or 0.0)

        # 批量编码：gold、pred_full、pred_sents
        # gold_vecs: [valid, d]
        gold_texts = [g for g, _, _ in samples]
        pred_full_texts = [p for _, p, _ in samples]
        gold_vecs = embedder.encode(gold_texts, batch_size=32)
        pred_full_vecs = embedder.encode(pred_full_texts, batch_size=32)

        # 逐样本算 EmbSimFull / EmbSimSentMax
        for idx, (gold, pred, sents) in enumerate(samples):
            gvec = gold_vecs[idx]
            pvec = pred_full_vecs[idx]
            emb_full = cosine_from_normed(pvec, gvec)
            emb_full_sum += emb_full

            sent_vecs = embedder.encode(sents, batch_size=16) if sents else None
            emb_sentmax = max((cosine_from_normed(v, gvec) for v in sent_vecs), default=0.0) if sent_vecs is not None else 0.0
            emb_sentmax_sum += emb_sentmax
            emb_acc += 1.0 if emb_sentmax >= args.acc_thres else 0.0

            tokf1_sum += token_f1(pred, gold)

        row = {
            "Setting": setting,
            "N": n,
            "ValidN": valid,
            "EmbSimFull": emb_full_sum / max(1, valid),
            "EmbSimSentMax": emb_sentmax_sum / max(1, valid),
            f"EmbAccSentMax@{args.acc_thres:.2f}": emb_acc / max(1, valid),
            "TokF1": tokf1_sum / max(1, valid),
            "AvgLatency(ms)": lat_sum / max(1, n),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    Path(args.out_md).write_text(df.to_markdown(index=False), encoding="utf-8")

    print(df.to_dict(orient="records"))
    print(f"[ok] wrote: {args.out_csv}")
    print(f"[ok] wrote: {args.out_md}")


if __name__ == "__main__":
    main()
