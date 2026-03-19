# -*- coding: utf-8 -*-
"""
eval_matrics_answer_only_v8.py

你现在的现象（AnswerSim 很低）主要来自：gold_answer 很短、pred_answer 很长且包含额外解释。
简单的“整段 vs 整段”相似度会被长度/噪声严重稀释。

v8 解决：提供“对短 gold 友好”的评估方式（不看引用、不看证据）：
1) Sentence-Max Similarity：把 pred 切成句子，计算每句与 gold 的相似度，取最大值（对“先解释再给答案”特别有效）
2) Token-F1：英文按词、中文按字的 F1（更像信息覆盖率）
3) 可选 Embedding 语义相似度：用 sentence-transformers 编码（推荐 bge-small-en-v1.5 / bge-small-zh-v1.5）
   - 支持 --emb_model_path 本地路径；不提供则跳过

输出指标（每个 setting 一行）：
- N / ValidN
- SimFull: pred整段 vs gold（参考值）
- SimSentMax: pred句子max vs gold（推荐主指标）
- TokF1: token-level F1（推荐辅助指标）
- AccSentMax@T: SimSentMax >= T 的比例（T 默认 0.65，你也可以改成 0.5）
- (可选) EmbSimFull / EmbSimSentMax / EmbAccSentMax@E
- AvgLatency(ms)

依赖（仅当你启用 embedding）：
pip install -U sentence-transformers torch
"""

from __future__ import annotations
import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

_RE_WS = re.compile(r"\s+")
_RE_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_RE_SENT = re.compile(r"(?<=[\.\!\?。！？])\s+")


# ============== IO ==============
def read_jsonl(path: Path) -> List[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ============== Text utils ==============
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

    # 去掉重复句（保序）
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
    # 如果没切开，至少返回整段
    return out if out else [s]


# ============== Similarity (lexical cosine) ==============
def _cosine_counter(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def word_ngrams(s: str, n: int = 1) -> List[str]:
    toks = _RE_WORD.findall((s or "").lower())
    if not toks:
        return []
    if n == 1:
        return toks
    if len(toks) < n:
        return [" ".join(toks)]
    return [" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)]


def char_ngrams(s: str, n: int = 2) -> List[str]:
    s = (s or "").lower().strip()
    s = _RE_WS.sub("", s)
    if not s:
        return []
    if len(s) < n:
        return [s]
    return [s[i:i+n] for i in range(len(s)-n+1)]


def lexical_sim(a: str, b: str) -> float:
    """
    英文：词级（unigram+bigram cosine 平均）
    中文/混合：char bigram cosine
    """
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0
    if _ascii_ratio(a + b) >= 0.75:
        c1a = Counter(word_ngrams(a, 1)); c1b = Counter(word_ngrams(b, 1))
        c2a = Counter(word_ngrams(a, 2)); c2b = Counter(word_ngrams(b, 2))
        return 0.5 * _cosine_counter(c1a, c1b) + 0.5 * _cosine_counter(c2a, c2b)
    else:
        ca = Counter(char_ngrams(a, 2)); cb = Counter(char_ngrams(b, 2))
        return _cosine_counter(ca, cb)


# ============== Token F1 ==============
def tokenize_for_f1(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    if _ascii_ratio(s) >= 0.75:
        return _RE_WORD.findall(s.lower())
    # 中文/混合：按“中日韩统一表意文字”单字 + 数字/英文词
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
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ============== Optional Embedding (sentence-transformers) ==============
class Embedder:
    def __init__(self, model_path: str, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model = SentenceTransformer(model_path, device=(device or None))

    def encode(self, texts: List[str]):
        # normalize_embeddings=True => cosine just dot
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)

def emb_cos(a_vec, b_vec) -> float:
    # embeddings already normalized => dot is cosine
    return float((a_vec * b_vec).sum())


# ============== Pred answer extraction ==============
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


# ============== Main ==============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--settings", default="sft")

    ap.add_argument("--use_clean", action="store_true")
    ap.add_argument("--acc_thres", type=float, default=0.65, help="用于 AccSentMax@T 的阈值（建议 0.50~0.65）")

    ap.add_argument("--out_csv", default="tables/answer_only_v8.csv")
    ap.add_argument("--out_md", default="tables/answer_only_v8.md")

    # optional embedding
    ap.add_argument("--emb_model_path", default="", help="本地 embedding 模型路径（例如 bge-small-en-v1.5）")
    ap.add_argument("--emb_device", default="", help="cuda/cpu, 留空让库自动")

    args = ap.parse_args()

    test = read_jsonl(Path(args.test))
    gold_map: Dict[str, dict] = {str(it.get("qid")): it for it in test if it.get("qid") is not None}

    embedder = None
    if args.emb_model_path:
        embedder = Embedder(args.emb_model_path, device=(args.emb_device or None))

    rows = []
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]

    for setting in settings:
        run_path = Path(args.runs_dir) / f"{setting}.jsonl"
        runs = read_jsonl(run_path)

        n = 0
        valid = 0
        lat_sum = 0.0

        sim_full_sum = 0.0
        sim_sentmax_sum = 0.0
        tokf1_sum = 0.0
        acc_sentmax = 0.0

        emb_full_sum = 0.0
        emb_sentmax_sum = 0.0
        emb_acc_sentmax = 0.0
        emb_n = 0

        for r in runs:
            qid = str(r.get("qid", ""))
            g = gold_map.get(qid)
            if not g:
                continue
            n += 1

            gold = str(g.get("gold_answer") or "")
            pred = extract_pred_answer(r)

            if args.use_clean:
                gold_eval = clean_answer(gold)
                pred_eval = clean_answer(pred)
            else:
                gold_eval = gold
                pred_eval = pred

            if gold_eval.strip() and pred_eval.strip():
                valid += 1
                # lexical sims
                sim_full = lexical_sim(pred_eval, gold_eval)
                sim_full_sum += sim_full

                sents = split_sentences(pred_eval)
                sim_sentmax = max((lexical_sim(s, gold_eval) for s in sents), default=0.0)
                sim_sentmax_sum += sim_sentmax

                tokf1 = token_f1(pred_eval, gold_eval)
                tokf1_sum += tokf1

                acc_sentmax += 1.0 if sim_sentmax >= args.acc_thres else 0.0

                # embeddings
                if embedder is not None:
                    # encode gold once per sample; pred sentence-max
                    vecs = embedder.encode([gold_eval] + sents[:12])  # cap sentences for speed
                    gvec = vecs[0]
                    pvec = vecs[1:]
                    e_full = emb_cos(embedder.encode([pred_eval])[0], gvec)
                    emb_full_sum += e_full
                    e_sentmax = max((emb_cos(v, gvec) for v in pvec), default=0.0)
                    emb_sentmax_sum += e_sentmax
                    emb_acc_sentmax += 1.0 if e_sentmax >= args.acc_thres else 0.0
                    emb_n += 1

            lat_sum += float(r.get("latency_ms") or 0.0)

        row = {
            "Setting": setting,
            "N": n,
            "ValidN": valid,
            "SimFull": sim_full_sum / max(1, valid),
            "SimSentMax": sim_sentmax_sum / max(1, valid),
            "TokF1": tokf1_sum / max(1, valid),
            f"AccSentMax@{args.acc_thres:.2f}": acc_sentmax / max(1, valid),
            "AvgLatency(ms)": lat_sum / max(1, n),
        }

        if embedder is not None:
            row.update({
                "EmbN": emb_n,
                "EmbSimFull": emb_full_sum / max(1, emb_n),
                "EmbSimSentMax": emb_sentmax_sum / max(1, emb_n),
                f"EmbAccSentMax@{args.acc_thres:.2f}": emb_acc_sentmax / max(1, emb_n),
            })

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
