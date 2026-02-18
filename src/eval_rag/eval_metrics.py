#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/eval_rag/eval_metrics.py

RAG 离线评测指标汇总（支持 baseline / +rerank / +refusal）
输入：runs/*.jsonl（每行一个样本的评测结果）
输出：
- tables/rag_metrics.md  （Markdown 表）
- tables/rag_metrics.csv （同内容 CSV）

指标定义（与常见 RAG 评测一致）：
1) Recall@10：
   - 看 top10 的 retrieved_chunk_ids 是否命中任意 gold_citations
   - 命中=1，否则=0，最后取平均

2) MRR@10：
   - 在 top10 里找到第一个命中 gold 的位置 rank（从1开始）
   - 得分=1/rank；若 top10 无命中则 0，最后取平均

3) CitePrec（citation precision）：
   - 每个样本： precision_i = |used_citations ∩ gold_citations| / |used_citations|
   - used_citations 为空时 precision_i=0（惩罚“没给引用/引用缺失”）
   - 最后对所有样本取平均（macro average）

4) RefusalRate：
   - final_refusal 的平均值

5) AvgLatency(ms)：
   - 优先取 wall_ms；若没有则取 timing_ms.total；都没有则跳过该样本（不计入均值分母）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# IO：读取 JSONL（鲁棒，跳过坏行）
# -----------------------------
def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = str(path)
    items: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"jsonl not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    items.append(obj)
            except Exception:
                # 坏行直接跳过，避免一次脏数据导致全表崩
                print(f"[WARN] skip bad json line: {path}:{ln}")
                continue
    return items


def ensure_parent_dir(p: str | Path) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 工具函数：安全取字段
# -----------------------------
def as_list_str(x: Any) -> List[str]:
    """把输入尽可能转成 string list；不合法就返回空"""
    if x is None:
        return []
    if isinstance(x, str):
        x = [x]
    if not isinstance(x, list):
        return []
    out: List[str] = []
    for it in x:
        if isinstance(it, str) and it.strip():
            out.append(it.strip())
    return out


def as_bool(x: Any) -> bool:
    """把输入尽可能转成 bool"""
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
    return False


def get_latency_ms(obj: Dict[str, Any]) -> Optional[float]:
    """
    延迟字段兜底：
    - 优先 wall_ms（你 eval_run 里有）
    - 否则 timing_ms.total
    """
    v = obj.get("wall_ms", None)
    if isinstance(v, (int, float)):
        return float(v)

    timing = obj.get("timing_ms", None)
    if isinstance(timing, dict):
        v2 = timing.get("total", None)
        if isinstance(v2, (int, float)):
            return float(v2)

    return None


# -----------------------------
# 指标计算（单样本）
# -----------------------------
def recall_at_k(gold: List[str], retrieved: List[str], k: int = 10) -> float:
    """top-k 是否命中任一 gold"""
    if not gold:
        return 0.0
    topk = retrieved[:k]
    gold_set = set(gold)
    return 1.0 if any(x in gold_set for x in topk) else 0.0


def mrr_at_k(gold: List[str], retrieved: List[str], k: int = 10) -> float:
    """top-k 的 reciprocal rank（第一个命中 gold 的位置）"""
    if not gold:
        return 0.0
    gold_set = set(gold)
    topk = retrieved[:k]
    for i, cid in enumerate(topk, 1):  # rank 从 1 开始
        if cid in gold_set:
            return 1.0 / float(i)
    return 0.0


def cite_precision(gold: List[str], used: List[str]) -> float:
    """
    引用精度：
    - used 为空 -> 0（惩罚）
    - 否则 |used∩gold| / |used|
    """
    if not used:
        return 0.0
    gold_set = set(gold)
    hit = sum(1 for c in used if c in gold_set)
    return float(hit) / float(len(used))


# -----------------------------
# 汇总结果结构
# -----------------------------
@dataclass
class MetricRow:
    setting: str
    n: int
    recall10: float
    mrr10: float
    citeprec: float
    refusal_rate: float
    avg_latency_ms: float


def infer_setting_label(run_path: str, items: List[Dict[str, Any]]) -> str:
    """
    setting 名称推断：
    - 优先用文件里每行自带的 setting（若存在）
    - 否则按文件名 baseline/rerank/refusal 映射成你图里的格式
    """
    # 1) 从内容里取（更靠谱）
    s = None
    if items:
        s = items[0].get("setting", None)
        if isinstance(s, str) and s.strip():
            # 统一成你图里的展示：baseline / +rerank / +refusal
            ss = s.strip().lower()
            if ss == "baseline":
                return "baseline"
            if "rerank" in ss:
                return "+rerank"
            if "refusal" in ss:
                return "+refusal"
            return s.strip()

    # 2) 从文件名猜
    name = Path(run_path).name.lower()
    if "baseline" in name:
        return "baseline"
    if "rerank" in name:
        return "+rerank"
    if "refusal" in name:
        return "+refusal"
    return Path(run_path).stem


def eval_one_run(run_path: str) -> MetricRow:
    items = read_jsonl(run_path)
    setting = infer_setting_label(run_path, items)

    # 用于累计（macro average）
    n = 0
    sum_recall10 = 0.0
    sum_mrr10 = 0.0
    sum_citeprec = 0.0
    sum_refusal = 0.0

    # 延迟：允许某些样本缺字段，分母单独算
    sum_latency = 0.0
    cnt_latency = 0

    for obj in items:
        n += 1

        gold = as_list_str(obj.get("gold_citations"))
        retrieved = as_list_str(obj.get("retrieved_chunk_ids"))
        used = as_list_str(obj.get("used_citations"))

        sum_recall10 += recall_at_k(gold, retrieved, k=10)
        sum_mrr10 += mrr_at_k(gold, retrieved, k=10)
        sum_citeprec += cite_precision(gold, used)

        sum_refusal += 1.0 if as_bool(obj.get("final_refusal")) else 0.0

        lat = get_latency_ms(obj)
        if lat is not None:
            sum_latency += float(lat)
            cnt_latency += 1

    # 防止空文件除 0
    denom = float(n) if n > 0 else 1.0
    avg_latency = (sum_latency / float(cnt_latency)) if cnt_latency > 0 else 0.0

    return MetricRow(
        setting=setting,
        n=n,
        recall10=sum_recall10 / denom,
        mrr10=sum_mrr10 / denom,
        citeprec=sum_citeprec / denom,
        refusal_rate=sum_refusal / denom,
        avg_latency_ms=avg_latency,
    )


# -----------------------------
# 输出：Markdown + CSV
# -----------------------------
def format_md_table(rows: List[MetricRow]) -> str:
    # 表头与图一致
    header = ["Setting", "N", "Recall@10", "MRR@10", "CitePrec", "RefusalRate", "AvgLatency(ms)"]
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for r in rows:
        lines.append(
            "| {setting} | {n} | {recall:.3f} | {mrr:.3f} | {citeprec:.3f} | {refusal:.3f} | {lat:.3f} |".format(
                setting=r.setting,
                n=r.n,
                recall=r.recall10,
                mrr=r.mrr10,
                citeprec=r.citeprec,
                refusal=r.refusal_rate,
                lat=r.avg_latency_ms,
            )
        )
    return "\n".join(lines) + "\n"


def write_csv(out_csv: str | Path, rows: List[MetricRow]) -> None:
    ensure_parent_dir(out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Setting", "N", "Recall@10", "MRR@10", "CitePrec", "RefusalRate", "AvgLatency(ms)"])
        for r in rows:
            w.writerow(
                [
                    r.setting,
                    r.n,
                    f"{r.recall10:.3f}",
                    f"{r.mrr10:.3f}",
                    f"{r.citeprec:.3f}",
                    f"{r.refusal_rate:.3f}",
                    f"{r.avg_latency_ms:.3f}",
                ]
            )


def write_md(out_md: str | Path, content: str) -> None:
    ensure_parent_dir(out_md)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(content)


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        nargs="*",
        default=["runs/baseline.jsonl", "runs/rerank.jsonl", "runs/refusal.jsonl"],
        help="jsonl run paths (default: baseline/rerank/refusal)",
    )
    ap.add_argument("--out_md", type=str, default="tables/rag_metrics.md")
    ap.add_argument("--out_csv", type=str, default="tables/rag_metrics.csv")
    args = ap.parse_args()

    rows: List[MetricRow] = []
    for p in args.runs:
        row = eval_one_run(p)
        rows.append(row)

    # 排序：baseline、+rerank、+refusal（更像你截图）
    order = {"baseline": 0, "+rerank": 1, "+refusal": 2}
    rows.sort(key=lambda r: order.get(r.setting, 999))

    md = format_md_table(rows)
    write_md(args.out_md, md)
    write_csv(args.out_csv, rows)

    print(f"[eval_metrics] wrote: {args.out_md} and {args.out_csv}")
    print("[eval_metrics] preview:")
    for r in rows:
        print(
            {
                "Setting": r.setting,
                "N": r.n,
                "Recall@10": f"{r.recall10:.3f}",
                "MRR@10": f"{r.mrr10:.3f}",
                "CitePrec": f"{r.citeprec:.3f}",
                "RefusalRate": f"{r.refusal_rate:.3f}",
                "AvgLatency(ms)": f"{r.avg_latency_ms:.3f}",
            }
        )


if __name__ == "__main__":
    main()
