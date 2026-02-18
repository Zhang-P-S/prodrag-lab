# scripts/eval_metrics_v2.py
# -*- coding: utf-8 -*-
"""
从 runs/*.jsonl 计算指标，并输出对比表（Markdown + CSV）

为什么要 v2：
- v1 更偏 “pipeline（检索/重排/拒答阈值）”评估：Recall/MRR/CitePrec/RefusalRate/Latency
- v2 增加 “SFT/LoRA 行为评估” 必需指标：
  1) Refusal Precision / Recall / F1：该拒才拒、不该拒不拒
  2) CitationSupportRate：引用是否被检索证据支撑（更硬的证据闭环）
  3) HallucinationRate（启发式）：证据链不闭环的风险比例

⚠️ 注意：
- HallucinationRate 是工程 proxy（不做事实级别核验），但对“RAG 可信/可控”非常实用。
- 真正论文级 hallucination 需要：原文证据 + claim-level 标注 或 LLM judge。

中文注释占比>40%（便于你写笔记/给面试官讲清楚）。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -------------------------
# 1) IO：读 jsonl
# -------------------------
def read_jsonl(p: Path) -> List[dict]:
    """读取一份 jsonl，返回 list[dict]。"""
    items: List[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


# -------------------------
# 2) Retrieval 指标
# -------------------------
def recall_at_k(retrieved: List[str], gold: List[str], k: int) -> float:
    """
    Recall@K：gold 里任意一个出现在 topK 就算命中。
    你的 gold 通常是 1 个 chunk（但也兼容多个）。
    """
    topk = set(retrieved[:k])
    return 1.0 if any(g in topk for g in gold) else 0.0


def mrr_at_k(retrieved: List[str], gold: List[str], k: int) -> float:
    """
    MRR@K：gold 第一次出现的排名倒数；没出现=0。
    """
    topk = retrieved[:k]
    for idx, cid in enumerate(topk, 1):
        if cid in gold:
            return 1.0 / idx
    return 0.0


# -------------------------
# 3) 引用/证据闭环指标
# -------------------------
def citation_precision(used: List[str], evidence: List[str]) -> float:
    """
    CitePrec（引用精度）：
    - used_citations 是否都在 evidence（retrieved_chunk_ids）里
    - used 为空：记为 0（说明没引用，不符合“强制引用”目标）
    """
    if not used:
        return 0.0
    ev = set(evidence)
    hit = sum(1 for c in used if c in ev)
    return hit / max(len(used), 1)


def citation_support_ok(used: List[str], evidence: List[str]) -> bool:
    """
    引用是否“证据闭环（硬约束）”：
    - used 不为空
    - used 的每一个引用都必须来自 retrieved evidence
    """
    if not used:
        return False
    ev = set(evidence)
    return all(c in ev for c in used)


# -------------------------
# 4) Refusal（拒答）评估：Precision/Recall/F1
# -------------------------
@dataclass
class PRF1:
    precision: float
    recall: float
    f1: float


def prf1(tp: int, fp: int, fn: int) -> PRF1:
    """计算 Precision/Recall/F1，处理除零。"""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return PRF1(precision=prec, recall=rec, f1=f1)


# -------------------------
# 5) 核心聚合逻辑
# -------------------------
def aggregate(run_items: List[dict], k: int = 10) -> Dict[str, float]:
    """
    汇总一份 runs/*.jsonl 的指标。

    你现在的 jsonl 字段习惯（根据你项目日志）：
    - retrieved_chunk_ids: List[str]  检索到的 chunk ids
    - gold_citations: List[str]       标准答案对应的 gold chunk ids
    - used_citations: List[str]       模型输出中使用的引用
    - final_refusal: bool             “最终应拒答”（真值标签，y_true）
    - model_refusal: bool             模型是否拒答（预测标签，y_pred）
    - timing_ms.total / wall_ms       用于延迟统计
    """
    n = len(run_items)
    if n == 0:
        return {}

    # ---------- retrieval ----------
    r10 = 0.0
    mrr10 = 0.0

    # ---------- citation ----------
    citep = 0.0

    # ---------- refusal ----------
    # y_true = final_refusal, y_pred = model_refusal
    tp = fp = fn = 0
    refusals_pred = 0  # model_refusal 的比例（和旧 RefusalRate 对齐更合理）
    refusals_true = 0  # final_refusal 的比例（数据集“应拒答”占比）

    # ---------- support / hallucination ----------
    non_refusal_cnt = 0
    support_ok_cnt = 0
    halluc_cnt = 0

    # ---------- latency ----------
    lat = 0.0

    for it in run_items:
        retrieved = it.get("retrieved_chunk_ids", []) or []
        gold = it.get("gold_citations", []) or []
        used = it.get("used_citations", []) or []

        # 1) retrieval
        r10 += recall_at_k(retrieved, gold, k)
        mrr10 += mrr_at_k(retrieved, gold, k)

        # 2) citation precision（平均意义）
        cp = citation_precision(used, retrieved)
        citep += cp

        # 3) refusal：用 final_refusal 作为真值，用 model_refusal 作为预测
        y_true = bool(it.get("final_refusal", False))
        y_pred = bool(it.get("model_refusal", False))

        if y_pred:
            refusals_pred += 1
        if y_true:
            refusals_true += 1

        # confusion matrix（拒答作为“正类”）
        if y_true and y_pred:
            tp += 1
        elif (not y_true) and y_pred:
            fp += 1
        elif y_true and (not y_pred):
            fn += 1

        # 4) 只在“非拒答样本”上计算证据闭环/胡编 proxy
        #    因为拒答本身不需要引用，也不涉及 factual claims。
        if not y_pred:
            non_refusal_cnt += 1

            ok = citation_support_ok(used, retrieved)
            if ok:
                support_ok_cnt += 1

            # HallucinationRate（启发式）：
            # - 没引用（used 为空）或者
            # - 引用里有不在 retrieved 的（cp < 1.0）
            # 这两个都意味着“证据链不闭环”，在企业场景一般当作高风险。
            if (not used) or (cp < 1.0):
                halluc_cnt += 1

        # 5) latency
        tms = it.get("timing_ms", {}) or {}
        lat += float(tms.get("total", it.get("wall_ms", 0.0)))

    # refusal 的 precision/recall/f1
    pr = prf1(tp=tp, fp=fp, fn=fn)

    # support / hallucination 的分母：非拒答样本数
    support_rate = (support_ok_cnt / non_refusal_cnt) if non_refusal_cnt > 0 else 0.0
    halluc_rate = (halluc_cnt / non_refusal_cnt) if non_refusal_cnt > 0 else 0.0

    return {
        "n": float(n),

        # retrieval
        "recall@10": r10 / n,
        "mrr@10": mrr10 / n,

        # citation
        "cite_prec": citep / n,
        "citation_support_rate": support_rate,

        # refusal
        "refusal_rate_pred": refusals_pred / n,  # 预测拒答比例（你原来的 RefusalRate 更接近这个）
        "refusal_rate_true": refusals_true / n,  # 数据集应拒答比例（用于 sanity check）
        "refusal_precision": pr.precision,
        "refusal_recall": pr.recall,
        "refusal_f1": pr.f1,

        # hallucination（proxy）
        "hallucination_rate": halluc_rate,

        # efficiency
        "avg_latency_ms": lat / n,
    }


# -------------------------
# 6) 输出：Markdown + CSV
# -------------------------
def write_markdown_table(out_md: Path, rows: List[Dict[str, str]]) -> None:
    """写 Markdown 表：方便你贴 README / 报告。"""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with out_md.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[h]) for h in headers) + " |\n")


def write_csv(out_csv: Path, rows: List[Dict[str, str]]) -> None:
    """写 CSV：方便你 Excel/脚本二次处理。"""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


# -------------------------
# 7) CLI：支持多份 runs 对比
# -------------------------
def parse_runs(run_args: List[str]) -> List[Tuple[str, Path]]:
    """
    解析 --run 形如：
      --run baseline=artifacts/rag_eval/baseline.jsonl
      --run sft=artifacts/rag_eval/baseline_sft.jsonl

    返回：[(name, path), ...]
    """
    runs: List[Tuple[str, Path]] = []
    for s in run_args:
        if "=" not in s:
            raise ValueError(f"--run 参数格式应为 name=path，但你给的是: {s}")
        name, path = s.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name:
            raise ValueError(f"--run name 不能为空: {s}")
        runs.append((name, Path(path)))
    return runs


def main():
    ap = argparse.ArgumentParser()

    # ✅ 推荐用法：多 runs 对比
    ap.add_argument(
        "--run",
        action="append",
        default=[],
        help="对比输入，格式 name=path，可重复多次，例如 --run baseline=a.jsonl --run sft=b.jsonl",
    )

    # ✅ 兼容旧用法：只给一份 data（会当作 setting=single）
    ap.add_argument("--data", type=str, default=None, help="（兼容旧版本）单一输入 jsonl")
    ap.add_argument("--setting", type=str, default="single", help="配合 --data 使用的 setting 名")

    ap.add_argument("--k", type=int, default=10, help="Recall/MRR 的 K（默认 10）")

    ap.add_argument("--out_md", type=str, default="tables/compare_sft_v2.md")
    ap.add_argument("--out_csv", type=str, default="tables/compare_sft_v2.csv")
    args = ap.parse_args()

    # 1) 组装 runs
    runs: List[Tuple[str, Path]] = parse_runs(args.run) if args.run else []
    if args.data:
        runs.append((args.setting, Path(args.data)))

    if not runs:
        raise ValueError("你需要至少提供一个输入：用 --run name=path 或者 --data path")

    # 2) 计算每个 run 的指标
    def fmt(x: float) -> str:
        return f"{x:.3f}"

    rows: List[Dict[str, str]] = []
    for name, path in runs:
        items = read_jsonl(path)
        agg = aggregate(items, k=args.k)

        row = {
            "Setting": name,
            "N": int(agg["n"]),
            "Recall@10": fmt(agg["recall@10"]),
            "MRR@10": fmt(agg["mrr@10"]),
            "CitePrec": fmt(agg["cite_prec"]),
            "CitationSupport": fmt(agg["citation_support_rate"]),
            "HallucRate": fmt(agg["hallucination_rate"]),
            "RefusalRate(pred)": fmt(agg["refusal_rate_pred"]),
            "RefusalPrec": fmt(agg["refusal_precision"]),
            "RefusalRec": fmt(agg["refusal_recall"]),
            "RefusalF1": fmt(agg["refusal_f1"]),
            "AvgLatency(ms)": fmt(agg["avg_latency_ms"]),
        }
        rows.append(row)

    # 3) 输出 Markdown + CSV
    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)
    write_markdown_table(out_md, rows)
    write_csv(out_csv, rows)

    # 4) 终端预览（面试官常看你日志输出）
    print("[eval_metrics_v2] wrote:", str(out_md), "and", str(out_csv))
    print("[eval_metrics_v2] preview:")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
