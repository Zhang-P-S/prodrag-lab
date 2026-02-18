# scripts/eval_metrics_strict.py
# -*- coding: utf-8 -*-
"""
严格评测 runs/*.jsonl，并输出 Markdown + CSV 对比表

为什么要“严格版”：
- 你的 run 里经常出现：
  llm_parse_meta.parsed_json=false, fallback=true, citations_autofilled=true
  这意味着 used_citations 可能是系统补的，而非模型真实输出
- 如果不区分“模型引用 vs 系统补引用”，CitePrec / CitationSupport / HallucRate 会虚高

本脚本的核心理念（面试官喜欢）：
A. Retrieval 评估：Recall@K / MRR@K
B. Citation 评估拆两类：
   1) Evidence-closure：引用是否来自检索证据（retrieved_chunk_ids）
   2) Gold-alignment：引用是否命中 gold_citations
C. Refusal 行为：用 final_refusal 作为真值标签，算 PRF1
D. 解析与格式：ParseSuccessRate / AutoFillRate / NoCitationRate（避免“指标看起来好但其实是 parser 在擦屁股”）
E. Hallucination 是 proxy：证据链不闭环的风险，而非事实核验

中文注释占比>40%（便于你写项目笔记/面试讲解）。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# -------------------------
# 1) IO：读 jsonl
# -------------------------
def read_jsonl(p: Path) -> List[dict]:
    """读取 jsonl：每行一个 dict。"""
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
    Recall@K：
    - gold 里任意一个出现在 topK 就算命中（你的 gold 通常是 1 个，但兼容多个）
    """
    if not gold:
        return 0.0
    topk = set(retrieved[:k])
    return 1.0 if any(g in topk for g in gold) else 0.0


def mrr_at_k(retrieved: List[str], gold: List[str], k: int) -> float:
    """
    MRR@K：
    - gold 第一次出现的排名倒数；没出现=0
    """
    if not gold:
        return 0.0
    for idx, cid in enumerate(retrieved[:k], 1):
        if cid in gold:
            return 1.0 / idx
    return 0.0


# -------------------------
# 3) Refusal（拒答）PRF1
# -------------------------
@dataclass
class PRF1:
    precision: float
    recall: float
    f1: float


def prf1(tp: int, fp: int, fn: int) -> PRF1:
    """标准 PRF1，处理除零。"""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return PRF1(precision=p, recall=r, f1=f)


# -------------------------
# 4) Citation 指标（严格版）
# -------------------------
def is_autofilled(it: dict) -> bool:
    """
    判断该样本的引用是否可能由系统 autofill 而来。
    你的 run 里有 llm_parse_meta.citations_autofilled 字段（非常关键）
    """
    meta = it.get("llm_parse_meta", {}) or {}
    return bool(meta.get("citations_autofilled", False))


def is_parse_success(it: dict) -> bool:
    """
    判断是否解析成功：
    - parsed_json=true 表示模型按格式输出可解析 JSON
    - 这会影响“我们是否相信 used_citations 是模型真实产物”
    """
    meta = it.get("llm_parse_meta", {}) or {}
    return bool(meta.get("parsed_json", False))


def sanitize_used_citations(it: dict, strict_ignore_autofill: bool = True) -> List[str]:
    """
    得到“可信 used_citations”。

    strict_ignore_autofill=True（推荐）：
    - 如果 citations_autofilled=true，则认为 used_citations 不可信，直接置空
    - 这样 Cite 指标不会被系统补引用“刷分”
    """
    used = it.get("used_citations", []) or []
    if strict_ignore_autofill and is_autofilled(it):
        return []
    # 去重但保持顺序（面试可讲：避免重复引用影响 precision）
    seen = set()
    dedup: List[str] = []
    for c in used:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


def cite_precision_against_evidence(used: List[str], retrieved: List[str]) -> float:
    """
    Evidence-closure Precision（证据闭环精度）：
    - 模型给出的引用中，有多少来自检索证据 retrieved_chunk_ids
    - used 为空：记 0（因为你强调“必须引用”）
    """
    if not used:
        return 0.0
    ev = set(retrieved)
    hit = sum(1 for c in used if c in ev)
    return hit / len(used)


def cite_support_all_in_evidence(used: List[str], retrieved: List[str]) -> float:
    """
    CitationSupport（严格版，样本级）：
    - used 非空 且 used 全部属于 retrieved 才算 1，否则 0
    - 这是“硬闭环”的定义：引用必须来自证据池
    """
    if not used:
        return 0.0
    ev = set(retrieved)
    return 1.0 if all(c in ev for c in used) else 0.0


def gold_hit_rate(used: List[str], gold: List[str]) -> float:
    """
    GoldHit（命中 gold 的比例，样本级）：
    - used 中是否包含任意 gold：有则 1，无则 0
    - 用来回答：模型引用有没有“至少碰到正确证据”
    """
    if not used or not gold:
        return 0.0
    g = set(gold)
    return 1.0 if any(c in g for c in used) else 0.0


def cite_precision_against_gold(used: List[str], gold: List[str]) -> float:
    """
    Gold-alignment Precision：
    - 模型引用里有多少是 gold
    - used 为空：0
    """
    if not used:
        return 0.0
    g = set(gold)
    hit = sum(1 for c in used if c in g)
    return hit / len(used)


def cite_recall_against_gold(used: List[str], gold: List[str]) -> float:
    """
    Gold-alignment Recall：
    - gold 被模型引用覆盖了多少
    - gold 为空：0（或你也可以选择跳过，这里保守返回 0）
    """
    if not gold:
        return 0.0
    if not used:
        return 0.0
    u = set(used)
    hit = sum(1 for g in gold if g in u)
    return hit / len(gold)


def f1_from_pr(p: float, r: float) -> float:
    """从 P/R 计算 F1。"""
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


# -------------------------
# 5) 聚合逻辑（严格、可解释）
# -------------------------
def aggregate(run_items: List[dict], k: int = 10, strict_ignore_autofill: bool = True) -> Dict[str, float]:
    """
    汇总指标。

    strict_ignore_autofill：
    - True：把 citations_autofilled 的样本当成“无有效引用”（推荐，避免刷分）
    - False：沿用 used_citations（适合做对照实验：看 autofill 把指标抬高多少）
    """
    n = len(run_items)
    if n == 0:
        return {}

    # ---------- retrieval ----------
    r_at_k = 0.0
    mrr_k = 0.0

    # ---------- citation：evidence-closure ----------
    ev_cite_prec_sum = 0.0
    citation_support_sum = 0.0

    # ---------- citation：gold-alignment ----------
    gold_hit_sum = 0.0
    gold_prec_sum = 0.0
    gold_rec_sum = 0.0
    gold_f1_sum = 0.0

    # ---------- refusal ----------
    tp = fp = fn = 0
    refusals_pred = 0
    refusals_true = 0

    # ---------- parsing / hygiene（非常关键：解释指标可信度） ----------
    parse_ok = 0
    autofill_cnt = 0
    no_citation_cnt = 0

    # ---------- hallucination proxy（严格版分解） ----------
    # 注意：HallucinationRate 是 proxy，建议拆出 NoCitationRate / BadEvidenceRate
    non_refusal_cnt = 0
    bad_evidence_cnt = 0  # used 非空但不全在 retrieved
    halluc_proxy_cnt = 0  # (无引用) 或 (引用不闭环) 的风险比例

    # ---------- latency ----------
    lat_sum = 0.0

    for it in run_items:
        retrieved = it.get("retrieved_chunk_ids", []) or []
        gold = it.get("gold_citations", []) or []

        # 1) retrieval
        r_at_k += recall_at_k(retrieved, gold, k)
        mrr_k += mrr_at_k(retrieved, gold, k)

        # 2) refusal labels
        y_true = bool(it.get("final_refusal", False))
        y_pred = bool(it.get("model_refusal", False))

        if y_pred:
            refusals_pred += 1
        if y_true:
            refusals_true += 1

        if y_true and y_pred:
            tp += 1
        elif (not y_true) and y_pred:
            fp += 1
        elif y_true and (not y_pred):
            fn += 1

        # 3) parse / autofill 状态（用于 sanity check）
        if is_parse_success(it):
            parse_ok += 1
        if is_autofilled(it):
            autofill_cnt += 1

        # 4) citations：先“消毒”，避免把 autofill 当模型能力
        used = sanitize_used_citations(it, strict_ignore_autofill=strict_ignore_autofill)
        if not used:
            no_citation_cnt += 1

        # 5) evidence-closure 指标（对全部样本统计，也可只在 non-refusal 统计；这里保留全量平均）
        ev_p = cite_precision_against_evidence(used, retrieved)
        ev_cite_prec_sum += ev_p

        sup = cite_support_all_in_evidence(used, retrieved)
        citation_support_sum += sup

        # 6) gold-alignment 指标：衡量“引用是否对齐标准答案”
        gh = gold_hit_rate(used, gold)
        gold_hit_sum += gh

        gp = cite_precision_against_gold(used, gold)
        gr = cite_recall_against_gold(used, gold)
        gf = f1_from_pr(gp, gr)
        gold_prec_sum += gp
        gold_rec_sum += gr
        gold_f1_sum += gf

        # 7) hallucination proxy（严格版只在“非拒答”样本上看）
        if not y_pred:
            non_refusal_cnt += 1

            # 7.1 无引用（在强制引用设置下，这通常意味着：要么模型没按格式输出，要么在胡说）
            # 7.2 引用不闭环（used 有，但不全来自 retrieved）
            if not used:
                halluc_proxy_cnt += 1
            else:
                if sup < 1.0:
                    bad_evidence_cnt += 1
                    halluc_proxy_cnt += 1

        # 8) latency
        tms = it.get("timing_ms", {}) or {}
        lat_sum += float(tms.get("total", it.get("wall_ms", 0.0)))

    # refusal PRF1
    pr = prf1(tp=tp, fp=fp, fn=fn)

    # 注意：HallucinationRate 分母用“非拒答样本数”，更符合语义
    halluc_rate = (halluc_proxy_cnt / non_refusal_cnt) if non_refusal_cnt > 0 else 0.0
    bad_evidence_rate = (bad_evidence_cnt / non_refusal_cnt) if non_refusal_cnt > 0 else 0.0

    return {
        "n": float(n),

        # --- retrieval ---
        "recall@k": r_at_k / n,
        "mrr@k": mrr_k / n,

        # --- citation: evidence-closure ---
        "cite_prec_evidence": ev_cite_prec_sum / n,          # 引用在证据池内的比例（平均）
        "citation_support_rate": citation_support_sum / n,   # 严格闭环（样本级全包含）比例

        # --- citation: gold-alignment ---
        "gold_hit_rate": gold_hit_sum / n,                   # 至少命中一个 gold 的比例（样本级）
        "cite_prec_gold": gold_prec_sum / n,                 # gold precision（平均）
        "cite_rec_gold": gold_rec_sum / n,                   # gold recall（平均）
        "cite_f1_gold": gold_f1_sum / n,                     # gold F1（平均）

        # --- refusal ---
        "refusal_rate_pred": refusals_pred / n,
        "refusal_rate_true": refusals_true / n,
        "refusal_precision": pr.precision,
        "refusal_recall": pr.recall,
        "refusal_f1": pr.f1,

        # --- parsing/hygiene ---
        "parse_success_rate": parse_ok / n,                  # parsed_json=true 的比例
        "autofill_rate": autofill_cnt / n,                   # citations_autofilled=true 的比例
        "no_citation_rate": no_citation_cnt / n,             # “有效引用为空”的比例（严格模式下会更真实）

        # --- hallucination proxy (strict) ---
        "hallucination_rate": halluc_rate,                   # 非拒答样本中：无引用 或 引用不闭环 的比例
        "bad_evidence_rate": bad_evidence_rate,              # 非拒答样本中：引用不闭环 的比例（拆出来更可解释）

        # --- efficiency ---
        "avg_latency_ms": lat_sum / n,
    }


# -------------------------
# 6) 输出：Markdown + CSV
# -------------------------
def write_markdown_table(out_md: Path, rows: List[Dict[str, str]]) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with out_md.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[h]) for h in headers) + " |\n")


def write_csv(out_csv: Path, rows: List[Dict[str, str]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def parse_runs(run_args: List[str]) -> List[Tuple[str, Path]]:
    """
    --run baseline=xxx.jsonl --run sft=yyy.jsonl
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
    ap.add_argument("--run", action="append", default=[],
                    help="对比输入：--run name=path，可重复多次")
    ap.add_argument("--k", type=int, default=10, help="Recall/MRR 的 K")
    ap.add_argument("--strict_ignore_autofill", action="store_true",default=True,
                    help="严格模式：忽略 citations_autofilled 样本的 used_citations（推荐开启）")
    ap.add_argument("--out_md", type=str, default="tables/compare_strict.md")
    ap.add_argument("--out_csv", type=str, default="tables/compare_strict.csv")
    args = ap.parse_args()

    runs = parse_runs(args.run)
    if not runs:
        raise ValueError("至少提供一个 --run name=path")

    def fmt(x: float) -> str:
        return f"{x:.3f}"

    rows: List[Dict[str, str]] = []
    for name, path in runs:
        items = read_jsonl(path)
        agg = aggregate(items, k=args.k, strict_ignore_autofill=args.strict_ignore_autofill)

        row = {
            "Setting": name,
            "N": int(agg["n"]),

            # Retrieval
            f"Recall@{args.k}": fmt(agg["recall@k"]),
            f"MRR@{args.k}": fmt(agg["mrr@k"]),

            # Evidence-closure
            "CitePrec(Ev)": fmt(agg["cite_prec_evidence"]),
            "CitationSupport": fmt(agg["citation_support_rate"]),

            # Gold-alignment（你质疑“是否真命中 gold”，这组能回答）
            "GoldHit": fmt(agg["gold_hit_rate"]),
            "CitePrec(Gold)": fmt(agg["cite_prec_gold"]),
            "CiteRec(Gold)": fmt(agg["cite_rec_gold"]),
            "CiteF1(Gold)": fmt(agg["cite_f1_gold"]),

            # Refusal
            "RefusalRate(pred)": fmt(agg["refusal_rate_pred"]),
            "RefusalPrec": fmt(agg["refusal_precision"]),
            "RefusalRec": fmt(agg["refusal_recall"]),
            "RefusalF1": fmt(agg["refusal_f1"]),

            # Hygiene（让指标“可信”）
            "ParseOK": fmt(agg["parse_success_rate"]),
            "AutoFill": fmt(agg["autofill_rate"]),
            "NoCite": fmt(agg["no_citation_rate"]),

            # Hallucination proxy（拆解更清晰）
            "BadEvidence": fmt(agg["bad_evidence_rate"]),
            "HallucRate": fmt(agg["hallucination_rate"]),

            # Efficiency
            "AvgLatency(ms)": fmt(agg["avg_latency_ms"]),
        }
        rows.append(row)

    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)
    write_markdown_table(out_md, rows)
    write_csv(out_csv, rows)

    print("[eval_metrics_strict] wrote:", str(out_md), "and", str(out_csv))
    print("[eval_metrics_strict] preview:")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
