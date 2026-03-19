# -*- coding: utf-8 -*-
"""
eval_matrics.py (v5)
改动要点（针对你遇到的“答案很好但指标很差”的现象）：
1) AnswerSim/AnswerAcc：对 pred_ans 做轻量清洗（去重复/去口头禅/截断），减少“长答案+重复句”导致的相似度偏低。
2) DeepSeek Judge：默认把清洗后的答案喂给 Judge（可用 --judge_use_clean_answer 关闭）。
3) Judge 并发鲁棒性：单条 judge 失败不再导致整批异常；统计 JudgeFailN / JudgeFailRate，便于定位。
4) JudgeN 对齐：只对 parse_ok 且 pred_ans 非空且 gold_evidence 非空 的样本发起 judge，避免 JudgeN 低于预期但不知原因。

注意：
- 这个脚本是“测评”，不会改变你模型输出的引用；如果你模型经常把 citations 指到错误 chunk，
  SoftCite 会低、Judge 也会低，这是正常反映“证据不支撑”的问题。
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import requests

_WS = re.compile(r"\s+")


# =========================
# IO
# =========================
def read_jsonl(path: Path) -> List[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def safe_list(x) -> List[str]:
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    return []


# =========================
# Similarity: char ngram cosine (中文/公式稳定)
# =========================
def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    # 去空白，避免中文无空格导致分词失效
    s = _WS.sub("", s)
    return s


def char_ngrams(s: str, n: int) -> List[str]:
    if len(s) < n:
        return [s] if s else []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def cosine_sim_charngram(a: str, b: str, ngram: int = 2) -> float:
    a = norm_text(a)
    b = norm_text(b)
    if not a or not b:
        return 0.0
    ca = Counter(char_ngrams(a, ngram))
    cb = Counter(char_ngrams(b, ngram))
    dot = sum(v * cb.get(k, 0) for k, v in ca.items())
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


# =========================
# Answer cleaning (for AnswerSim/Judge)
# =========================
def clean_answer_for_eval(ans: str, max_chars: int = 900) -> str:
    """
    用于 AnswerSim/Judge 的轻量清洗：
    - 去掉 </think> 之前的内容
    - 去掉常见“口头禅”前缀
    - 去掉显著重复（例如 “The answer is:” + 再重复一遍）
    - 截断，避免超长
    """
    ans = (ans or "").strip()

    # 1) 去掉 think 之前的内容（如果有）
    if "</think>" in ans:
        ans = ans.split("</think>", 1)[1].strip()

    # 2) 去掉常见废话前缀（多语言）
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

    # 3) 合并空白
    ans = re.sub(r"\s+", " ", ans).strip()

    # 4) 去掉明显重复：如果后半段是前半段的重复或近似重复，就保留较短那段
    #    简单策略：按句号/问号/叹号/中文句号切句，然后去重保序
    parts = re.split(r"(?<=[\.\!\?。！？])\s+", ans)
    uniq = []
    seen = set()
    for s in parts:
        s2 = s.strip()
        if not s2:
            continue
        key = norm_text(s2)[:120]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s2)
    ans = " ".join(uniq).strip()

    # 5) 截断
    return ans[:max_chars]


# =========================
# chunk texts
# =========================
def load_chunk_texts(path: Path) -> Dict[str, str]:
    """
    chunks.jsonl: 每行至少包含 chunk_id + text/chunk/content
    """
    m: Dict[str, str] = {}
    for it in read_jsonl(path):
        cid = str(it.get("chunk_id") or it.get("id") or "").strip()
        if not cid:
            continue
        txt = it.get("text") or it.get("chunk") or it.get("content") or ""
        txt = str(txt)
        if txt:
            m[cid] = txt
    return m


# =========================
# Soft Citation (evidence text tolerance)
# =========================
def soft_cite_prec_rec(
    pred_cits: List[str],
    gold_cits: List[str],
    chunk_text: Dict[str, str],
    thres: float,
) -> Tuple[float, float]:
    if not gold_cits:
        return 0.0, 0.0

    # precision: pred citations 中有多少能匹配任一 gold（文本相似度 >= thres）
    if pred_cits:
        hit = 0
        for pc in pred_cits:
            pt = chunk_text.get(pc, "")
            if not pt:
                continue
            ok = False
            for gc in gold_cits:
                gt = chunk_text.get(gc, "")
                if not gt:
                    continue
                if cosine_sim_charngram(pt, gt) >= thres:
                    ok = True
                    break
            if ok:
                hit += 1
        prec = hit / max(1, len(pred_cits))
    else:
        prec = 0.0

    # recall: gold citations 有多少被 pred 覆盖
    covered = 0
    for gc in gold_cits:
        gt = chunk_text.get(gc, "")
        if not gt:
            continue
        ok = False
        for pc in pred_cits:
            pt = chunk_text.get(pc, "")
            if not pt:
                continue
            if cosine_sim_charngram(pt, gt) >= thres:
                ok = True
                break
        if ok:
            covered += 1
    rec = covered / max(1, len(gold_cits))
    return prec, rec


# =========================
# DeepSeek Judge (LLM-as-a-judge) with cache
# =========================
def sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def build_judge_prompt(question: str, evidence: str, model_answer: str) -> str:
    # 注意：不要让 judge 看 gold_answer，避免变成“对齐某种表述”
    return f"""You are a strict academic evaluator. You MUST only use the provided Evidence.

Question:
{question}

Evidence:
{evidence}

Model Answer:
{model_answer}

Task:
1) Determine whether the model answer is fully supported by the Evidence.
2) The answer must not include unsupported/hallucinated claims.
3) Minor wording differences are allowed.
4) Score from 0 to 1:
   - 1.0 = fully correct and supported
   - 0.5 = partially correct / missing key parts
   - 0.0 = incorrect or unsupported

Return ONLY valid JSON:
{{
  "score": <float between 0 and 1>,
  "reason": "<short reason>"
}}
"""


def deepseek_chat_completion(
    api_key: str,
    model: str,
    prompt: str,
    api_base: str = "https://api.deepseek.com",
    timeout: int = 60,
    max_retries: int = 3,
) -> dict:
    url = api_base.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }

    last_err = None
    for i in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"DeepSeek API failed after retries: {last_err}")


def parse_judge_json(text: str) -> Optional[dict]:
    """
    Judge 可能会输出多余字符；我们尽量从中提取 JSON。
    """
    if not text:
        return None
    text = text.strip()
    # 直接尝试
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "score" in obj:
            return obj
    except Exception:
        pass

    # 从中找第一个 {...} 片段
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict) and "score" in obj:
            return obj
    except Exception:
        return None
    return None


class JudgeCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db: Dict[str, dict] = {}
        if self.path.exists():
            try:
                self.db = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.db = {}

    def get(self, key: str) -> Optional[dict]:
        return self.db.get(key)

    def set(self, key: str, val: dict):
        self.db[key] = val

    def save(self):
        self.path.write_text(json.dumps(self.db, ensure_ascii=False, indent=2), encoding="utf-8")


def judge_one(
    api_key: str,
    model: str,
    question: str,
    evidence: str,
    model_answer: str,
    cache: JudgeCache,
    api_base: str,
) -> Tuple[float, str]:
    # 以（question+evidence+answer）做缓存键，保证可复用
    cache_key = sha1_text(question + "\n" + evidence + "\n" + model_answer + "\n" + model)
    hit = cache.get(cache_key)
    if hit is not None:
        return float(hit.get("score", 0.0)), str(hit.get("reason", ""))

    prompt = build_judge_prompt(question, evidence, model_answer)
    raw = deepseek_chat_completion(api_key=api_key, model=model, prompt=prompt, api_base=api_base)
    content = raw["choices"][0]["message"]["content"]
    obj = parse_judge_json(content)
    if obj is None:
        # 解析失败当 0 分，但把原文记下来便于排查
        score, reason = 0.0, f"judge_parse_fail: {content[:180]}"
    else:
        score = float(obj.get("score", 0.0))
        reason = str(obj.get("reason", ""))[:240]
        score = max(0.0, min(1.0, score))

    cache.set(cache_key, {"score": score, "reason": reason})
    return score, reason


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="测试集 jsonl（含 gold_answer/gold_citations）")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--settings", default="sft")

    ap.add_argument("--cite_sim_thres", type=float, default=0.65)
    ap.add_argument("--ans_sim_thres", type=float, default=0.65)
    ap.add_argument("--ans_use_clean", action="store_true", help="AnswerSim 使用清洗后的 pred_ans（建议开启）")

    # refusal
    ap.add_argument("--refusal_threshold", type=float, default=None, help="若设置，则 top1_score < 阈值 视为应拒答")

    # outputs
    ap.add_argument("--out_csv", default="tables/metrics_v5.csv")
    ap.add_argument("--out_md", default="tables/metrics_v5.md")

    # deepseek judge
    ap.add_argument("--use_deepseek_judge", action="store_true")
    ap.add_argument("--deepseek_api_key", default=os.getenv("DEEPSEEK_API_KEY", ""))
    ap.add_argument("--deepseek_api_base", default="https://api.deepseek.com")
    ap.add_argument("--judge_model", default="deepseek-chat")
    ap.add_argument("--judge_workers", type=int, default=6)
    ap.add_argument("--judge_acc_thres", type=float, default=0.8)
    ap.add_argument("--judge_cache", default="tables/judge_cache/deepseek_judge_cache.json")
    ap.add_argument("--judge_use_clean_answer", action="store_true", help="Judge 输入使用清洗后的 pred_ans（建议开启）")

    args = ap.parse_args()

    # load gold
    test = read_jsonl(Path(args.test))
    gold_map: Dict[str, dict] = {str(it.get("qid")): it for it in test if it.get("qid") is not None}

    # load chunks
    chunk_text = load_chunk_texts(Path(args.chunks))
    if not chunk_text:
        raise RuntimeError(f"chunk_text 为空，请检查 --chunks: {args.chunks}")

    # judge cache
    cache = JudgeCache(Path(args.judge_cache))

    rows = []
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]

    for setting in settings:
        run_path = Path(args.runs_dir) / f"{setting}.jsonl"
        runs = read_jsonl(run_path)

        n = 0

        # answer sim
        ans_sim_sum = 0.0
        ans_acc_sum = 0.0
        ans_count = 0

        # soft cite
        cite_prec_sum = 0.0
        cite_rec_sum = 0.0
        cite_count = 0

        # refusal
        pred_refuse_cnt = 0
        tp = fp = fn = 0

        # latency
        lat_sum = 0.0

        # judge
        judge_score_sum = 0.0
        judge_acc_sum = 0.0
        judge_count = 0
        judge_fail = 0

        # 为并发准备任务
        judge_tasks = []

        for r in runs:
            qid = str(r.get("qid", ""))
            g = gold_map.get(qid)
            if not g:
                continue

            n += 1
            question = str(g.get("question") or "")
            gold_ans = str(g.get("gold_answer") or "")
            gold_cits = safe_list(g.get("gold_citations", []))
            # gold evidence：把所有 gold_citations 文本拼起来
            gold_evidence_parts = [chunk_text.get(cid, "") for cid in gold_cits if chunk_text.get(cid, "")]
            gold_evidence = "\n\n".join(gold_evidence_parts).strip()

            parsed = r.get("parsed", {}) or {}
            parse_ok = bool(parsed.get("parse_ok", True))  # runs 里一般 parse_ok=true 或缺省
            pred_refuse = bool(parsed.get("refusal")) if parse_ok else False
            pred_ans_raw = str(parsed.get("answer") or "") if parse_ok else ""
            pred_cits = safe_list(parsed.get("citations", [])) if parse_ok else []

            pred_ans_for_sim = clean_answer_for_eval(pred_ans_raw) if args.ans_use_clean else pred_ans_raw
            pred_ans_for_judge = clean_answer_for_eval(pred_ans_raw) if args.judge_use_clean_answer else pred_ans_raw

            # should_refuse label
            if args.refusal_threshold is not None:
                top1 = r.get("retrieve_meta", {}) or {}
                top1_score = float(top1.get("top1_score", 1.0))
                should_refuse = top1_score < float(args.refusal_threshold)
            else:
                should_refuse = (len(gold_cits) == 0)

            # refusal metrics
            if pred_refuse:
                pred_refuse_cnt += 1
            if should_refuse and pred_refuse:
                tp += 1
            elif (not should_refuse) and pred_refuse:
                fp += 1
            elif should_refuse and (not pred_refuse):
                fn += 1

            # answer sim (只在不该拒答时算)
            if not should_refuse:
                sim = cosine_sim_charngram(pred_ans_for_sim, gold_ans)
                ans_sim_sum += sim
                ans_acc_sum += 1.0 if sim >= args.ans_sim_thres else 0.0
                ans_count += 1

            # soft cite (只在 gold 有 citations 时算)
            if gold_cits:
                prec, rec = soft_cite_prec_rec(pred_cits, gold_cits, chunk_text, args.cite_sim_thres)
                cite_prec_sum += prec
                cite_rec_sum += rec
                cite_count += 1

            lat_sum += float(r.get("latency_ms") or 0.0)

            # judge tasks（只在不该拒答且 evidence 非空 且 parse_ok 且 pred_ans 非空）
            if (
                args.use_deepseek_judge
                and (not should_refuse)
                and bool(gold_evidence)
                and parse_ok
                and bool(pred_ans_for_judge.strip())
            ):
                judge_tasks.append((qid, question, gold_evidence, pred_ans_for_judge))

        # run judge in parallel
        if args.use_deepseek_judge:
            if not args.deepseek_api_key:
                raise RuntimeError(
                    "你启用了 --use_deepseek_judge 但没有提供 --deepseek_api_key 或环境变量 DEEPSEEK_API_KEY"
                )

            with ThreadPoolExecutor(max_workers=max(1, args.judge_workers)) as ex:
                futs = []
                for qid, question, evidence, pred_ans in judge_tasks:
                    futs.append(
                        ex.submit(
                            judge_one,
                            args.deepseek_api_key,
                            args.judge_model,
                            question,
                            evidence,
                            pred_ans,
                            cache,
                            args.deepseek_api_base,
                        )
                    )
                for fut in as_completed(futs):
                    try:
                        score, _reason = fut.result()
                        judge_score_sum += score
                        judge_acc_sum += 1.0 if score >= args.judge_acc_thres else 0.0
                        judge_count += 1
                    except Exception as e:
                        judge_fail += 1

            # save cache
            cache.save()

        # finalize refusal
        refusal_rate = pred_refuse_cnt / max(1, n)
        refusal_prec = tp / max(1, tp + fp)
        refusal_rec = tp / max(1, tp + fn)
        refusal_f1 = (
            0.0
            if (refusal_prec + refusal_rec) == 0
            else (2 * refusal_prec * refusal_rec / (refusal_prec + refusal_rec))
        )

        row = {
            "Setting": setting,
            "N": n,
            "AnswerSim": ans_sim_sum / max(1, ans_count),
            f"AnswerAcc@{args.ans_sim_thres:.2f}": ans_acc_sum / max(1, ans_count),
            f"SoftCitePrec@{args.cite_sim_thres:.2f}": cite_prec_sum / max(1, cite_count),
            f"SoftCiteRec@{args.cite_sim_thres:.2f}": cite_rec_sum / max(1, cite_count),
            "RefusalRate(pred)": refusal_rate,
            "RefusalF1": refusal_f1,
            "AvgLatency(ms)": lat_sum / max(1, n),
        }

        if args.use_deepseek_judge:
            total_judge_intended = len(judge_tasks)
            row.update(
                {
                    "JudgeN": judge_count,
                    "JudgeIntendedN": total_judge_intended,
                    "JudgeFailN": judge_fail,
                    "JudgeFailRate": judge_fail / max(1, total_judge_intended),
                    "JudgeScore": judge_score_sum / max(1, judge_count),
                    f"JudgeAcc@{args.judge_acc_thres:.2f}": judge_acc_sum / max(1, judge_count),
                }
            )

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
