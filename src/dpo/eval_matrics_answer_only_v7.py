# -*- coding: utf-8 -*-
"""
eval_matrics_answer_only_v7.py

目标：只评“模型答案 vs gold_answer”的相似度/正确性，不评引用、不评证据支撑、不评拒答。
适用于你现在要做的“先验证模型输出质量”的场景。

输出指标：
- AnswerSim: 语义/文本相似度（默认：英文用词级 cosine，中文/混合用 char-bigram cosine）
- AnswerAcc@T: 相似度 >= T 的比例（T 默认 0.65）
- AvgLatency(ms): 来自 runs 中 latency_ms 的均值（若没有则 0）
- JudgeScore / JudgeAcc@J（可选）：用 DeepSeek 对 “pred_answer vs gold_answer” 打分（0~1）

输入：
- --test: data/dpo/qa_raw_test.jsonl（含 qid, question, gold_answer）
- --runs_dir: runs/（例如 runs/sft.jsonl）
- --settings: 逗号分隔多个 setting（对应 runs/{setting}.jsonl）

注意：
- 该脚本不会读取 chunks，也不会用 citations。
- runs 文件中 pred answer 的提取顺序：
  1) r["parsed"]["answer"]
  2) r["llm_raw"]
  3) r["answer"]
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

_RE_WS = re.compile(r"\s+")
_RE_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")  # 简单英文/数字词


# ============== IO ==============
def read_jsonl(path: Path) -> List[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ============== Similarity ==============
def _ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    ascii_cnt = sum(1 for ch in s if ord(ch) < 128)
    return ascii_cnt / max(1, len(s))


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


def answer_similarity(a: str, b: str) -> float:
    """
    英文：词级（unigram + bigram）cosine 平均
    中文/混合：char bigram cosine
    """
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0

    if _ascii_ratio(a + b) >= 0.75:
        c1a = Counter(word_ngrams(a, 1))
        c1b = Counter(word_ngrams(b, 1))
        c2a = Counter(word_ngrams(a, 2))
        c2b = Counter(word_ngrams(b, 2))
        return 0.5 * _cosine_counter(c1a, c1b) + 0.5 * _cosine_counter(c2a, c2b)
    else:
        ca = Counter(char_ngrams(a, 2))
        cb = Counter(char_ngrams(b, 2))
        return _cosine_counter(ca, cb)


def clean_answer_for_eval(ans: str, max_chars: int = 900) -> str:
    """
    清洗用于相似度 / judge：
    - 去掉 </think> 之前内容
    - 去掉常见“口头禅”前缀
    - 句子去重
    - 截断
    """
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

    # sentence de-dup
    sents = re.split(r"(?<=[\.\!\?。！？])\s+", ans)
    uniq = []
    seen = set()
    for s in sents:
        s = s.strip()
        if not s:
            continue
        key = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s.lower())[:140]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    ans = " ".join(uniq).strip()
    return ans[:max_chars]


# ============== DeepSeek Judge (Answer-only) ==============
def sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def build_answer_only_judge_prompt(question: str, gold_answer: str, model_answer: str) -> str:
    return f"""You are a strict grader.

Question:
{question}

Gold Answer:
{gold_answer}

Model Answer:
{model_answer}

Task:
Score how correct the Model Answer is compared to the Gold Answer.
- 1.0 = fully correct (paraphrase allowed)
- 0.5 = partially correct / missing key info
- 0.0 = incorrect

Return ONLY valid JSON (no markdown, no code fence):
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
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}

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


def parse_json_robust(text: str) -> Optional[dict]:
    if not text:
        return None
    t = text.strip()
    # strip code fences
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)

    # direct
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # extract by braces
    l = t.find("{")
    r = t.rfind("}")
    if l == -1 or r == -1 or r <= l:
        return None
    frag = t[l:r+1]
    try:
        obj = json.loads(frag)
        if isinstance(obj, dict):
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


def judge_one_answer_only(
    api_key: str,
    model: str,
    question: str,
    gold_answer: str,
    model_answer: str,
    cache: JudgeCache,
    api_base: str,
) -> Tuple[float, str, bool, str]:
    """
    returns: (score, reason, parse_fail, raw_prefix)
    """
    cache_key = sha1_text(question + "\n" + gold_answer + "\n" + model_answer + "\n" + model)
    hit = cache.get(cache_key)
    if hit is not None:
        return float(hit.get("score", 0.0)), str(hit.get("reason", "")), bool(hit.get("parse_fail", False)), str(hit.get("raw_prefix", ""))

    prompt = build_answer_only_judge_prompt(question, gold_answer, model_answer)
    raw = deepseek_chat_completion(api_key=api_key, model=model, prompt=prompt, api_base=api_base)
    content = raw["choices"][0]["message"]["content"]
    obj = parse_json_robust(content)
    raw_prefix = (content or "").strip().replace("\n", "\\n")[:240]

    if obj is None or "score" not in obj:
        score, reason, parse_fail = 0.0, "judge_parse_fail", True
    else:
        try:
            score = float(obj.get("score", 0.0))
        except Exception:
            score = 0.0
        reason = str(obj.get("reason", ""))[:260]
        score = max(0.0, min(1.0, score))
        parse_fail = False

    cache.set(cache_key, {"score": score, "reason": reason, "parse_fail": parse_fail, "raw_prefix": raw_prefix})
    return score, reason, parse_fail, raw_prefix


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

    ap.add_argument("--ans_sim_thres", type=float, default=0.65)
    ap.add_argument("--use_clean", action="store_true", help="对 pred/gold 都做清洗再算相似度（建议开启）")

    ap.add_argument("--out_csv", default="tables/answer_only_v7.csv")
    ap.add_argument("--out_md", default="tables/answer_only_v7.md")

    # optional judge
    ap.add_argument("--use_deepseek_judge", action="store_true")
    ap.add_argument("--deepseek_api_key", default=os.getenv("DEEPSEEK_API_KEY", ""))
    ap.add_argument("--deepseek_api_base", default="https://api.deepseek.com")
    ap.add_argument("--judge_model", default="deepseek-chat")
    ap.add_argument("--judge_workers", type=int, default=6)
    ap.add_argument("--judge_acc_thres", type=float, default=0.8)
    ap.add_argument("--judge_cache", default="tables/judge_cache/deepseek_judge_answer_only_cache.json")
    ap.add_argument("--dump_judge_debug", default="", help="dump judge debug jsonl (qid,score,reason,raw_prefix)")

    args = ap.parse_args()

    test = read_jsonl(Path(args.test))
    gold_map: Dict[str, dict] = {str(it.get("qid")): it for it in test if it.get("qid") is not None}

    rows = []
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]

    cache = JudgeCache(Path(args.judge_cache))

    for setting in settings:
        run_path = Path(args.runs_dir) / f"{setting}.jsonl"
        runs = read_jsonl(run_path)

        n = 0
        sim_sum = 0.0
        acc_sum = 0.0
        lat_sum = 0.0
        valid = 0

        judge_tasks = []
        judge_debug = []
        judge_score_sum = 0.0
        judge_acc_sum = 0.0
        judge_n = 0
        judge_fail = 0
        judge_parse_fail = 0

        for r in runs:
            qid = str(r.get("qid", ""))
            g = gold_map.get(qid)
            if not g:
                continue
            n += 1

            question = str(g.get("question") or "")
            gold_answer = str(g.get("gold_answer") or "")

            pred_answer = extract_pred_answer(r)

            if args.use_clean:
                gold_eval = clean_answer_for_eval(gold_answer)
                pred_eval = clean_answer_for_eval(pred_answer)
            else:
                gold_eval = gold_answer
                pred_eval = pred_answer

            if gold_eval.strip() and pred_eval.strip():
                sim = answer_similarity(pred_eval, gold_eval)
                sim_sum += sim
                acc_sum += 1.0 if sim >= args.ans_sim_thres else 0.0
                valid += 1

                if args.use_deepseek_judge:
                    judge_tasks.append((qid, question, gold_eval, pred_eval))

            lat_sum += float(r.get("latency_ms") or 0.0)

        if args.use_deepseek_judge:
            if not args.deepseek_api_key:
                raise RuntimeError("启用了 --use_deepseek_judge 但没有提供 deepseek_api_key / DEEPSEEK_API_KEY")

            with ThreadPoolExecutor(max_workers=max(1, args.judge_workers)) as ex:
                fut_map = {}
                for qid, question, gold_eval, pred_eval in judge_tasks:
                    fut = ex.submit(
                        judge_one_answer_only,
                        args.deepseek_api_key,
                        args.judge_model,
                        question,
                        gold_eval,
                        pred_eval,
                        cache,
                        args.deepseek_api_base,
                    )
                    fut_map[fut] = qid

                for fut in as_completed(fut_map):
                    qid = fut_map[fut]
                    try:
                        score, reason, parse_fail, raw_prefix = fut.result()
                        judge_score_sum += score
                        judge_acc_sum += 1.0 if score >= args.judge_acc_thres else 0.0
                        judge_n += 1
                        if parse_fail:
                            judge_parse_fail += 1
                        if args.dump_judge_debug:
                            judge_debug.append({"qid": qid, "score": score, "reason": reason, "raw_prefix": raw_prefix})
                    except Exception:
                        judge_fail += 1

            cache.save()
            if args.dump_judge_debug:
                Path(args.dump_judge_debug).parent.mkdir(parents=True, exist_ok=True)
                with Path(args.dump_judge_debug).open("w", encoding="utf-8") as f:
                    for it in judge_debug:
                        f.write(json.dumps(it, ensure_ascii=False) + "\n")

        row = {
            "Setting": setting,
            "N": n,
            "ValidN": valid,
            "AnswerSim": sim_sum / max(1, valid),
            f"AnswerAcc@{args.ans_sim_thres:.2f}": acc_sum / max(1, valid),
            "AvgLatency(ms)": lat_sum / max(1, n),
        }

        if args.use_deepseek_judge:
            row.update({
                "JudgeN": judge_n,
                "JudgeIntendedN": len(judge_tasks),
                "JudgeFailN": judge_fail,
                "JudgeParseFailN": judge_parse_fail,
                "JudgeScore": judge_score_sum / max(1, judge_n),
                f"JudgeAcc@{args.judge_acc_thres:.2f}": judge_acc_sum / max(1, judge_n),
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
