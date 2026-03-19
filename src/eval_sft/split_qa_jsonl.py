#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把一个 jsonl QA 文件按样本数均匀切成 N 份，方便多进程并行评测。

用法示例：

  python src/eval_sft/split_qa_jsonl.py \
      --input data/eval/qa_v1_200.jsonl \
      --num_shards 4

会在同目录下生成：
  data/eval/qa_v1_200.part1.jsonl
  data/eval/qa_v1_200.part2.jsonl
  data/eval/qa_v1_200.part3.jsonl
  data/eval/qa_v1_200.part4.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Any


def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(p: Path, items: List[Dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="待切分的 QA jsonl 路径")
    ap.add_argument("--num_shards", type=int, required=True, help="切成多少份")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"input not found: {in_path}")

    num_shards = int(args.num_shards)
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")

    items = read_jsonl(in_path)
    n = len(items)
    if n == 0:
        raise ValueError(f"input is empty: {in_path}")

    shard_size = math.ceil(n / num_shards)
    print(f"[split_qa_jsonl] input={in_path} n={n} num_shards={num_shards} shard_size≈{shard_size}")

    stem = in_path.stem  # e.g. qa_v1_200
    suffix = in_path.suffix  # .jsonl

    for i in range(num_shards):
        start = i * shard_size
        end = min((i + 1) * shard_size, n)
        if start >= end:
            break
        shard_items = items[start:end]
        out_path = in_path.with_name(f"{stem}.part{i+1}{suffix}")
        write_jsonl(out_path, shard_items)
        print(f"[split_qa_jsonl] shard {i+1}: {len(shard_items)} items -> {out_path}")

    print("[split_qa_jsonl] done.")


if __name__ == "__main__":
    main()

