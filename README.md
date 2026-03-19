# ProdRAG-Lab

ProdRAG-Lab is a production-oriented RAG (Retrieval-Augmented Generation) system with a focus on evaluation, controllability, and reliability. It integrates retrieval, reranking, model fine-tuning (LoRA/SFT), preference optimization (DPO), and agent-based multi-step reasoning.

## Overview

The project is designed to move beyond “a working RAG demo” toward a system that is:

- measurable (quantitative evaluation)
- controllable (refusal and citation constraints)
- reproducible (local + API models)
- extensible (agent and training pipelines)

Core pipeline:

User Query  
→ Hybrid Retrieval (BM25 + Dense)  
→ Reranker  
→ RAG Pipeline (citation + refusal control)  
→ LLM (API / Local / LoRA / DPO)  
→ Answer or Refusal  

## Key Features

### 1. Hybrid Retrieval
- BM25 (keyword-based)
- Dense retrieval with FAISS
- Combined recall for better coverage

### 2. Reranking
- Cross-encoder based reranker
- Significant improvement in Recall and ranking quality

### 3. Citation-Constrained Generation
- Answers must include supporting chunk IDs
- Improves faithfulness and traceability

### 4. Refusal Mechanism
- Rejects queries when evidence is insufficient
- Based on score thresholds and evidence coverage

### 5. Evaluation System
- Automatically generated QA dataset with citations
- Metrics:
  - Recall@K / MRR
  - Citation Precision
  - Faithfulness
  - Refusal Accuracy
  - Hallucination Rate
  - Latency

### 6. SFT (Supervised Fine-Tuning)
- Trains model to follow RAG format:
  - answer
  - citations
  - refusal
- Focus on behavior rather than knowledge memorization

### 7. DPO (Preference Optimization)
- Optimizes:
  - correct citation
  - correct refusal
  - reduced hallucination

### 8. Agent Extension
- Multi-step retrieval and reasoning
- Query rewriting, conflict detection, evidence completion

## Project Structure

```
prodrag-lab/
├── configs/
├── data/
├── scripts/
├── src/
│   ├── rag/
│   ├── eval/
│   ├── sft/
│   ├── dpo/
│   ├── agent/
├── runs/
├── tables/
├── environment.yml
└── pyproject.toml
```

## Setup

```
conda create -n prodrag python=3.11
conda activate prodrag
pip install -r requirements.txt
```

## Model Setup

Models are not included in this repository.

Example:

Embedding:
```
hf download BAAI/bge-small-zh-v1.5
hf download BAAI/bge-small-en-v1.5
```

Reranker:
```
hf download BAAI/bge-reranker-base
```

LLM (example):
```
unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit
```

## Data Pipeline

1. Fetch data
```
python scripts/fetch_arxiv.py ...
```

2. Build knowledge base
```
python src/rag/build_kb.py configs/build.yaml
```

Pipeline:
PDF → text → chunks → index

## Run RAG

```
python src/rag/cli_chat.py configs/rag.yaml
```

## Evaluation

Build dataset:
```
python src/eval_rag/build_evalset.py ...
```

Run evaluation:
```
python src/eval_rag/eval_run.py ...
python src/eval_rag/eval_metrics.py ...
```

## SFT

```
python src/sft/build_sftdata.py ...
```

## DPO

```
python src/dpo/build_dpo_dataset.py ...
```

## Agent

```
python src/agent/agent_cli.py
```

## Notes

- Large models and checkpoints are not tracked by git
- Use .gitignore to exclude data, runs, and model weights
- Designed for experimentation and reproducibility

## Author

Zhang Pusheng
