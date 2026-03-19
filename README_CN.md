# ProdRAG-Lab

ProdRAG-Lab 是一个面向生产环境的 RAG（检索增强生成）系统，重点关注系统的可评测性、可控性与可靠性。项目集成了检索、重排、模型微调（LoRA/SFT）、偏好优化（DPO）以及基于 Agent 的多步推理能力。

## 项目概述

本项目的目标不是简单实现一个 RAG Demo，而是构建一个具备工程价值的系统，使其具备：

- 可量化评测能力
- 可控的生成行为（引用与拒答）
- 可复现的实验流程
- 可扩展的架构设计

核心流程如下：

用户问题  
→ 混合检索（BM25 + Dense）  
→ 重排（Reranker）  
→ RAG Pipeline（引用约束 + 拒答机制）  
→ 大模型（API / 本地 / LoRA / DPO）  
→ 输出答案或拒答  

## 核心功能

### 1. 混合检索（Hybrid Retrieval）
- BM25（关键词检索）
- Dense Retrieval（语义检索，基于向量）
- FAISS 向量索引

用于提升召回覆盖率与鲁棒性。

---

### 2. 重排（Reranker）
- 基于 cross-encoder 的精排模型
- 显著提升 Recall 与排序质量

---

### 3. 引用约束生成（Citation Constraint）
- 所有回答必须附带引用（chunk_id）
- 保证答案可追溯、可验证

---

### 4. 拒答机制（Refusal）
- 当证据不足时拒绝回答
- 基于检索置信度与证据覆盖度判断

用于降低 hallucination（幻觉）。

---

### 5. 评测体系（Evaluation）
自动构建带引用的 QA 数据集，支持多维度评测：

- Recall@K / MRR
- Citation Precision
- Faithfulness（是否基于证据）
- Refusal Accuracy
- Hallucination Rate
- 延迟与成本

---

### 6. SFT（监督微调）
训练模型学习“RAG行为”，而不是记忆知识：

- 输入：query + 检索到的证据
- 输出：answer + citations + refusal

---

### 7. DPO（偏好优化）
基于偏好数据优化模型行为：

- 更偏好正确引用
- 更稳定触发拒答
- 减少无证据生成

---

### 8. Agent（多步推理）
支持多轮决策流程：

- query 重写
- 多跳检索
- 冲突检测
- 补充证据
- 生成或拒答

---

## 项目结构

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

---

## 环境配置

```
conda create -n prodrag python=3.11
conda activate prodrag
pip install -r requirements.txt
```

---

## 模型说明

仓库不包含模型文件，请自行下载。

示例：

Embedding 模型：
```
hf download BAAI/bge-small-zh-v1.5
hf download BAAI/bge-small-en-v1.5
```

Reranker：
```
hf download BAAI/bge-reranker-base
```

大模型：
```
unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit
```

---

## 数据处理流程

1. 抓取数据：
```
python scripts/fetch_arxiv.py ...
```

2. 构建知识库：
```
python src/rag/build_kb.py configs/build.yaml
```

流程：
PDF → 文本 → 分块（chunks）→ 向量索引

---

## 运行 RAG

```
python src/rag/cli_chat.py configs/rag.yaml
```

---

## 构建评测集

```
python src/eval_rag/build_evalset.py ...
```

---

## 评测

```
python src/eval_rag/eval_run.py ...
python src/eval_rag/eval_metrics.py ...
```

---

## SFT 训练

```
python src/sft/build_sftdata.py ...
```

---

## DPO 训练

```
python src/dpo/build_dpo_dataset.py ...
```

---

## Agent

```
python src/agent/agent_cli.py
```

---

## 注意事项

- 模型权重、数据集、实验结果未纳入版本控制
- 请使用 .gitignore 排除大文件
- 项目侧重实验与工程验证

---

## 作者

Zhang Pusheng
