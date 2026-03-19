# src/agent_cli.py
from __future__ import annotations

import yaml
import jieba

from llm.base import build_llm
from rag.retrieval import DualIndexHybridRetriever

# 你项目里如果有非流式版本最好：
# from rag.pipeline import run_rag
from rag.pipeline import run_rag_stream,run_rag_once  # 如果你只有 stream，用它拼出来

from agent.tools import default_tool_registry
from agent.react_agent import ReActAgent, AgentConfig
from feishu_webhook import feishu_send_text, feishu_send_card

def load_llm_cfg():
    cfg = yaml.safe_load(open("configs/rag.yaml", "r", encoding="utf-8"))
    return cfg["llm"]


def make_run_rag_fn(llm, retriever):
    """
    返回一个 run_rag_fn(query)->out 的函数，供 rag_search tool 使用
    """

    # 方案A：你有 run_rag（推荐）
    def run_rag_fn(query: str):
        return run_rag_once(query, llm, retriever)  # 具体签名按你项目改
    return run_rag_fn

    # # 方案B：你只有 run_rag_stream：把流拼成完整文本
    # def run_rag_fn(query: str):
    #     chunks = []
    #     for delta in run_rag_stream(query, llm, retriever):
    #         chunks.append(delta)
    #     return "".join(chunks)

    return run_rag_fn


def main():
    print("🧩 Agent CLI (type 'exit' to quit)")

    jieba.initialize()

    llm_cfg = load_llm_cfg()
    llm = build_llm(llm_cfg)
    # retriever = DualIndexHybridRetriever(index_root="data/index")

    # run_rag_fn = make_run_rag_fn(llm, retriever)

    # tools = default_tool_registry(run_rag_fn)
    # 临时：不加载 DualIndexHybridRetriever
    # retriever = DualIndexHybridRetriever(index_root="data/index")
    # run_rag_fn = make_run_rag_fn(llm, retriever)
    # tools = default_tool_registry(run_rag_fn)

    tools = default_tool_registry(run_rag_fn=None)  # 先只保留不依赖 RAG 的工具
    agent = ReActAgent(llm=llm, tools=tools, cfg=AgentConfig(max_steps=6))

    while True:
        q = input("\n👤 You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        out = agent.run(q)
        print("\n🤖 Agent:", out["final_answer"])

        # 想 debug 就打开
        print(out["steps"])

        # 上传到飞书
        final = out["final_answer"]
        print("\n🤖 Agent:", final)

        # 推送到飞书
        md = f"**用户问题：**\n{q}\n\n---\n**Agent回答：**\n{final}"
        feishu_send_card(title="🤖 RAG Agent 回答", md=md)

if __name__ == "__main__":
    main()