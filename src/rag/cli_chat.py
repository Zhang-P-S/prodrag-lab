# rag/cli_chat.py
import yaml
import jieba

from rag.pipeline import run_rag_stream
from rag.retrieval import DualIndexHybridRetriever
from llm.base import build_llm


def load_llm_cfg():
    cfg = yaml.safe_load(open("configs/rag.yaml", "r", encoding="utf-8"))
    return cfg["llm"]


def main():
    print("🧠 RAG Assistant（输入 exit 退出）")

    # ---------- 0) 初始化 ----------
    jieba.initialize()

    # ---------- 1) LLM ----------
    llm_cfg = load_llm_cfg()
    llm = build_llm(llm_cfg)

    # ---------- 2) Retriever（只初始化一次，非常重要） ----------
    retriever = DualIndexHybridRetriever(index_root="data/index")

    # ---------- 3) REPL ----------
    while True:
        query = input("\n👤 你：").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("👋 再见")
            break

        # ---------- 4) 流式输出 ----------
        print("🤖 助手：", end="", flush=True)
        for delta in run_rag_stream(query, llm, retriever):
            print(delta, end="", flush=True)

        # ---------- 5) 流式结束后：结构化证据 ----------
        result = getattr(run_rag_stream, "last_result", None)
        if not result:
            continue

        print("\n\n📌 citations:")
        for c in result["citations"]:
            print(f"  - {c}")

        print("\n📄 top_chunks:")
        for i, c in enumerate(result["top_chunks"], 1):
            print(
                f"  {i}) {c['chunk_id']} | page={c['page']} | score={c['rerank_score']:.4f}"
            )


if __name__ == "__main__":
    main()
