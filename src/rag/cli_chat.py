# rag/cli_chat.py
import os
import warnings
import contextlib
import yaml

# ✅ 这些要放在尽量靠前的位置（在 import jieba/transformers 等之前更好）
# 1) 静音 Python warnings（例如 pkg_resources deprecated）
warnings.filterwarnings("ignore")

# 2) 静音 transformers 日志（loading / deprecated）
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 3) 尽量关掉 tqdm 进度条（很多 from_pretrained 会用到）
os.environ.setdefault("TQDM_DISABLE", "1")

import jieba  # 放到环境变量设置后再 import

from rag.pipeline import run_rag_stream
from rag.retrieval import DualIndexHybridRetriever
from llm.base import build_llm


@contextlib.contextmanager
def suppress_stdout_stderr():
    """
    只在模型/索引加载阶段用：吞掉第三方库的 print 和 warning 输出，避免刷屏。
    注意：异常仍会抛出，只是不打印中间噪音。
    """
    with open(os.devnull, "w") as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            yield


def load_llm_cfg(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    return cfg["llm"]


def main():
    print("🧠 RAG Assistant（输入 exit 退出）")

    # ---------- 0) 初始化（静音） ----------
    with suppress_stdout_stderr():
        jieba.initialize()

    # ---------- 1) LLM（静音加载） ----------
    llm_cfg = load_llm_cfg("configs/rag.yaml")
    with suppress_stdout_stderr():
        llm = build_llm(llm_cfg)

    # ---------- 2) Retriever（静音加载） ----------
    with suppress_stdout_stderr():
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
            print(f"  {i}) {c['chunk_id']} | page={c['page']} | score={c['rerank_score']:.4f}")


if __name__ == "__main__":
    main()
