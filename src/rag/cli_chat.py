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


def load_cfg(cfg_path: str) -> dict:
    """加载完整 RAG 配置（llm + retrieval + eval 等）。"""
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    print("🧠 RAG Assistant（输入 exit 退出）")

    # ---------- 0) 初始化（静音） ----------
    with suppress_stdout_stderr():
        jieba.initialize()

    # ---------- 1) 加载配置 ----------
    cfg = load_cfg("configs/rag.yaml")

    # ---------- 2) LLM（静音加载） ----------
    llm_cfg = cfg.get("llm", {})
    with suppress_stdout_stderr():
        llm = build_llm(llm_cfg)

    # ---------- 3) Retriever（静音加载，完全由 YAML 控制） ----------
    paths_cfg = cfg.get("paths", {})
    retrieval_cfg = cfg.get("retrieval", {})
    dense_cfg = retrieval_cfg.get("dense", {})
    rerank_cfg = cfg.get("rerank", {})

    index_root = paths_cfg.get("index_dir", "data/index")
    embed_model_zh = dense_cfg.get("model_zh", "BAAI/bge-small-zh-v1.5")
    embed_model_en = dense_cfg.get("model_en", "BAAI/bge-small-en-v1.5")
    reranker_name = rerank_cfg.get("model_name", "BAAI/bge-reranker-base")

    with suppress_stdout_stderr():
        retriever = DualIndexHybridRetriever(
            index_root=index_root,
            embed_model_zh=embed_model_zh,
            embed_model_en=embed_model_en,
            reranker_name=reranker_name,
        )

    # ---------- 4) 检索/生成参数 ----------
    eval_cfg = cfg.get("eval", {})
    bm25_cfg = retrieval_cfg.get("bm25", {})

    dual_lang = bool(eval_cfg.get("dual_lang", True))
    dense_topk = int(dense_cfg.get("topk", 30))
    bm25_topk = int(bm25_cfg.get("topk", 30))
    merge_topk = int(retrieval_cfg.get("fusion", {}).get("rrf_k", 60))
    rerank_topk = int(rerank_cfg.get("topk", 8))

    # ---------- 5) REPL ----------
    while True:
        query = input("\n👤 你：").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("👋 再见")
            break

        # ---------- 6) 流式输出 ----------
        print("🤖 助手：", end="", flush=True)
        for delta in run_rag_stream(
            query,
            llm,
            retriever,
            dual_lang=dual_lang,
            dense_topk=dense_topk,
            bm25_topk=bm25_topk,
            merge_topk=merge_topk,
            rerank_topk=rerank_topk,
        ):
            print(delta, end="", flush=True)

        # ---------- 7) 流式结束后：结构化证据 ----------
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
