# src/agent/tools.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import re
import math


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[[str], str]


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        if tool.name in self.tools:
            raise ValueError(f"tool already exists: {tool.name}")
        self.tools[tool.name] = tool

    def call(self, name: str, tool_input: str) -> str:
        if name not in self.tools:
            return f"[tool_error] unknown tool: {name}"
        try:
            return self.tools[name].fn(tool_input)
        except Exception as e:
            return f"[tool_error] {name} crashed: {type(e).__name__}: {e}"

    def render_tool_list(self) -> str:
        lines = []
        for t in self.tools.values():
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)


# -----------------------------
# Tool: calculator（安全版）
# -----------------------------
_ALLOWED_EXPR = re.compile(r"^[0-9\.\+\-\*\/\(\)\s]+$")

def calculator_tool(expr: str) -> str:
    """
    极简计算器：仅允许数字 + - * / () . 空格
    防止注入：不允许字母/下划线/方括号等
    """
    expr = expr.strip()
    if not expr:
        return "[calculator] empty expression"
    if not _ALLOWED_EXPR.match(expr):
        return "[calculator] invalid characters"
    # 进一步限制长度，避免滥用
    if len(expr) > 200:
        return "[calculator] expression too long"

    # 仅使用 eval 的安全子集：无 builtins
    try:
        val = eval(expr, {"__builtins__": {}}, {})
    except Exception as e:
        return f"[calculator] error: {type(e).__name__}: {e}"
    return f"[calculator] {val}"


# -----------------------------
# Tool: refuse（统一拒答口）
# -----------------------------
def refuse_tool(reason: str) -> str:
    reason = reason.strip() or "insufficient information"
    return f"[refuse] {reason}"


# -----------------------------
# Tool: rag_search（你项目接入点）
# -----------------------------
def build_rag_search_tool(run_rag_fn: Callable[[str], Any]) -> Callable[[str], str]:
    """
    传入你现有的 run_rag(query) 或封装函数。
    run_rag_fn(query) -> 你自己的返回结构（建议含 answer / citations 等）
    这里统一转成字符串 Observation 给 Agent。
    """
    def rag_search(query: str) -> str:
        q = query.strip()
        if not q:
            return "[rag_search] empty query"

        out = run_rag_fn(q)

        # 兼容几种常见返回：
        # 1) out 是 str
        if isinstance(out, str):
            return f"[rag_search]\n{out}"

        # 2) out 有 answer/citations
        answer = getattr(out, "answer", None)
        citations = getattr(out, "citations", None)

        if answer is None:
            # 兜底：转 str
            return f"[rag_search]\n{str(out)}"

        obs = f"[rag_search]\n{answer}"
        if citations:
            obs += f"\n[citations] {citations}"
        return obs

    return rag_search


def default_tool_registry(run_rag_fn: Callable[[str], Any]) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(
        name="rag_search",
        description="Query the internal knowledge base (RAG) and return grounded info/citations.",
        fn=build_rag_search_tool(run_rag_fn),
    ))
    reg.register(Tool(
        name="calculator",
        description="Do basic arithmetic with + - * / and parentheses. Input: a math expression string.",
        fn=calculator_tool,
    ))
    reg.register(Tool(
        name="refuse",
        description="Use when the question cannot be answered safely or lacks evidence. Input: brief reason.",
        fn=refuse_tool,
    ))
    return reg