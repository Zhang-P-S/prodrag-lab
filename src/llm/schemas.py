# src/llm/schemas.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    """统一的对话消息结构（上层 pipeline 只依赖这个，不关心具体后端）。"""
    role: Role
    content: str


@dataclass
class GenerateConfig:
    """
    统一生成参数（尽量覆盖 API 和本地模型的常见参数）。
    - return_meta=True 时，provider 会尽量返回耗时、token usage 等信息（若后端支持）。
    """
    max_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 1.0

    # 采样/稳定性相关：本地模型常用
    do_sample: Optional[bool] = None  # None 表示自动：temperature>0 => True

    # 本地推理常见参数
    repetition_penalty: float = 1.0
    seed: Optional[int] = None

    # 是否返回 meta（耗时/usage/模型名等）
    return_meta: bool = False


@dataclass
class LLMResponse:
    """统一返回结构。"""
    text: str
    meta: Optional[Dict[str, Any]] = None
