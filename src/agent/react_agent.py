# src/agent/react_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re

from agent.prompt import REACT_SYSTEM_PROMPT
from agent.tools import ToolRegistry
from llm.schemas import ChatMessage, GenerateConfig, LLMResponse


@dataclass
class AgentConfig:
    max_steps: int = 6
    # 可选：控制模型温度/最大token等（看你 build_llm 支持什么）
    temperature: float = 0.2
    max_tokens: int = 800


_ACTION_RE = re.compile(
    r"Action:\s*(?P<tool>[a-zA-Z0-9_\-]+)\s*[\r\n]+Action Input:\s*(?P<input>.*)",
    re.DOTALL
)
_FINAL_RE = re.compile(r"Final Answer:\s*(?P<final>.*)", re.DOTALL)


def _extract_action(text: str) -> Optional[Tuple[str, str]]:
    m = _ACTION_RE.search(text)
    if not m:
        return None
    tool = m.group("tool").strip()
    tool_input = m.group("input").strip()
    return tool, tool_input


def _extract_final(text: str) -> Optional[str]:
    m = _FINAL_RE.search(text)
    if not m:
        return None
    return m.group("final").strip()


class ReActAgent:
    def __init__(self, llm: Any, tools: ToolRegistry, cfg: Optional[AgentConfig] = None):
        """
        llm: 你现有的 LLM provider（需提供 generate(messages, config) 或类似接口）
        tools: 工具注册表
        """
        self.llm = llm
        self.tools = tools
        self.cfg = cfg or AgentConfig()

        self.system_prompt = REACT_SYSTEM_PROMPT.format(tool_list=self.tools.render_tool_list())

    def run(self, user_query: str) -> Dict[str, Any]:
        """
        返回结构里带：
        - final_answer
        - steps（每一步模型输出、工具调用、observation）
        - messages（可用于debug）
        """
        steps: List[Dict[str, Any]] = []

        messages: List[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_query.strip()),
        ]

        for step_idx in range(self.cfg.max_steps):
            # 你项目里可能是 llm.generate(messages, GenerateConfig(...))
            # 这里做一个“适配层”：尽量只要求 llm.generate(messages, cfg)
            text = self._llm_generate(messages)

            steps.append({"llm_output": text})

            final = _extract_final(text)
            if final is not None:
                return {
                    "final_answer": final,
                    "steps": steps,
                    "messages": messages,
                    "stop_reason": "final",
                }

            act = _extract_action(text)
            if act is None:
                # 模型没按格式来：给它一个强制纠错提示继续
                repair = (
                    "Your last response did not follow the required format. "
                    "Either output an Action block (Thought/Action/Action Input) "
                    "or output Final Answer."
                )
                messages.append(ChatMessage(role="assistant", content=text))
                messages.append(ChatMessage(role="user", content=repair))
                steps[-1]["repair"] = True
                continue

            tool_name, tool_input = act
            observation = self.tools.call(tool_name, tool_input)

            steps[-1]["tool_name"] = tool_name
            steps[-1]["tool_input"] = tool_input
            steps[-1]["observation"] = observation

            # 把模型输出和工具 observation 都塞回上下文
            messages.append(ChatMessage(role="assistant", content=text))
            # 工具输出作为“用户后续提供的信息”喂回去，避免 DeepSeek 不支持 role="tool"
            messages.append(
                ChatMessage(
                    role="user",
                    content=f"[Tool:{tool_name}] {observation}",
                )
            )

        return {
            "final_answer": "I couldn't finish within the step limit.",
            "steps": steps,
            "messages": messages,
            "stop_reason": "max_steps",
        }

    def _llm_generate(self, messages: List[ChatMessage]) -> str:
        """
        适配当前 LLMProvider 接口（src/llm/base.py）：
        - generate(messages: List[ChatMessage], config: Optional[GenerateConfig]) -> LLMResponse
        """
        if not hasattr(self.llm, "generate"):
            raise RuntimeError("LLM provider does not implement .generate()")

        resp = self.llm.generate(
            messages,
            GenerateConfig(
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
            ),
        )
        if isinstance(resp, LLMResponse):
            return resp.text
        # 兼容性兜底：如果底层返回的是 str
        if isinstance(resp, str):
            return resp
        raise TypeError(f"Unexpected LLM.generate return type: {type(resp)}")