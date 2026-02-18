# src/llm/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

from .schemas import ChatMessage, GenerateConfig, LLMResponse


class LLMProvider(ABC):
    """
    LLM 后端统一接口：
    - generate：一次性返回完整文本
    - stream_generate：以 generator 的方式逐段产出文本（用于终端/网页流式输出）
    """

    @abstractmethod
    def generate(self, messages: List[ChatMessage], config: Optional[GenerateConfig] = None) -> LLMResponse:
        raise NotImplementedError

    def stream_generate(self, messages: List[ChatMessage], config: Optional[GenerateConfig] = None) -> Iterable[str]:
        """
        默认实现：用 generate 的结果做“伪流式”切片输出（保证所有后端至少可用）。
        子类可覆盖成真流式（API SSE / 本地 streamer）。
        """
        resp = self.generate(messages, config=config)
        text = resp.text or ""
        # 简单按字符切片（你也可以按 token/句子切分）
        for ch in text:
            yield ch


def build_llm(cfg: Dict[str, Any]) -> LLMProvider:
    """
    根据配置构造 LLM 后端（工厂函数）。
    cfg 示例（来自 configs/rag.yaml 的 llm 节点）：
      {
        "backend": "api",
        "api": {"provider":"deepseek","api_key":"...","model":"deepseek-chat","base_url":"https://api.deepseek.com"},
        "local": {"model_path":"...","dtype":"float16","device":"cuda","lora_path":None, ...}
      }
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"llm cfg must be dict, got {type(cfg)}")

    backend = (cfg.get("backend") or "api").lower()

    if backend == "api":
        api_cfg = cfg.get("api") or {}
        provider = (api_cfg.get("provider") or "deepseek").lower()

        if provider == "deepseek":
            from .api_deepseek import DeepSeekAPIProvider
            return DeepSeekAPIProvider(
                api_key=api_cfg.get("api_key", ""),
                model=api_cfg.get("model", "deepseek-chat"),
                base_url=api_cfg.get("base_url", "https://api.deepseek.com"),
                timeout_sec=float(api_cfg.get("timeout_sec", 60)),
            )

        raise ValueError(f"Unknown api provider: {provider}")

    if backend == "local":
        local_cfg = cfg.get("local") or {}
        from .local_hf import LocalHFProvider
        return LocalHFProvider(
            model_path=local_cfg.get("model_path", ""),
            lora_path=local_cfg.get("lora_path"),
            device=str(local_cfg.get("device", "cuda")),
            dtype=str(local_cfg.get("dtype", "float16")),
            load_in_4bit=bool(local_cfg.get("load_in_4bit", False)),
            device_map=local_cfg.get("device_map"),
            max_memory=local_cfg.get("max_memory"),
            llm_int8_enable_fp32_cpu_offload=bool(local_cfg.get("llm_int8_enable_fp32_cpu_offload", False)),
            max_input_tokens=int(local_cfg.get("max_input_tokens", 4096)),
        )

    raise ValueError(f"Unknown llm backend: {backend}")
