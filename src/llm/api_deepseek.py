# src/llm/api_deepseek.py
from __future__ import annotations

"""
DeepSeek API 后端
- generate：普通非流式
- stream_generate：SSE 流式（yield 字符串片段）
注意：
- 必须 import json
- 流式解析要兼容 choice.delta.content 与 choice.message.content
- 解析异常不要无声吞掉（提供 debug 开关）
"""

import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional

import requests

from .base import LLMProvider
from .schemas import ChatMessage, GenerateConfig, LLMResponse


class DeepSeekAPIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        timeout_sec: float = 60,
    ):
        # 1) 先 strip，避免空格/换行
        api_key = (api_key or "").strip()

        # 2) 支持 ${ENV_NAME} 这种写法（从环境变量读取）
        if api_key.startswith("${") and api_key.endswith("}"):
            env_name = api_key[2:-1].strip()
            api_key = (os.getenv(env_name, "") or "").strip()

        # 3) 防止用户把 "Bearer xxx" 也写进 key
        if api_key.lower().startswith("bearer "):
            api_key = api_key.split(None, 1)[1].strip()

        if not api_key:
            raise ValueError("DeepSeek api_key is empty. Please set env var or pass a real key string.")

        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec


    def _url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate(self, messages: List[ChatMessage], config: Optional[GenerateConfig] = None) -> LLMResponse:
        cfg = config or GenerateConfig()

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "top_p": cfg.top_p,
            "stream": False,
        }

        t0 = time.time()
        r = requests.post(self._url(), headers=self._headers(), json=payload, timeout=self.timeout_sec)

        # 关键：把401的body打印出来，否则你永远只能看到Unauthorized
        if r.status_code == 401:
            key_tail = self.api_key[-4:] if self.api_key else "NONE"
            raise RuntimeError(
                "DeepSeek API 401 Unauthorized.\n"
                f"URL={self._url()}\n"
                f"api_key_len={len(self.api_key) if self.api_key else 0}, api_key_tail={key_tail}\n"
                f"ResponseBody={r.text[:800]}"
            )

        try:
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"DeepSeek API HTTP error: {e}\nURL={self._url()}\nResponseBody={r.text[:800]}"
            ) from e

        data = r.json()
        dt = time.time() - t0

        # DeepSeek 与 OpenAI 风格一致：choices[0].message.content
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""

        meta: Optional[Dict[str, Any]] = None
        if cfg.return_meta:
            meta = {
                "latency_sec": dt,
                "usage": data.get("usage"),
                "model": self.model,
                "backend": "api",
            }
        return LLMResponse(text=text, meta=meta)

    def stream_generate(self, messages: List[ChatMessage], config: Optional[GenerateConfig] = None) -> Iterable[str]:
        """
        DeepSeek 流式生成：yield 文本片段（通常是增量 token 或小片段）。
        你在 pipeline 中写：
          for delta in llm.stream_generate(...):
              yield delta
        这里就会真实吐 token。
        """
        cfg = config or GenerateConfig()

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "top_p": cfg.top_p,
            "stream": True,
        }

        debug = os.getenv("DEEPSEEK_STREAM_DEBUG", "0") == "1"

        with requests.post(
            self._url(),
            headers=self._headers(),
            json=payload,
            stream=True,
            timeout=self.timeout_sec,
        ) as r:
            r.raise_for_status()

            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue

                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                    choice = (chunk.get("choices") or [{}])[0]

                    # 兼容两类：delta 风格（stream）与 message 风格（部分实现/边界 chunk）
                    delta_text = (choice.get("delta") or {}).get("content")
                    if not delta_text:
                        delta_text = (choice.get("message") or {}).get("content")

                    if delta_text:
                        yield delta_text
                except Exception as e:
                    if debug:
                        yield f"\n[DeepSeekStreamParseError] {repr(e)} | line={line}\n"
                    continue
