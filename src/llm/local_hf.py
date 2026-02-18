# src/llm/local_hf.py
from __future__ import annotations

"""
本地 HuggingFace 推理后端（支持 base + 可选 LoRA）
特点：
- 启动时加载模型一次（重）
- generate：一次性返回完整文本
- stream_generate：使用 TextIteratorStreamer 真流式输出（推荐）
"""

import time
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from threading import Thread

try:
    from peft import PeftModel
except Exception:
    PeftModel = None  # 允许没有 peft 的环境也能跑 base

from .base import LLMProvider
from .schemas import ChatMessage, GenerateConfig, LLMResponse


def _pick_dtype(dtype: str) -> torch.dtype:
    d = (dtype or "float16").lower()
    if d in {"fp16", "float16"}:
        return torch.float16
    if d in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


class LocalHFProvider(LLMProvider):
    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "float16",
        load_in_4bit: bool = False,
        max_input_tokens: int = 4096,
    ):
        if not model_path:
            raise ValueError("local model_path is empty")

        self.model_path = model_path
        self.lora_path = lora_path
        self.device = device
        self.dtype = _pick_dtype(dtype)
        self.load_in_4bit = load_in_4bit
        self.max_input_tokens = max_input_tokens

        # 1) tokenizer
        # ✅ 1) 先加载 tokenizer（关键：赋给 self.tokenizer）
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )

        # 强制整模型都上 GPU
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # 2) model
        # 中文说明：
        # - 如果你用 4bit，需要 bitsandbytes 环境支持；否则直接 fp16/bf16
        # - device_map="auto" 能自动放 GPU/CPU（多卡也行）
        model_kwargs: Dict[str, Any] = dict(
            torch_dtype=self.dtype,
            device_map="auto" if device != "cpu" else None,
        )

        if load_in_4bit:
            # 这里不强制写 BitsAndBytesConfig，避免你环境没有 bitsandbytes 时直接崩
            # 如果你确定要 4bit，可在此处按你项目已有写法接 BitsAndBytesConfig
            model_kwargs["load_in_4bit"] = True
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        # 3) optional LoRA
        if lora_path:
            if PeftModel is None:
                raise RuntimeError("peft is not installed, but lora_path is provided.")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model.eval()
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def _build_prompt(self, messages: List[ChatMessage]) -> str:
        """
        把 ChatMessage 列表拼成单段 prompt。
        说明：
        - 不同模型 chat template 不同；最稳妥是使用 tokenizer.apply_chat_template（若支持）
        - 若 tokenizer 没有 chat_template，就用简单拼接（通用但不如模板质量高）
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": m.role, "content": m.content} for m in messages],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # fallback：简易拼接
        parts = []
        for m in messages:
            if m.role == "system":
                parts.append(f"[SYSTEM]\n{m.content}\n")
            elif m.role == "user":
                parts.append(f"[USER]\n{m.content}\n")
            else:
                parts.append(f"[ASSISTANT]\n{m.content}\n")
        parts.append("[ASSISTANT]\n")
        return "\n".join(parts)

    def generate(self, messages: List[ChatMessage], config: Optional[GenerateConfig] = None) -> LLMResponse:
        cfg = config or GenerateConfig()
        prompt = self._build_prompt(messages)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = cfg.do_sample
        if do_sample is None:
            do_sample = cfg.temperature > 1e-6

        gen_kwargs = dict(
            max_new_tokens=cfg.max_tokens,
            do_sample=do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        dt = time.time() - t0

        # 只取新增部分（避免把 prompt 原样回显）
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        meta = None
        if cfg.return_meta:
            meta = {"latency_sec": dt, "model": self.model_path, "backend": "local"}
        return LLMResponse(text=text, meta=meta)

    def stream_generate(self, messages: List[ChatMessage], config: Optional[GenerateConfig] = None) -> Iterable[str]:
        """
        真流式：边 generate 边 yield。
        注意：本地真流式需要线程 + TextIteratorStreamer。
        """
        cfg = config or GenerateConfig()
        prompt = self._build_prompt(messages)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = cfg.do_sample
        if do_sample is None:
            do_sample = cfg.temperature > 1e-6

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,            # 不输出 prompt
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=cfg.max_tokens,
            do_sample=do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        # 生成放后台线程，主线程从 streamer 迭代输出
        def _worker():
            with torch.no_grad():
                self.model.generate(**gen_kwargs)

        t = Thread(target=_worker, daemon=True)
        t.start()

        for piece in streamer:
            if piece:
                yield piece
