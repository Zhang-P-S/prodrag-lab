# src/llm/local_hf.py
from __future__ import annotations

"""
本地 HuggingFace 推理后端（支持 base + 可选 LoRA）

设计目标：
- 启动时加载模型一次（重）
- generate：一次性返回完整文本
- stream_generate：使用 TextIteratorStreamer 真流式输出（推荐）
- 可选 4bit 量化（bitsandbytes），不开就用 fp16/bf16

⚠️ 注意：
- 4bit 量化需要安装 bitsandbytes（以及合适 CUDA 版本）
- LoRA 需要安装 peft
"""

import time
from threading import Thread
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

# 只有需要 4bit 时才会用到；这里导入，环境没装也能正常跑（会在使用时再报错）
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None  # type: ignore

try:
    from peft import PeftModel
except Exception:
    PeftModel = None  # 允许没有 peft 的环境也能跑 base

from .base import LLMProvider
from .schemas import ChatMessage, GenerateConfig, LLMResponse


def _pick_dtype(dtype: str) -> torch.dtype:
    """把字符串 dtype 转成 torch.dtype。"""
    d = (dtype or "float16").lower()
    if d in {"fp16", "float16"}:
        return torch.float16
    if d in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


class LocalHFProvider(LLMProvider):
    """
    本地 HuggingFace 推理后端

    参数说明：
    - model_path: HF 模型路径（本地目录或仓库名）
    - lora_path: LoRA adapter 路径（可选）
    - device: "cuda" / "cpu"（一般用 cuda）
    - dtype: "float16" / "bfloat16" / "float32"
    - load_in_4bit: 是否启用 4bit 量化
    - max_input_tokens: prompt 截断长度（避免超长）
    """

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

        # -------------------------------------------------
        # 1) tokenizer（必须挂到 self.tokenizer）
        # -------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )

        # 一些模型没有 pad_token，推理时可能 warning 或报错；这里做兜底
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 有的 causal LM 更适合 left padding（尤其 batch 推理时）
        # 不强制也行；如果你后续做 batch，可以考虑打开
        # self.tokenizer.padding_side = "left"

        # -------------------------------------------------
        # 2) model：按是否 4bit 分支加载
        # -------------------------------------------------
        model_kwargs: Dict[str, Any] = {}

        # device_map 选择：
        # - 你想“强制整模型上单卡0”：可以 device_map={"":0}
        # - 更通用更稳：device_map="auto"（单卡、多卡、CPU offload 都能处理）
        if device == "cpu":
            model_kwargs["device_map"] = None
        else:
            model_kwargs["device_map"] = "auto"

        # dtype 选择：
        # - 4bit 情况下，torch_dtype 主要影响部分计算/权重加载策略
        # - 非 4bit 情况下 torch_dtype 直接决定加载精度（fp16/bf16）
        model_kwargs["torch_dtype"] = self.dtype
        model_kwargs["low_cpu_mem_usage"] = True
        model_kwargs["trust_remote_code"] = True

        # 只有 load_in_4bit=True 才配置量化
        if self.load_in_4bit:
            if BitsAndBytesConfig is None:
                raise RuntimeError(
                    "你开启了 load_in_4bit=True，但 transformers.BitsAndBytesConfig 不可用；"
                    "请确认 transformers 版本与 bitsandbytes 已安装。"
                )
            # 4bit 量化配置（nf4 通常是主流选择）
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                # compute_dtype 建议 bf16（若显卡支持），否则 fp16
                bnb_4bit_compute_dtype=torch.bfloat16
                if torch.cuda.is_available()
                else torch.float16,
            )
            model_kwargs["quantization_config"] = bnb_config

        # ✅ 正确加载模型（使用 model_kwargs，不要写死）
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        )
        self.model.eval()

        # -------------------------------------------------
        # 3) optional LoRA
        # -------------------------------------------------
        if lora_path:
            if PeftModel is None:
                raise RuntimeError("peft is not installed, but lora_path is provided.")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model.eval()

    def _build_prompt(self, messages: List[ChatMessage]) -> str:
        """
        把 ChatMessage 列表拼成 prompt。

        最优做法：优先用 tokenizer.apply_chat_template（如果模型提供 chat_template）
        兜底：简单拼接（通用但可能不如模板对齐）
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

        # tokenize + 截断（避免 prompt 过长 OOM）
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        # 把输入放到正确设备：
        # - 如果 device_map="auto"，model.device 可能不是一个固定 device（尤其多卡）
        # - 但 inputs 一般放到 cuda:0 即可，HF 会处理跨设备（大多数情况下没问题）
        # - 更稳方式：取 inputs 送到 self.model.device（对单卡最稳）
        try:
            target_device = self.model.device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
        except Exception:
            # 多卡情况下 self.model.device 可能不可用；退化到 cuda:0 或 cpu
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

        # do_sample：若温度极低，则默认为贪心
        do_sample = cfg.do_sample
        if do_sample is None:
            do_sample = cfg.temperature > 1e-6

        gen_kwargs = dict(
            max_new_tokens=cfg.max_tokens,
            do_sample=do_sample,
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,  # ✅ 加速，尤其是流式
        )

        # ✅ 只有采样模式才传这些
        if do_sample:
            gen_kwargs.update(
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )

        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        dt = time.time() - t0

        # 只取新增部分（避免把 prompt 原样回显）
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        meta = None
        if cfg.return_meta:
            meta = {
                "latency_sec": dt,
                "model": self.model_path,
                "backend": "local",
                "load_in_4bit": bool(self.load_in_4bit),
                "lora": bool(self.lora_path),
            }
        return LLMResponse(text=text, meta=meta)

    def stream_generate(self, messages: List[ChatMessage], config: Optional[GenerateConfig] = None) -> Iterable[str]:
        """
        真流式：边 generate 边 yield。

        原理：
        - TextIteratorStreamer 会在模型生成时不断产出 token 文本
        - 需要把 generate 放到后台线程，主线程迭代 streamer 输出
        """
        cfg = config or GenerateConfig()
        prompt = self._build_prompt(messages)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        # 同 generate 的设备处理逻辑
        try:
            target_device = self.model.device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
        except Exception:
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

        do_sample = cfg.do_sample
        if do_sample is None:
            do_sample = cfg.temperature > 1e-6

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,          # 不输出 prompt
            skip_special_tokens=True,  # 过滤特殊 token
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=cfg.max_tokens,
            do_sample=do_sample,
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
            use_cache=True,  # ✅
        )

        if do_sample:
            gen_kwargs.update(
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )


        def _worker():
            with torch.no_grad():
                self.model.generate(**gen_kwargs)

        t = Thread(target=_worker, daemon=True)
        t.start()

        for piece in streamer:
            if piece:
                yield piece
