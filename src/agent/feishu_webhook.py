import os
import json
import requests

def feishu_send_text(text: str, webhook: str | None = None, timeout: int = 10) -> dict:
    """
    发送纯文本到飞书群自定义机器人 Webhook
    """
    webhook = webhook or os.getenv("FEISHU_WEBHOOK", "").strip()
    if not webhook:
        raise ValueError("Missing FEISHU_WEBHOOK env or webhook argument")

    payload = {
        "msg_type": "text",
        "content": {"text": text},
    }

    r = requests.post(webhook, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def feishu_send_card(title: str, md: str, webhook: str | None = None, timeout: int = 10) -> dict:
    """
    发送一个简单“消息卡片”（interactive）
    """
    webhook = webhook or os.getenv("FEISHU_WEBHOOK", "").strip()
    if not webhook:
        raise ValueError("Missing FEISHU_WEBHOOK env or webhook argument")

    payload = {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {"title": {"tag": "plain_text", "content": title}},
            "elements": [
                {"tag": "markdown", "content": md},
            ],
        },
    }

    r = requests.post(webhook, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()