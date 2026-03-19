import os
import json
import time
import requests
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()

FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "").strip()
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "").strip()
FEISHU_VERIFICATION_TOKEN = os.getenv("FEISHU_VERIFICATION_TOKEN", "").strip()

TOKEN_CACHE = {"token": "", "expire_at": 0}


def get_tenant_access_token() -> str:
    """
    自建应用用 app_id/app_secret 换 tenant_access_token（有效期约2小时）
    """
    now = int(time.time())
    if TOKEN_CACHE["token"] and now < TOKEN_CACHE["expire_at"] - 60:
        return TOKEN_CACHE["token"]

    if not FEISHU_APP_ID or not FEISHU_APP_SECRET:
        raise RuntimeError("Missing FEISHU_APP_ID / FEISHU_APP_SECRET")

    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    resp = requests.post(url, json={"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"get token failed: {data}")

    token = data["tenant_access_token"]
    expire = int(data.get("expire", 7200))
    TOKEN_CACHE["token"] = token
    TOKEN_CACHE["expire_at"] = now + expire
    return token


def reply_text(message_id: str, text: str):
    """
    直接“回复”收到的那条消息（最简单）
    """
    token = get_tenant_access_token()
    url = f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"}
    payload = {"msg_type": "text", "content": json.dumps({"text": text}, ensure_ascii=False)}
    r = requests.post(url, headers=headers, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def run_agent(user_text: str) -> str:
    """
    TODO：替换成你的 Agent：
    out = agent.run(user_text)
    return out["final_answer"]
    """
    return f"（Demo）我收到了：{user_text}"


@app.post("/feishu/event")
async def feishu_event(request: Request):
    body = await request.json()

    # 1) URL 验证：飞书会发 url_verification，必须原样回 challenge
    # 官方事件订阅配置案例里就是这个流程
    if body.get("type") == "url_verification":
        return {"challenge": body.get("challenge")}  # 让飞书通过校验

    # 2) 事件回调：校验 Verification Token（防伪造）
    token = body.get("token")
    if FEISHU_VERIFICATION_TOKEN and token != FEISHU_VERIFICATION_TOKEN:
        raise HTTPException(status_code=401, detail="invalid verification token")

    event = (body.get("event") or {})
    # 机器人收到消息事件
    message = event.get("message") or {}
    message_id = message.get("message_id")
    content = message.get("content")  # 注意：通常是 JSON 字符串，如 {"text":"..."}
    msg_type = message.get("message_type")

    if not message_id or not msg_type:
        return {"ok": True}

    user_text = ""
    if msg_type == "text" and content:
        try:
            user_text = json.loads(content).get("text", "")
        except Exception:
            user_text = str(content)

    if not user_text.strip():
        return {"ok": True}

    answer = run_agent(user_text)
    # 回复消息
    reply_text(message_id, answer)

    # 飞书只要求你返回 200 即可
    return {"ok": True}