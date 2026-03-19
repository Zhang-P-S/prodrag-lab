from feishu_webhook import feishu_send_text, feishu_send_card

feishu_send_text("✅ ProdRAG-Agent 已上线：这是一条测试消息")
feishu_send_card(
    title="ProdRAG-Agent 测试卡片",
    md="**Hello** 飞书！\n\n- 我可以把评测结果/回答推送到群里\n- 下一步可接 RAG + web_search"
)