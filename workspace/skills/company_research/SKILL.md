---
name: Company Research
description: 帮助用户对指定公司进行多维度研究，覆盖产品、市场、团队、融资、竞争等维度，并生成结构化 Markdown 报告。
script: "Company_Ressearch.py"
scriptArgs: ["{company}", "--depth", "{depth}"]
metadata: {"openclaw":{"requires":{"env":["DASHSCOPE_API_KEY"]}}}
---

Use this when the user asks for company research, due diligence, or a report on a specific company.

The script produces a structured Markdown report. When invoking via the chatbot, pass:
- **company**: company name (e.g. OpenAI, 腾讯)
- **depth**: brief | standard | deep | investment (default standard)
