## Company Research 脚本使用说明

该脚本用于对指定公司进行多维度研究，并生成结构化的 Markdown 报告，已接入阿里云 DashScope 大模型。

### 1. 环境准备

- 安装依赖：

```bash
pip install dashscope
```

- 配置环境变量 `DASHSCOPE_API_KEY`（推荐在项目根目录的 `.env` 中设置，或直接在系统环境中设置）：

```bash
export DASHSCOPE_API_KEY=你的DashScope密钥
```

### 2. 运行方式

**命令行**（在项目根或 `workspace/skills` 下）：

```bash
# 从项目根
python workspace/skills/company_research/Company_Ressearch.py OpenAI

# 或进入 skill 目录
cd workspace/skills/company_research

# 标准深度
python Company_Ressearch.py OpenAI

# 指定深度和输出目录
python Company_Ressearch.py OpenAI --depth deep --output ./reports
```

- `company`：必填，公司名称（如 `OpenAI`、`腾讯` 等）。
- `--depth`：研究深度，可选值：
  - `brief`：简要版，150–250 字。
  - `standard`：标准版（默认）。
  - `deep`：详细版。
  - `investment`：偏投资视角的深度研究。
- `--output`：报告输出目录，默认当前目录。

生成的报告文件名形如：

```text
OpenAI_research_20260210.md
```

### 3. 交互式模式

如果不在命令中提供公司名：

```bash
python Company_Ressearch.py
```

脚本会提示输入公司名，按回车后自动生成并在终端打印报告内容，同时按上述规则保存为 Markdown 文件。

### 4. 在 ChatBot 中通过技能执行当 **Skill script execution** 开启时（UI 勾选或设置 `SKILL_EXEC_ENABLED=1`），聊天中的模型可以请求执行本技能：在回复中输出：```
[[SKILL:Company Research]]
company=OpenAI
depth=standard
[[/SKILL]]
```系统会运行 `Company_Ressearch.py` 并将脚本输出交给模型总结后返回给用户。技能仅在 `workspace/skills/` 下且于 `SKILL.md` 中声明了 `script` 的目录中执行（允许列表）。超时由 `SKILL_EXEC_TIMEOUT`（秒，默认 300）控制。