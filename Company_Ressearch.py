#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenClaw Skill: Company Research
功能：对指定公司进行多维度研究，并生成结构化 Markdown 报告。

本文件已接入阿里云 DashScope，大模型调用依赖环境变量 `DASHSCOPE_API_KEY`。
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # 未安装时仅依赖系统环境变量

# 启动时从 .env 加载环境变量（当前工作目录或脚本所在目录）
if load_dotenv is not None:
    load_dotenv()
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

try:
    # DashScope 官方 SDK（需先安装：pip install dashscope）
    import dashscope
    from http import HTTPStatus
except ImportError:
    dashscope = None  # 允许导入失败，在实际调用时再报错，便于静态分析
    HTTPStatus = None


def _get_dashscope_api_key() -> str:
    """从环境变量中获取 DashScope API Key，没有则抛出明确错误。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "环境变量 DASHSCOPE_API_KEY 未配置，无法调用 DashScope 大模型。\n"
            "请在 .env 或系统环境中设置 DASHSCOPE_API_KEY 后重试。"
        )
    return api_key


def call_llm(prompt: str, model: str = "qwen-plus") -> str:
    """
    使用 DashScope 调用大模型生成内容。

    :param prompt: 完整提示词（中文/英文均可）
    :param model: DashScope 模型名称，默认使用 qwen-plus
    :return: 模型生成的纯文本内容
    """
    if dashscope is None:
        raise RuntimeError(
            "未安装 dashscope SDK，无法调用 DashScope 大模型。\n"
            "请先运行：pip install dashscope"
        )

    api_key = _get_dashscope_api_key()
    dashscope.api_key = api_key

    try:
        response = dashscope.Generation.call(
            model=model,
            prompt=prompt,
        )
    except Exception as e:
        raise RuntimeError(f"调用 DashScope 接口失败：{e}") from e

    # 兼容新版/旧版返回结构：优先按官方推荐字段解析
    # 常见结构：response.output.text 或 response["output"]["text"]
    text = None
    if hasattr(response, "output") and getattr(response.output, "text", None):
        text = response.output.text
    elif isinstance(response, dict):
        # dict 形式
        output = response.get("output") or {}
        text = output.get("text") or output.get("choices", [{}])[0].get("message", {}).get("content")

    if not text:
        # 尝试直接转字符串作为兜底
        text = str(response)

    return text


class CompanyResearchSkill:
    """公司研究 Skill 主逻辑"""

    NAME = "Company Research"
    DESCRIPTION = "帮助用户对特定公司进行系统性研究，覆盖产品、市场、团队、融资、竞争等维度"

    # 研究维度定义（可扩展）
    DIMENSIONS = [
        {
            "id": "basic_info",
            "title": "1. 基本信息",
            "prompt_template": "提取 {company_name} 的基本信息：成立时间、总部地点、当前状态（上市/私有）、官网链接、所属主要行业。"
        },
        {
            "id": "products",
            "title": "2. 核心产品与服务",
            "prompt_template": "详细描述 {company_name} 的核心产品/服务，包括旗舰产品、技术特点、目标用户群体、核心技术壁垒。"
        },
        {
            "id": "business_model",
            "title": "3. 商业模式",
            "prompt_template": "分析 {company_name} 的商业模式：主要收入来源、定价策略、主要客户类型、变现路径、毛利率水平（如果可得）。"
        },
        {
            "id": "market",
            "title": "4. 市场与规模",
            "prompt_template": "评估 {company_name} 的市场情况：TAM/SAM/SOM、市场占有率、增长趋势、主要市场区域、用户/客户规模（尽可能使用最新数据）。"
        },
        {
            "id": "team",
            "title": "5. 团队与创始人",
            "prompt_template": "介绍 {company_name} 的核心团队：创始人背景、关键高管、技术/产品/销售团队实力、公司人员规模（最新数据）。"
        },
        {
            "id": "funding",
            "title": "6. 融资历史",
            "prompt_template": "列出 {company_name} 的融资历史：各轮时间、金额、估值、主要投资机构、最近一轮情况（尽量完整，按时间倒序列出）。"
        },
        {
            "id": "competition",
            "title": "7. 竞争格局",
            "prompt_template": "分析 {company_name} 的竞争环境：主要竞争对手、与竞品的对比（功能/价格/市场份额）、自身竞争优势与劣势、护城河。"
        },
        {
            "id": "recent_news",
            "title": "8. 最新动态（近12个月）",
            "prompt_template": "总结 {company_name} 近12个月内的重要事件：新产品发布、重大合作、诉讼、裁员、融资、政策影响等。"
        },
        {
            "id": "risks",
            "title": "9. 主要风险与挑战",
            "prompt_template": "列出 {company_name} 目前面临的主要风险与挑战，包括经营风险、法律风险、宏观环境风险、核心依赖风险等。"
        },
        {
            "id": "summary",
            "title": "10. 总体评价与结论",
            "prompt_template": "对 {company_name} 进行综合评价，给出一句核心结论，并列出 3-5 个最关键的投资/业务看点。"
        }
    ]

    def __init__(self, company_name: str, depth: str = "standard"):
        self.company_name = company_name.strip()
        self.depth = depth.lower()  # "brief", "standard", "deep", "investment"
        self.report = []
        self.metadata = {
            "company": self.company_name,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "depth": self.depth,
            "version": "1.0"
        }

    def run(self) -> str:
        """执行研究并生成完整报告"""
        self.report.append(self._generate_header())

        for dim in self.DIMENSIONS:
            section_content = self._research_dimension(dim)
            self.report.append(section_content)

        return "\n\n".join(self.report)

    def _generate_header(self) -> str:
        return f"""# {self.company_name} 公司研究报告

**生成时间**：{self.metadata["generated_at"]}  
**研究深度**：{self.depth.capitalize()}  
**目的**：全面了解公司现状与发展潜力

"""

    def _research_dimension(self, dim: Dict[str, Any]) -> str:
        prompt = dim["prompt_template"].format(company_name=self.company_name)

        # 根据深度调整提示词强度
        if self.depth == "deep" or self.depth == "investment":
            prompt += "\n请尽可能提供具体数据、数字、日期、机构/人物名称，并引用可靠来源（如Crunchbase、PitchBook、公司官网、财报等）。"
        elif self.depth == "brief":
            prompt += "\n请简洁回答，控制在150-250字以内。"

        # 调用 LLM 获取内容
        content = call_llm(prompt)

        return f"""## {dim["title"]}

{content}

"""

    def save(self, output_dir: str = "."):
        """保存报告

        1. 为每个研究维度生成一个临时 Markdown 文件；
        2. 将所有内容整合成一个最终报告 Markdown 文件；
        3. 删除各维度的临时 Markdown 文件，仅保留最终报告。
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        safe_company = self.company_name.replace(" ", "_")
        today_str = datetime.now().strftime("%Y%m%d")

        # self.report[0] 是总报告头部，其余依次为各维度内容
        header = self.report[0] if self.report else ""
        sections = self.report[1:] if len(self.report) > 1 else []

        # 1. 为每个维度生成独立的临时 Markdown 文件
        section_files = []
        for idx, (dim, section_md) in enumerate(zip(self.DIMENSIONS, sections), start=1):
            section_filename = f"{safe_company}_{idx:02d}_{dim['id']}.md"
            section_path = os.path.join(output_dir, section_filename)
            with open(section_path, "w", encoding="utf-8") as sf:
                sf.write(section_md)
            section_files.append(section_path)

        # 2. 生成最终整合报告
        final_filename = f"{safe_company}_research_{today_str}.md"
        final_path = os.path.join(output_dir, final_filename)
        final_content = "\n\n".join(self.report)
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        # 3. 删除临时的维度 Markdown 文件
        for section_path in section_files:
            try:
                if os.path.exists(section_path):
                    os.remove(section_path)
            except OSError:
                # 删除失败不影响主流程
                pass

        print(f"报告已保存至：{final_path}")
        return final_path


def main():
    parser = argparse.ArgumentParser(description="Company Research Skill")
    parser.add_argument("company", type=str, help="要研究的公司名称")
    parser.add_argument("--depth", type=str, default="standard",
                        choices=["brief", "standard", "deep", "investment"],
                        help="研究深度")
    parser.add_argument("--output", type=str, default=".",
                        help="输出目录")

    args = parser.parse_args()

    skill = CompanyResearchSkill(args.company, args.depth)
    report_content = skill.run()
    # 统一采用本地保存逻辑，使脚本可以独立运行
    skill.save(args.output)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # 交互式测试示例
        company = input("请输入要研究的公司名称：").strip()
        if company:
            skill = CompanyResearchSkill(company, "standard")
            print(skill.run())