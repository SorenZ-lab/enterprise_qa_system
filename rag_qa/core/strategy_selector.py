#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: strategy_selector.py
作者: Zhi FANG
项目: 企业知识库问答
描述: 
"""

# 导入 LangChain 提示模板
from langchain_core.prompts import PromptTemplate
# 导入日志和配置
# todo 导入基础配置
from base.config import config
from base.logger import logger
# 导入 OpenAI
from openai import OpenAI



class StrategySelector:

    def __init__(self):
        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=config.DASHSCOPE_API_KEY,
                             base_url=config.DASHSCOPE_BASE_URL)
        # 获取策略选择提示模板
        self.strategy_prompt_template = self._get_strategy_prompt()

    def call_dashscope(self, prompt):
        # 调用 DashScope API
        try:
            # 创建聊天完成请求
            completion = self.client.chat.completions.create(
                # todo 更改config名称
                # model=Config().LLM_MODEL,
                model = config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手，能够根据用户输入的Prompt严格执行并返回可靠的结果"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1
            )
            # 返回完成结果
            print(completion)
            # ChatCompletion(id='chatcmpl-23d98629-59d6-9d3a-8601-1aa49095a893',
            # choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='假设问题检索', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None))], created=1784174617, model='qwen-plus', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=3, prompt_tokens=538, total_tokens=541, completion_tokens_details=None, prompt_tokens_details=PromptTokensDetails(audio_tokens=None, cached_tokens=0)))
            print(completion.choices[0])
            print("最终的输出：",completion.choices[0].message.content)
            return completion.choices[0].message.content if completion.choices else "直接检索"
        except Exception as e:
            # 记录 API 调用失败
            logger.error(f"DashScope API 调用失败: {e}")
            # 默认返回直接检索
            return "直接检索"


    def _get_strategy_prompt(self):
        #   定义私有方法，获取策略选择 Prompt 模板
        return PromptTemplate(
            template="""
            你是一个智能助手，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

            以下是几种检索增强策略及其适用场景：

            1.  **直接检索：**
                * 描述：对用户查询直接进行检索，不进行任何增强处理。
                * 适用场景：适用于查询意图明确，需要从知识库中检索**特定信息**的问题，例如：
                    * 示例：
                        * 查询：公司产品如何收费？
                        * 策略：直接检索
                    * 查询：公司的技术架构是什么？
                        * 策略：直接检索
            2.  **假设问题检索（HyDE）：**
                * 描述：使用 LLM 生成一个假设的答案，然后基于假设答案进行检索。
                * 适用场景：适用于查询较为抽象，直接检索效果不佳的问题，例如：
                    * 示例：
                        * 查询：如何提升企业的运营效率？
                        * 策略：假设问题检索
            3.  **子查询检索：**
                * 描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                * 适用场景：适用于查询涉及多个实体或方面，需要分别检索不同信息的问题，例如：
                    * 示例：
                        * 查询：比较 Milvus 和 Zilliz Cloud 的优缺点。
                        * 策略：子查询检索
            4.  **回溯问题检索：**
                * 描述：将复杂的用户查询转化为更基础、更易于检索的问题，然后进行检索。
                * 适用场景：适用于查询较为复杂，需要简化后才能有效检索的问题，例如：
                    * 示例：
                        * 查询：我有一个包含 100 亿条记录的数据集，想把它存储到 Milvus 中进行查询。可以吗？
                        * 策略：回溯问题检索

            根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。
            只返回 直接检索，假设问题检索，子查询检索，回溯问题检索 这四种策略中的一种，【**不要带任何上下文**】！！！
            """
            ,
            input_variables=["query"],
        )

    #   定义方法，选择检索策略
    def select_strategy(self, query):
        #   调用 LLM 获取检索策略
        strategy = self.call_dashscope(self.strategy_prompt_template.format(query=query)).strip()
        logger.info(f"为查询 '{query}' 选择的检索策略：{strategy}")
        return strategy

if __name__ == '__main__':
    ss = StrategySelector()
    # ss.select_strategy('你好吗')
    ss.select_strategy('公司产品如何收费？')