#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: ragas_evaluate.py
作者: Zhi FANG
项目: 企业知识库问答
描述: 
"""

# 导入pandas库，用于数据处理和保存CSV文件
import pandas as pd
# 导入ragas库的evaluate函数，用于执行RAG评估
from ragas import evaluate
# 导入ragas的评估指标，包括忠实度、答案相关性、上下文相关性和上下文召回率
# todo 1:
#  1 如果你不想看 context_relevancy，换成 context_precision
#  2 如果你想看 context_relevancy，可以将ragas版本，从现在0.2.6 升级到 0.3.0
#  3 升级指令：切换到当前的环境下，然后执行 pip install ragas==0.3.0
from ragas.metrics import (
    faithfulness,       # 忠实度
    answer_relevancy,   # 答案相关性
    context_precision,  # 上下文精确度
    # context_relevancy,  # 上下文相关性
    ContextRelevance,   # 上下文相关性 ，如果你切换成了0.3.0版本，就使用 ContextRelevance 。
    context_recall      # 上下文召回率
)
# 导入datasets库的Dataset类，用于构建RAGAS所需的数据格式
from datasets import Dataset
# 导入langchain_openai的嵌入模型和聊天模型，用于评估时的语义计算和推理
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# 导入json库，用于加载JSON格式的评估数据集
import json
from base.config import config

# 1. 加载生成的数据集
# 使用with语句打开JSON文件，确保文件正确关闭，指定编码为utf-8
with open("rag_evaluate_data_small.json", "r", encoding="utf-8") as f:
    # 将JSON文件内容加载到data变量中，data为包含多个数据条目的列表
    data = json.load(f)

# 2. 转换为RAGAS格式
# 创建字典eval_data，将JSON数据转换为RAGAS要求的字段格式
eval_data = {
    # 提取每个数据条目的question字段，组成问题列表
    "question": [item["question"] for item in data],
    # 提取每个数据条目的answer字段，组成答案列表
    "answer": [item["answer"] for item in data],
    # 提取每个数据条目的context字段，组成上下文列表（每个context为列表）
    "contexts": [item["context"] for item in data],
    # 提取每个数据条目的ground_truth字段，组成真实答案列表
    "ground_truth": [item["ground_truth"] for item in data]
}
# 使用Dataset.from_dict将字典转换为RAGAS所需的Dataset对象
dataset = Dataset.from_dict(eval_data)

# 3. 配置RAGAS评估环境
# todo 2 初始化大模型和嵌入模型，使用本地embedding bme_m3也可以。milvus包下的嵌入模型报错：没有embed_query这个方法。
# 初始化ChatOpenAI模型，指定使用gpt-4模型，并设置OpenAI API密钥
# llm = ChatOpenAI(model="gpt-4", openai_api_key="your_openai_api_key")
llm = ChatOpenAI(api_key=config.DASHSCOPE_API_KEY,
                 base_url=config.DASHSCOPE_BASE_URL,
                 model_name = config.LLM_MODEL,
                 temperature=0.0
                 )

# 初始化OpenAI嵌入模型，用于计算语义相似度，设置API密钥
# embeddings = OpenAIEmbeddings(openai_api_key="your_openai_api_key")
from langchain_ollama import OllamaEmbeddings
# 初始化Ollama嵌入模型，使用mxbai-embed-large模型，温度设置为0
embeddings = OllamaEmbeddings(model="mxbai-embed-large", temperature=0)


# 4. 执行评估
# 调用evaluate函数，传入数据集、评估指标、LLM模型和嵌入模型
# todo 3 context_relevancy 改成 ContextRelevance，也可以将 context_precision 加进来。
#  ContextRelevance方法是非实例化的，使用时需要加括号。
result = evaluate(
    # 传入转换好的Dataset对象
    dataset=dataset,
    # 指定使用的评估指标列表
    metrics=[
        faithfulness,  # 忠实度：答案是否基于上下文
        answer_relevancy,  # 答案相关性：答案与问题的匹配度
        context_precision,  # 上下文精确度：上下文是否包含所有必要信息
        # context_relevancy,  # 上下文相关性：上下文是否仅包含相关信息
        ContextRelevance(),  # 上下文相关性：上下文是否仅包含相关信息
        context_recall  # 上下文召回率：上下文是否包含所有必要信息
    ],
    # 传入配置好的LLM模型
    llm=llm,
    # 传入配置好的嵌入模型
    embeddings=embeddings
)

# 5. 输出并保存结果
# 打印评估结果标题
print("RAGAS评估结果：")
# 打印评估结果，包含各指标的分数
print(result)
# 将评估结果转换为pandas DataFrame，便于保存
result_df = pd.DataFrame([result])
# 将DataFrame保存为CSV文件，文件名为ragas_evaluation_results.csv，不保存索引
result_df.to_csv("ragas_evaluation_results.csv", index=False)

# todo 4 详细的评估指标
print("详细评估指标：")
result = result.to_pandas()
result.to_excel("ragas_evaluation_results.xlsx", index=False)
