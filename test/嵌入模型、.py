#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: 嵌入模型、.py
作者: ZZS
项目: 3_代码
创建日期: 2026/7/14
描述: 
"""
# import dashscope
# from http import HTTPStatus
input_texts = "衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买"
from openai import OpenAI  # 使用 OpenAI 接口
from langchain_openai import ChatOpenAI

api_key = "sk-请填写你的DashScope密钥"

# client = OpenAI(api_key="https://dashscope.aliyuncs.com/compatible-mode/v1",
#                         base_url=api_key)

# 初始化客户端，替换base_url为dashscope兼容地址
client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


res = client.embeddings.create(model="text-embedding-v4", input=["测试文本"])
print(res.data[0].embedding)