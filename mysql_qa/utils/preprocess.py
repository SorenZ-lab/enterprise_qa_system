#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: preprocess.py
作者: Zhi FANG
项目: 企业知识库问答
创建日期: 2026/7/13
描述: 
"""
# 导入分词库
import jieba
# 导入日志
from base.logger import logger

def preprocess_text(text):
    # 预处理文本
    logger.info("开始预处理文本")
    try:
        # todo 以下是测试代码
        # print(jieba.lcut(text))
        # print(type(jieba.lcut(text)))
        # ['我', '是', '精通', 'dify', '，', '大', '模型', '的', 'yes', '工程师', '！']
        # <class 'list'>
        # print("*" * 100)
        # print(jieba.cut(text))
        # print(type(jieba.cut(text)))
        # print("*" * 100)
        # lcut_for_search 比cut更适合检索
        # print(jieba.lcut_for_search(text))
        # print(type(jieba.lcut_for_search(text)))
        # ['我', '是', '精通', 'dify', '，', '大', '模型', '的', 'yes', '工程', '工程师', '！']
        # <class 'list'>
        # print("*"*100)
        # todo 以上是测试代码
        # 分词并转换为小写
        # 为啥转转小写，将大小写统一
        return jieba.lcut(text.lower())
    except AttributeError as e:
        # 记录预处理失败
        logger.error(f"文本预处理失败: {e}")
        # 返回空列表
        return []


if __name__ == '__main__':
    print(preprocess_text("我是精通dify，大模型的yes工程师！"))
