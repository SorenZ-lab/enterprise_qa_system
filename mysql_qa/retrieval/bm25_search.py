#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: bm25_search.py
作者: ZZS
项目: 3_代码
创建日期: 2026/7/13
描述: 
"""
# 导入 BM25 算法
from rank_bm25 import BM25Okapi
# 导入数值计算库
import numpy as np
# 导入文本预处理
from mysql_qa.utils.preprocess import preprocess_text
# 导入日志
from base.logger import logger

class BM25Search:
    def __init__(self, redis_client, mysql_client):
        # 初始化日志
        self.logger = logger
        # 初始化 Redis 客户端
        self.redis_client = redis_client
        # 初始化 MySQL 客户端
        self.mysql_client = mysql_client
        # 初始化 BM25 模型
        self.bm25 = None
        # 初始化问题列表
        self.questions = None
        # 初始化原始问题
        self.original_questions = None
        # 加载数据
        self._load_data()

    def _load_data(self):
        # 加载数据
        original_key = "qa_original_questions"
        tokenized_key = "qa_tokenized_questions"
        # 从 Redis 获取原始问题
        self.original_questions = self.redis_client.get_data(original_key)
        # 从 Redis 获取分词问题
        tokenized_questions = self.redis_client.get_data(tokenized_key)
        # 如果 Redis 中没有数据，从 MySQL 加载
        if not self.original_questions or not tokenized_questions:
            # 从 MySQL 获取问题，获取所有的问题
            self.original_questions = self.mysql_client.fetch_questions()
            if not self.original_questions:
                # 记录无问题警告
                self.logger.warning("未加载到问题")
                return
            # 分词问题
            tokenized_questions = [preprocess_text(q[0]) for q in self.original_questions]
            # 存储原始问题到 Redis
            self.redis_client.set_data(original_key, [(q[0]) for q in self.original_questions])
            # 存储分词问题到 Redis
            self.redis_client.set_data(tokenized_key, tokenized_questions)
        # 设置问题列表
        self.questions = tokenized_questions
        # 初始化 BM25 模型
        self.bm25 = BM25Okapi(self.questions)
        # 记录 BM25 初始化成功
        self.logger.info("BM25 模型初始化完成")

    def _softmax(self, scores):
        # 计算 Softmax 分数
        exp_scores = np.exp(scores - np.max(scores))
        # 返回归一化分数
        return exp_scores / exp_scores.sum()

    def search(self, query, threshold=0.85):
        # 搜索查询
        # 如果查询的不是字符或者为空，直接返回None，False代表我们不要进行RAG检索,True代表要进行RAG检索。
        if not query or not isinstance(query, str):
            # 记录无效查询
            self.logger.error("无效查询")
            # 返回 None 和 False
            return None, False

        # 检查 Redis 缓存
        cached_answer = self.redis_client.get_answer(query)
        if cached_answer:
            # 返回缓存答案
            return cached_answer, False
        try:
            # 分词查询
            query_tokens = preprocess_text(query)
            logger.info(f'原始查询:{query}')
            logger.info(f'分词后的查询:{query_tokens}')
            # 计算 BM25 分数
            # scores 是一个列表，列表中的元素是这个查询文档对每个文档的得分。[34, 10, 0.4,180]
            scores = self.bm25.get_scores(query_tokens)
            logger.info(f'BM25得分:{scores[:10]}')
            logger.info(f'BM25得分列表长度:{len(scores)}')
            # 计算 Softmax 分数
            # softmax_scores 是一个列表，列表中的元素 是所有得分汇总后的概率分布，样式是 [0.1, 0.2, 0.3, 0.4]，加和是1
            softmax_scores = self._softmax(scores)
            logger.info(f'softmax_scores:{softmax_scores[:10]}')
            logger.info(f'softmax_scores列表长度:{len(softmax_scores)}')
            # 获取最高分索引
            best_idx = softmax_scores.argmax()
            logger.info(f'best_idx:{best_idx}')
            # 获取最高分
            best_score = softmax_scores[best_idx]
            logger.info(f'best_score:{best_score}')
            # 检查是否超过阈值
            if best_score >= threshold:
                # 获取原始问题
                original_question = self.original_questions[best_idx]
                logger.info(f'原始问题:{original_question}')
                # 获取答案
                # 第一次应该什么都没有，返回None，因为没有缓存过。
                redis_result = self.redis_client.get_answer(original_question)
                logger.info(f'从redis中获取到的答案:{redis_result}')
                # 获取答案
                if redis_result:
                    answer = redis_result
                    logger.info(f'从redis中获取到了问题的答案:{answer}')
                else:
                    # 如果redis中没有这个问题的答案，那么就从mysql中获取
                    answer = self.mysql_client.fetch_answer(original_question)
                    logger.info(f'从mysql中获取到了问题的答案:{answer}')

                if answer:
                    # 缓存答案
                    # 缓存的是 key用户的问题 + value和用户问题相似的问题对应的答案。
                    self.redis_client.set_data(f"answer:{query}", answer)
                    # 记录搜索成功
                    self.logger.info(f"搜索成功，Softmax 相似度: {best_score:.3f}")
                    # 返回答案和 False
                    return answer, False
            # 记录无可靠答案
            self.logger.info(f"未找到可靠答案，最高 Softmax 相似度: {best_score:.3f}")
            # 返回 None 和 True，True代表我们进行RAG检索
            return None, True
        except Exception as e:
            # 记录搜索失败
            self.logger.error(f"搜索失败: {e}")
            # 返回 None 和 True
            return None, True

if __name__ == '__main__':
    from mysql_qa.cache.redis_client import RedisClient
    from mysql_qa.db.mysql_client import MySQLClient
    redis_client = RedisClient()
    mysql_client = MySQLClient()
    bm25_search = BM25Search(redis_client, mysql_client)
    bm25_search.search("我精通dify能不能月薪10万？", threshold=0.22)
