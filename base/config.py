#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: config.py
作者: Zhi FANG
项目: 企业知识库问答
描述: 集中管理项目配置，从 config.ini 与环境变量加载。
"""
import ast
import configparser
import os


class Config:
    """加载 config.ini 与环境变量，提供统一的配置访问入口。"""

    def __init__(self, config_file=None):
        # 启用插值功能：不同 section 之间可用 ${section:key} 互相引用
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa', 'data')
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa', 'models')
        self.ENTERPRISE_DOCUMENT_LOADERS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa', 'enterprise_document_loaders')

        if config_file is None:
            config_file = os.path.join(self.PROJECT_ROOT, 'config.ini')
        self.config.read(config_file, encoding='utf-8')

        # MySQL 配置
        self.MYSQL_HOST = os.getenv('MYSQL_HOST', self.config.get('mysql', 'host', fallback='localhost'))
        self.MYSQL_PORT = int(os.getenv('MYSQL_PORT', self.config.get('mysql', 'port', fallback=3306)))
        self.MYSQL_USER = os.getenv('MYSQL_USER', self.config.get('mysql', 'user', fallback='root'))
        self.MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', self.config.get('mysql', 'password', fallback='your_mysql_password'))
        self.MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', self.config.get('mysql', 'database', fallback='enterprise_kg'))

        # Redis 配置
        self.REDIS_HOST = os.getenv('REDIS_HOST', self.config.get('redis', 'host', fallback='localhost'))
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', self.config.get('redis', 'port', fallback=6379)))
        self.REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', self.config.get('redis', 'password', fallback='your_redis_password'))
        self.REDIS_DB = int(os.getenv('REDIS_DB', self.config.get('redis', 'db', fallback=0)))

        # Milvus 配置
        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='localhost'))
        self.MILVUS_PORT = os.getenv('MILVUS_PORT', self.config.get('milvus', 'port', fallback='19530'))
        self.MILVUS_DATABASE_NAME = os.getenv('MILVUS_DATABASE_NAME', self.config.get('milvus', 'database_name', fallback='enterprise'))
        self.MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME', self.config.get('milvus', 'collection_name', fallback='enterprise_rag'))

        # LLM 配置
        self.LLM_MODEL = self.config.get('llm', 'model', fallback='qwen-plus')
        self.DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', self.config.get('llm', 'dashscope_api_key', fallback='your_dashscope_api_key'))
        self.DASHSCOPE_BASE_URL = self.config.get('llm', 'dashscope_base_url', fallback='https://dashscope.aliyuncs.com/compatible-mode/v1')

        # 检索参数
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        # 应用配置
        self.VALID_SOURCES = ast.literal_eval(
            self.config.get('app', 'valid_sources', fallback='["product", "tech", "hr", "admin"]'))
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='企业客服')

        # 日志文件路径
        self.LOG_FILE = os.path.join(self.LOG_DIR, 'app.log')


config = Config()


if __name__ == '__main__':
    conf = Config()
    print(conf.MYSQL_USER)
