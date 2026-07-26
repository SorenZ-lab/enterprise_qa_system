#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: config.py
作者: Zhi FANG
项目: 企业知识库问答

 
"""
# 导入配置解析库
import configparser
# 导入路径操作库
import os
# 导入日志记录库
import logging


class Config1:
    # 初始化配置，加载 config.ini 文件
    def __init__(self, config_file='../config.ini'):
        # 创建配置解析器
        self.config = configparser.ConfigParser()
        # 读取配置文件
        self.config.read(config_file)
        # 整体有什么结构看一下。
        print(self.config.sections())
        # 查看全部的配置
        print(self.config._sections)
        # {
        #   'mysql':
        #       {'host': 'localhost',
        #       'user': 'root',
        #       'password': 'your_mysql_password',
        #       'database': 'enterprise_kg'},
        #   'redis':
        #       {'host': 'localhost',
        #       'port': '16379',
        #       'password': 'your_redis_password',
        #       'db': '0'},
        #   'logger':
        #       {'log_file': 'logs/app.log'},
        #   'chunk':
        #       {'chunk_size': '300'}
        # }
        # MySQL 配置
        # MySQL 主机地址
        self.MYSQL_HOST = self.config.get('mysql', 'host', fallback='localhost')
        # MySQL 用户名
        self.MYSQL_USER = self.config.get('mysql', 'user', fallback='root')
        # MySQL 密码
        self.MYSQL_PASSWORD = self.config.get('mysql', 'password', fallback='your_mysql_password')
        # MySQL 数据库名
        self.MYSQL_DATABASE = self.config.get('mysql', 'database', fallback='enterprise_kg')

        # Redis 配置
        # Redis 主机地址
        self.REDIS_HOST = self.config.get('redis', 'host', fallback='localhost')
        # Redis 端口
        self.REDIS_PORT = self.config.getint('redis', 'port', fallback=6379)
        # Redis 密码
        self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='your_redis_password')
        # Redis 数据库编号
        self.REDIS_DB = self.config.getint('redis', 'db', fallback=0)
        # 日志文件路径
        self.LOG_FILE = self.config.get('logger', 'log_file', fallback='logs/app.log')

        # 子块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('chunk', 'chunk_size', fallback=1000)



class Config:
    # 初始化配置，加载 config.ini 文件
    def __init__(self, config_file=None):
        # 创建配置解析器，启用插值功能
        # 差值功能可以让不同section之间进行引用，引用的方式为 ${section_name:key_name}
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        # 如果没有提供配置文件路径，则使用默认路径
        # 在实际开发中，你本地的路径和线上、测试的的代码路径都是不一样。
        print("当前文件路径为：",__file__)
        print("当前文件目录为：", os.path.dirname(__file__))
        print("当前文件目录的目录为：",os.path.dirname(os.path.dirname(__file__)))
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        print("日志目录为：", self.LOG_DIR)
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa\data')
        print("数据目录为：", self.DATA_DIR)
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa\models')
        self.ENTERPRISE_DOCUMENT_LOADERS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa\enterprise_document_loaders')

        if config_file is None:
            config_file = os.path.join(self.PROJECT_ROOT, 'config.ini')
        # 读取配置文件
        # todo 增加编码，避免中文乱码
        self.config.read(config_file, encoding='utf-8')

        # MySQL 配置
        # MySQL 主机地址
        self.MYSQL_HOST = os.getenv('MYSQL_HOST', self.config.get('mysql', 'host', fallback='localhost'))
        # MySQL 用户名
        self.MYSQL_USER = os.getenv('MYSQL_USER', self.config.get('mysql', 'user', fallback='root'))
        # MySQL 密码
        self.MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', self.config.get('mysql', 'password', fallback='your_mysql_password'))
        # MySQL 数据库名
        self.MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', self.config.get('mysql', 'database', fallback='enterprise_kg'))

        # Redis 配置
        # Redis 主机地址
        self.REDIS_HOST = os.getenv('REDIS_HOST', self.config.get('redis', 'host', fallback='localhost'))
        # Redis 端口
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', self.config.get('redis', 'port', fallback=6379)))
        # Redis 密码
        self.REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', self.config.get('redis', 'password', fallback='your_redis_password'))
        # Redis 数据库编号
        self.REDIS_DB = int(os.getenv('REDIS_DB', self.config.get('redis', 'db', fallback=0)))

        # Milvus 配置
        # Milvus 主机地址
        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='localhost'))
        # Milvus 端口
        self.MILVUS_PORT = os.getenv('MILVUS_PORT', self.config.get('milvus', 'port', fallback='19530'))
        # Milvus 数据库名
        self.MILVUS_DATABASE_NAME = os.getenv('MILVUS_DATABASE_NAME', self.config.get('milvus', 'database_name', fallback='enterprise'))
        # Milvus 集合名
        self.MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME', self.config.get('milvus', 'collection_name', fallback='enterprise_rag'))

        # LLM 配置
        # LLM 模型名
        self.LLM_MODEL = self.config.get('llm', 'model', fallback='qwen-plus')
        # DashScope API 密钥
        self.DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', self.config.get('llm', 'dashscope_api_key', fallback='your_dashscope_api_key'))
        # DashScope API 地址
        self.DASHSCOPE_BASE_URL = self.config.get('llm', 'dashscope_base_url',
                                                  fallback='https://dashscope.aliyuncs.com/compatible-mode/v1')

        # 检索参数
        # 父块大小
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # 子块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # 块重叠大小
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # 检索返回数量
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        # 最终候选数量
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        # 应用配置
        # 有效来源列表
        self.VALID_SOURCES = eval(
            self.config.get('app', 'valid_sources', fallback='["product", "tech", "hr", "admin"]'))
        # 客服电话
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='企业客服')

        # 日志文件路径
        self.LOG_FILE = os.path.join(self.LOG_DIR, 'app.log')

        # 路径配置


config = Config()

if __name__ == '__main__':
    conf = Config()
    print(conf.MYSQL_USER)