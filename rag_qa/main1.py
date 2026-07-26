#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: main1.py.py
作者: ZZS
项目: 3_代码
创建日期: 2026/7/15
描述: 
"""
# run.py （项目入口，永远运行这个，不直接跑main.py）
import ssl
import certifi

# 全局覆盖ssl创建逻辑，彻底跳过Windows证书仓库
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# 再导入所有业务代码
from rag_qa.main import *
main()
