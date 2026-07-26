#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: document_processor.py
作者: ZZS
项目: 3_代码
创建日期: 2026/7/15
描述: 
"""
import os
import sys

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownTextSplitter
from datetime import datetime
# todo 1. 可以使用下面的 文件完整目录的导入方式，也可以使用 在目录下加一个init方法，然后导入。
from rag_qa.enterprise_text_spliter import ChineseRecursiveTextSplitter
from rag_qa.enterprise_text_spliter.enterprise_chinese_recursive_text_splitter import ChineseRecursiveTextSplitter

from rag_qa.enterprise_text_spliter import AliTextSplitter
from rag_qa.enterprise_document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader
from base.config import config
from base.logger import logger

# todo 2. 因为在配置文件中定义了conf，所以不需要实例化。
# conf = Config()
conf = config

# 定义支持的文件类型及其对应的加载器字典
document_loaders = {
    # 文本文件使用 TextLoader
    ".txt": TextLoader,
    # PDF 文件使用 OCRPDFLoader
    ".pdf": OCRPDFLoader,
    # Word 文件使用 OCRDOCLoader
    ".docx": OCRDOCLoader,
    # PPT 文件使用 OCRPPTLoader
    ".ppt": OCRPPTLoader,
    # PPTX 文件使用 OCRPPTLoader
    ".pptx": OCRPPTLoader,
    # JPG 文件使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # PNG 文件使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 文件使用 UnstructuredMarkdownLoader
    ".md": UnstructuredMarkdownLoader
}

# 定义函数，从指定文件夹加载多种类型文件并添加元数据
def load_documents_from_directory(directory_path):
    # 初始化空列表，用于存储加载的文档
    documents = []
    # 获取支持的文件扩展名集合
    # 拿到的就是[.txt, .pdf , .ppt,,,,]
    supported_extensions = document_loaders.keys()
    print("文档加载的类型：",supported_extensions)

    # 从目录名提取知识分类（如 "tech_data" -> "tech"）
    # 拿到的就是tech
    source = os.path.basename(directory_path).replace("_data", "")
    print("当前处理的知识分类：",source)


    # 遍历指定目录及其子目录
    for root, _, files in os.walk(directory_path):
        print("遍历信息：",root, _, files)
        # 遍历当前目录下的所有文件
        for file in files:
            # 构造文件的完整路径
            file_path = os.path.join(root, file)
            # 获取文件扩展名并转换为小写
            file_extension = os.path.splitext(file_path)[1].lower()
            # 检查文件类型是否在支持的扩展名列表中
            if file_extension in supported_extensions:
                # 使用 try-except 捕获加载过程中的异常
                try:
                    # 根据文件扩展名获取对应的加载器类  比如拿到 .txt 文件，document_loaders[".txt"]拿到 TextLoader 类
                    loader_class = document_loaders[file_extension]
                    # 实例化加载器对象，传入文件路径
                    if file_extension == ".txt":
                        loader = loader_class(file_path, encoding="utf-8")
                    else:
                        loader = loader_class(file_path)


                    # 调用加载器加载文档内容，返回文档列表
                    loaded_docs = loader.load()
                    # print("加载的文档内容：",loaded_docs)

                    print("文档数量：",len(loaded_docs))

                    # 遍历加载的每个文档
                    for doc in loaded_docs:
                        print(type( doc))
                        # print("文档读取内容：",doc)
                        # 格式：<class 'langchain_core.documents.base.Document'>
                        # page_content='LLM背景知识介绍
                        # 学习⽬标
                        # 了解LLM背景的知识
                        # 掌握什么是语⾔模型
                        # ，，，，，
                        # P(wW...WNPP(S)
                        # ∑log(P(ui))
                        # '
                        # metadata={'source': 'rag_qa/data/ai_data\\LLM基础知识.pdf'}

                        # 为文档添加知识分类元数据
                        doc.metadata["source"] = source
                        # 为文档添加文件路径元数据
                        doc.metadata["file_path"] = file_path
                        # 为文档添加当前时间戳元数据  isoformat() 函数将时间戳转换为 ISO 格式字符串 YYYY-MM-DD HH:MM:SS.mmmmmm
                        doc.metadata["timestamp"] = datetime.now().isoformat()

                        # print("文档处理之后加上元数据内容：", doc)
                        # page_content='LLM背景知识介绍
                        # 学习⽬标
                        # ∑log(P(ui))
                        # '
                        # metadata={
                        #   'source': 'ai',
                        #   'file_path': 'rag_qa/data/ai_data\\产品介绍.docx',
                        #   'timestamp': '2026-07-15T14:30:15.564885'
                        #   }
                    # 将加载的文档添加到总列表中
                    documents.extend(loaded_docs)
                    # 记录成功加载文件的日志
                    logger.info(f"成功加载文件: {file_path}")
                # 捕获加载过程中可能出现的异常
                except Exception as e:
                    # 记录加载失败的日志，包含错误信息
                    logger.error(f"加载文件 {file_path} 失败: {str(e)}")
            # 如果文件类型不在支持列表中
            else:
                # 记录警告日志，提示不支持的文件类型
                logger.warning(f"不支持的文件类型: {file_path}")
    # 返回加载的所有文档列表
    return documents

# 定义函数，处理文档并进行分层切分，返回子块结果
# 父子块的切分。
def process_documents(directory_path,
                      parent_chunk_size=conf.PARENT_CHUNK_SIZE,
                      child_chunk_size=conf.CHILD_CHUNK_SIZE,
                      chunk_overlap=conf.CHUNK_OVERLAP
                      ):
    # 从指定目录加载所有文档
    # todo 加载文档
    documents = load_documents_from_directory(directory_path)
    # sys.exit(111111)

    # 记录加载的文档总数日志
    logger.info(f"加载的文档数量: {len(documents)}")

    # 初始化父块和子块分词器（通用）
    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    # 初始化 Markdown 专用分词器
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    # 初始化空列表，用于存储所有子块
    child_chunks = []
    # 遍历每个原始文档，带上索引 i
    for i, doc in enumerate(documents):
        # print(doc)
        # 获取文件扩展名
        file_extension = os.path.splitext(doc.metadata.get("file_path", ""))[1].lower()

        # 选择切分器
        is_markdown = (file_extension == ".md")
        parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
        # print(f'parent_splitter_to_use-->{parent_splitter_to_use}')
        child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter
        logger.info(f"处理文档: {doc.metadata['file_path']}, 使用切分器: {'Markdown' if is_markdown else 'ChineseRecursive'}")

        # 使用父块分词器将文档切分为父块
        parent_docs = parent_splitter_to_use.split_documents([doc])
        print("父块内容：",parent_docs)
        print("父块数量：",len(parent_docs))

        #  [
        #  Document(
        #       metadata={
        #           'source': 'ai',
        #           'file_path': 'rag_qa/data/ai_data\\LLM基础知识.pdf',
        #           'timestamp': '2026-07-15T15:16:24.406352'
        #           },
        #       page_content='LLM背景知识介绍\n \n学习⾔模型型代表是BA参数规模步⼊千万'
        #      ),
        #   Document(
        #      metadata={
        #           'source': 'ai',
        #           'file_path': 'rag_qa/data/ai_data\\LLM基础知识.pdf',
        #           'timestamp': '2026-07-15T15:16:24.406352'
        #           },
        #     page_content='Evolutionary\nGLM\nO够计算出'
        #     ),
        #      ，，，，
        #   ]

        # 遍历每个父块，带上索引 j
        for j, parent_doc in enumerate(parent_docs):
            # 为父块生成唯一 ID，格式为 "doc_i_parent_j"
            parent_id = f"doc_{i}_parent_{j}"
            # 将父块 ID 添加到元数据
            parent_doc.metadata["parent_id"] = parent_id
            # 将父块内容存储到元数据
            parent_doc.metadata["parent_content"] = parent_doc.page_content

            # 使用子块分词器将父块切分为子块
            sub_chunks = child_splitter_to_use.split_documents([parent_doc])
            print("子块内容：",sub_chunks)
            print("子块数量：",len(sub_chunks))
            # 子块内容：
            # [
            #      Document(
            #         metadata={
            #             'source': 'ai',
            #             'file_path': 'rag_qa/data/ai_data\\LLM基础知识.pdf',
            #             'timestamp': '2026-07-15T15:25:13.624969',
            #             'parent_id': 'doc_0_parent_0',
            #             'parent_content': 'LLM背景知识介绍\n \n语⾔任nt）时代，模型参数规模步⼊千万'
            #             },
            #         page_content='LLM背景知识介绍\n \⽌23年，语⾔模型发展⾛过了三个阶段：'
            #         ),
            #        Document(,,,,)
            # ]
            # 子块数量:2


            # 遍历每个子块，带上索引 k
            for k, sub_chunk in enumerate(sub_chunks):
                # 为子块添加父块 ID 到元数据
                sub_chunk.metadata["parent_id"] = parent_id
                # 为子块添加父块内容到元数据
                sub_chunk.metadata["parent_content"] = parent_doc.page_content
                # 为子块生成唯一 ID，格式为 "parent_id_child_k"
                sub_chunk.metadata["id"] = f"{parent_id}_child_{k}"
                # 将子块添加到子块列表中
                child_chunks.append(sub_chunk)
                print("处理后的子块内容：",sub_chunk)
                # page_content='LLM背景知识介绍学习⽬标,了解LLM背景的知识,,,⾔模型发展⾛过了三个阶段：'
                # metadata={
                # 	'source': 'ai',
                # 	'file_path': 'rag_qa/data/ai_data\\LLM基础知识.pdf',
                # 	'timestamp': '2026-07-15T15:32:55.158096',
                # 	'parent_id': 'doc_0_parent_0',
                # 	'parent_content': 'LLM背景知识介绍\n \n学习⽬标\n \n了解,,,,,和训练语料规模，探索不同类型的参数规模步⼊千万',
                # 	'id': 'doc_0_parent_0_child_0'
                # }
                # id: 第几个文档，父块第几个，子块第几个。


                # break

            # break

        # break

    # 记录子块总数日志
    logger.info(f"子块数量: {len(child_chunks)}")
    # 返回所有子块列表
    return child_chunks


if __name__ == '__main__':

    chunks = process_documents(
        r'rag_qa/data/ai_data',
        conf.PARENT_CHUNK_SIZE,
        conf.CHILD_CHUNK_SIZE,
        conf.CHUNK_OVERLAP,  # 5% ~ 20%，和token有关
    )
    print(len(chunks))
    # for chunk in chunks:
    #     print(chunk)