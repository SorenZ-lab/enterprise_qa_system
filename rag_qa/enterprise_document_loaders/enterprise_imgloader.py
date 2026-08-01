#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: enterprise_imgloader.py
作者: Zhi FANG
项目: 企业知识库问答
创建日期: 2026/7/15
描述: 
"""
# enterprise_document_loaders/enterprise_imgloader.py 源码
from typing import Iterator
# todo 使用从根目录开始的路径导入。
from rag_qa.enterprise_document_loaders.enterprise_ocr import get_ocr
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader


class OCRIMGLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, img_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            img_path: The path to the img to load.
        """
        self.img_path = img_path

    def lazy_load(self) -> Iterator[Document]:
        # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """

        line = self.img2text()
        yield Document(page_content=line, metadata={"source": self.img_path})

    def img2text(self):
        resp = ""
        ocr = get_ocr()
        result, _ = ocr(self.img_path)
        if result:
            ocr_result = [line[1] for line in result]
            resp += "\n".join(ocr_result)
        return resp


if __name__ == '__main__':
    img_loader = OCRIMGLoader(img_path='./data/test_img.png')
    doc = img_loader.load()
    print(doc)