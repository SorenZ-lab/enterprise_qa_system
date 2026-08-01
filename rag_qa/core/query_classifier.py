#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: query_classifier.py
作者: Zhi FANG
项目: 企业知识库问答
创建日期: 2026/7/16
描述: 
"""
# 导入标准库
import json
import os
# 导入 PyTorch
import torch
# 导入日志
import sys
# todo 导入日志
from base.logger import logger
# 导入numpy
import numpy as np
# 导入 Transformers 库
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
# 导入train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

current_dir = os.path.dirname(os.path.abspath(__file__))
rag_qa_path = os.path.abspath(os.path.dirname(os.path.abspath(current_dir)))
project_root = os.path.abspath(os.path.dirname(os.path.abspath(rag_qa_path)))
sys.path.insert(0, project_root)

class QueryClassifier:
    # todo 更改了模型位置
    # def __init__(self, model_path='models/bert_query_classifier'):
    def __init__(self, model_path='models\\bert_query_classifier'):

        # 加载bert
        # todo 预训练模型路径
        self.pre_trained_model_path = f'{rag_qa_path}\\models\\bert-base-chinese'
        # 模型训练以后保存的位置
        self.model_path = model_path
        logger.info(f"模型保存路径: {self.model_path}")
        print(f"加载 BERT 预训练模型: {self.pre_trained_model_path}")
        # 加载 BERT 分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.pre_trained_model_path)
        print(f"加载 BERT 分词器特殊token: {self.tokenizer.all_special_tokens}")
        print(f"加载 BERT 分词器特殊id: {self.tokenizer.all_special_ids}")
        # 加载 BERT 分词器特殊token: ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
        # 加载 BERT 分词器特殊id: [100, 102, 0, 101, 103]
        # sys.exit(1111)
        # 初始化模型
        self.model = None
        # 确定设备（GPU 或 CPU）
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu" )
        # 记录设备信息
        logger.info(f"使用设备: {self.device}")
        # 定义标签映射
        self.label_map = {"通用知识": 0, "企业咨询": 1}
        # 加载模型
        self.load_model()

    def load_model(self):
        # 检查模型路径是否存在
        print(f"模型路径查询: {self.model_path}")
        if os.path.exists(self.model_path):
            # 加载预训练模型
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            # 将模型移到指定设备
            self.model.to(self.device)
            # 记录加载成功的日志
            logger.info(f"加载模型: {self.model_path}")
        else:
            # 初始化新模型
            # todo 把模型位置设置好。
            self.model = BertForSequenceClassification.from_pretrained(self.pre_trained_model_path, num_labels=2)
            # 将模型移到指定设备
            self.model.to(self.device)
            # 记录初始化模型的日志
            logger.info("初始化新 BERT 模型")

    def save_model(self):
        """保存模型"""
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f"模型保存至: {self.model_path}")

    def preprocess_data(self, texts, labels):
        """预处理数据为 BERT 输入格式"""
        # truncation=True 表示将文本截断为指定长度，上限max_length
        # padding=True，表示填充文本，使其长度达到max_length
        # 张量类型为pt，pytorch
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        return encodings, [self.label_map[label] for label in labels]

    def create_dataset(self, encodings, labels):
        """创建 PyTorch 数据集"""

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        return Dataset(encodings, labels)

    def train_model(self, data_file="training_dataset_hybrid_5000.json"):
        """训练 BERT 分类模型"""
        # 加载数据集
        if not os.path.exists(data_file):
            logger.error(f"数据集文件 {data_file} 不存在")
            raise FileNotFoundError(f"数据集文件 {data_file} 不存在")

        # with open：读完就自动关闭
        # 按行读取readlines，并用json解析loads。
        with open(data_file, "r", encoding="utf-8") as f:
            data = [json.loads(value) for value in f.readlines()]

        texts = [item["query"] for item in data]
        labels = [item["label"] for item in data]

        # 数据划分，80%用于训练，20%用于验证，随机种子设为42。
        # 分别对应的是 x_train, x_test, y_train , y_test
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # 预处理
        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
        print(f'train_encodings--》{train_encodings[:5]}')
        print(f'train_labels--》{train_labels[:5]}')
        val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)

        # 创建数据集
        train_dataset = self.create_dataset(train_encodings, train_labels)
        print(f'train_dataset--》{train_dataset[0]}')
        # 把文本转成了token
        # train_dataset--》
        # {
        # 'input_ids':
        #       tensor([ 101, 3844, 6407, 6440, 4923,  833, 3136, 2595, 5543, 3844, 6407, 1408,
        #         8043,  102,  0, 0,  0, 0, 0,  0,  0,  0,  0,0,0,  0,  0,  0]),
        # 'token_type_ids':
        #           tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0]),
        # 'attention_mask':
        #           tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0]),
        #  'labels': tensor(1)
        #  }

        # input_ids：词对应的数字 id（文本本体），每个id对应一个tokenizer的词
        # token_type_ids：句子分段标记（区分上下句），当文本分类、语言理解任务中，句子分段标记通常为0
        # attention_mask：1 = 真实 token，0=padding 填充位，屏蔽无效 padding。填充1表明在实际反向传播时要计算梯度


        val_dataset = self.create_dataset(val_encodings, val_labels)
        #
        # 设置训练参数
        training_args = TrainingArguments(
            # 设置模型和检查点保存的目录路径
            output_dir="./bert_results",
            # 核心：设置训练的总轮数为3轮
            num_train_epochs=3,

            # 核心：学习率，有默认值5e-5
            #     learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
            # 设置每个设备（GPU/CPU）上的训练批次大小为8
            per_device_train_batch_size=8,
            # 设置每个设备（GPU/CPU）上的评估批次大小为8
            per_device_eval_batch_size=8,
            # 核心：设置学习率预热步数为500步，训练初期学习率从0逐渐增加到设定值
            # 出现损失函数一直波动太大，尝试调整下。总体步数 = 训练轮数 * （样本数/批次大小）
            # 预热步数一般占比10%左右。
            warmup_steps=500,
            # 设置权重衰减系数为0.01，用于防止过拟合
            weight_decay=0.01,
            # 设置日志文件保存的目录路径
            logging_dir="./bert_logs",
            # 设置每10个训练步骤记录一次日志
            logging_steps=10,
            # 设置评估策略为每个epoch结束后进行评估
            evaluation_strategy="epoch",
            # 设置模型保存策略为每个epoch结束后保存
            save_strategy="epoch",
            # 设置训练结束后加载最佳模型而非最后一个模型
            # 只加载评估效果好的模型。
            load_best_model_at_end=True,
            # 设置最多保存1个检查点文件，超出时自动删除旧的
            save_total_limit=1,
            # 设置用于判断最佳模型的指标为评估损失
            metric_for_best_model="eval_loss",
            # 禁用FP16混合精度训练，使用FP32精度，int8: -127~+127
            # 使用原精度训练，训练速度比较慢，精度更高。经验值大概损失1%精度。
            fp16=False,
        )

        # 初始化 Trainer
        trainer = Trainer(
            # 传入要训练的模型实例
            model=self.model,
            # 传入上面定义的训练参数配置
            args=training_args,
            # 传入训练数据集
            train_dataset=train_dataset,
            # 传入验证数据集，用于训练过程中评估模型性能
            eval_dataset=val_dataset,
            # 传入计算评估指标的函数，用于在验证集上计算准确率等指标
            compute_metrics=self.compute_metrics
        )
        # 训练模型
        logger.info("开始训练 BERT 模型...")
        trainer.train()
        self.save_model()

        # 评估模型
        self.evaluate_model(val_texts, val_labels)

    def compute_metrics(self, eval_pred):
        """计算评估指标，准确率"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    def evaluate_model(self, texts, labels):
        """评估模型性能"""
        # 仅对 texts 进行分词，labels 已为数字
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        dataset = self.create_dataset(encodings, labels)

        trainer = Trainer(model=self.model)
        predictions = trainer.predict(dataset)
        # 真实的模型预测的标签列表 + 真实的标签列表
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        true_labels = labels  # 直接使用数字标签

        logger.info("分类报告:")
        logger.info(classification_report(
            true_labels,
            pred_labels,
            target_names=["通用知识", "企业咨询"]
        ))
        logger.info("混淆矩阵:")
        logger.info(confusion_matrix(true_labels, pred_labels))

    def predict_category(self, query):
        # 检查模型是否加载
        if self.model is None:
            # 模型未加载，记录错误
            logger.error("模型未训练或加载")
            # 默认返回通用知识
            return "通用知识"
        # 对查询进行编码
        encoding = self.tokenizer(query, truncation=True, padding=True, max_length=128, return_tensors="pt")
        # 将编码移到指定设备
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        # 不计算梯度，进行预测
        with torch.no_grad():
            # 获取模型输出
            outputs = self.model(**encoding)
            print("模型预测输出：",outputs.logits)
            # 模型预测输出： tensor([[-3.7800,  3.9381]])
            # 模型预测结果： 1
            # 获取预测结果
            prediction = torch.argmax(outputs.logits, dim=1).item()
            print("模型预测结果：",prediction)
        # 根据预测结果返回类别
        return "企业咨询" if prediction == 1 else "通用知识"

if __name__ == "__main__":
    # 初始化分类器
    classifier = QueryClassifier(model_path="bert_query_classifier")

    # 训练模型
    # classifier.train_model(data_file='../classify_data/model_generic_5000.json')
    # 示例预测
    test_queries = [
        "公司产品的核心功能是什么",
        "公司产品如何收费？",
        "5*9等于多少？",
        "公司的请假流程是什么？"
    ]
    for query in test_queries:
        category = classifier.predict_category(query)
        print(f"查询: {query} -> 分类: {category}")
        # 模型预测输出： tensor([[-3.7800,  3.9381]])
        # 模型预测结果： 1
        # 查询: 公司产品的核心功能是什么 -> 分类: 企业咨询
        # 模型预测输出： tensor([[-3.6629,  4.1277]])
        # 模型预测结果： 1
        # 查询: 公司产品如何收费？ -> 分类: 企业咨询
        # 模型预测输出： tensor([[ 2.4184, -1.8165]])
        # 模型预测结果： 0
        # 查询: 5*9等于多少？ -> 分类: 通用知识
        # 模型预测输出： tensor([[-3.7361,  3.9717]])
        # 模型预测结果： 1
        # 查询: 公司的请假流程是什么？ -> 分类: 企业咨询