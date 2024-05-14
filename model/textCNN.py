import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers.models.auto import AutoTokenizer, AutoModel


class MyConfig(object):
    def __init__(self, task):
        """

        :param task:任务名
        """
        bert = "bert-base-chinese"
        robert = "hfl/chinese-roberta-wwm-ext"
        longformer = 'schen/longformer-chinese-base-4096'

        self.task_name = task
        self.model_name = longformer  # 模型名称
        self.model_path = './pretrain/longformer/'
        self.data_path = './dataset/data.csv'
        self.epochs = 5  # 训练周期
        self.max_length = 4096  # 句子最大长度
        self.vocab_size = len(open(self.model_path + 'vocab.txt', 'r', encoding='utf-8').readlines())  # 语料库大小
        self.embedding_dim = 128
        self.hidden_size = 128
        self.in_channels = 1  # 输入通道数，因为是文本，所以为1
        self.n_filter = 3  # 卷积核个数
        self.kernel_size = (20, self.embedding_dim)  # 卷积核尺寸

        self.batch_size = 1  # 批处理个数
        self.lr = 2e-5  # 学习率
        self.dropout_rate = 0.1
        self.class_list = [i.replace("\n", "") for i in
                           open("./dataset/class.txt", encoding='utf-8').readlines()]  # 类别名称
        self.feature_name = "权利要求书"  # 特征名
        self.device = "cuda:0"  # 训练设备
        self.save_path = "evaluation/textCNN.ckpt"  # 模型结果
        self.require_improvement = 200000  # 若超过多少batch效果还没提升，则提前结束训练


class MyDataset(Dataset):
    def __init__(self, config, data):
        super(MyDataset, self).__init__()
        self.config = config
        self.claim1 = data['claim_1']
        self.claim2 = data['claim_2']
        self.label = data['label']

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        claim1 = self.claim1.loc[idx]
        claim2 = self.claim2.loc[idx]
        label = self.label.loc[idx]
        return {
            'claim1': claim1, 'claim2': claim2, 'label': label
        }


class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        # Conv层：in_channels 通道数，out_channels 卷积核个数，kernel_size 卷积核尺寸
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=config.n_filter, kernel_size=config.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.classifier = nn.Linear(6114, 2)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.parameters(), lr=config.lr)

    def forward(self, batch):
        # 获取数据
        claim1, claim2, label = batch['claim1'], batch['claim2'], batch['label']
        encoding1 = self.tokenizer(
            claim1,
            padding='max_length',  # Pad to the maximum sequence length
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt',  # Return PyTorch tensors
        ).to(self.config.device)

        encoding2 = self.tokenizer(
            claim2,
            padding='max_length',  # Pad to the maximum sequence length
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt',  # Return PyTorch tensors
        ).to(self.config.device)

        # textCNN 模型提取语义特征
        x1 = self.embedding(encoding1['input_ids'])
        x2 = self.embedding(encoding2['input_ids'])

        x1, x2 = x1.unsqueeze(1), x2.unsqueeze(1)  # 添加通道数 [batch_size, channels, seq_length, embedding_dim]

        x1 = self.conv1(x1)
        x2 = self.conv1(x2)

        x1 = x1.flatten(start_dim=1)  # 进行flatten操作
        x2 = x2.flatten(start_dim=1)  # 进行flatten操作

        # 分类层
        output = F.softmax(self.classifier(x1 - x2), dim=1).cpu()
        loss = self.criterion(output, label)
        return output, loss
