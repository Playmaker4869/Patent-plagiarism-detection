import torch
import torch.nn as nn
import re
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers.models.auto import AutoTokenizer, AutoModel
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


class MyConfig(object):
    def __init__(self, task):
        """

        :param task:任务名
        """
        bert = "bert-base-chinese"
        robert = "hfl/chinese-roberta-wwm-ext"

        self.task_name = task
        self.model_name = bert  # 模型名称
        self.model_path = './pretrain/bert/'
        self.data_path = './dataset/data.csv'
        self.epochs = 5  # 训练周期
        self.max_length = 512  # 句子最大长度
        self.vocab_size = len(open(self.model_path + 'vocab.txt', 'r', encoding='utf-8').readlines())  # 语料库大小
        self.embedding_dim = 768
        self.hidden_size = 768

        # GAT参数
        self.in_channels = 768
        self.out_channels = 768
        self.gat_num_layers = 3  # 图注意力网络的层数
        self.gat_num_heads = 4
        self.num_classes = 2

        self.batch_size = 1  # 批处理个数
        self.lr = 2e-5  # 学习率
        self.dropout_rate = 0.1
        self.class_list = [i.replace("\n", "") for i in
                           open("./dataset/class.txt", encoding='utf-8').readlines()]  # 类别名称
        self.feature_name = "权利要求书"  # 特征名
        self.device = "cuda:0"  # 训练设备
        self.save_path = "evaluation/test.ckpt"  # 模型结果
        self.require_improvement = 200000  # 若超过多少batch效果还没提升，则提前结束训练


# class MyDataset(Dataset):
#     def __init__(self, config, data):
#         super(MyDataset, self).__init__()
#         self.config = config
#         self.claim1 = data['claim_1']
#         self.claim2 = data['claim_2']
#         self.label = data['label']
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, idx):
#         claim1 = self.claim1.loc[idx]
#         claim2 = self.claim2.loc[idx]
#         edge1 = get_x_edge(claim1)
#         edge2 = get_x_edge(claim2)
#         label = self.label.loc[idx]
#         return {
#             'claim1': claim1, 'claim2': claim2,
#             'edge1': edge1, 'edge2': edge2, 'label': label
#         }

class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        # if key == 'x1':
        #     return self.x1.size(0)
        # if key == 'x2':
        #     return self.x2.size(0)
        if key == 'edge1':
            return self.x1.size(0)
        if key == 'edge2':
            return self.x2.size(0)
        # if key == 'attn_mask1':
        #     return self.x1.size(0)
        # if key == 'attn_mask2':
        #     return self.x2.size(0)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'edge1' in key or 'edge2' in key:
            return 1
        else:
            return 0


def get_data_list(config, data):
    def is_valid_edge_index(edge_index):
        # 判断邻接矩阵是否合法
        if True in (edge_index[0] >= edge_index[1]):
            return False
        else:
            return True

    pattern = r'[1-9][0-9]?[\.|、]'
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    data_list = []
    for i in range(len(data)):
        claim1 = data.loc[i, 'claim_1']
        claim2 = data.loc[i, 'claim_2']
        x1 = re.split(pattern, str(claim1))[1:]
        x2 = re.split(pattern, str(claim2))[1:]
        if len(x1) == 0 or len(x2) == 0:
            continue
        edge1 = get_edge(x1)
        edge2 = get_edge(x2)
        if not (is_valid_edge_index(edge1) and is_valid_edge_index(edge2)):
            continue
        encoding1 = tokenizer(
            x1,
            padding='max_length',  # Pad to the maximum sequence length
            truncation=True,
            max_length=config.max_length,
            return_tensors='pt',  # Return PyTorch tensors
        )
        encoding2 = tokenizer(
            x2,
            padding='max_length',  # Pad to the maximum sequence length
            truncation=True,
            max_length=config.max_length,
            return_tensors='pt',  # Return PyTorch tensors
        )
        y = data.loc[i, 'label']
        data_list.append(PairData(x1=encoding1['input_ids'], x2=encoding2['input_ids'], edge1=edge1, edge2=edge2,
                                  attn_mask1=encoding1['attention_mask'], attn_mask2=encoding2['attention_mask'],
                                  y=torch.tensor(y, dtype=torch.long)))
    return data_list


def get_edge(sentence):
    # 初始化有向边矩阵
    edge_index = [[], []]
    substring = "权利要求"
    # 超参数,表示'权利要求'尾部截取字符串长度
    next_length = 10
    ls = ['-', '至', '或', '～', '‑', '所述']
    # 遍历所有权利要求，找到对应引用关系
    for index, text in enumerate(sentence):
        if index == 0:
            continue
        while True:
            # 找到下标
            i = text.find(substring)
            if i == -1:
                break
            t = text[i + len(substring): i + next_length]
            for l in ls:
                if l in t:
                    t = t.split(l)[0]
            numbers = [int(match) for match in re.findall(r'\d+', t) if int(match) < index]
            if len(numbers) > 1 and '或' not in t:
                numbers = [z for z in range(int(numbers[0]), int(numbers[1] + 1))]
            edge_index[0] += [z - 1 for z in numbers]
            edge_index[1] += [index] * len(numbers)
            text = text[i + len(substring):]
    # 得到边矩阵
    return torch.tensor(edge_index, dtype=torch.long)


class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.bert = AutoModel.from_pretrained(config.model_path)
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.gat1 = GATConv(in_channels=config.in_channels, out_channels=config.out_channels,
                            heads=config.gat_num_heads, dropout=config.dropout_rate)
        self.gat2 = GATConv(in_channels=config.hidden_size * config.gat_num_heads, out_channels=config.hidden_size,
                            heads=config.gat_num_heads, dropout=config.dropout_rate)
        self.gat3 = GATConv(in_channels=config.hidden_size * config.gat_num_heads, out_channels=config.hidden_size,
                            heads=1, dropout=config.dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(config.dropout_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.parameters(), lr=config.lr)

    def forward(self, batch):
        # 获取数据
        x1, x2 = batch.x1.to(self.config.device), batch.x2.to(self.config.device)
        edge1, edge2 = batch.edge1.to(self.config.device), batch.edge2.to(self.config.device)
        attn_mask1, attn_mask2 = batch.attn_mask1.to(self.config.device), batch.attn_mask2.to(self.config.device)
        y = batch.y

        # BERT模型提取语义特征
        with torch.no_grad():
            # 将BERT部分的前向传播置于no_grad上下文中
            x1 = self.bert(x1, attention_mask=attn_mask1).pooler_output
            x2 = self.bert(x2, attention_mask=attn_mask2).pooler_output
        x1 = self.fc1(x1)
        x2 = self.fc1(x2)

        # GAT图注意力网络提取权利要求中的关联特征
        x1 = self.gat1(x=x1, edge_index=edge1)
        # 应用LeakyReLU激活函数 and 添加Dropout正则化
        x1 = self.dropout(self.leakyrelu(x1))
        x1 = self.gat2(x=x1, edge_index=edge1)
        x1 = self.dropout(self.leakyrelu(x1))
        x1 = self.gat3(x=x1, edge_index=edge1)

        x2 = self.gat1(x=x2, edge_index=edge2)
        x2 = self.dropout(self.leakyrelu(x2))
        x2 = self.gat2(x=x2, edge_index=edge2)
        x2 = self.dropout(self.leakyrelu(x2))
        x2 = self.gat3(x=x2, edge_index=edge2)

        # 分类层
        output = F.softmax(self.classifier(torch.abs(torch.mean(x1, dim=0) - torch.mean(x2, dim=0))),
                           dim=0).cpu().unsqueeze(0)
        loss = self.criterion(output, y)
        return output, loss
