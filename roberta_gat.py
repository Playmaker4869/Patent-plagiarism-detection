import pandas as pd
import torch
import numpy as np
from torch_geometric.loader import DataLoader

from utils import data_split, train
from model.roberta_gat import MyConfig, get_data_list, MyModel

if __name__ == '__main__':
    # 配置参数
    task = '权利要求书抄袭识别'
    config = MyConfig(task)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    # 数据读取和划分
    data = pd.read_csv('dataset/data.csv')
    train_data, test_data, val_data = data_split(data)
    train_set, test_set, val_set = get_data_list(config, train_data), get_data_list(config, test_data), get_data_list(
        config, val_data)
    train_loader = DataLoader(train_set, batch_size=config.batch_size)
    test_loader = DataLoader(test_set, batch_size=config.batch_size)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)

    # 定义模型
    model = MyModel(config).to(config.device)
    train(config, model, train_loader, test_loader, val_loader)
