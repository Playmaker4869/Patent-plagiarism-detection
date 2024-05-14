import torch
import time
from sklearn import metrics
from datetime import timedelta
from sklearn.model_selection import train_test_split


def data_split(data, test_size=0.1, random_state=42):
    X = data[['claim_1', 'claim_2']]
    y = data['label']
    # 首先，划分为训练集和临时集（包含验证集和测试集）
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y,
                                                                  random_state=random_state)
    # 然后，再将临时集划分为验证集和测试集
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=1 / 9,
                                                      stratify=y_train_temp,
                                                      random_state=random_state)
    # 将特征和标签合并
    X_train['label'], X_test['label'], X_val['label'] = y_train, y_test, y_val
    train = X_train.reset_index(drop=True)
    test = X_test.reset_index(drop=True)
    val = X_val.reset_index(drop=True)
    return train, test, val


def train(config, model, train_loader, test_loader, dev_loader):
    start_time = time.time()
    print("*** Training... ***")
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    model.train()
    for epoch in range(config.epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.epochs))
        model.train()
        for batch in train_loader:
            model.optimizer.zero_grad()
            output, loss = model(batch)
            loss.backward()
            model.optimizer.step()
            # 清理GPU缓存
            torch.cuda.empty_cache()
            # 模型评估
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                try:
                    true = batch.y
                except:
                    true = batch['label']
                predict = output.argmax(dim=1)
                train_acc = metrics.accuracy_score(true, predict)
                dev_acc, dev_loss = evaluate(config, model, dev_loader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(
                    msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif,
                               improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_loader)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    # 保存结果到本地evaluation中的txt文件
    logging(config, test_loss, test_acc, test_report, test_confusion, time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = []
    labels_all = []
    with torch.no_grad():
        for batch in data_iter:
            # 获取标签
            try:
                labels = batch.y
            except:
                labels = batch['label']
            output, loss = model(batch)
            loss_total += loss
            predict_all += output.argmax(dim=1).tolist()
            labels_all += labels.tolist()

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def logging(config, test_loss, test_acc, test_report, test_confusion, time_dif):
    logs = ''
    logs += f'Task Name:{config.task_name},  Model Name:{config.model_name}\n'
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    logs += msg.format(test_loss, test_acc) + '\n'
    logs += "Precision, Recall and F1-Score...\n"
    logs += test_report
    logs += "Confusion Matrix..."
    print("type:", type(time_dif))
    logs += str(test_confusion)
    logs += "Time usage:", time_dif
    with open('evaluation/' + config.task_name + '.txt') as f:
        f.write(logs)
