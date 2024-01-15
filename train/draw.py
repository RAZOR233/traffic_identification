import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import paddle
import pandas as pd
from settings import *
from flowpicset import FlowPicSet
from model2 import CRNN
import warnings

warnings.filterwarnings("ignore", category=Warning)

# 设置中文字体, 根据系统中文字体设置即可
sns.set(font='Noto Sans CJK JP')
plt.gcf().subplots_adjust(bottom=0.3)
plt.rcParams['font.sans-serif']=['Noto Sans CJK JP'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 #有中文出现的情况，需要u'内容'


# 归一化
def normalise_matrix(matrix):
    with np.errstate(all='ignore'):
        normalised_rs = matrix / matrix.sum(axis=1, keepdims=True)
        normalised_rs = np.nan_to_num(normalised_rs)
        return normalised_rs


# 绘制混淆矩阵
def plot_confusion_matrix(matrix, labels):
    normalised_cm = normalise_matrix(matrix)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        data=normalised_cm, cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        annot=True, ax=ax, fmt='.2f',
    )

    ax.set_xlabel('Predict labels')
    ax.set_ylabel('True labels')
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    plt.show()


# 计算混淆矩阵
def confusion_matrix(data_loader, model, num_class):
    # 加载训练好的模型参数
    model.eval()
    cf_mtx = np.zeros((num_class, num_class))
    for batch_id, data in enumerate(data_loader()):
        x_data, y_data = data
        output = model(x_data)
        # 获取预测
        predicts = output[:, :num_classes]
        # 获取预测标签
        pred_label = paddle.argmax(predicts, 1).numpy()
        true_label = paddle.squeeze(y_data).numpy()
        # 填充混淆矩阵
        for i in range(len(true_label)):
            cf_mtx[true_label[i], pred_label[i]] += 1
    return cf_mtx


# 计算precision
def get_precision(matrix, i):
    tp = matrix[i, i]
    tp_fp = matrix[:, i].sum()

    return tp / tp_fp


# 计算recall
def get_recall(matrix, i):
    tp = matrix[i, i]
    p = matrix[i, :].sum()

    return tp / p


# 计算f1值
def get_f1_score(precision, recall):
    return 2.0 * precision * recall / (precision + recall)


# 报告各项指标
def get_classification_report(cf_mtx, labels=None):
    rows = []
    for i in range(cf_mtx.shape[0]):
        # 计算各项指标
        precision = get_precision(cf_mtx, i)
        recall = get_recall(cf_mtx, i)
        f1_score = get_f1_score(precision, recall)
        if labels:
            label = labels[i]
        else:
            label = i

        row = {
            'label': label,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == '__main__':
    # 获取文字标签
    label_list = list(id2label.values())
    # 获取test_loader
    test_data = FlowPicSet(test_path)
    test_loader = paddle.io.DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
    # 加载训练好的模型参数
    state_dict = paddle.load("model_params/model.pdparams")
    # 构建网络模型
    crnn_model = CRNN(in_size=256, in_channels=1, num_classes=num_classes, num_hidden=128, num_rnn=2)
    # 将训练好的参数读取到网络中
    crnn_model.set_state_dict(state_dict)
    # 计算混淆矩阵
    cm = confusion_matrix(test_loader, crnn_model, num_classes)
    # 计算各项指标
    report = get_classification_report(cm, label_list)
    print(report)
    # 绘制混淆矩阵
    plot_confusion_matrix(cm, label_list)
