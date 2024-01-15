import paddle
import numpy as np
from loss import CenterLoss, FocalLoss
from model3 import CRNN
from flowpicset import FlowPicSet
from settings import *

import warnings

warnings.filterwarnings("ignore", category=Warning)


# 定义训练函数
def train_model(model, train_data_loader, valid_data_loader):
    # 损失函数
    # center_loss_fn = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)
    focal_loss_fn = FocalLoss(num_classes=num_classes)
    # 优化参数
    params = list(model.parameters())
    # 优化器
    optimizer = paddle.optimizer.Adam(parameters=params, learning_rate=learning_rate)
    best_acc = 0.0
    # 训练
    for epoch in range(epochs):
        model.train()
        print('epoch {}:'.format(epoch + 1))
        accuracies = []
        losses = []
        for batch_id, data in enumerate(train_data_loader()):
            x_data, y_data = data
            output = model(x_data)
            # 获取输出及特征
            predicts, feats = output[:, :num_classes], output[:, num_classes:]
            # 计算两种损失，此处只使用focal_loss
            # center_loss = center_loss_fn(feats, y_data)
            focal_loss = focal_loss_fn(predicts, y_data)
            # 计算总的损失
            loss = focal_loss
            # 计算准确率
            acc = paddle.metric.accuracy(predicts, y_data)
            # 更新梯度
            focal_loss.backward() #两种loss训练效果不理想，可只采用focal_loss训练
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
            optimizer.step()
            optimizer.clear_grad()
        print("[train] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
        # 验证
        model.eval()
        accuracies = []
        for batch_id, data in enumerate(valid_data_loader()):
            x_data, y_data = data
            output = model(x_data)
            predicts = output[:, :num_classes]
            acc = paddle.metric.accuracy(predicts, y_data)
            accuracies.append(acc.numpy())
        print("[validation] accuracy: {}".format(np.mean(accuracies)))
        if np.mean(accuracies) >= best_acc:
            save_params(model, optimizer, focal_loss_fn)
            best_acc = np.mean(accuracies)


# 保存参数
def save_params(trained_model, trained_optimizer, focal_loss_fn):
    paddle.save(trained_model.state_dict(), "model_params/model.pdparams")
    paddle.save(trained_optimizer.state_dict(), "model_params/optimizer.pdopt")
    paddle.save(focal_loss_fn.state_dict(), "model_params/center.pdparams")


# 加载参数
def load_params(trained_model, trained_optimizer, focal_loss_fn):
    model_state_dict = paddle.load("model_params/model.pdparams")
    opt_state_dict = paddle.load("model_params/optimizer.pdopt")
    clf_state_dict = paddle.load("model_params/center.pdparams")
    trained_model.set_state_dict(model_state_dict)
    trained_optimizer.set_state_dict(opt_state_dict)
    focal_loss_fn.set_state_dict(clf_state_dict)

def load_model(model):
    focal_loss_fn = FocalLoss(num_classes=num_classes)
    # 优化参数
    params = list(model.parameters())
    # 优化器
    optimizer = paddle.optimizer.Adam(parameters=params, learning_rate=learning_rate)
    load_params(model, optimizer, focal_loss_fn)
    model.eval()
    return model

if __name__ == '__main__':
    # 定义训练和验证数据
    train_data = FlowPicSet(train_path)
    valid_data = FlowPicSet(valid_path)
    print("已完成定义训练和验证数据")
    # 数据加载
    train_loader = paddle.io.DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    valid_loader = paddle.io.DataLoader(valid_data, batch_size=valid_batch_size, shuffle=False)
    print("已完成数据加载")
    # 定义模型
    crnn_model = CRNN(in_size=128, in_channels=1, num_classes=num_classes, num_hidden=64, num_rnn=2)
    #crnn_model = load_model(CRNN(in_size=128, in_channels=1, num_classes=num_classes, num_hidden=64, num_rnn=2))
    print("已完成定义模型")
    # 训练模型00
    train_model(crnn_model, train_loader, valid_loader)
    print("已完成训练模型")
