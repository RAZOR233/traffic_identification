import paddle
import numpy as np
from paddle.nn import LSTM, Conv2D, BatchNorm2D, MaxPool2D, Linear, Dropout
from paddle.nn.functional import relu

# 过程张量维度均经过手动计算，牵一发而动全身，勿改

# 双向LSTM
# https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/LSTM_cn.html#lstm
class Bi_LSTM(paddle.nn.Layer):
    def __init__(self, nIn, nHidden):
        super(Bi_LSTM, self).__init__()
        self.rnn = LSTM(nIn, nHidden, direction='bidirect')

    def forward(self, x):
        output, _ = self.rnn(x)
        return output #输出与输入形状相同

# 空洞卷积残差块
# https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Conv2D_cn.html
class BasicBlock(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=[3, 2],
                 stride=1, padding=[1, 1], dilation=[1, 2]):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=kernel_size[0],
                            stride=stride, padding=padding[0], dilation=dilation[0])
        self.bn1 = BatchNorm2D(out_channels)
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=kernel_size[1],
                            stride=stride, padding=padding[1], dilation=dilation[1])
        self.bn2 = BatchNorm2D(out_channels)
        self.downsample = paddle.nn.Sequential(
            Conv2D(in_channels, out_channels, kernel_size=1, stride=1),
            BatchNorm2D(out_channels),
        )

    def forward(self, x):
        residual = x # x 32,in_channels,256,256
        out = self.conv1(x) # out 32,out_channels,256,256
        out = self.bn1(out) # out 32,out_channels,256,256
        out = relu(out) # out 32,out_channels,256,256
        out = self.conv2(out) # out 32,out_channels,256,256
        out = self.bn2(out) # out 32,out_channels,256,256
        residual = self.downsample(residual) # residual 32,out_channels,256,256
        out += residual
        out = relu(out)
        return out # 32,out_channels,256,256

# 组合网络：空洞卷积+残差连接+双向LSTM
class CRNN(paddle.nn.Layer):
    def __init__(self, in_size, in_channels, num_classes, num_hidden=128, num_rnn=2):
        super(CRNN, self).__init__()

        cnn = paddle.nn.Sequential(
            BasicBlock(in_channels, 64),  # 64*256*256
            MaxPool2D(4),  # 64*64*64
            BasicBlock(64, 128),  # 128*64*64
            MaxPool2D(4),  # 128*16*16
            BasicBlock(128, 256),  # 256*16*16
            MaxPool2D((4, 1)),  # 256*4*16
            BasicBlock(256, 512),  # 512*4*16
            MaxPool2D((4, 1)),  # 512*1*16
            BatchNorm2D(512) # 512*1*16
        )
        self.cnn = cnn
        self.rnn = paddle.nn.Sequential(
            Bi_LSTM(512, num_hidden),
            Bi_LSTM(num_hidden * 2, num_hidden)
        )
        self.fc1 = Linear(16 * num_hidden * 2, 64)
        self.dropout = Dropout(0.5)
        self.fc2 = Linear(64, num_classes)

    def forward(self, x):
        # conv features
        conv = self.cnn(x) # conv 32,512,1,16
        conv = paddle.squeeze(conv, 2) # conv 32,512,16
        conv = paddle.transpose(conv, perm=[0, 2, 1]) # conv 32,16,512
        # rnn features
        bi_lstm = self.rnn(conv) # bi_lstm 32,16,256
        bi_lstm = paddle.reshape(bi_lstm, [bi_lstm.shape[0], -1])
        # classifier
        feats = self.fc1(bi_lstm)
        feats = relu(feats)
        output = self.dropout(feats)
        output = self.fc2(output)
        # 级联输出和特征
        output = paddle.concat([output, feats], axis=-1)
        return output