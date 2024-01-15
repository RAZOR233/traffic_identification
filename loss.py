import paddle

'''
    损失函数定义，包括不平衡数据训练的focal loss和度量学习的center loss
'''


# focal loss (包装Paddle既有的API)
class FocalLoss(paddle.nn.Layer):
    def __init__(self,
                 num_classes,
                 reduction='mean',
                 alpha=0.25,
                 gamma=2.0,
                 name=None):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma
        self.name = name

    def forward(self, x, label):
        # assert x.dtype == paddle.float32, '数据类型必须为float32'
        label = paddle.squeeze(label, 1)
        label = paddle.nn.functional.one_hot(label, num_classes=self.num_classes)
        one = paddle.to_tensor([1.], dtype='float32')
        fg_label = paddle.greater_equal(label, one)
        fg_num = paddle.sum(paddle.cast(fg_label, dtype='float32'))
        loss = paddle.nn.functional.sigmoid_focal_loss(
            x,
            label,
            normalizer=fg_num,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
            name=self.name)
        return loss


# center loss (修改实现)
class CenterLoss(paddle.nn.Layer):
    def __init__(self,
                 num_classes,
                 feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = paddle.create_parameter(shape=[num_classes, feat_dim], dtype='float32')

    # 这里的输入input为特征向量
    def forward(self, x, label):
        batch_size = x.shape[0]
        n_center = paddle.index_select(self.centers, index=label, axis=0)
        distance = paddle.dist(x, n_center)
        loss = (1 / 2.0 / batch_size) * distance
        return loss
