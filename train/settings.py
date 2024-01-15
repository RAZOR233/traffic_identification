# 类别数目
num_classes = 8
# 输入矩阵大小
resolution = 128
# 训练epoch
epochs = 20
# 训练batch大小
train_batch_size = 16
# 学习率
learning_rate = 0.001
# centerloss中center学习率
center_lr = 0.01
# lambda
lmbda = 0.1
# 验证batch大小
valid_batch_size = 16
# 测试batch大小
test_batch_size = 16
# 特征维度
feat_dim = 64
# 训练集路径
train_path = '../preprocessed_data/data_stat/train.csv'
# 验证集路径
valid_path = '../preprocessed_data/data_stat/valid.csv'
# 测试集路径
test_path = '../preprocessed_data/data_stat/test.csv'
# id到标签的字典映射
id2label = {0: 'bilibili', 1: 'gaodemap', 2: 'jingdong', 3: 'meituan',4:'netease',
            5:'qqmusic',6:'taobao',7:'weibo'}
label2id = { 'bilibili':0,  'gaodemap':1, 'jingdong':2, 'meituan':3, 'netease':4,
             'qqmusic':5, 'taobao':6, 'weibo':7}
