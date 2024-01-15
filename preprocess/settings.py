
# 类别数目
num_classes = 8

# id与传输层协议及类别标签的字典映射
id2proto = {6: 'TCP', 17: 'UDP'}
id2label = {0: 'bilibili', 1: 'gaodemap', 2: 'jingdong', 3: 'meituan',4:'netease',
            5:'qqmusic',6:'taobao',7:'weibo'}
label2id = { 'bilibili':0,  'gaodemap':1, 'jingdong':2, 'meituan':3, 'netease':4,
             'qqmusic':5, 'taobao':6, 'weibo':7}
# 数据集相关路径
dataset_dir = '../dataset'
features_dir = '../preprocessed_data/features'
data_stat_dir = '../preprocessed_data/data_stat'
total_path = data_stat_dir + '/total.csv'
train_path = data_stat_dir + '/train.csv'
valid_path = data_stat_dir + '/valid.csv'
test_path = data_stat_dir + '/test.csv'

# 时间窗口大小
block_size = 5
# 窗口包数目阈值
threshold = 10
# 包过滤规则
bpf_filter = "!broadcast && !multicast"
# 图像/矩阵尺寸
resolution = 128
# 随机数种子
random_state = 10
# 验证集比例
valid_ratio = 0.2
# 测试集比例
test_ratio = 0.2