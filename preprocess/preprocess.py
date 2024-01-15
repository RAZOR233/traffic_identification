import os.path
import pandas as pd
from nfstream import NFStreamer, NFPlugin
from settings import *
import glob
from sklearn.model_selection import train_test_split

# 根据官方API定义新的类FlowPic，继承NFPlugin
# https://www.nfstream.org/docs/api
class FlowPic(NFPlugin):
    '''
    on_init(self, packet, flow):Method called at flow creation.
    You must initiate your udps values if you plan to compute ones.
    '''
    def on_init(self, packet, flow):
        # 创建一个字典列表，每一个字典含有两组键对值，存储两个列表
        # 用于表达包到达时间以及对应大小
        # 包大小使用正负值表示方向
        flow.udps.blocks = [{'arrival_time':[], 'packet_size':[]}]

        # 提取包信息
        # 此处到达时间是包到达的时间戳
        packet_size = packet.ip_size
        arrival_time = packet.time

        # 过长包截断
        if(packet_size > 1500):
            packet_size = 1500

        # 判断流方向
        # direction 0 for src2dst ; 1 for dst2src
        # direction 0 for c->s ; 1 for s->c
        # 定义一个流中第一个包的方向为c -> s 并取负
        packet_size = packet_size * -1

        # 将信息加入字典中
        flow.udps.blocks[0]['arrival_time'].append(arrival_time)
        flow.udps.blocks[0]['packet_size'].append(packet_size)

    '''
    on_update(self, packet, flow): Method called to update each
    flow with its belonging packet.
    '''
    def on_update(self, packet, flow):
        # 提取包信息
        # 此处到达时间是包到达的时间戳
        packet_size = packet.ip_size
        arrival_time = packet.time

        # 过长包截断
        if(packet_size > 1500):
            packet_size = 1500

        # 判断流方向
        # direction 0 for src2dst ; 1 for dst2src
        # direction 0 for c->s ; 1 for s->c
        # 定义一个流中第一个包的方向为c -> s 并取负
        if(packet.direction == 0):
            packet_size = packet_size * -1

        # 将信息加入字典中
        flow.udps.blocks[-1]['arrival_time'].append(arrival_time)
        flow.udps.blocks[-1]['packet_size'].append(packet_size)

        if(arrival_time - flow.udps.blocks[-1]['arrival_time'][0] >= self.block_size * 1000):
            # 时间窗口中的数据包数目达到预定阈值
            if(len(flow.udps.blocks[-1]['arrival_time']) >= self.threshold):
                # 开辟新的时间窗口存储字典，供下一个分块使用
                flow.udps.blocks.append({'arrival_time': [], 'packet_size': []})
            else:
                # 否则置空该字典，意为该块数据包数目未达到阀值，给予抛弃
                flow.udps.blocks[-1] = {'arrival_time': [], 'packet_size': []}

    '''
    on_expire(self, flow):Method called at flow expiration
    '''
    def on_expire(self, flow):
        # 最后一个时间窗口数据包不足阈值或时间周期未达到设置窗口大小的0.6
        if len(flow.udps.blocks[-1]['arrival_time']) < self.threshold \
                or flow.udps.blocks[-1]['arrival_time'][-1] - flow.udps.blocks[-1]['arrival_time'][
            0] < 0.6 * self.block_size * 1000:
            # 弹出最后一个时间窗口信息
            flow.udps.blocks.pop()

# 将block转化为dataframe后输出到csv
# 输入：block（df），csv_path
# 返回：无
# 写入dataframe到csv文件
def write_csv(df, csv_path):
    # 若csv文件已存在
    if os.path.exists(csv_path):
        # 采用'a+'模式, 无需写入header
        df.to_csv(csv_path, mode='a+', index=False, header=False)
    else:
        # 否则采用'w'模式
        df.to_csv(csv_path, mode='w', index=False)


# 判断该流是否为有效流（判断dns等）
# 输入：flow对象
# 返回：True or False
def flow_filter(flow):
    proto_list = ['dns', 'stun', 'icmp', 'arp', 'ssdp', 'llmnr', 'dhcp',
                  'netbios', 'nbns', 'nbdd', 'gquic', 'igmp', 'ntp', 'snmp','mdns',]
    if(flow.protocol == 6 or flow.protocol == 17):
        return True
    for proto in proto_list:
        if(proto in flow.application_name.lower()):
            return False
    return True

# 保存该流所生成的特征
# 输入：flow对象
# 返回：无
def save_flow(flow, class_name):
    flow_id = "_".join(
        [str(flow.id), flow.src_ip, str(flow.src_port), flow.dst_ip, str(flow.dst_port), id2proto[flow.protocol]])
    save_blocks(flow.udps.blocks, class_name, flow_id)

# 保存一个流中的flowpic特征，保存格式
# 输入：flow对应blocks（参考FlowPic中定义），分类名，flow对应id
# 返回：无
def save_blocks(blocks, class_name, flow_id):
    # 创建会话记录目录
    flow_dir = os.path.join(features_dir, class_name)
    flow_dir = os.path.join(flow_dir, flow_id)

    if(not os.path.exists(flow_dir) and len(blocks) > 0):
        os.makedirs(flow_dir)

    # 依次保存时间窗口记录
    for i in range(len(blocks)):
        block_id = i + 1
        block = pd.DataFrame.from_dict(blocks[i])
        block_name = 'block_' + str(block_id) + '.csv'
        csv_path = os.path.join(flow_dir, block_name)
        write_csv(block, csv_path)
        #print(block)

# 预处理单个文件
# 输入：文件路径
# 返回：无
def preprocess(file_path):
    # 获得该文件的分类
    class_name = file_path.split('\\')[-2]
    print(class_name)
    # 扫描并读取流
    streamer = NFStreamer(source=file_path, bpf_filter=bpf_filter, udps=FlowPic(block_size=block_size, threshold=threshold))
    for flow in streamer:
        # 跳过无效流
        if(flow_filter(flow)):
            save_flow(flow, class_name)

# 预处理多个文件
# 流水线处理pcap文件，无输入返回
def pipeline():
    file_paths = glob.glob(os.path.join(dataset_dir + '/*', "*"))
    print("%d preprocessed files found."% len(file_paths))
    count = 1
    for file_path in file_paths:
        print("processing file {}".format(count))
        print(file_path)
        preprocess(file_path)
        print("finish processing file {}".format(count))
        count = count + 1

# 获取数据描述
# 无输入返回
def get_data_stat():
    files = glob.glob(os.path.join(features_dir + '/*/*', "*"))
    print("%d files found."% len(files))
    # 构建path和label的dataframe
    data = {'path': [], 'label': []}
    for file in files:
        class_name = file.split('\\')[-3]
        label = label2id[class_name]
        data['path'].append(file)
        data['label'].append(label)
    df = pd.DataFrame.from_dict(data)
    class_stat = df.label.value_counts()
    print(class_stat)
    # 写入该dataframe
    if not os.path.exists(data_stat_dir):
        os.makedirs(data_stat_dir)
    write_csv(df, total_path)

# 将total.csv划分为train.csv,test.csv,valid.csv
# 无输入输出
def split():
    data = pd.read_csv(total_path)
    for i in range(num_classes):
        class_data = data.loc[data['label'] == i]
        # 先切分出测试集
        train_valid_matrix, class_test_matrix, train_valid_label, class_test_label = train_test_split(
            class_data['path'],
            class_data['label'],
            random_state=random_state,
            test_size=test_ratio,
            train_size=1-test_ratio,
            )
        # 再划分训练集和验证集
        class_train_matrix, class_valid_matrix, class_train_label, class_valid_label = train_test_split(
            train_valid_matrix,
            train_valid_label,
            random_state=random_state * 10 + 1,
            test_size=valid_ratio,
            train_size=1-valid_ratio)
        # 记录训练/验证/测试集的特征路径及对应标签
        class_train_data = pd.DataFrame.from_dict({'path': class_train_matrix, 'label': class_train_label})
        class_valid_data = pd.DataFrame.from_dict({'path': class_valid_matrix, 'label': class_valid_label})
        class_test_data = pd.DataFrame.from_dict({'path': class_test_matrix, 'label': class_test_label})
        write_csv(class_train_data, train_path)
        write_csv(class_valid_data, valid_path)
        write_csv(class_test_data, test_path)

if __name__ == '__main__':
    pipeline()
    get_data_stat()
    split()
    print('已经预处理完毕，若需要重新预处理请删除已生成数据后更改调用')