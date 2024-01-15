import os.path
import pandas as pd
from nfstream import NFStreamer,NFPlugin
from preprocess.settings import bpf_filter,block_size,threshold,id2proto
from dirs import *
import glob

#下同preprocess中的FlowPic,为方便调用直接贴过来了
class FlowPic(NFPlugin):
    def on_init(self, packet, flow):
        flow.udps.blocks = [{'arrival_time':[], 'packet_size':[]}]

        packet_size = packet.ip_size
        arrival_time = packet.time

        if(packet_size > 1500):
            packet_size = 1500

        packet_size = packet_size * -1

        flow.udps.blocks[0]['arrival_time'].append(arrival_time)
        flow.udps.blocks[0]['packet_size'].append(packet_size)

    def on_update(self, packet, flow):

        packet_size = packet.ip_size
        arrival_time = packet.time

        if(packet_size > 1500):
            packet_size = 1500

        if(packet.direction == 0):
            packet_size = packet_size * -1

        flow.udps.blocks[-1]['arrival_time'].append(arrival_time)
        flow.udps.blocks[-1]['packet_size'].append(packet_size)

        if(arrival_time - flow.udps.blocks[-1]['arrival_time'][0] >= self.block_size * 1000):

            if(len(flow.udps.blocks[-1]['arrival_time']) >= self.threshold):

                flow.udps.blocks.append({'arrival_time': [], 'packet_size': []})
            else:

                flow.udps.blocks[-1] = {'arrival_time': [], 'packet_size': []}

    def on_expire(self, flow):

        if len(flow.udps.blocks[-1]['arrival_time']) < self.threshold \
                or flow.udps.blocks[-1]['arrival_time'][-1] - flow.udps.blocks[-1]['arrival_time'][
            0] < 0.6 * self.block_size * 1000:

            flow.udps.blocks.pop()

def preprocess(file_path,stamp):
    #stamp为文件名，同时也是当时采样的时间戳
    streamer = NFStreamer(source=file_path, bpf_filter=bpf_filter, udps=FlowPic(block_size=block_size, threshold=threshold))
    for flow in streamer:
        # 跳过无效流
        if(flow_filter(flow)):
            save_flow(flow,stamp)

def flow_filter(flow):
    proto_list = ['dns', 'stun', 'icmp', 'arp', 'ssdp', 'llmnr', 'dhcp',
                  'netbios', 'nbns', 'nbdd', 'gquic', 'igmp', 'ntp', 'snmp','mdns',]
    if(flow.protocol == 6 or flow.protocol == 17):
        return True
    for proto in proto_list:
        if(proto in flow.application_name.lower()):
            return False
    return True

def save_flow(flow,stamp):
    flow_id = "_".join(
        [str(flow.id), flow.src_ip, str(flow.src_port), flow.dst_ip, str(flow.dst_port), id2proto[flow.protocol]])
    save_blocks(flow.udps.blocks, flow_id,stamp)

def save_blocks(blocks, flow_id,stamp):
    # 创建会话记录目录
    flow_dir = os.path.join(features_dir,stamp)
    flow_dir = os.path.join(flow_dir,flow_id)

    if(not os.path.exists(flow_dir) and len(blocks) > 0):
        os.makedirs(flow_dir)

    # 依次保存时间窗口记录
    for i in range(len(blocks)):
        block_id = i + 1
        block = pd.DataFrame.from_dict(blocks[i])
        block_name = 'block_' + str(block_id) + '.csv'
        csv_path = os.path.join(flow_dir, block_name)
        write_csv(block, csv_path)

def write_csv(df, csv_path):
    # 若csv文件已存在
    if os.path.exists(csv_path):
        # 采用'a+'模式, 无需写入header
        df.to_csv(csv_path, mode='a+', index=False, header=False)
    else:
        # 否则采用'w'模式
        df.to_csv(csv_path, mode='w', index=False)

def get_data_stat(stamp):
    files = glob.glob(os.path.join(os.path.join(features_dir,stamp)+'/*','*'))
    print("%d files generated."% (len(files)-1))
    # 构建path和label的dataframe
    data = {'path': [], 'label': []}
    for file in files:
        label = -1
        data['path'].append(file)
        data['label'].append(label)
    df = pd.DataFrame.from_dict(data)
    class_stat = df.label.value_counts()
    # 写入该dataframe
    if not os.path.exists(data_stat_dir):
        os.makedirs(data_stat_dir)
    write_csv(df, data_stat_dir+'/'+stamp+'.csv')


def single_packet_preprocess(filepath):
    stamp = filepath.split('\\')[-1].split('.')[0]
    preprocess(filepath,stamp)
    get_data_stat(stamp)

if __name__ == '__main__':
    single_packet_preprocess('predicted_data\dataset\\2022_12_13 16_09_27_955218.pcap')