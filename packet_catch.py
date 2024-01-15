from scapy.all import *
import datetime
from dirs import *

my_iface = "Intel(R) Wi-Fi 6 AX200 160MHz"

def catch_packet(my_iface):
    pack = sniff(count=10000,iface=my_iface)
    output_file = get_timestamp()+".pcap"
    output_file_path = os.path.join(dataset_dir,output_file)
    wrpcap(output_file_path,pack)
    return output_file

def get_timestamp():
    return datetime.datetime.now().strftime('%Y_%m_%d %H_%M_%S_%f')

if __name__ == '__main__':
    filename = catch_packet(my_iface)
