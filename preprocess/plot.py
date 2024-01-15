from  matplotlib import pyplot as plt
from random import randint
from settings import *
import pandas as pd

print('随机从总数据中随机抽取一个进行绘图')

# 随机选择类别
rand_class = randint(0,num_classes-1)
print('抽取类别为{}'.format(id2label[rand_class]))

# 随机选择该类别中一个
class_data = pd.read_csv(total_path)
class_data = class_data.loc[class_data['label'] == rand_class]
rand_flow_id = randint(0,len(class_data))

data = pd.read_csv(class_data.iloc[rand_flow_id,0])
first_arrival_time = data.iloc[0,0]
arrival_time = []
packet_size = []

for index, row in data.iterrows():
    arrival_time.append(row[0] - first_arrival_time)
    packet_size.append(row[1])

plt.title("FLowpic of {}".format(id2label[rand_class] + '\nXgj is lazy so that it goes run with Chinese\nHe\'ll make it correct when drawing the true pic'))
plt.xlabel("arrival_time")
plt.ylabel("packet_size")
plt.scatter(arrival_time,packet_size,marker='.')
plt.show()