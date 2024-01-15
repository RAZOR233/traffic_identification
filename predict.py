from loss import FocalLoss
from train.settings import learning_rate, num_classes,valid_batch_size
from train.model import CRNN
from train.flowpicset import FlowPicSet
import paddle
from packet_catch import *
from single_packet_preprocess import single_packet_preprocess

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

def predict(model,predict_data_loader):
    count = 0
    for batch_id, data in enumerate(predict_data_loader()):
        count += 1
        x_data, y_data = data
        output = model(x_data)
        predicts = output[:, :num_classes]
        tmp = result(predicts)
        print("predicts:", tmp)
        summary(tmp)


def result(predicts):
    output = []
    for list in predicts:
        # print(list[0])
        # print(list[0].item())
        min = abs(list[0].item())
        minnum = 0
        for k in range(len(list)):
            if abs(list[k].item()) < min:
                min = abs(list[k].item())
                minnum = k
        output.append(minnum)
    return output

def summary(lists):
    count_dist = dict()
    for i in lists:
        if i in count_dist:
            count_dist[i] += 1
        else:
            count_dist[i] = 1
    s = len(lists)
    count_dist_1 = sorted(count_dist.items(),key= lambda x:x[1],reverse=True)
    for i in range(len(count_dist_1)):
        print('{:.2%}:'.format(count_dist_1[i][1]/s),id2label[count_dist_1[i][0]],end=' ')
    print()

if __name__ == '__main__':
    file_name = catch_packet(my_iface)
    print("Real time file caught!")
    single_packet_preprocess(os.path.join(dataset_dir,file_name))
    print("Preprocess done!")
    model =load_model(CRNN(in_size=256, in_channels=1, num_classes=num_classes, num_hidden=128, num_rnn=2))
    predict_data = FlowPicSet(data_stat_dir+'/'+file_name[:-5]+".csv")
    predict_data_loader = paddle.io.DataLoader(predict_data, batch_size=valid_batch_size, shuffle=False)
    predict(model,predict_data_loader)