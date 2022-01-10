import torch
import time
import argparse
from models.Auto_Encoder import Auto_Encoder
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import mean_squared_error
import numpy as np

from dataloader.dataloader_RGB import Facedata_Loader
from loger import Logger
from utils import plot_roc_curve, cal_metrics, plot_3_kind_data, plot_real_fake_data

import os

# 모델 생성
# loss 생성 -> MSE loss 사용
# 옵티마이저, 스케줄러 생성

parser = argparse.ArgumentParser(description='face anto-spoofing')
parser.add_argument('--batch-size', default='64', type=int, help='train batch size') 
parser.add_argument('--test-size', default='64', type=int, help='test batch size') 
parser.add_argument('--save-path', default='../ad_output/RGB/logs/Valid/', type=str, help='logs save path')
parser.add_argument('--checkpoint', default='model.pth', type=str, help='pretrained model checkpoint')
parser.add_argument('--message', default='', type=str, help='pretrained model checkpoint')
parser.add_argument('--model', default='', type=str, help='model directory')
parser.add_argument('--train', default=False, type=bool, help='train')
parser.add_argument('--skf', default=0, type=int, help='stratified k-fold')
args = parser.parse_args()

time_object = time.localtime(time.time())
time_string = time.strftime('%Y-%m-%d_%I:%M_%p', time_object)

save_path = args.save_path + f'{args.message}' + '_' + f'{time_string}'
if not os.path.exists(save_path):
    os.makedirs(save_path)

logger = Logger(f'{save_path}/logs.logs')
logger.Print(time_string + " - " + args.message  + "\n")


def valid(data_loader, checkpoint):

    model = Auto_Encoder()

    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda:
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))) #  device_ids=[0, 1, 2]
        model.cuda()
        model.load_state_dict(torch.load(checkpoint))
    else:
        print("Something Wrong, Cuda Not Used")

    # 1. 전체 데이터에 대한 MSE 구하기 
    mse = []

    y_true = []
    # y_pred = []

    data_high = []
    data_mid = []
    data_low = []
    
    data_real = []
    data_fake = []

    for _, data in enumerate(data_loader):
        rgb_image, label, rgb_path = data
        rgb_image = rgb_image.cuda()        
        label = label.cuda()

        recons_image = model(rgb_image)

        # 모든 데이터에 대한 MSE 구하기 
        for i in range(len(rgb_image)):
            np_image = rgb_image[i].cpu().detach().numpy()
            np_recons_image = recons_image[i].cpu().detach().numpy()
            np_image = np_image * 255
            np_recons_image = np_recons_image * 255

            diff = []
            for d in range(np_image.shape[0]) :        
                val = mean_squared_error(np_image[d].flatten(), np_recons_image[d].flatten())
                diff.append(val)
            mse_by_sklearn = np.array(diff).mean()
            mse.append(mse_by_sklearn)

            # light 에 따라 데이터 분류하기 
            path = rgb_path[i].split('/')[-5]
            if "High" in path:
                data_high.append(mse_by_sklearn)
            elif "Mid" in path:
                data_mid.append(mse_by_sklearn)
            elif "Low" in path:
                data_low.append(mse_by_sklearn)
            else:
                print("Data Classification Error - High, Mid, Low")

            # mask 유무에 따라 데이터 분류하기
            if label[i].item() == 1:
                data_real.append(mse_by_sklearn)
            else:
                data_fake.append(mse_by_sklearn)

            # y_true 는 넣어.
            y_true.append(label[i].cpu().detach().numpy())
    print("MSE Caculation Finished")
    print(f"Max MSE: {max(mse)}, Min MSE: {min(mse)}")

    # 2. MSE 분포에 따른 threshold 리스트 결정 
    threshold = np.arange(round(min(mse), -1)+10, round(max(mse), -1)-10, 10)

    # 3. threshold 에 대한 accuracy 구하고 max accuracy에 대한 threshold 리턴 
    accuracy_per_thres = []
    precision_per_thres = []
    recall_per_thres = []
    f1_per_thres = []
    for thres in threshold:   

        print(f"***** Threshold: {thres}")

        y_pred = []
        for i in range(len(mse)):
            if mse[i] < thres:
                y_pred.append(1)
            else:
                y_pred.append(0)

        if len(y_true) != len(y_pred):
            logger.Print("length error - y_true, y_pred")

        # 한번 계산을 해주고, 그 계산에 대한 비교 해줘야 함. 
        accu, prec, reca, f = cal_metrics(y_true, y_pred) 
        accuracy_per_thres.append(accu)
        precision_per_thres.append(prec)
        recall_per_thres.append(reca)
        f1_per_thres.append(f)

    accuracy_max = max(accuracy_per_thres)
    index = accuracy_per_thres.index(accuracy_max)
    precision_max = precision_per_thres[index]
    recall_max = recall_per_thres[index]
    f1_max = f1_per_thres[index]
    threhold_max = threshold[index]

    # 데이터 분포도 그래프로 그리기 
    if not os.path.exists(f'{save_path}/graph'):
        os.makedirs(f'{save_path}/graph')
    plot_3_kind_data(f"{save_path}/graph", f"Light_Distribution_Threshold_{thres}_", data_high, data_mid, data_low)
    plot_real_fake_data(f"{save_path}/graph", f"Mask_Distribution_Threshold_{thres}_", data_real, data_fake)

    return threhold_max, accuracy_max, precision_max, recall_max, f1_max

if __name__ == "__main__":

    _, valid_loader, _ = Facedata_Loader(train_size=64, test_size=64)

    threshold_per_epoch = []
    accuracy_per_epoch = []
    precision_per_epoch = []
    recall_per_epoch = []
    f1_per_epoch = []
    epochs = []

    for i in range(2999, 3000, 1):

        print(f"***** Epoch {i} start")

        checkpoint = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/RGB/checkpoint/{args.model}/epoch_{i}_model.pth"
        thres, accu, prec, reca, f1 = valid(valid_loader, checkpoint)

        threshold_per_epoch.append(thres)
        accuracy_per_epoch.append(accu) 
        precision_per_epoch.append(prec)
        recall_per_epoch.append(reca)
        f1_per_epoch.append(f1)
        epochs.append(i)

    accuracy_max = max(accuracy_per_epoch)
    index = accuracy_per_epoch.index(accuracy_max)
    epoch_max = epochs[index]
    threshold_max = threshold_per_epoch[index]
    precision_max = precision_per_epoch[index]
    recall_max = recall_per_epoch[index]
    f1_max = f1_per_epoch[index]

    logger.Print(f"***** Total Threshold per epoch")
    logger.Print(threshold_per_epoch)
    logger.Print(f"***** Total Accuracy per epoch")
    logger.Print(accuracy_per_epoch)
    logger.Print(f"***** Total Precision per epoch")
    logger.Print(precision_per_epoch)
    logger.Print(f"***** Total Recall per epoch")
    logger.Print(recall_per_epoch)
    logger.Print(f"***** Total F1 per epoch")
    logger.Print(f1_per_epoch)
    logger.Print(f"***** Total epoch ")
    logger.Print(epochs)

    logger.Print(f"***** Result")
    logger.Print(f"***** Max Accuracy: {accuracy_max:3f}")
    logger.Print(f"***** Epoch(=real-1): {epoch_max}")
    logger.Print(f"***** Threshold: {threshold_max:3f}")
    logger.Print(f"***** Precision: {precision_max:3f}")
    logger.Print(f"***** Recall: {recall_max:3f}")
    logger.Print(f"***** F1: {f1_max:3f}")