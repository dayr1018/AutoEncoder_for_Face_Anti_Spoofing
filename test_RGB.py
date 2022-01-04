import torch
import time
import argparse

from models.Auto_Encoder import Auto_Encoder
from dataloader.dataloader_RGB import Facedata_Loader
from loger import Logger
from utils import plot_roc_curve, cal_metrics, plot_3_kind_data, plot_real_fake_data, plot_result

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from PIL import Image

import os

# 모델 생성
# loss 생성 -> MSE loss 사용
# 옵티마이저, 스케줄러 생성

model = Auto_Encoder()
weight_path = "/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/RGB/checkpoint/original/epoch_2999_model.pth"

use_cuda = True if torch.cuda.is_available() else False
if use_cuda:
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))) #  device_ids=[0, 1, 2]
    model.cuda()
    model.load_state_dict(torch.load(weight_path))
else:
    print("Something Wrong, Cuda Not Used")

parser = argparse.ArgumentParser(description='face anto-spoofing')
parser.add_argument('--batch-size', default='64', type=int, help='train batch size') 
parser.add_argument('--test-size', default='64', type=int, help='test batch size') 
parser.add_argument('--save-path', default='../ad_output/RGB/logs/Test/', type=str, help='logs save path')
parser.add_argument('--checkpoint', default='model.pth', type=str, help='pretrained model checkpoint')
parser.add_argument('--message', default='', type=str, help='pretrained model checkpoint')
parser.add_argument('--train', default=False, type=bool, help='train')
parser.add_argument('--skf', default=0, type=int, help='stratified k-fold')
args = parser.parse_args()

time_object = time.localtime(time.time())
time_string = time.strftime('%Y-%m-%d_%I:%M_%p', time_object)

save_path = args.save_path + f'{time_string}' + '_' + f'{args.message}'
if not os.path.exists(save_path):
    os.makedirs(save_path)

logger = Logger(f'{save_path}/logs.logs')
logger.Print(time_string + " - " + args.message + "\n")

def test(data_loader, threshold):
    
    model.eval()

    dist=[]
    dist_noblack=[]

    y_true = []
    y_prob = []
    y_pred = []

    data_high = []
    data_mid = []
    data_low = []

    data_real = []
    data_fake = []

    logger.Print(f"***** << Test threshold({threshold}) >>")  

    for batch, data in enumerate(data_loader, 1):

        size = len(data_loader.dataset)
        rgb_image, label, rgb_path = data
        
        if use_cuda:
            rgb_image.cuda()
            label.cuda()
        else:
            print("Something Wrong, Cuda Not Used")
        
        # 모델 태우기 
        recons_image = model(rgb_image)

        # batch 단위로 되어있는 텐서를 넘파이 배열로 전환 후, 픽셀 값(0~255)으로 전환
        # y_true, y_prod, y_pred 값도 여기서 같이 처리
        for i in range(len(rgb_image)):
            np_image = rgb_image[i].numpy()
            np_recons_image = recons_image[i].cpu().detach().numpy()
            np_image = np_image * 255
            np_recons_image = np_recons_image * 255

            # mse 구하기 
            diff = []
            for d in range(np_image.shape[0]) :        
                val = mean_squared_error(np_image[d].flatten(), np_recons_image[d].flatten())
                diff.append(val)
            mse_by_sklearn = np.array(diff).mean()

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

            # 모델 결과값 
            y_true.append(label[i].numpy())
            y_prob.append(mse_by_sklearn)
            if mse_by_sklearn < threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
                
    # light, mask유무 에 따른 데이터분포 그리기
    if not os.path.exists(f'{save_path}/graph'):
        os.makedirs(f'{save_path}/graph')
    plot_3_kind_data(f"{save_path}/graph", "data_distribution(light)", data_high, data_mid, data_low)       
    plot_real_fake_data(f"{save_path}/graph", "data_distribution(real,fake)", data_fake, data_real)

    # 모델 평가지표 및 그리기 
    plot_roc_curve(f"{save_path}/graph", f"threshold({threshold})", y_true, y_prob)
    
    accuracy, precision, recall = cal_metrics(y_true, y_pred)
    return accuracy, precision, recall

if __name__ == "__main__":
    

    ## Threshold 에 따른 모델 성능 출력 
    train_loader, valid_loader, test_loader = Facedata_Loader(train_size=64, test_size=64)
    print(f"Test data set Size: {len(test_loader)}")

    accuracy = []
    precisioin = []
    recall = []

    threshold = np.arange(300,380,5) 
    for value in threshold:
        result = test(test_loader, value)
        
        logger.Print(f"Threshold: {value}")
        logger.Print(f"accuracy: {result[0]:3f}")
        logger.Print(f"precisioin: {result[1]:3f}")
        logger.Print(f"recall: {result[2]:3f}")

        accuracy.append(f"{result[0]:3f}")
        precisioin.append(f"{result[1]:3f}")
        recall.append(f"{result[2]:3f}")

    if not os.path.exists(f'{save_path}/graph'):
        os.makedirs(f'{save_path}/graph')
    plot_result(f"{save_path}/graph", threshold, accuracy, precisioin, recall)

    ## valid 에 따른? 
    