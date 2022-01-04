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
from utils import plot_roc_curve, plot_eval_per_epoch, cal_metrics, plot_3_kind_data

import os

# 모델 생성
# loss 생성 -> MSE loss 사용
# 옵티마이저, 스케줄러 생성

model = Auto_Encoder()
mse = torch.nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

use_cuda = True if torch.cuda.is_available() else False
if use_cuda:
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))) #  device_ids=[0, 1, 2]
    model.cuda()
    mse.cuda()
else:
    print("Something Wrong, Cuda Not Used")
    
parser = argparse.ArgumentParser(description='face anto-spoofing')
parser.add_argument('--batch-size', default='64', type=int, help='train batch size') 
parser.add_argument('--test-size', default='64', type=int, help='test batch size') 
parser.add_argument('--save-path', default='../ad_output/RGB/logs/Train/', type=str, help='logs save path')
parser.add_argument('--checkpoint', default='model.pth', type=str, help='pretrained model checkpoint')
parser.add_argument('--message', default='', type=str, help='pretrained model checkpoint')
parser.add_argument('--epochs', default=3000, type=int, help='train epochs')
parser.add_argument('--train', default=True, type=bool, help='train')
parser.add_argument('--skf', default=0, type=int, help='stratified k-fold')
args = parser.parse_args()

time_object = time.localtime(time.time())
time_string = time.strftime('%Y-%m-%d_%I:%M_%p', time_object)

save_path = args.save_path + f'{time_string}' + '_' + f'{args.message}'
if not os.path.exists(save_path):
    os.makedirs(save_path)

logger = Logger(f'{save_path}/logs.logs')
logger.Print(time_string + " - " + args.message + "\n")

weight_dir = f'../ad_output/RGB/checkpoint/{args.message}'
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

def train(epochs, data_loader):

    model.train()

    y_true = []
    y_prob = []
    # y_pred = []

    # accuracy_list = []
    # precision_list = []
    # recall_list = []
    # epoch_list = []

    for epoch in range(epochs):

        logger.Print(f"***** << Training epoch:{epoch} >>")  

        # 데이터 불러오기
        # 데이터 to cuda  

        data_all = []
        data_high = []
        data_mid = []
        data_low = []
        error = []

        for batch, data in enumerate(data_loader, 0):

            size = len(data_loader.dataset)
            rgb_image, label, rgb_path = data
            
            if use_cuda:
                rgb_image = rgb_image.cuda()
                label = label.cuda()
            else:
                print("Something Wrong, Cuda Not Used")

            # 모델 태우기 
            recons_image = model(rgb_image)

            # loss 불러오기
            loss = mse(rgb_image, recons_image)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss 출력
            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(data[0])
                logger.Print(f"***** loss: {loss:>7f}  [{current:>5d}/{size:>5d}] [{batch}:{len(data[0])}]")

            diff = []
            for d in range(rgb_image.data.cpu().shape[0]) :        
                val = mean_squared_error(rgb_image[d].data.cpu().flatten(), recons_image[d].data.cpu().flatten())
                diff.append(val)
            mse_by_sklearn = np.array(diff).mean()
          
            # light 별로 데이터 분류해서 저장
            for i in range(len(rgb_path)):
                part = rgb_path[i].split('/')[-5]
                data_all.append()
                if "High" in part:
                    data_high.append(mse_by_sklearn)
                elif "Mid" in part:
                    data_mid.append(mse_by_sklearn)
                elif "Low" in part:
                    data_low.append(mse_by_sklearn)
                else:
                    print("Data Classification Error - High, Mid, Low")
                    error.append(rgb_path[i])
    
        # weight 저장 
        ckpt_name = f'{weight_dir}/epoch_{epoch}'
        torch.save(model.state_dict(), f"{ckpt_name}_model.pth")

        # high, mid, low 별로 구분해서 데이터분포 그리기 
        if not os.path.exists(f'{save_path}/data_distribution'):
            os.makedirs(f'{save_path}/data_distribution')
        plot_3_kind_data(f"{save_path}/data_distribution", f"epoch_{epoch}", data_high, data_mid, data_low)
        if len(error) != 0:
            logger.Print("Data Distribution Error List")
            logger.Print(error)        

        # evaluation function   
        # plot_roc_curve(save_path, f"epoch {args.epochs}", y_true, y_prob)
        
if __name__ == "__main__":
    
    train_loader, valid_loader, test_loader = Facedata_Loader(train_size=64, test_size=64)

    train(args.epochs, train_loader)


