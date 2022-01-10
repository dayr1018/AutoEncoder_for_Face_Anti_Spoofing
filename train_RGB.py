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

save_path = args.save_path + f'{args.message}' + '_' + f'{time_string}' 
if not os.path.exists(save_path):
    os.makedirs(save_path)

logger = Logger(f'{save_path}/logs.logs')
logger.Print(time_string + " - " + args.message + "\n")

weight_dir = f'../ad_output/RGB/checkpoint/{args.message}'
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

def train(epochs, data_loader, valid_loader):

    model.train()

    epoch_for_valid = []
    threshold_per_epoch = []
    accuracy_per_epoch = []
    precision_per_epoch = []
    recall_per_epoch = []
    f1_per_epoch = []

    for epoch in range(epochs):

        logger.Print(f"***** << Training epoch:{epoch} >>")  

        data_high = []
        data_mid = []
        data_low = []
        
        data_real = []
        data_fake = []
        
        # 데이터 불러오기
        # 데이터 to cuda  
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

            # batch 단위로 되어있는 텐서를 넘파이 배열로 전환 후, 픽셀 값(0~255)으로 전환
            for i in range(len(rgb_image)):
                np_image = rgb_image[i].cpu().detach().numpy()
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
          
        # weight 저장 
        checkpoint = f'{weight_dir}/epoch_{epoch}_model.pth'
        torch.save(model.state_dict(), checkpoint)
        
        # 결과 나타내기 
        # high, mid, low 별로 구분해서 데이터분포 그리기 
        if epoch == 1 or epoch == 1000 or epoch == 2000 or epoch == 2999: 
            if not os.path.exists(f'{save_path}/graph'):
                os.makedirs(f'{save_path}/graph')
            plot_3_kind_data(f"{save_path}/graph", f"Light_Distribution_Epoch_{epoch}_", data_high, data_mid, data_low)
            plot_real_fake_data(f"{save_path}/graph", f"Mask_Distribution_Epoch_{epoch}_", data_real, data_fake)
        
        # epoch 2900 이상인 경우에만 validation 수행
        if(epoch >= 2899):
            thres, accu, prec, reca, f1 = validate(valid_loader, checkpoint)
            # epoch 별 validation 결과(쓰레시홀드) 저장해두기 
            epoch_for_valid.append(epoch)
            threshold_per_epoch.append(thres)
            accuracy_per_epoch.append(accu)
            precision_per_epoch.append(prec)
            recall_per_epoch.append(reca)
            f1_per_epoch.append(f1)
    
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

    accuracy_max = max(accuracy_per_epoch)
    index = accuracy_per_epoch.index(accuracy_max)
    epoch_max = epoch_for_valid[index]
    threshold_max = threshold_per_epoch[index]
    precision_max = precision_per_epoch[index]
    recall_max = recall_per_epoch[index]
    f1_max = f1_per_epoch[index]

    logger.Print(f"***** Result")
    logger.Print(f"***** Max Accuracy: {accuracy_max:3f}")
    logger.Print(f"***** Epoch(=real-1): {epoch_max}")
    logger.Print(f"***** Threshold: {threshold_max:3f}")
    logger.Print(f"***** Precision: {precision_max:3f}")
    logger.Print(f"***** Recall: {recall_max:3f}")
    logger.Print(f"***** F1: {f1_max:3f}")

def validate(valid_loader, checkpoint):
    
    mse = []

    # 현재 checkpoint를 사용한 모델생성
    model = Auto_Encoder()
    if use_cuda:
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))) #  device_ids=[0, 1, 2]
        model.cuda()
        model.load_state_dict(torch.load(checkpoint))
    else:
        print("Something Wrong, Cuda Not Used - validation")        

    # 1. 전체 데이터에 대한 MSE 구하기 
    for i, data in enumerate(valid_loader):
        rgb_image, _, _ = data
        rgb_image = rgb_image.cuda()        

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

    print("MSE Caculation Finished")
    print(f"Max MSE: {max(mse)}, Min MSE: {min(mse)}")
    # 2. MSE 분포에 따른 threshold 리스트 결정 
    threshold = np.arange(min(mse), max(mse), 10)

    # 3. threshold 에 대한 accuracy 구하고 max accuracy에 대한 threshold 리턴 
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for thres in threshold:   

        print(f"***** Threshold: {thres}")
        y_true = []
        y_pred = []

        for _, data in enumerate(valid_loader):
            rgb_image, label, _ = data
            rgb_image = rgb_image.cuda()
            # label = label.cuda()            

            recons_image = model(rgb_image)

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

                # y_true.append(label[i].cpu().detach().numpy())
                y_true.append(label[i])
                if mse_by_sklearn < thres:
                    y_pred.append(1)
                else:
                    y_pred.append(0)

        # 한번 계산을 해주고, 그 계산에 대한 비교 해줘야 함. 
        accu, prec, reca, f = cal_metrics(y_true, y_pred) 
        accuracy.append(accu)
        precision.append(prec)
        recall.append(reca)
        f1.append(f)

    accuracy_max = max(accuracy)
    index = accuracy.index(accuracy_max)

    precision_max = precision[index]
    recall_max = recall[index]
    f1_max = f1[index]
    threhold_max = threshold[index]

    return threhold_max, accuracy_max, precision_max, recall_max, f1_max

if __name__ == "__main__":
    
    train_loader, valid_loader, _ = Facedata_Loader(train_size=64, test_size=64)

    train(args.epochs, train_loader, valid_loader)
