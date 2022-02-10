import torch
import time
from datetime import datetime
import argparse
from models.Auto_Encoder_RGB import AutoEncoder_Original, AutoEncoder_Dropout, AutoEncoder_layer4
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import mean_squared_error
import numpy as np
import torch 
from torch.utils.tensorboard import SummaryWriter

from ad_dataloader.dataloader_RGB import Facedata_Loader_RGB
from loger import Logger
from utility import plot_roc_curve, cal_metrics, plot_3_kind_data, plot_real_fake_data, plot_histogram

import os

def booltype(str):
    if isinstance(str, bool):
        return str
    if str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentError("Boolean value expected")

parser = argparse.ArgumentParser(description='face anto-spoofing')
parser.add_argument('--save-path', default='../ad_output/logs/Train/', type=str, help='logs save path')
parser.add_argument('--checkpoint', default='model.pth', type=str, help='pretrained model checkpoint')
parser.add_argument('--message', default='', type=str, help='pretrained model checkpoint')
parser.add_argument('--model', default='', type=str, help='model')
parser.add_argument('--epochs', default=3000, type=int, help='train epochs')
parser.add_argument('--lowdata', default=True, type=booltype, help='whether low data is included')
parser.add_argument('--skf', default=0, type=int, help='stratified k-fold')
args = parser.parse_args()

time_object = time.localtime(time.time())
time_string = time.strftime('%Y-%m-%d_%I:%M_%p', time_object)

save_path = args.save_path + f'{args.message}' + '_' + f'{time_string}' 
save_path_valid = save_path + '/valid'
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(save_path_valid):
    os.makedirs(save_path_valid)

logger = Logger(f'{save_path}/logs.logs')
logger.Print(time_string + " - " + args.message + "\n")

weight_dir = f'../ad_output/checkpoint/{args.message}'
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

# 모델 생성
# loss 생성 -> MSE loss 사용
# 옵티마이저, 스케줄러 생성

if "original" in args.model:
    model = AutoEncoder_Original()        
    print("***** You're training 'original' model.")
elif "dropout" in args.model:
    model = AutoEncoder_Dropout()
    print("***** You're training 'dropout' model.")
elif "layer4" in args.model:
    model = AutoEncoder_layer4()
    print("***** You're training 'layer4' model.")

mse = torch.nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

use_cuda = True if torch.cuda.is_available() else False
if use_cuda : 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('device :', device)

writer = SummaryWriter(f"runs/{args.message}")

def train(epochs, data_loader, valid_loader):
    start = datetime.now()

    threshold_per_epoch = []
    accuracy_per_epoch = []
    precision_per_epoch = []
    recall_per_epoch = []
    f1_per_epoch = []
    epoch_list = []

    for epoch in range(epochs):

        model.train()

        logger.Print(f"***** << Training epoch:{epoch} >>")  

        data_high = []
        data_mid = []
        data_low = []
        
        data_real = []
        data_fake = []
        
        # 데이터 load & to cuda  
        for batch, data in enumerate(data_loader, 0):

            size = len(data_loader.dataset)
            rgb_image, label, rgb_path = data
            rgb_image = rgb_image.to(device)
            label = label.to(device)

            # 가우시안 노이즈 추가 
            if "original" in args.model:
                input_image = rgb_image   
            elif "dropout" in args.model:
                gaussian_ratio = 100
                input_image = rgb_image + torch.clip(torch.randn_like(rgb_image), 0, 1) * gaussian_ratio  
                # input_image = rgb_image 



            # 모델 태우기 
            recons_image = model(input_image)

            # loss 불러오기
            loss = mse(rgb_image, recons_image)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/Epoch", loss, epoch)

            # loss 출력
            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(data[0])
                logger.Print(f"***** loss_py_: {loss:5f}  [{current:>5d}/{size:>5d}]")

            loss_by_cal = []
            # batch 단위로 되어있는 텐서를 넘파이 배열로 전환 후, 픽셀 값(0~255)으로 전환
            for i in range(len(rgb_image)):
                np_image = rgb_image[i].cpu().detach().numpy()
                np_recons_image = recons_image[i].cpu().detach().numpy()

                # mse 구하기 
                diff = []
                for d in range(np_image.shape[0]) :        
                    val = mean_squared_error(np_image[d].flatten(), np_recons_image[d].flatten())
                    val = 128 * 128 * val
                    diff.append(val)
                mse_by_sklearn = np.array(diff).sum()                
     
                # light 에 따라 데이터 분류하기 
                path = rgb_path[i].split('/')[-5]
                if "High" in path:
                    data_high.append(mse_by_sklearn)
                elif "Mid" in path:
                    data_mid.append(mse_by_sklearn)
                elif "Low" in path:
                    data_low.append(mse_by_sklearn)
                else:
                    logger.Print("Data Classification Error - High, Mid, Low")

                # mask 유무에 따라 데이터 분류하기
                if label[i].item() == 1:
                    data_real.append(mse_by_sklearn)
                else:
                    data_fake.append(mse_by_sklearn)
                
                loss_by_cal.append(mse_by_sklearn)

            if batch % 10 == 0:
                logger.Print(f"***** loss_sc_: {np.array(loss_by_cal).sum():5f}")

        if (epoch % 10) == 0 or epoch == (epochs-1):
            # validation 수행
            threshold, accuracy, precision, recall, f1 = valid(valid_loader, epoch, epochs) 
            logger.Print(f"Current Epoch: {epoch}, Accuracy: {accuracy}")

            writer.add_scalar("Threshold/Epoch", threshold, epoch)
            writer.add_scalar("Accuracy/Epoch", accuracy, epoch)
            
            threshold_per_epoch.append(threshold)
            accuracy_per_epoch.append(accuracy)
            precision_per_epoch.append(precision)
            recall_per_epoch.append(recall)
            f1_per_epoch.append(f1)
            epoch_list.append(epoch)

            # 0, 10, 20, ... 순서대로 weight들 저장   
            checkpoint = f'{weight_dir}/epoch_{epoch}_model.pth'
            torch.save(model.state_dict(), checkpoint)
            
            # 결과 나타내기 
            plot_3_kind_data(f"{save_path}", f"Light_Distribution_Epoch_{epoch}_", epoch, data_high, data_mid, data_low)
            plot_real_fake_data(f"{save_path}", f"Mask_Distribution_Epoch_{epoch}_", epoch, data_real, data_fake)
            plot_histogram(f"{save_path}", f"Train_Historgram_{epoch}_", epoch, data_real, data_fake)

    # 모든 게 끝났을 때, epoch 이 언제일 때 가장 큰 accuracy를 갖는지 확인 
    accuracy_max = max(accuracy_per_epoch)
    index = accuracy_per_epoch.index(accuracy_max)
    threshold_max = threshold_per_epoch[index]
    precision_max = precision_per_epoch[index]
    recall_max = recall_per_epoch[index]
    f1_max = f1_per_epoch[index]
    epoch_max = epoch_list[index]

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
    logger.Print(epoch_list)

    logger.Print(f"***** Result")
    logger.Print(f"Accuracy: {accuracy_max:3f}")
    logger.Print(f"Precision: {precision_max:3f}")
    logger.Print(f"Recall: {recall_max:3f}")
    logger.Print(f"F1: {f1_max:3f}")
    logger.Print(f"Epoch: {epoch_max}")
    logger.Print(f"Threshold: {threshold_max:3f}")


def valid(valid_loader, epoch, epochs):

    model.eval()

    # 1. 전체 데이터에 대한 MSE 구하기 
    mse_list = []

    y_true = []
    # y_pred = []

    data_high = []
    data_mid = []
    data_low = []
    
    data_real = []
    data_fake = []

    with torch.no_grad():
        for _, data in enumerate(valid_loader):
            rgb_image, label, rgb_path = data
            rgb_image = rgb_image.to(device)        
            label = label.to(device)

            recons_image = model(rgb_image)

            # 모든 데이터에 대한 MSE 구하기 
            for i in range(len(rgb_image)):
                np_image = rgb_image[i].cpu().detach().numpy()
                np_recons_image = recons_image[i].cpu().detach().numpy()

                diff = []
                for d in range(np_image.shape[0]) :        
                    val = mean_squared_error(np_image[d].flatten(), np_recons_image[d].flatten())
                    val = 128 * 128 * val
                    diff.append(val)
                mse_by_sklearn = np.array(diff).sum() 
                mse_list.append(mse_by_sklearn)

                # light 에 따라 데이터 분류하기 
                path = rgb_path[i].split('/')[-5]
                if "High" in path:
                    data_high.append(mse_by_sklearn)
                elif "Mid" in path:
                    data_mid.append(mse_by_sklearn)
                elif "Low" in path:
                    data_low.append(mse_by_sklearn)
                else:
                    logger.Print("------ Data Classification Error - High, Mid, Low")

                # mask 유무에 따라 데이터 분류하기
                if label[i].item() == 1:
                    data_real.append(mse_by_sklearn)
                else:
                    data_fake.append(mse_by_sklearn)

                # y_true 는 넣어.
                y_true.append(label[i].cpu().detach().numpy())

    print("------ MSE Caculation Finished")
    logger.Print(f"------ Max MSE: {max(mse_list)}, Min MSE: {min(mse_list)}")

    # 2. MSE 분포에 따른 threshold 리스트 결정 
    threshold = np.arange(round(min(mse_list), -1)+10, round(max(mse_list), -1)-10, 10)

    # 3. threshold 에 대한 accuracy 구하고 max accuracy에 대한 threshold 리턴 
    accuracy_per_thres = []
    precision_per_thres = []
    recall_per_thres = []
    f1_per_thres = []
    for thres in threshold:   

        y_pred = []
        for i in range(len(mse_list)):
            if mse_list[i] < thres:
                y_pred.append(1)
            else:
                y_pred.append(0)

        if len(y_true) != len(y_pred):
            logger.Print("------ length error - y_true, y_pred")

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
    threshold_max = threshold[index]

    # 데이터 분포도 그래프로 그리기 
    plot_3_kind_data(save_path_valid, f"Light_Distribution_Epoch_{epoch}_", epoch, data_high, data_mid, data_low)
    plot_real_fake_data(save_path_valid, f"Mask_Distribution_Epoch_{epoch}_", epoch, data_real, data_fake)
    plot_histogram(f"{save_path_valid}", f"Valid_Historgram_{epoch}_", epoch, data_real, data_fake)

    return threshold_max, accuracy_max, precision_max, recall_max, f1_max

if __name__ == "__main__":

    train_loader, valid_loader, _ = Facedata_Loader_RGB(train_size=64, test_size=64, use_lowdata=args.lowdata)

    train(args.epochs, train_loader, valid_loader)

    writer.close()