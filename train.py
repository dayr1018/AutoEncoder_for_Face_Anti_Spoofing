
# python train.py --depth true --model layer4_4to4 --cuda 0 --gr 50 --dr 0.5 --message 0202_layer4_4to4_g50 

import torch
import torch.nn.init as init
import torch.nn.functional as f
import time
from datetime import datetime
import argparse
from models.Auto_Encoder_RGB import AutoEncoder_Original_layer2, AutoEncoder_Original_layer3, AutoEncoder_Original_layer4, AutoEncoder_Original_layer5, AutoEncoder_Dropout, AutoEncoder_layer4
from models.Auto_Encoder_RGB_Depth import Depth_layer2, Depth_layer3, Depth_layer4, Depth_layer5
from models.Auto_Encoder_RGB_Depth import Depth_layer3_1to1, Depth_layer4_1to1, Depth_layer5_1to1
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import mean_squared_error
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import cv2
from skimage.util import random_noise


from ad_dataloader.dataloader import Facedata_Loader

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

def train(args, train_loader, valid_loader):
    
    # logger 만들고 현재 args 다 기록해두기
    # tensorboard 코드도 여기에 넣기 
    # 모델, 옵티마이저, 스케줄러, loss함수 여기서 만들기

    # logger 
    logger = Logger(f'{args.save_path}/logs.logs')
    logger.Print(args)

    # Tensorboard 
    writer = SummaryWriter(f"runs/{args.message}")

    # RGB 모델
    if "original_layer2" in args.model:
        model = AutoEncoder_Original_layer2().to(args.device)        
        print("***** You're training 'original layer2' model.")
    if "original_layer3" in args.model:
        model = AutoEncoder_Original_layer3().to(args.device)        
        print("***** You're training 'original layer3' model.")
    elif "original_layer4" in args.model:
        model = AutoEncoder_Original_layer4().to(args.device)        
        print("***** You're training 'original layer3' model.")    
    elif "original_layer5" in args.model:
        model = AutoEncoder_Original_layer5().to(args.device)        
        print("***** You're training 'original layer4' model.")
    elif "dropout" in args.model:
        model = AutoEncoder_Dropout(use_drop=args.usedrop, dropout_rate=args.dr).to(args.device)
        print("***** You're training 'dropout' model.")
    elif "layer4" in args.model:
        model = AutoEncoder_layer4(use_drop=args.usedrop, dropout_rate=args.dr).to(args.device)
        print("***** You're training 'layer4' model.")

    # Depth 모델 
    if "depth_layer2" in args.model:
        model = Depth_layer2(use_drop=args.usedrop, dropout_rate=args.dr).to(args.device)
        print("**** You're training 'Depth_layer2' model.")
    elif "depth_layer3" in args.model:
        model = Depth_layer3(use_drop=args.usedrop, dropout_rate=args.dr).to(args.device)
        print("**** You're training 'Depth_layer3' model.")
    elif "depth_layer4" in args.model:
        model = Depth_layer4(use_drop=args.usedrop, dropout_rate=args.dr).to(args.device)
        print("**** You're training 'Depth_layer4' model.")
    elif "depth_layer5" in args.model:
        model = Depth_layer5(use_drop=args.usedrop, dropout_rate=args.dr).to(args.device)
        print("**** You're training 'Depth_layer5' model.")

    # 옵티마이저, 스케줄러
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # loss 함수
    mse = torch.nn.MSELoss(reduction='sum')

    # train 부분 시작
    start = datetime.now()

    thresholds_when_accuracy_criteria = []
    accuracys_when_accuracy_criteria = []
    precisions_when_accuracy_criteria = []
    recalls_when_accuracy_criteria = []
    f1s_when_accuracy_criteria = []
    epochs_when_accuracy_criteria = []

    thresholds_when_f1_criteria = []
    accuracys_when_f1_criteria = []
    precisions_when_f1_criteria = []
    recalls_when_f1_criteria = []
    f1s_when_f1_criteria = []
    epochs_when_f1_criteria = []

    ###### 매 에폭 텐서 고정!!!!
    # # # Gaussian Noise 
    # gaussian_mean = 0
    # gaussian_std = args.gr

    # noise_3channel = init.normal_(torch.zeros(3, 128, 128), gaussian_mean, gaussian_std).to(args.device)
    # noise_1channel = init.normal_(torch.zeros(1, 128, 128), gaussian_mean, gaussian_std).to(args.device)
    # noise_4channel = init.normal_(torch.zeros(4, 128, 128), gaussian_mean, gaussian_std).to(args.device)

    # noise_3channel = f.normalize(noise_3channel, dim=0)
    # noise_4channel = f.normalize(noise_4channel, dim=0)

    for epoch in range(args.epochs):

        logger.Print(f"***** << Training epoch:{epoch} >>")  
        
        model.train()

        data_high = []
        data_mid = []
        data_low = []
        
        data_real = []
        data_fake = []
        
        # 데이터 load & to cuda  
        for batch, data in enumerate(train_loader, 0):

            size = len(train_loader.dataset)
            rgb_image, depth_image, _, _, label, rgb_path = data
            label = label.to(args.device)

            total_image = torch.cat((rgb_image, depth_image), dim=1)

            # RGB
            if args.depth == False:        
                rgb_image = torch.FloatTensor(random_noise(rgb_image, mode='gaussian', mean=0, var=args.gr, clip=True)).to(args.device)    
                standard_image = rgb_image  
                input_image = standard_image
            # Depth
            elif args.depth == True:
                total_image = torch.FloatTensor(random_noise(total_image, mode='gaussian', mean=0, var=args.gr, clip=True)).to(args.device)
                standard_image = total_image
                input_image = standard_image
            

        

            # size = len(train_loader.dataset)
            # rgb_image, depth_image, _, _, label, rgb_path = data
            # rgb_image = torch.FloatTensor(random_noise(rgb_image, mode='gaussian', mean=0, var=args.gr, clip=True))
            # depth_image = torch.FloatTensor(random_noise(depth_image, mode='gaussian', mean=0, var=args.gr, clip=True))
            # rgb_image = rgb_image.to(args.device)
            # depth_image = depth_image.to(args.device)
            # label = label.to(args.device)

            # # RGB
            # if args.depth == False:            
            #     standard_image = rgb_image
            #     input_image = standard_image  
            # # Depth
            # elif args.depth == True:
            #     standard_image = torch.cat((rgb_image, depth_image), dim=1)
            #     input_image = standard_image





            # Gaussian Noise 
            # gaussian_mean = 0
            # gaussian_std = args.gr
#1.
            # ###### 일반적인 정규분포 정규화
            # if args.depth == False:
            #     noise = torch.nn.init.normal_(torch.zeros(len(input_image), 3, 128, 128), gaussian_mean, gaussian_std).to(args.device)
            # elif args.depth == True:
            #     noise = torch.nn.init.normal_(torch.zeros(len(input_image), 4, 128, 128), gaussian_mean, gaussian_std).to(args.device)
            
            # #.1 그냥 했을 때.

            # #.2 abs 했을 때
            # #noise = noise.abs()
            
            # #.3 일반 정규화
            # #noise = torch.nn.functional.normalize(noise)
            
            # #.4 일반 정규화 후 abs
            # # noise = torch.nn.functional.normalize(noise).abs()
            
            # #.5 minmax 정규화 
            # noise = ((noise - noise.min()) / (noise.max() - noise.min()))

#2.
            ##### 매 에폭 텐서 고정!!!!
            # if args.depth == False:
            #     noise = torch.zeros(len(rgb_image), 3, 128, 128).to(args.device)
            #     for i in range(len(rgb_image)):
            #         noise[i] = noise_3channel
            # elif args.depth == True:
            #     noise = torch.zeros(len(rgb_image), 4, 128, 128).to(args.device)
            #     for i in range(len(rgb_image)):
            #         noise[i] = noise_4channel
            # noise = torch.nn.functional.normalize(noise)
            # noise = torch.nn.functional.normalize(noise).abs()
            # noise = (noise - noise.min()) / (noise.max() - noise.min())
            # noise = noise.abs()

#3. 
            ###### 일반적인 정규분포 정규화
            # noise_3 = torch.nn.init.normal_(torch.zeros(len(input_image), 3, 128, 128), gaussian_mean, gaussian_std)
            # noise_1 = torch.nn.init.normal_(torch.zeros(len(input_image), 1, 128, 128), gaussian_mean, gaussian_std)
            # noise_3 = f.normalize(noise_3)
            # noise_1 = f.normalize(noise_1)

            # noise_4 = torch.cat((noise_3, noise_1), dim=1)

            # if args.depth == False:
            #     noise = noise_3.to(args.device)
            # elif args.depth == True:
            #     noise = noise_4.to(args.device)
            
            # #.1 그냥 했을 때.

            # #.2 abs 했을 때
            # #noise = noise.abs()
            
            # #.3 일반 정규화
            # noise = torch.nn.functional.normalize(noise)
            
            # #.4 일반 정규화 후 abs
            # #noise = torch.nn.functional.normalize(noise).abs()
            
            # #.5 minmax 정규화 
            # # noise = ((noise - noise.min()) / (noise.max() - noise.min()))


            # if args.gr != 0:
            #     input_image = input_image + noise

            # 모델 태우기 
            recons_image = model(input_image)
          
            # loss 불러오기
            loss = mse(standard_image, recons_image)

            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/Epoch", loss, epoch)

            # batch 단위로 되어있는 텐서를 넘파이 배열로 전환 후, 픽셀 값(0~255)으로 전환
            mse_list = []
            red = []
            green = []
            blue = []
            depth = []  
            for i in range(len(standard_image)):
                
                # 하나의 텐서에 대한 MSE 값 확인 
                mse_per_tensor = mse(standard_image[i], recons_image[i]).cpu().detach().numpy()
                mse_list.append(mse_per_tensor)                   
                     
                # light 에 따라 데이터 분류하기 
                path = rgb_path[i].split('/')[-5]
                if "High" in path:
                    data_high.append(mse_per_tensor)
                elif "Mid" in path:
                    data_mid.append(mse_per_tensor)
                elif "Low" in path:
                    data_low.append(mse_per_tensor)
                else:
                    logger.Print("Data Classification Error - High, Mid, Low")

                # mask 유무에 따라 데이터 분류하기
                if label[i].item() == 1:
                    data_real.append(mse_per_tensor)
                else:
                    data_fake.append(mse_per_tensor)

                # channel 별 MSE 확인 
                for c in range(len(standard_image[i])) :
                    mse_per_chnnel = mse(standard_image[i][c], recons_image[i][c])
                    if c == 0:
                        red.append(mse_per_chnnel)
                    elif c == 1:
                        green.append(mse_per_chnnel)
                    elif c == 2:
                        blue.append(mse_per_chnnel)
                    elif c == 3:
                        depth.append(mse_per_chnnel) 

            # loss 출력 & 채널별 mse 총합 출력 
            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(data[0])
                logger.Print(f"***** loss (std): {loss:5f}  [{current:>5d}/{size:>5d}]")
                logger.Print(f"***** loss (mse): {np.array(mse_list).sum():5f}")
                logger.Print(f"***** loss (cal): {np.array(red+green+blue+depth).sum():5f}")
                logger.Print(f"***** red: {np.array(red).sum():5f}, green: {np.array(green).sum():5f}, blue: {np.array(blue).sum():5f}, depth: {np.array(depth).sum():5f}")

        scheduler.step()

        if (epoch % 5) == 0 or epoch == (args.epochs-1):
            # validation 수행
            result_when_accuracy_max, result_when_f1_max = valid(args, valid_loader, model, epoch, logger, mse, writer) 
            logger.Print(f"Current Epoch: {epoch}, Accuracy: {result_when_accuracy_max[1]}, F1: {result_when_f1_max[4]}")

            writer.add_scalar("Threshold/Epoch (criteria: accuracy)", result_when_accuracy_max[0], epoch)
            writer.add_scalar("Accuracy/Epoch (criteria: accuracy)", result_when_accuracy_max[1], epoch)
            writer.add_scalar("F1/Epoch (criteria: accuracy)", result_when_accuracy_max[4], epoch)
            
            writer.add_scalar("Threshold/Epoch (criteria: f1)", result_when_f1_max[0], epoch)
            writer.add_scalar("Accuracy/Epoch (criteria: f1)", result_when_f1_max[1], epoch)
            writer.add_scalar("F1/Epoch (criteria: f1)", result_when_f1_max[4], epoch)

            # accuracy 를 기준으로 했을 때
            thresholds_when_accuracy_criteria.append(result_when_accuracy_max[0])
            accuracys_when_accuracy_criteria.append(result_when_accuracy_max[1])
            precisions_when_accuracy_criteria.append(result_when_accuracy_max[2])
            recalls_when_accuracy_criteria.append(result_when_accuracy_max[3])
            f1s_when_accuracy_criteria.append(result_when_accuracy_max[4])
            epochs_when_accuracy_criteria.append(epoch)

            # f1 score를 기준으로 했을 때 
            thresholds_when_f1_criteria.append(result_when_f1_max[0])
            accuracys_when_f1_criteria.append(result_when_f1_max[1])
            precisions_when_f1_criteria.append(result_when_f1_max[2])
            recalls_when_f1_criteria.append(result_when_f1_max[3])
            f1s_when_f1_criteria.append(result_when_f1_max[4])
            epochs_when_f1_criteria.append(epoch)

            # 0, 10, 20, ... 순서대로 weight들 저장   
            checkpoint = f'{args.checkpoint_path}/epoch_{epoch}_model.pth'
            torch.save(model.state_dict(), checkpoint)
            
            # 결과 나타내기 
            plot_3_kind_data(f"{args.save_path}", f"Light_Distribution_Epoch_{epoch}_", epoch, data_high, data_mid, data_low)
            plot_real_fake_data(f"{args.save_path}", f"Mask_Distribution_Epoch_{epoch}_", epoch, data_real, data_fake)
            plot_histogram(f"{args.save_path}", f"Train_Historgram_{epoch}_", epoch, data_real, data_fake)

    # 모든 게 끝났을 때, accuracy를 기준으로 했던 결과들 중 가장 큰 accuracy를 갖는 때  
    max_accuracy = max(accuracys_when_accuracy_criteria)
    ma_index = accuracys_when_accuracy_criteria.index(max_accuracy)
    ma_precision = precisions_when_accuracy_criteria[ma_index]
    ma_recall = recalls_when_accuracy_criteria[ma_index]
    ma_f1 = f1s_when_accuracy_criteria[ma_index]
    ma_threshold = thresholds_when_accuracy_criteria[ma_index]
    ma_epoch = epochs_when_accuracy_criteria[ma_index]

    # 모든 게 끝났을 때, f1 score를 기준으로 했던 결과들 중 가장 큰 f1 score를 갖는 때 
    max_f1 = max(f1s_when_f1_criteria)
    mf_index = f1s_when_f1_criteria.index(max_f1)
    mf_precision = precisions_when_f1_criteria[mf_index]
    mf_recall = recalls_when_f1_criteria[mf_index]
    mf_accuracy = accuracys_when_f1_criteria[mf_index]
    mf_threshold = thresholds_when_f1_criteria[mf_index]
    mf_epoch = epochs_when_f1_criteria[mf_index]

    logger.Print(f"\n***** Total Threshold per epoch (criteria: accuracy)")
    logger.Print(thresholds_when_accuracy_criteria)
    logger.Print(f"***** Total Accuracy per epoch (criteria: accuracy)")
    logger.Print(accuracys_when_accuracy_criteria)
    logger.Print(f"***** Total Precision per epoch (criteria: accuracy)")
    logger.Print(precisions_when_accuracy_criteria)
    logger.Print(f"***** Total Recall per epoch (criteria: accuracy)")
    logger.Print(recalls_when_accuracy_criteria)
    logger.Print(f"***** Total F1 per epoch (criteria: accuracy)")
    logger.Print(f1s_when_accuracy_criteria)
    logger.Print(f"***** Total epoch (criteria: accuracy)")
    logger.Print(epochs_when_accuracy_criteria)

    logger.Print(f"\n***** Total Threshold per epoch (criteria: f1)")
    logger.Print(thresholds_when_f1_criteria)
    logger.Print(f"***** Total Accuracy per epoch (criteria: f1)")
    logger.Print(accuracys_when_f1_criteria)
    logger.Print(f"***** Total Precision per epoch (criteria: f1)")
    logger.Print(precisions_when_f1_criteria)
    logger.Print(f"***** Total Recall per epoch (criteria: f1)")
    logger.Print(recalls_when_f1_criteria)
    logger.Print(f"***** Total F1 per epoch (criteria: f1)")
    logger.Print(f1s_when_f1_criteria)
    logger.Print(f"***** Total epoch (criteria: f1)")
    logger.Print(epochs_when_f1_criteria) 

    logger.Print(f"\n***** Result (criteria: accuracy)")
    logger.Print(f"Accuracy: {max_accuracy:3f}")
    logger.Print(f"Precision: {ma_precision:3f}")
    logger.Print(f"Recall: {ma_recall:3f}")
    logger.Print(f"F1: {ma_f1:3f}")
    logger.Print(f"Epoch: {ma_epoch}")
    logger.Print(f"Threshold: {ma_threshold:3f}")

    logger.Print(f"\n***** Result (criteria: f1)")
    logger.Print(f"Accuracy: {mf_accuracy:3f}")
    logger.Print(f"Precision: {mf_precision:3f}")
    logger.Print(f"Recall: {mf_recall:3f}")
    logger.Print(f"F1: {max_f1:3f}")
    logger.Print(f"Epoch: {mf_epoch}")
    logger.Print(f"Threshold: {mf_threshold:3f}")

    writer.close()

def valid(args, valid_loader, model, epoch, logger, loss_function, writer):

    model.eval()

    # 1. 전체 데이터에 대한 MSE 구하기 
    mse_total = []

    y_true = []
    # y_pred = []

    data_high = []
    data_mid = []
    data_low = []
    
    data_real = []
    data_fake = []
    
    with torch.no_grad():
        for _, data in enumerate(valid_loader):

            rgb_image, depth_image, _, _, label, rgb_path = data
            rgb_image = rgb_image.to(args.device)        
            depth_image = depth_image.to(args.device)
            label = label.to(args.device)

            # RGB
            if args.depth == False:            
                standard_image = rgb_image
                input_image = standard_image  
            # Depth
            elif args.depth == True:
                standard_image = torch.cat((rgb_image, depth_image), dim=1)
                input_image = standard_image

            # 모델 태우기 
            recons_image = model(input_image)

            # Validation 할 때 loss 값 
            loss_valid = loss_function(standard_image, recons_image)
            writer.add_scalar("Loss/Epoch (Valid)", loss_valid, epoch)

            # 모든 데이터에 대한 MSE 구하기            
            for i in range(len(standard_image)):

                # 하나의 텐서에 대한 MSE 값 확인 
                mse_per_tensor = loss_function(standard_image[i], recons_image[i]).cpu().detach().numpy()
                mse_total.append(mse_per_tensor)  
                
                # light 에 따라 데이터 분류하기 
                path = rgb_path[i].split('/')[-5]
                if "High" in path:
                    data_high.append(mse_per_tensor)
                elif "Mid" in path:
                    data_mid.append(mse_per_tensor)
                elif "Low" in path:
                    data_low.append(mse_per_tensor)
                else:
                    logger.Print("Data Classification Error - High, Mid, Low")

                # mask 유무에 따라 데이터 분류하기
                if label[i].item() == 1:
                    data_real.append(mse_per_tensor)
                else:
                    data_fake.append(mse_per_tensor)

                # y_true 는 넣어.
                y_true.append(label[i].cpu().detach().numpy())

    print("------ MSE Caculation Finished")
    logger.Print(f"------ Max MSE: {max(mse_total)}, Min MSE: {min(mse_total)}")

    # 2. MSE 분포에 따른 threshold 리스트 결정 
    threshold = np.arange(np.floor(min(mse_total))+10, np.floor(max(mse_total))-10, 1)

    # 3. threshold 에 대한 accuracy 구하고 max accuracy에 대한 threshold 리턴 
    accuracy_per_thres = []
    precision_per_thres = []
    recall_per_thres = []
    f1_per_thres = []
    for thres in threshold:   

        y_pred = []
        for i in range(len(mse_total)):
            if mse_total[i] < thres:
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

    # Accuracy 가 최대일 때의 일련의 결과값 
    max_accuracy = max(accuracy_per_thres)
    ma_index = accuracy_per_thres.index(max_accuracy)
    ma_precision = precision_per_thres[ma_index]
    ma_recall = recall_per_thres[ma_index]
    ma_f1 = f1_per_thres[ma_index]
    ma_threshold = threshold[ma_index]

    result_when_accuracy_max = [ma_threshold, max_accuracy, ma_precision, ma_recall, ma_f1]

    # F1 score 가 최대일 때의 일련의 결과값
    max_f1 = max(f1_per_thres)
    mf_index = f1_per_thres.index(max_f1)
    mf_precision = precision_per_thres[mf_index]
    mf_recall = recall_per_thres[mf_index]
    mf_accuracy = accuracy_per_thres[mf_index]
    mf_threshold = threshold[mf_index]

    result_when_f1_max = [mf_threshold, mf_accuracy, mf_precision, mf_recall, max_f1]

    # 데이터 분포도 그래프로 그리기  
    plot_3_kind_data(args.save_path_valid, f"Light_Distribution_Epoch_{epoch}", epoch, data_high, data_mid, data_low)
    plot_real_fake_data(args.save_path_valid, f"Mask_Distribution_Epoch_{epoch}", epoch, data_real, data_fake)
    plot_histogram(f"{args.save_path_valid}", f"Valid_Historgram_{epoch}_", epoch, data_real, data_fake)

    return result_when_accuracy_max, result_when_f1_max


if __name__ == "__main__":

    # args option
    parser = argparse.ArgumentParser(description='face anto-spoofing')

    parser.add_argument('--save-path', default='../ad_output/logs/Train/', type=str, help='train logs path')
    parser.add_argument('--save-path-valid', default='', type=str, help='valid logs path')

    parser.add_argument('--model', default='', type=str, help='model')                                              # essential
    parser.add_argument('--checkpoint-path', default='', type=str, help='checkpoint path')
    parser.add_argument('--depth', default=True, type=booltype, help='RGB or Depth(default: Depth)')                # essential

    parser.add_argument('--message', default='', type=str, help='pretrained model checkpoint')                      # essential
    parser.add_argument('--epochs', default=300, type=int, help='train epochs')                                     # essential
    parser.add_argument('--lowdata', default=True, type=booltype, help='whether low data is included')
    parser.add_argument('--usedrop', default=False, type=booltype, help='whether dropout layer is used')            
    parser.add_argument('--dataset', default=0, type=int, help='data set type')
    parser.add_argument('--loss', default=0, type=int, help='0: mse, 1:rapp')
    
    parser.add_argument('--gr', default=0.01, type=float, help='gaussian rate(default: 0.01)')
    parser.add_argument('--dr', default=0.5, type=float, help='dropout rate(default: 0.1)')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate(default: 0.001)')
    parser.add_argument('--skf', default=0, type=int, help='stratified k-fold')

    parser.add_argument('--seed', default=1, type=int, help='Seed for random number generator')
    parser.add_argument('--cuda', default=0, type=int, help='gpu number')                                           # essential
    parser.add_argument('--device', default='', type=str, help='device when cuda is available')

    args = parser.parse_args()

    # 결과 파일 path 설정 
    time_string = time.strftime('%Y-%m-%d_%I:%M_%p', time.localtime(time.time()))
    args.save_path = args.save_path + f'{args.message}' + '_' + f'{time_string}'  
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    args.save_path_valid = args.save_path + '/valid'
    if not os.path.exists(args.save_path_valid): os.makedirs(args.save_path_valid)

    # weight 파일 path
    args.checkpoint_path = f'../ad_output/checkpoint/{args.message}'
    if not os.path.exists(args.checkpoint_path): os.makedirs(args.checkpoint_path)

    # cuda 관련 코드
    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda : 
        if args.cuda == 0:
            args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        elif args.cuda == 1:
            args.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print('device :', args.device)
    
    # random 요소 없애기 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    # data loader
    train_loader, valid_loader, _ = Facedata_Loader(train_size=64, test_size=64, use_lowdata=args.lowdata, dataset=args.dataset, gaussian_radius=args.gr)
    
    # train 코드
    train(args, train_loader, valid_loader)
    
    



