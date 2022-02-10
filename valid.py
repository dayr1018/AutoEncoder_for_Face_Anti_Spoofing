# python valid.py --depth true --model layer4_4to4 --checkpoint_path 0202_layer4_4to4_g50 --message valid_0202_layer4_4to4_g50 

import torch
import time
from datetime import datetime
import argparse
from models.Auto_Encoder_RGB import AutoEncoder_Original, AutoEncoder_Dropout, AutoEncoder_layer4
from models.Auto_Encoder_RGB_Depth import Depth_layer3, Depth_layer4, Depth_layer5
from models.Auto_Encoder_RGB_Depth import Depth_layer3_1to1, Depth_layer4_1to1, Depth_layer5_1to1
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import mean_squared_error
import numpy as np
import torch 
from torch.utils.tensorboard import SummaryWriter

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

def main(args, valid_loader):

    # logger 
    logger = Logger(f'{args.save_path_valid}/logs.logs')
    logger.Print(args)

    # Tensorboard 
    writer = SummaryWriter(f"runs/{args.message}")

    # RGB 모델
    if "original" in args.model:
        model = AutoEncoder_Original().to(args.device)        
        print("***** You're training 'original' model.")
    elif "dropout" in args.model:
        model = AutoEncoder_Dropout(use_drop=args.usedrop, dropout_rate=args.dr).to(args.device)
        print("***** You're training 'dropout' model.")
    elif "layer4" in args.model:
        model = AutoEncoder_layer4(use_drop=args.usedrop, dropout_rate=args.dr).to(args.device)
        print("***** You're training 'layer4' model.")

    # Depth 모델 
    if "depth_layer3" in args.model:
        model = Depth_layer3(use_drop=args.usedrop, dropout_rate=args.dr).to(args.device)
        print("**** You're training 'Depth_layer3' model.")
    elif "depth_layer4" in args.model:
        model = Depth_layer4(use_drop=args.usedrop, dropout_rate=args.dr).to(args.device)
        print("**** You're training 'Depth_layer4' model.")
    elif "depth_layer5" in args.model:
        model = Depth_layer5(use_drop=args.usedrop, dropout_rate=args.dr).to(args.device)
        print("**** You're training 'Depth_layer5' model.") 

    # loss 함수 - valid loss확인하기 위해 
    mse = torch.nn.MSELoss(reduction='sum')

    # valid 부분 시작
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

    model.eval()

    for epoch in range(args.epochs):

        if (epoch % 10) == 0 or epoch == (args.epochs-1):
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
    
    checkpoint=f"{args.checkpoint_path}/epoch_{epoch}_model.pth"
    model.load_state_dict(torch.load(checkpoint))
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

            rgb_image, depth_image, label, rgb_path = data
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

    parser.add_argument('--save-path-valid', default='../ad_output/logs/Valid/', type=str, help='valid logs path')

    parser.add_argument('--model', default='', type=str, help='model')
    parser.add_argument('--checkpoint-path', default='', type=str, help='checkpoint path')
    parser.add_argument('--depth', default=True, type=booltype, help='RGB or Depth(default: Depth)')

    parser.add_argument('--message', default='', type=str, help='pretrained model checkpoint')
    parser.add_argument('--epochs', default=1000, type=int, help='train epochs')
    parser.add_argument('--lowdata', default=True, type=booltype, help='whether low data is included')
    parser.add_argument('--usedrop', default=True, type=booltype, help='whether dropout layer is used')
    parser.add_argument('--datatype', default=0, type=int, help='data set type')
    parser.add_argument('--loss', default=0, type=int, help='0: mse, 1:rapp')
    
    parser.add_argument('--gr', default=0, type=float, help='guassian rate(default: 0)')
    parser.add_argument('--dr', default=0.5, type=float, help='dropout rate(default: 0.1)')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate(default: 0.001)')
    parser.add_argument('--skf', default=0, type=int, help='stratified k-fold')

    parser.add_argument('--seed', default=1, type=int, help='Seed for random number generator')
    parser.add_argument('--cuda', default=0, type=int, help='gpu number')    
    parser.add_argument('--device', default='', type=str, help='device when cuda is available')

    args = parser.parse_args()

    # 결과 파일 path 설정 
    time_string = time.strftime('%Y-%m-%d_%I:%M_%p', time.localtime(time.time()))
    args.save_path_valid = args.save_path_valid + f'{args.message}' + '_' + f'{time_string}'  
    if not os.path.exists(args.save_path_valid): os.makedirs(args.save_path_valid)

    # weight 파일 path
    args.checkpoint_path = f'../ad_output/checkpoint/{args.checkpoint_path}'
    if not os.path.exists(args.checkpoint_path): os.makedirs(args.checkpoint_path)

    # cuda 관련 코드
    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda : 
        if args.cuda == 0:
            args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        elif args.cuda == 1:
            args.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print('device :', args.device)
    
    # torch seed 정하기 
    torch.manual_seed(args.seed)

    # data loader
    _, valid_loader, _ = Facedata_Loader(train_size=64, test_size=64, use_lowdata=args.lowdata, dataset=args.datatype)
    
    # main 함수
    main(args, valid_loader)

