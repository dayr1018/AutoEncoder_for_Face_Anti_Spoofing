
# python train.py --depth true --model layer4_4to4 --cuda 0 --gr 50 --dr 0.5 --message 0202_layer4_4to4_g50 

import torch
import torch.nn.init as init
import torch.nn.functional as f
import torch.optim as optim
from torch.optim import lr_scheduler
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from models.AutoEncoder import AutoEncoder_RGB, AutoEncoder_Depth
from models.AutoEncoder import AutoEncoder_Intergrated_Basic, AutoEncoder_Intergrated_Proposed
from ad_dataloader.dataloader import Facedata_Loader

import numpy as np
import random
import cv2
import time
from datetime import datetime
import argparse
from loger import Logger
import os
import sys

from sklearn.metrics import mean_squared_error
from utility import plot_roc_curve, cal_metrics, plot_3_kind_data, plot_real_fake_data, plot_histogram
from skimage.util import random_noise


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

    # Tensorboard 
    global writer    
    writer = SummaryWriter(f"runs/{args.message}")

    # args 출력 
    logger.Print(args)

    # 모델 선택 
    if "rgb" in args.model:
        model = AutoEncoder_RGB(args.layer, args.batchnorm, args.dr).to(args.device)
    elif "depth" in args.model:
        model = AutoEncoder_Depth(args.layer, args.batchnorm, args.dr).to(args.device)
    elif "both" in args.model:
        model = AutoEncoder_Intergrated_Basic(args.layer, args.batchnorm, args.dr).to(args.device)
    elif "proposed" in args.model:
        model = AutoEncoder_Intergrated_Proposed(3, 5, args.batchnorm, args.dr).to(args.device)
    else:
        print("args.model is not correct")
        sys.exit(0)


    # Model 아키텍쳐 출력 
    summary(model)

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

            rgb_image, depth_image, label, rgb_path = data

            # 가우시안 노이즈 세팅 
            if args.model == "rgb":
                if args.gr != 0:
                    rgb_image = torch.FloatTensor(random_noise(rgb_image, mode='gaussian', mean=0, var=args.gr, clip=True))
                input_image = rgb_image    

            elif args.model == "depth":
                if args.gr != 0:
                    depth_image = torch.FloatTensor(random_noise(depth_image, mode='gaussian', mean=0, var=args.gr, clip=True))
                input_image = depth_image   

            elif args.model == "both":
                if args.gr != 0:
                    rgb_image = torch.FloatTensor(random_noise(rgb_image, mode='gaussian', mean=0, var=args.gr, clip=True))
                    depth_image = torch.FloatTensor(random_noise(depth_image, mode='gaussian', mean=0, var=args.gr, clip=True))
                input_image = torch.cat((rgb_image, depth_image), dim=1)  

            elif args.model == "proposed":
                if args.gr != 0:
                    rgb_image = torch.FloatTensor(random_noise(rgb_image, mode='gaussian', mean=0, var=args.gr, clip=True))
                input_image = torch.cat((rgb_image, depth_image), dim=1)  

            # 텐서화 
            rgb_image = rgb_image.to(args.device)
            depth_image = depth_image.to(args.device)
            label = label.to(args.device)
            input_image = input_image.to(args.device)

            # 모델 태우기
            if args.model == "rgb":
                recons_image = model(rgb_image)
            elif args.model == "depth":
                recons_image = model(depth_image)
            else:
                recons_image = model(rgb_image, depth_image)
          
            # loss 불러오기
            loss = mse(input_image, recons_image)

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
            for i in range(len(input_image)):
                
                # 하나의 텐서에 대한 MSE 값 확인 
                mse_per_tensor = mse(input_image[i], recons_image[i]).cpu().detach().numpy()
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
                for c in range(len(input_image[i])) :
                    mse_per_chnnel = mse(input_image[i][c], recons_image[i][c])
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
                logger.Print(f"***** loss (std): {loss:5f}  [{current:>5d}/{len(train_loader.dataset):>5d}]")
                logger.Print(f"***** loss (mse): {np.array(mse_list).sum():5f}")
                logger.Print(f"***** loss (cal): {np.array(red+green+blue+depth).sum():5f}")
                logger.Print(f"***** red: {np.array(red).sum():5f}, green: {np.array(green).sum():5f}, blue: {np.array(blue).sum():5f}, depth: {np.array(depth).sum():5f}")

        scheduler.step()

        if (epoch % 5) == 0 or epoch == (args.epochs-1):
            # validation 수행
            result_when_accuracy_max = valid(args, valid_loader, model, epoch) 
            logger.Print(f"Current Epoch: {epoch}, Accuracy: {result_when_accuracy_max[1]}")

            writer.add_scalar("Threshold/Epoch (criteria: accuracy)", result_when_accuracy_max[0], epoch)
            writer.add_scalar("Accuracy/Epoch (criteria: accuracy)", result_when_accuracy_max[1], epoch)
            writer.add_scalar("F1/Epoch (criteria: accuracy)", result_when_accuracy_max[4], epoch)

            # accuracy 를 기준으로 했을 때
            thresholds_when_accuracy_criteria.append(result_when_accuracy_max[0])
            accuracys_when_accuracy_criteria.append(result_when_accuracy_max[1])
            precisions_when_accuracy_criteria.append(result_when_accuracy_max[2])
            recalls_when_accuracy_criteria.append(result_when_accuracy_max[3])
            f1s_when_accuracy_criteria.append(result_when_accuracy_max[4])
            epochs_when_accuracy_criteria.append(epoch)

            # 0, 5, 10, ... 순서대로 weight들 저장   
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

    logger.Print(f"\n***** Result (Valid)")
    logger.Print(f"Accuracy: {max_accuracy:3f}")
    logger.Print(f"Precision: {ma_precision:3f}")
    logger.Print(f"Recall: {ma_recall:3f}")
    logger.Print(f"F1: {ma_f1:3f}")
    logger.Print(f"Epoch: {ma_epoch}")
    logger.Print(f"Threshold: {ma_threshold:3f}")

    writer.close()

    return ma_epoch, ma_threshold

def valid(args, valid_loader, model, epoch):

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
    
    mse = torch.nn.MSELoss(reduction='sum')

    with torch.no_grad():
        for _, data in enumerate(valid_loader):

            rgb_image, depth_image, label, rgb_path = data
            rgb_image = rgb_image.to(args.device)        
            depth_image = depth_image.to(args.device)
            label = label.to(args.device)

            # input_iamge 세팅 
            if args.model == "rgb":
                input_image = rgb_image    
            elif args.model == "depth":
                input_image = depth_image   
            elif args.model == "both":
                input_image = torch.cat((rgb_image, depth_image), dim=1)  
            elif args.model == "proposed":
                input_image = torch.cat((rgb_image, depth_image), dim=1)  

            # 모델 태우기
            if args.model == "rgb":
                recons_image = model(rgb_image)
            elif args.model == "depth":
                recons_image = model(depth_image)
            else:
                recons_image = model(rgb_image, depth_image)

            # Validation 할 때 loss 값 
            loss_valid = mse(input_image, recons_image)
            writer.add_scalar("Loss/Epoch (Valid)", loss_valid, epoch)

            # 모든 데이터에 대한 MSE 구하기            
            for i in range(len(input_image)):

                # 하나의 텐서에 대한 MSE 값 확인 
                mse_per_tensor = mse(input_image[i], recons_image[i]).cpu().detach().numpy()
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

    # 데이터 분포도 그래프로 그리기  
    plot_3_kind_data(args.save_path_valid, f"Light_Distribution_Epoch_{epoch}", epoch, data_high, data_mid, data_low)
    plot_real_fake_data(args.save_path_valid, f"Mask_Distribution_Epoch_{epoch}", epoch, data_real, data_fake)
    plot_histogram(f"{args.save_path_valid}", f"Valid_Historgram_{epoch}_", epoch, data_real, data_fake)

    return result_when_accuracy_max


def test(args, test_loader, weight_path, threshold):

    global logger

    # 모델 선택 
    if "rgb" in args.model:
        model = AutoEncoder_RGB(args.layer, args.batchnorm, args.dr).to(args.device)
    elif "depth" in args.model:
        model = AutoEncoder_Depth(args.layer, args.batchnorm, args.dr).to(args.device)
    elif "both" in args.model:
        model = AutoEncoder_Intergrated_Basic(args.layer, args.batchnorm, args.dr).to(args.device)
    elif "proposed" in args.model:
        model = AutoEncoder_Intergrated_Proposed(3, 5, args.batchnorm, args.dr).to(args.device)
    else:
        print("args.model is not correct")
        sys.exit(0)


    model.load_state_dict(torch.load(weight_path))
    model.eval()

    # 1. 전체 데이터에 대한 MSE 구하기 
    mse_total = []

    y_true = []
    y_prob = []
    y_pred = []

    data_high = []
    data_mid = []
    data_low = []

    data_real = []
    data_fake = []

    mse = torch.nn.MSELoss(reduction='sum')

    with torch.no_grad():
        for data in test_loader:

            rgb_image, depth_image, label, rgb_path = data
            rgb_image = rgb_image.to(args.device)        
            depth_image = depth_image.to(args.device)
            label = label.to(args.device)

            # input_iamge 세팅 
            if args.model == "rgb":
                input_image = rgb_image    
            elif args.model == "depth":
                input_image = depth_image   
            elif args.model == "both":
                input_image = torch.cat((rgb_image, depth_image), dim=1)  
            elif args.model == "proposed":
                input_image = torch.cat((rgb_image, depth_image), dim=1)  

            # 모델 태우기
            if args.model == "rgb":
                recons_image = model(rgb_image)
            elif args.model == "depth":
                recons_image = model(depth_image)
            else:
                recons_image = model(rgb_image, depth_image)


            # 모든 데이터에 대한 MSE 구하기            
            for i in range(len(input_image)):

                # 하나의 텐서에 대한 MSE 값 확인 
                mse_per_tensor = mse(input_image[i], recons_image[i]).cpu().detach().numpy() 
                
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

                # 모델 결과값 
                y_true.append(label[i].cpu().detach().numpy())
                y_prob.append(mse_per_tensor)
                if mse_per_tensor < threshold:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
                
    # light, mask유무 에 따른 데이터분포 그리기
    if not os.path.exists(f'{args.save_path_test}/graph'):
        os.makedirs(f'{args.save_path_test}/graph')
    plot_3_kind_data(f"{args.save_path_test}/graph", "data_distribution(light)", "x", data_high, data_mid, data_low)       
    plot_real_fake_data(f"{args.save_path_test}/graph", "data_distribution(real,fake)", "x", data_real, data_fake)
    plot_histogram(f"{args.save_path_test}/graph", f"Test_Historgram_", "x", data_real, data_fake)

    # 모델 평가지표 및 그리기 
    plot_roc_curve(f"{args.save_path_test}/graph", f"threshold({threshold})", y_true, y_prob)
    
    accuracy, precision, recall, f1 = cal_metrics(y_true, y_pred)

    ## 결과 파일 따로 저장 
    logger.Print(f"***** Result (Test)")
    logger.Print(f"Accuracy: {accuracy:3f}")
    logger.Print(f"Precisioin: {precision:3f}")
    logger.Print(f"Recall: {recall:3f}")
    logger.Print(f"F1: {f1:3f}")
    logger.Print(f"Threshold: {threshold}")


if __name__ == "__main__":

    # args option
    parser = argparse.ArgumentParser(description='face anto-spoofing')

    parser.add_argument('--save-path', default='../ad_output/logs/Train/', type=str, help='train logs path')
    parser.add_argument('--save-path-valid', default='', type=str, help='valid logs path')
    parser.add_argument('--save-path-test', default='../ad_output/logs/Test/', type=str, help='test logs path')

    parser.add_argument('--model', default='', type=str, help='rgb, depth, both, proposed')                                              # essential
    parser.add_argument('--checkpoint-path', default='', type=str, help='checkpoint path')
    parser.add_argument('--layer', default=4, type=int, help='number of layers (default: 4)')                # essential

    parser.add_argument('--message', default='', type=str, help='pretrained model checkpoint')                      # essential
    parser.add_argument('--epochs', default=300, type=int, help='train epochs')                                     # essential
    parser.add_argument('--lowdata', default=True, type=booltype, help='whether low data is included')
    parser.add_argument('--dataset', default=0, type=int, help='data set type')
    parser.add_argument('--loss', default=0, type=int, help='0: mse, 1:rapp')
    
    parser.add_argument('--gr', default=0.0, type=float, help='gaussian rate(default: 0.01)')
    parser.add_argument('--dr', default=0.5, type=float, help='dropout rate(default: 0.1)')
    parser.add_argument('--batchnorm', default=False, type=booltype, help='batch normalization(default: False)')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate(default: 0.001)')
    parser.add_argument('--skf', default=0, type=int, help='stratified k-fold')

    parser.add_argument('--seed', default=1, type=int, help='Seed for random number generator')
    parser.add_argument('--cuda', default=0, type=int, help='gpu number')                                           # essential
    parser.add_argument('--device', default='', type=str, help='device when cuda is available')

    args = parser.parse_args()

    # 체크 
    if args.model not in ["rgb", "depth", "both", "proposed"]:
         print("You need to checkout model [rgb, depth, both, proposed]")
         sys.exit(0)

    # 결과 파일 path 설정 
    time_string = time.strftime('%Y-%m-%d_%I:%M_%p', time.localtime(time.time()))
    args.save_path = args.save_path + f'{args.message}' + '_' + f'{time_string}'
    args.save_path_valid = args.save_path + '/valid'
    args.save_path_test = args.save_path + '/test'

    if not os.path.exists(args.save_path): 
        os.makedirs(args.save_path)    
    if not os.path.exists(args.save_path_valid): 
        os.makedirs(args.save_path_valid)
    if not os.path.exists(args.save_path_test): 
        os.makedirs(args.save_path_test)
    
    # weight 파일 path
    args.checkpoint_path = f'/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.message}'
    if not os.path.exists(args.checkpoint_path): 
        os.makedirs(args.checkpoint_path)

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

    # logger 
    global logger 
    logger = Logger(f'{args.save_path}/logs.logs')

    # data loader
    train_loader, valid_loader, test_loader = Facedata_Loader(train_size=64, test_size=64, use_lowdata=args.lowdata, dataset=args.dataset)
    
    # train 코드
    epoch, threshold = train(args, train_loader, valid_loader)
    weight_path = f"{args.checkpoint_path}/epoch_{epoch}_model.pth"

    # test 코드
    test(args, test_loader, weight_path, threshold)
   


