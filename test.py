# python test.py --depth true --model layer4_4to4 --checkpoint_path 0202_layer4_4to4_g50 --threshold dddd --message test_0202_layer4_4to4_g50 

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
    logger = Logger(f'{args.save_path_test}/logs.logs')
    logger.Print(args)

    checkpoint_test_original = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_15_model.pth"
    checkpoint_test_depth = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_45_model.pth"


    if args.datatype == 0:
        if args.lowdata == True:
            checkpoint_original = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_30_model.pth"
            checkpoint_dropout = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_590_model.pth"
            checkpoint_depth_layer3 = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_0_model.pth"
            checkpoint_depth_layer4 = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_0_model.pth"
            checkpoint_depth_layer5 = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_0_model.pth"
            checkpoint_depth_layer3_nodrop = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_50_model.pth"
            checkpoint_depth_layer4_nodrop = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_20_model.pth"
            checkpoint_depth_layer5_nodrop = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_10_model.pth"
            checkpoint_depth_layer3_dr10 = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_0_model.pth"
            checkpoint_depth_layer4_dr10 = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_0_model.pth"
            checkpoint_depth_layer5_dr10 = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_0_model.pth"

        elif args.lowdata == False:
            checkpoint_original = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_30_model.pth"
            checkpoint_depth_layer3_nodrop = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_50_model.pth"
    
            # 0210 버전은 lowdata 포함시키지 않은 모델임
            checkpoint_0210_original = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_10_model.pth"
            checkpoint_0210_dropout = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch__model.pth"
            checkpoint_0210_depth_layer3_nodrop = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_50_model.pth"
            checkpoint_0210_depth_layer4_nodrop = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_30_model.pth"
            checkpoint_0210_depth_layer5_nodrop = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_20_model.pth"

  
            

    elif args.datatype == 1:
        if args.lowdata == True:
            checkpoint_original = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_10_model.pth"
            checkpoint_depth_layer3_nodrop = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_190_model.pth"

        elif args.lowdata == False:
            checkpoint_original = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_320_model.pth"
            checkpoint_depth_layer3_nodrop = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint_path}/epoch_370_model.pth"


    logger.Print(f"You're conducting '{args.checkpoint_path}' weight.")
    checkpoint=""
    if "0209_original" == args.checkpoint_path:
        checkpoint = checkpoint_original        
    elif "0209_dropout" == args.checkpoint_path:
        checkpoint = checkpoint_dropout    
    elif "0209_depth_layer3" == args.checkpoint_path:
        checkpoint = checkpoint_depth_layer3 
    elif "0209_depth_layer4" == args.checkpoint_path:
        checkpoint = checkpoint_depth_layer4    
    elif "0209_depth_layer5" in args.checkpoint_path:
        checkpoint = checkpoint_depth_layer5 
    elif "0209_depth_layer3_nodrop" == args.checkpoint_path:
        checkpoint = checkpoint_depth_layer3_nodrop 
    elif "0209_depth_layer4_nodrop" == args.checkpoint_path:
        checkpoint = checkpoint_depth_layer4_nodrop    
    elif "0209_depth_layer5_nodrop" in args.checkpoint_path:
        checkpoint = checkpoint_depth_layer5_nodrop 
    elif "0209_depth_layer3_dr10" == args.checkpoint_path:
        checkpoint = checkpoint_depth_layer3_dr10 
    elif "0209_depth_layer4_dr10" == args.checkpoint_path:
        checkpoint = checkpoint_depth_layer4_dr10    
    elif "0209_depth_layer5_dr10" in args.checkpoint_path:
        checkpoint = checkpoint_depth_layer5_dr10 
    elif "0210_original" == args.checkpoint_path:
        checkpoint = checkpoint_0210_original        
    elif "0210_dropout" == args.checkpoint_path:
        checkpoint = checkpoint_0210_dropout    
    elif "0210_depth_layer3_nodrop" == args.checkpoint_path:
        checkpoint = checkpoint_0210_depth_layer3_nodrop 
    elif "0210_depth_layer4_nodrop" == args.checkpoint_path:
        checkpoint = checkpoint_0210_depth_layer4_nodrop    
    elif "0210_depth_layer5_nodrop" in args.checkpoint_path:
        checkpoint = checkpoint_0210_depth_layer5_nodrop 
    elif "0213_TEST_original_v21" in args.checkpoint_path:
        checkpoint = checkpoint_test_original 
    elif "0213_TEST_depth_v21" in args.checkpoint_path:
        checkpoint = checkpoint_test_depth 
    logger.Print(f"Weight file is '{checkpoint}'.")

    # loss 함수 - test MSE 구하기 위해 
    mse = torch.nn.MSELoss(reduction='sum')

    accuracy, precision, recall, f1 = test(args, test_loader, checkpoint, mse)

    ## 결과 파일 따로 저장 
    logger.Print(f"Result (criteria: accuracy)")
    logger.Print(f"Accuracy: {float(accuracy):3f}")
    logger.Print(f"Precisioin: {float(precision):3f}")
    logger.Print(f"Recall: {float(recall):3f}")
    logger.Print(f"F1: {float(f1):3f}")
    logger.Print(f"Threshold: {args.threshold}")


def test(args, test_loader, checkpoint, loss_function):

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

    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    y_true = []
    y_prob = []
    y_pred = []

    data_high = []
    data_mid = []
    data_low = []

    data_real = []
    data_fake = []

    with torch.no_grad():
        for data in test_loader:

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

            # 모든 데이터에 대한 MSE 구하기 
            mse_list = []            
            for i in range(len(standard_image)):

                # 하나의 텐서에 대한 MSE 값 확인 
                mse_per_tensor = loss_function(standard_image[i], recons_image[i]).cpu().detach().numpy()
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

                # 모델 결과값 
                y_true.append(label[i].cpu().detach().numpy())
                y_prob.append(mse_per_tensor)
                if mse_per_tensor < args.threshold:
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
    plot_roc_curve(f"{args.save_path_test}/graph", f"threshold({args.threshold})", y_true, y_prob)
    
    accuracy, precision, recall, f1 = cal_metrics(y_true, y_pred)

    return accuracy, precision, recall, f1

if __name__ == "__main__":

    # args option
    parser = argparse.ArgumentParser(description='face anto-spoofing')

    parser.add_argument('--save-path-test', default='../ad_output/logs/Test/', type=str, help='test logs path')

    parser.add_argument('--model', default='', type=str, help='model')
    parser.add_argument('--checkpoint-path', default='', type=str, help='checkpoint path')
    parser.add_argument('--depth', default=True, type=booltype, help='RGB or Depth(default: Depth)')

    parser.add_argument('--message', default='', type=str, help='pretrained model checkpoint')
    parser.add_argument('--epochs', default=1000, type=int, help='train epochs')
    parser.add_argument('--lowdata', default=True, type=booltype, help='whether low data is included')
    parser.add_argument('--usedrop', default=False, type=booltype, help='whether dropout layer is used')
    parser.add_argument('--datatype', default=0, type=int, help='data set type')
    parser.add_argument('--loss', default=0, type=int, help='0: mse, 1:rapp')
    
    parser.add_argument('--gr', default=1, type=float, help='guassian rate(default: 0)')
    parser.add_argument('--dr', default=0.5, type=float, help='dropout rate(default: 0.1)')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate(default: 0.001)')
    parser.add_argument('--skf', default=0, type=int, help='stratified k-fold')
    parser.add_argument('--threshold', default=0, type=int, help='threshold')

    parser.add_argument('--seed', default=1, type=int, help='Seed for random number generator')
    parser.add_argument('--cuda', default=0, type=int, help='gpu number')    
    parser.add_argument('--device', default='', type=str, help='device when cuda is available')

    args = parser.parse_args()

    # 결과 파일 path 설정 
    time_string = time.strftime('%Y-%m-%d_%I:%M_%p', time.localtime(time.time()))
    args.save_path_test = args.save_path_test + f'{args.message}' + '_' + f'{time_string}'  
    if not os.path.exists(args.save_path_test): os.makedirs(args.save_path_test)

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
    _, _, test_loader = Facedata_Loader(train_size=64, test_size=64, use_lowdata=args.lowdata, dataset=args.datatype)
    
    # main 함수
    main(args, test_loader)

