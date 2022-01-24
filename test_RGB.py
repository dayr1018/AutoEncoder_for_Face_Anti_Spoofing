import torch
import time
import argparse

from models.Auto_Encoder_RGB import Auto_Encoder_Original, Auto_Encoder_Dropout, Auto_Encoder_layer4
from ad_dataloader.dataloader_RGB import Facedata_Loader
from loger import Logger
from utils import plot_roc_curve, cal_metrics, plot_3_kind_data, plot_real_fake_data, plot_result, find_max_accuracy

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from PIL import Image

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

# argument parser
parser = argparse.ArgumentParser(description='face anto-spoofing')
parser.add_argument('--save-path', default='../ad_output/RGB/logs/Test/', type=str, help='logs save path')
parser.add_argument('--checkpoint', default='', type=str, help='checkpoint path')
parser.add_argument('--message', default='', type=str, help='pretrained model checkpoint')
parser.add_argument('--lowdata', default=True, type=booltype, help='whether low data is included')
parser.add_argument('--datatype', default=0, type=int, help='data set type')
parser.add_argument('--threshold', default='', type=int, help='threshold')
parser.add_argument('--skf', default=0, type=int, help='stratified k-fold')
args = parser.parse_args()

time_object = time.localtime(time.time())
time_string = time.strftime('%Y-%m-%d_%I:%M_%p', time_object)

save_path = args.save_path + f'{args.message}' + '_' + f'{time_string}'
if not os.path.exists(save_path):
    os.makedirs(save_path)

logger = Logger(f'{save_path}/logs.logs')
logger.Print(time_string + " - " + args.message + "\n")

def test(data_loader, threshold, checkpoint):

    if "original" in args.checkpoint:
        model = Auto_Encoder_Original()        
        print("***** You're training 'original' model.")
    elif "dropout" in args.checkpoint:
        model = Auto_Encoder_Dropout()
        print("***** You're training 'dropout' model.")
    elif "layer4" in args.checkpoint:
        model = Auto_Encoder_layer4()
        print("***** You're training 'layer4' model.")

    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda:
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))) #  device_ids=[0, 1, 2]
        model.cuda()
        model.load_state_dict(torch.load(checkpoint))
    else:
        print("Something Wrong, Cuda Not Used")

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

    for data in data_loader:

        rgb_image, label, rgb_path = data
        rgb_image = rgb_image.cuda()
        label = label.cuda()

        # 모델 태우기 
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
            y_true.append(label[i].cpu().detach().numpy())
            y_prob.append(mse_by_sklearn)
            if mse_by_sklearn < threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
                
    # light, mask유무 에 따른 데이터분포 그리기
    if not os.path.exists(f'{save_path}/graph'):
        os.makedirs(f'{save_path}/graph')
    plot_3_kind_data(f"{save_path}/graph", "data_distribution(light)", "x", data_high, data_mid, data_low)       
    plot_real_fake_data(f"{save_path}/graph", "data_distribution(real,fake)", "x", data_real, data_fake)

    # 모델 평가지표 및 그리기 
    plot_roc_curve(f"{save_path}/graph", f"threshold({threshold})", y_true, y_prob)
    
    accuracy, precision, recall, f1 = cal_metrics(y_true, y_pred)

    return accuracy, precision, recall, f1

if __name__ == "__main__":

    ## Threshold 에 따른 모델 성능 출력 
    _, _, test_loader = Facedata_Loader(train_size=64, test_size=64, use_lowdata=args.lowdata, dataset=args.datatype)

    if args.threshold == '':
        print("--threshold option is required")
        exit(1)

    checkpoint_original = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/RGB/checkpoint/{args.model}/epoch_1000_model.pth"
    checkpoint_dropout = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/RGB/checkpoint/{args.model}/epoch_1370_model.pth"
    checkpoint_layer4 = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/RGB/checkpoint/{args.model}/epoch_2990_model.pth"

    logger.Print(f"You're conducting '{args.checkpoint}' weight.")
    checkpoint=""
    if "original" in args.checkpoint:
        checkpoint = checkpoint_original
    elif "dropout" in args.checkpoint:
        checkpoint = checkpoint_dropout
    elif "layer4" in args.checkpoint:
        checkpoint = checkpoint_layer4    
    logger.Print(f"Weight file is '{checkpoint}'.")

    accuracy, precision, recall, f1 = test(test_loader, args.threshold, checkpoint)

    ## 결과 파일 따로 저장 
    logger.Print(f"***** Result (Max Accuracy)")
    logger.Print(f"***** Threshold: {args.threshold}")
    logger.Print(f"***** Accuracy: {float(accuracy):3f}")
    logger.Print(f"***** Precisioin: {float(precision):3f}")
    logger.Print(f"***** Recall: {float(recall):3f}")
    logger.Print(f"***** F1: {float(f1):3f}")
    