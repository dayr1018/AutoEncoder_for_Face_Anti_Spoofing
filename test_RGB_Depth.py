import torch
import time
import argparse

from models.Auto_Encoder_RGB_Depth import Depth_layer3_4to3, Depth_layer3_4to4, Depth_layer4_4to3, Depth_layer4_4to4, Depth_layer5_4to4
from models.Auto_Encoder_RGB_Depth import Depth_layer3_1to1, Depth_layer4_1to1, Depth_layer5_1to1
from ad_dataloader.dataloader_RGB_Depth import Facedata_Loader
from loger import Logger
from utility import plot_roc_curve, cal_metrics, plot_3_kind_data, plot_real_fake_data, plot_histogram

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
parser.add_argument('--save-path', default='../ad_output/logs/Test/', type=str, help='logs save path')
parser.add_argument('--checkpoint', default='', type=str, help='checkpoint path')
parser.add_argument('--message', default='', type=str, help='pretrained model checkpoint')
parser.add_argument('--model', default='', type=str, help='model')
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

    if "layer3_4to3" in args.model:
        model = Depth_layer3_4to3()
        print("**** You're training 'depth_layer3_4to3' model.")
    elif "layer3_4to4" in args.model:
        model = Depth_layer3_4to4()
        print("**** You're training 'depth_layer3_4to4' model.")
    elif "layer4_4to3" in args.model:
        model = Depth_layer4_4to3()
        print("**** You're training 'depth_layer4_4to3' model.")
    elif "layer4_4to4" in args.model:
        model = Depth_layer4_4to4()
        print("**** You're training 'Depth_layer4_4to4' model.")
    elif "layer5_4to4" in args.model:
        model = Depth_layer5_4to4()
        print("**** You're training 'Depth_layer5_4to4' model.")
    elif "layer3_1to1" in args.model:
        model = Depth_layer3_1to1()
        print("**** You're training 'Depth_layer3_1to1' model.")
    elif "layer4_1to1" in args.model:
        model = Depth_layer4_1to1()
        print("**** You're training 'Depth_layer4_1to1' model.")    
    elif "layer5_1to1" in args.model:
        model = Depth_layer5_1to1()
        print("**** You're training 'Depth_layer5_1to1' model.") 

    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print('device :', device)
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

    logger.Print(f"***** << Test threshold({threshold}) >>")  

    with torch.no_grad():
        for data in data_loader:

            rgb_image, depth_image, label, rgb_path = data
            rgb_image = rgb_image.to(device)
            depth_image = depth_image.to(device)
            label = label.to(device)

            # depth 모델에 따라 기준 input 달라짐 (v1: 4channel, v2: 3channel)
            if "4to3" in args.model:
                input_image = torch.cat((rgb_image, depth_image), dim=1)
                standard_image = rgb_image
            elif "4to4" in args.model:
                input_image = torch.cat((rgb_image, depth_image), dim=1)
                standard_image = input_image
            elif "1to1" in args.model:
                input_image = depth_image
                standard_image = depth_image

            # 모델 태우기 
            # recons_rgb, recons_depth = model(rgb_image, depth_image)
            recons_image = model(input_image)

            # 모든 데이터에 대한 MSE 구하기 
            for i in range(len(standard_image)):
                np_image = standard_image[i].cpu().detach().numpy()
                np_recons_image = recons_image[i].cpu().detach().numpy()

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
    plot_histogram(f"{save_path}/graph", f"Valid_Historgram_", "x", data_real, data_fake)

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

    checkpoint_layer3_4to4_g10 = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint}/epoch_1100_model.pth"
    checkpoint_layer3_4to4_g50 = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint}/epoch_1070_model.pth"
    checkpoint_layer3_4to4_g100 = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint}/epoch_290_model.pth"

    checkpoint_layer4_4to4_g10 = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint}/epoch_0_model.pth"
    checkpoint_layer4_4to4_g50 = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint}/epoch_1170_model.pth"
    checkpoint_layer4_4to4_g100 = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint}/epoch_1690_model.pth"

    checkpoint_layer5_4to4_g10 = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint}/epoch_1000_model.pth"
    checkpoint_layer5_4to4_g50 = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint}/epoch_0_model.pth"    
    checkpoint_layer5_4to4_g100 = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint}/epoch_0_model.pth"


    checkpoint_layer3_1to1 = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint}/epoch_0_model.pth"
    checkpoint_layer4_1to1 = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint}/epoch_0_model.pth"
    checkpoint_layer5_1to1 = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint}/epoch_0_model.pth"

    c_uni = f"/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/{args.checkpoint}/epoch_110_model.pth"
    
    logger.Print(f"You're conducting '{args.checkpoint}' weight.")
    checkpoint=""
    if "0128_layer3_4to4" in args.checkpoint:
        checkpoint = checkpoint_layer3_4to4_g10
    elif "0128_50_layer3_4to4" in args.checkpoint:
        checkpoint = checkpoint_layer3_4to4_g50
    elif "0128_100_layer3_4to4" in args.checkpoint:
        checkpoint = checkpoint_layer3_4to4_g100     

    elif "0128_layer4_4to4" in args.checkpoint:
        checkpoint = checkpoint_layer4_4to4_g10
    elif "0128_50_layer4_4to4" in args.checkpoint:
        # checkpoint = checkpoint_layer4_4to4_g50
        checkpoint = c_uni
    elif "0128_100_layer4_4to4" in args.checkpoint:
        checkpoint = checkpoint_layer4_4to4_g100

    elif "0128_layer5_4to4" in args.checkpoint:
        checkpoint = checkpoint_layer5_4to4_g10
    elif "0128_50_layer5_4to4" in args.checkpoint:
        checkpoint = checkpoint_layer5_4to4_g50
    elif "0128_100_layer5_4to4" in args.checkpoint:
        checkpoint = checkpoint_layer5_4to4_g100

    elif "0128_layer3_1to1" in args.checkpoint:
        checkpoint = checkpoint_layer3_1to1
    elif "0128_layer4_1to1" in args.checkpoint:
        checkpoint = checkpoint_layer4_1to1

    elif "0128_layer5_1to1" in args.checkpoint:
        checkpoint = checkpoint_layer5_1to1
    logger.Print(f"Weight file is '{checkpoint}'.")

    accuracy, precision, recall, f1 = test(test_loader, args.threshold, checkpoint)

    ## 결과 파일 따로 저장 
    logger.Print(f"Result (Max Accuracy)")
    logger.Print(f"Accuracy: {float(accuracy):3f}")
    logger.Print(f"Precisioin: {float(precision):3f}")
    logger.Print(f"Recall: {float(recall):3f}")
    logger.Print(f"F1: {float(f1):3f}")
    logger.Print(f"Threshold: {args.threshold}")
    