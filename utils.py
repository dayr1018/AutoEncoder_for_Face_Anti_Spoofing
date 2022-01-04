import matplotlib
import matplotlib.pyplot as plt
from torch.serialization import location_tag  
matplotlib.use('Agg')
plt.switch_backend('agg')
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

def image_crop_f(image):
    image_crops = []
    for i in range(3):
        for j in range(3):
            x_ = i*24
            y_ = j*24
            w_ = x_+64
            h_ = y_+64
            # img_crop = image[:,x_:w_,y_:h_]
            img_crop = image[:,:,x_:w_,y_:h_]
            image_crops.append(img_crop)
    return image_crops

def plot_figure(path, train_loss, eval_loss):

    train_x = np.array([i for i in range(len(train_loss))])
    train_y = np.array(train_loss)
    eval_x = np.array([i for i in range(len(eval_loss))])
    eval_y = np.array(eval_loss)

    fig = plt.figure()
    plt.title("Loss Graph")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_x, train_y, lw=1, label='train')
    plt.plot(eval_x, eval_y, lw=1, label='evaluation')    
    plt.legend(loc='upper left')
    plt.savefig(path+f'/loss.png')
    plt.close(fig)

def plot_roc_curve(path, title_info, y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = auc(fpr, tpr)

    fig = plt.figure()
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")    
    plt.plot(fpr, tpr, color='red', label=title_info)
    plt.plot([0,1], [0,1], color='green', linestyle='--') 
    plt.legend(loc='upper left')
    plt.text(0.55, 0.1, f"Area Under the Curve:{ auc_value:4f}")
    plt.savefig(path+f'/{title_info}roc_curve.png')
    plt.close(fig)

    return auc_value

# def plot_eval_per_epoch(path, title_info, accuracy, precision, recall, epoch):
#     fig = plt.figure()
#     plt.title(f"Accuracy, Precision, Recall - {title_info}")
#     plt.xlabel("Epoch")
#     plt.ylabel("Result")
#     plt.plot(epoch, accuracy, color="red", label="Accuracy")
#     plt.plot(epoch, precision, color="blue", label="Precision")
#     plt.plot(epoch, recall, color="green", label="Recall")
#     plt.legend(loc='upper left')
#     plt.savefig(path+f'/{title_info}_accuracy_reesult.png')
#     plt.close(fig)


def cal_metrics(y_true, y_pred):
    
    if len(y_true) != len(y_pred):
        print("Evalution length Error - y_true, y_pred")
        return 0,0,0
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    return accuracy, precision, recall

def plot_3_kind_data(path, filename, y1, y2, y3):
    x1 = np.arange(1, len(y1)+1, 1)
    x2 = np.arange(1, len(y2)+1, 1)
    x3 = np.arange(1, len(y3)+1, 1)
    
    fig = plt.figure()
    plt.title(f"Training Data Distrbution (Light)")
    plt.xlabel("Data Number")
    plt.ylabel("MSE Value")
    plt.plot(x1, y1, 'ro', label=f"High Light Data({len(y1)})")
    plt.plot(x2, y2, 'go', label=f"Mid Light Data({len(y2)})")
    plt.plot(x3, y3, 'bo', label=f"Low Light Data({len(y3)})")
    plt.legend(loc='upper right', fontsize='x-small')
    plt.savefig(path+f'/{filename}.png')
    plt.close(fig)

def plot_real_fake_data(path, filename, y1, y2):
    x1 = np.arange(1, len(y1)+1, 1)
    x2 = np.arange(1, len(y2)+1, 1)
    
    fig = plt.figure()
    plt.title(f"Training Data Distrbution (Real, Fake)")
    plt.xlabel("Data Number")
    plt.ylabel("MSE Value")
    plt.plot(x1, y1, 'ro', label=f"Real Data({len(y1)})")
    plt.plot(x2, y2, 'bo', label=f"Fake Data({len(y2)})")
    plt.legend(loc='upper right', fontsize='x-small')
    plt.savefig(path+f'/{filename}.png')
    plt.close(fig)

def plot_result(path, x, y1, y2, y3):

    # x : threshold
    # y1 : accuracy
    # y2 : precision
    # y3 : recall

    x1 = [] 
    x2 = [] 
    x3 = []
    for i in range(len(x)):
        x1.append(x[i])
        x2.append(x[i]+1)
        x3.append(x[i]+2)

    fig = plt.figure()
    plt.title(f"Accuracy, Precision, Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Result")
    bar1 = plt.bar(x1, y1, color='r', label="Accuracy", width=1) 
    bar2 = plt.bar(x2, y2, color='g', label="Precision", width=1)
    bar3 = plt.bar(x3, y3, color='b', label="Recall", width=1)   

    # for rect in bar1:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width()/2.0, height, height, color='r', ha='center', va='bottom', size='xx-small')
    # for rect in bar2:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width()/2.0, height, height, color='b', ha='center', va='bottom', size='xx-small')
    # for rect in bar3:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width()/2.0, height, height, color='b', ha='center', va='bottom', size='xx-small')
    
    for i, v in enumerate(x):
        plt.text(v, y1[i], y1[i], 
        fontsize='x-small', 
        horizontalalignment='center', 
        verticalalignment='top')
        plt.text(v, y2[i], y2[i], 
        fontsize='x-small', 
        horizontalalignment='center', 
        verticalalignment='top')
        plt.text(v, y3[i], y3[i], 
        fontsize='x-small', 
        horizontalalignment='center', 
        verticalalignment='top')
    plt.legend(loc='upper right', fontsize='x-small')
    plt.savefig(path+f'/Result_Accuracy_Presicion_Recall.png')
    plt.close(fig)