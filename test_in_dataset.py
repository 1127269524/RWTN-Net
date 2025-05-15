# 计算损失与测试数据集

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


criteon = nn.BCEWithLogitsLoss()  # sigmoid + 交叉熵


def fs_(pred, gt):
    ref = gt.flatten() == 255
    hyf = pred.flatten() >= 0.5
    ac, re, fscore, _ = precision_recall_fscore_support(
    ref, hyf, pos_label=1, average='binary')
    auc = roc_auc_score(ref, hyf)
    return ac, re, fscore, auc



def VGG19_LOSS_2(outputs,label):

   loss0 = criteon(outputs[0], label)
   loss1 = criteon(outputs[1], label)


   loss = loss0 + loss1

   return loss

def test_loss_and_f1score_(model,test_loader,bs,device,loss_func):
    F1_Score = []
    total_loss = 0
    total = len(test_loader.dataset) / bs

    with torch.no_grad():
        for step, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast():
                logits = model(x)
                y = torch.mean(y, dim=1).unsqueeze(1)
                loss = loss_func(logits, y)  # 交叉熵误差
                #防止损失值出现nan
            if loss.item() == loss.item():
                total_loss += loss.item() / total
            elif loss.item() != loss.item(): # 如果loss出现nan，则跳过本次循环
                print('损失值出错,','准备跳过本step')
                continue

            logits0 = logits[0]

            # 计算F1 Score

            sig_logits0 = torch.sigmoid(logits0)

            for i in range(logits0.shape[0]):
                F1_Score.append(get_F1(sig_logits0[i][0], y[i][0]))


        f1_len = len(F1_Score)
        F1_Score = np.array(F1_Score)


        f1 = np.sum(F1_Score) / f1_len



        return total_loss,f1



def test_Pre_Rec_f1score(model,test_loader,device):
    Pre = []
    Rec = []
    F1_Score = []

    with torch.no_grad():
        for step, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast():
                logits = model(x)
                y = torch.mean(y, dim=1).unsqueeze(1)

            logits0 = logits[0]

            # 计算F1 Score

            sig_logits0 = torch.sigmoid(logits0)

            for i in range(logits0.shape[0]):
                P, R, F = get_Precision_Recall_F1(sig_logits0[i][0], y[i][0])
                Pre.append(P)
                Rec.append(R)
                F1_Score.append(F)


        f1_len = len(F1_Score)

        F1_Score = np.array(F1_Score)
        f1 = np.sum(F1_Score) / f1_len

        Pre = np.array(Pre)
        pre = np.sum(Pre) / f1_len

        Rec = np.array(Rec)
        rec = np.sum(Rec) / f1_len

        return pre,rec,f1

# Recall
def get_sensitivity(SR, GT, threshold=0.5):


    SR = SR > threshold
    GT = GT > threshold




    TP = ((SR == 1).float() + (GT == 1).float()) == 2
    FN = ((SR == 0).float() + (GT == 1).float()) == 2



    if (float(torch.sum(TP + FN)))==0:
        SE=0
    else:
        SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)))
    return SE


# Precision
def get_precision(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT > threshold



    TP = ((SR == 1).float() + (GT == 1).float()) == 2
    FP = ((SR == 1).float() + (GT == 0).float()) == 2


    if (float(torch.sum(TP + FP)))==0:
        PC=0
    else:
        PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)))


    return PC


def get_F1(SR, GT, threshold=0.5):

    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC)

    return F1


def get_Precision_Recall_F1(SR, GT, threshold=0.5):

    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    if (SE+PC)==0:
        F1=0
    else:
        F1 = 2 * SE * PC / (SE + PC)

    return PC,SE,F1



