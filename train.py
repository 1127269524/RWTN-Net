
import os
import random

# import matplotlib.pyplot as plt
import matplotlib
import numpy

import test_in_dataset
from utils import Interpolation_Coefficient

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from tqdm import trange
from torchvision import transforms
from PIL import Image
import tqdm
import time
from RWTN_Net import RWTN_Net
from test_in_dataset import VGG19_LOSS_2 as VGG19_LOSS,test_loss_and_f1score_
import os


class MyDataset(torch.utils.data.Dataset):
    def __init__(self,root,mode,resize):
        super(MyDataset,self).__init__()
        self.root = root
        self.resize = resize
        self.mode = mode

        if mode == 'train':
            self.png_root = os.path.join(root, 'train_image')  # png文件路径
            self.all_root = os.path.join(root, 'train_gt')  # 标签文件路径
        elif mode == 'test':
            self.png_root = os.path.join(root, 'test_image')
            self.all_root = os.path.join(root, 'test_gt')
        else:
            print('mode参数设置错误')

        self.len_data = len(os.listdir(self.png_root))

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):

        img = os.path.join(self.png_root, f'image_{idx}.png')
        label = os.path.join(self.all_root, f'mask_{idx}.png')


        tf1 = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize((int(self.resize), int(self.resize))),
                transforms.ToTensor()])
        tf2 = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize((int(self.resize), int(self.resize))),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

        img,label  = tf2(img),tf1(label)



        return img,label



def test_Pre_Rec_f1score(model, test_loader, device):
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



            sig_logits0 = torch.sigmoid(logits0)

            for i in range(logits0.shape[0]):
                P, R, F = test_in_dataset.get_Precision_Recall_F1(sig_logits0[i][0], y[i][0])
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

        return pre, rec, f1



if __name__ == '__main__':

    img_size = 320 # 数据集图像大小
    bs = 16        # batch size 16
    lr_ = 0.01     # 学习率  0.01
    epochs = 100    # epoch数
    n = 20         # 测试频率
    begin_epoch = 0
    optima = '普通交叉熵与SGD'
    half_tensor = True


    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)
    # 权重文件夹，之后将在这里自动保存模型权重，以及记录日志信息
    weight_root = os.path.join(current_dir, 'weights')


    dataset_path ='D:/study/copymove_img'

    # USC-ISI训练集加载 ============================================================================================================
    train_db = MyDataset(os.path.join(dataset_path,'uscisi'), 'train', img_size)
    train_loader = torch.utils.data.DataLoader(train_db, batch_size=bs, shuffle=True,num_workers=4)

    test_loader=None
    # Coverage数据集
    coverage_db = MyDataset(os.path.join(dataset_path,'COVERAGE'), 'test', img_size)
    coverage_loader  = torch.utils.data.DataLoader(coverage_db , batch_size=bs, shuffle=False,num_workers=4)

    # CoMoFoD数据集
    CoMoFoD_db = MyDataset(os.path.join(dataset_path,'CoMoFoD_small_v2'), 'test', img_size)
    CoMoFoD_loader  = torch.utils.data.DataLoader(CoMoFoD_db , batch_size=bs, shuffle=False,num_workers=4)

    # CASIA数据集
    CASIA_db = MyDataset(os.path.join(dataset_path,'CASIA'), 'test', img_size)
    CASIA_loader = torch.utils.data.DataLoader(CASIA_db, batch_size=bs, shuffle=False,num_workers=4)



    load_pth =False
    # 是否测试USC-ISI的测试集
    test_USC = False

    # 训练设置
    device = torch.device('cuda:0')
    Coefficient_3 = Interpolation_Coefficient(3)
    Coefficient_3=Coefficient_3.to(device)

    model = RWTN_Net(Coefficient_3=Coefficient_3).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_, momentum=0.9, nesterov=True)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                     patience=30, verbose=True, cooldown=0, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler()


    # 加载预训练权重
    if load_pth:
        weight_path = os.path.join(weight_root, 'last.pth')
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        last_epoch = checkpoint['epoch']
        try:  # 防止原版本没有保存记录
            best_f1, best_cov_f1, best_comofod, best_CoMoFoD, best_CASIA,best_three_f1,best_nine = checkpoint['best_f1'], checkpoint['best_cov_f1'], checkpoint['best_comofod'], checkpoint['best_CoMoFoD'], checkpoint['best_CASIA'],checkpoint['best_three_f1'],checkpoint['best_nine'] # 记录最好的状态
        except:
            try:
                best_f1, best_cov_f1, best_comofod, best_CoMoFoD,best_three_f1 ,best_nine= checkpoint['best_f1'], checkpoint['best_cov_f1'], checkpoint['best_comofod'], checkpoint['best_CoMoFoD'],checkpoint['best_three_f1'],checkpoint['best_nine']  # 记录最好的状态
                best_CASIA = 0.30  # 防止以前没有记录best_CASIA
            except:
                best_f1, best_cov_f1, best_comofod, best_CoMoFoD, best_CASIA,best_three_f1,best_nine = 0.3, 0.2, 0.2, 0.20, 0.20,0.20,0.20  # 记录最好的状态
        print('上次epoch为：',last_epoch,' 已加载上次pth文件。','最佳记录：best_f1:%f, best_cov_f1:%f, best_comofod:%f, best_CoMoFoD:%f, best_CASIA:%f,best_three_f1:%f,best_nine:%f'%(best_f1, best_cov_f1, best_comofod, best_CoMoFoD, best_CASIA,best_three_f1,best_nine))
    else:
        last_epoch = -1
        best_f1, best_cov_f1, best_comofod, best_CoMoFoD, best_CASIA,best_three_f1,best_nine = 0.1, 0.1, 0.1, 0.1, 0.1,0.1,0.1 # 记录最好的状态

    logRoot = os.path.join(weight_root, 'log_.txt')
    # if load_pth == False:
    with open(logRoot, 'a') as f:
        str_ = '============================= img_size:%d, batch size:%d, 学习率:%.5f, 损失和优化器:%s ============================= \n' % (img_size, bs, lr_, optima)
        f.write(str_)

    print(time.strftime('当前时间：%H:%M:%S =====================================================================',time.localtime(time.time())))

    for epoch in trange(epochs):
        #训练===============================================================================================================
        epoch = epoch + last_epoch + 1  # 更新epoch数
        train_total_loss = 0  #
        train_total = len(train_loader.dataset) / bs  # 总的数量

        model.train() #训练模式
        for step, (x, y) in enumerate(train_loader):  # 循环取batch
            x, y = x.to(device), y.to(device) # 每次只把一个batch的数据放进显存
            if half_tensor:  # 是否开启半精度
                with torch.cuda.amp.autocast():   # 半精度训练
                    logits = model(x)
                    y = torch.mean(y, dim=1).unsqueeze(1)
                    loss = VGG19_LOSS(logits, y)  # 交叉熵误差
            else:
                logits = model(x)
                y = torch.mean(y, dim=1).unsqueeze(1)
                loss = VGG19_LOSS(logits, y)  # 交叉熵误差

            if loss.item() == loss.item():
                train_total_loss += loss.item() / train_total

                if half_tensor:  # 是否开启半精度
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()  # 看是否要增大scaler
                else:
                    optimizer.zero_grad()
                    loss.backward()  # 计算梯度
                    optimizer.step()  # 更新参数


            elif loss.item() != loss.item(): # 如果loss出现nan，则跳过本次循环
                optimizer.zero_grad()
                print('出错step为：',step,' ,已清空优化器梯度,准备跳过本step')
                continue




            # 打印训练进度
            percent = round((step + 1) * 100 / train_total,2)
            print('\r', end='')
            print(f'训练进度：{percent}%', end='')


            # 按step进行测试
            if ((step + 1) % int(train_total/n+1) == 0 and epoch >= begin_epoch) or (step + 1) == int(train_total):


                model.eval()  # 开启验证模式



                # 原数据集测试
                if test_USC and ((step + 1)  == int(train_total)):
                    total_loss , f1 = test_loss_and_f1score_(model=model, test_loader=test_loader, bs=bs, device=device,loss_func=VGG19_LOSS)
                else:
                    total_loss, f1 = 0. , 0.

                # Coverage数据集测试
                cov_pre,cov_rec,cov_f1 = test_Pre_Rec_f1score(model=model, test_loader=coverage_loader, device=device)

                # CoMoFoD数据集测试
                CoMoFoD_pre,CoMoFoD_rec,CoMoFoD_f1 = test_Pre_Rec_f1score(model=model, test_loader=CoMoFoD_loader, device=device)

                # CASIA数据集测试
                CASIA_pre,CASIA_rec,CASIA_f1 = test_Pre_Rec_f1score(model=model, test_loader=CASIA_loader, device=device)

                # Coverage与CoMoFoD的联合数据集数据集测试
                comofod_f1 = cov_f1 * 0.6667 + CoMoFoD_f1 * 0.3333
                # 三个数据集的精度
                three_f1 = (cov_f1 + CoMoFoD_f1 + CASIA_f1*2) / 4
                # 三个数据集九个指标的精度
                nine=(cov_pre+cov_rec+cov_f1+CoMoFoD_pre+CoMoFoD_rec+CoMoFoD_f1+CASIA_pre*2+CASIA_rec*2+CASIA_f1*2)/12

                try:
                    # print('epoch:%d, step:%d, 最大值：%.3f, 最小值：%.3f, 平均值：%.3f' % (epoch, step, torch.max(torch.sigmoid(logits[0])).item(), torch.min(torch.sigmoid(logits[0])).item(),torch.mean(torch.sigmoid(logits[0])).item()))
                    print('训练集损失:%f' % train_total_loss, ',测试集损失:%f' % total_loss, ',USC val F1 Score:%f' % f1, ',combine F1 Score:%4f' % comofod_f1,',Cov F1 Score:%4f' % cov_f1,"CoMoFoD_f1:%.4f" % CoMoFoD_f1, "CASIA_f1:%.4f"%CASIA_f1,"best_three_f1:%.4f"%best_three_f1)
                except:
                    print('测试集损失:%f' % total_loss, ',F1 Score:%f' % f1, ',Cov F1 Score:%f' % cov_f1)



                # 记录日志信息至txt文件
                with open(logRoot, 'a') as f:
                    str_ = ('epoch:%d, step:%d, 训练集损失:%f, comofod_f1:%.4f, three_f1:%.4f, nine:%.4f, COVERAGE: '
                            'PRE:%.4f, REC:%.4f, F1:%.4f, CoMoFoD: PRE:%.4f, REC:%.4f, F1:%.4f, '
                            'CASIA: PRE:%.4f, REC:%.4f, F1:%.4f, best_comofod:%.4f, best_three_f1:%.4f, best_nine:%.4f') % (
                               epoch, step, train_total_loss, comofod_f1, three_f1, nine,
                               cov_pre, cov_rec, cov_f1,
                               CoMoFoD_pre, CoMoFoD_rec, CoMoFoD_f1,
                               CASIA_pre, CASIA_rec, CASIA_f1,
                               best_comofod, best_three_f1, best_nine
                           )
                    f.write(str_)
                train_total_loss = 0


                flag = 0

                # 如果模型达到新的最优状态，则保存权重文件pt文件
                if f1  > best_f1:
                    best_f1 = f1
                    # 保存模型权重pt文件
                    f1Root = os.path.join(weight_root, 'best_val.pth')
                    torch.save({'epoch': epoch,'best_f1':best_f1,'best_cov_f1':best_cov_f1,'best_comofod':best_comofod,'best_CoMoFoD':best_CoMoFoD,'best_CASIA':best_CASIA,
                                'model': model.state_dict(),'optimizer': optimizer.state_dict(),'scaler': scaler.state_dict()}, f1Root)
                    print('f1达到新记录，已保存最佳状态; ')
                    flag += 1

                if comofod_f1 > best_comofod:
                    best_comofod = comofod_f1
                    # 保存模型权重pt文件
                    comofodRoot = os.path.join(weight_root, 'best_comofod.pth')
                    torch.save({'epoch': epoch,'best_f1':best_f1,'best_cov_f1':best_cov_f1,'best_comofod':best_comofod,'best_CoMoFoD':best_CoMoFoD,'best_CASIA':best_CASIA,
                                'model': model.state_dict(),'optimizer': optimizer.state_dict(),'scaler': scaler.state_dict()}, comofodRoot)
                    print('comofod达到新记录，已保存最佳状态; ')
                    flag += 1

                    with open(logRoot, 'a') as f:
                        str_ = '两者同时达成: best_comofod:%.4f,  ' % (best_comofod)
                        f.write(str_)

                if three_f1 > best_three_f1:
                    best_three_f1 = three_f1
                    # 保存模型权重pt文件
                    threeRoot = os.path.join(weight_root, 'best_three_f1.pth')
                    torch.save(
                        {'epoch': epoch, 'best_f1': best_f1, 'best_cov_f1': best_cov_f1, 'best_comofod': best_comofod,
                         'best_CoMoFoD': best_CoMoFoD, 'best_CASIA': best_CASIA,'best_three_f1':best_three_f1,
                         'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'scaler': scaler.state_dict()}, threeRoot)
                    print('three_f1达到新记录，已保存最佳状态; ')
                    flag += 1

                    with open(logRoot, 'a') as f:
                        str_ = '三者同时达成 best_three_f1:%.4f ' % (best_three_f1)
                        f.write(str_)

                if nine > best_nine:
                    best_nine = nine
                    # 保存模型权重pt文件
                    nineRoot = os.path.join(weight_root, 'best_nine.pth')
                    torch.save(
                        {'epoch': epoch, 'best_f1': best_f1, 'best_cov_f1': best_cov_f1, 'best_comofod': best_comofod,
                         'best_CoMoFoD': best_CoMoFoD, 'best_CASIA': best_CASIA,'best_three_f1':best_three_f1,'best_nine':best_nine,
                         'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'scaler': scaler.state_dict()}, nineRoot)
                    print('nine达到新记录，已保存最佳状态; ')
                    flag += 1

                    with open(logRoot, 'a') as f:
                        str_ = ('同时达成: COVERAGE: PRE:%.4f, REC:%.4f,'
                                ' F1:%.4f, CoMoFoD: PRE:%.4f, REC:%.4f, F1:%.4f, '
                                'CASIA: PRE:%.4f, REC:%.4f, F1:%.4f, best_nine:%.4f') % (
                                   cov_pre, cov_rec, cov_f1,
                                   CoMoFoD_pre, CoMoFoD_rec, CoMoFoD_f1,
                                   CASIA_pre, CASIA_rec, CASIA_f1,
                                   best_nine
                               )
                        f.write(str_)

                if flag >0:
                    with open(logRoot, 'a') as f:
                        str_ = ',  新记录***\n'
                        f.write(str_)
                else:
                    with open(logRoot, 'a') as f:
                        str_ = '\n'
                        f.write(str_)


                if (step+1) == int(train_total):
                    # 直接保存最后一个epoch的权重
                    lastRoot = os.path.join(weight_root, 'last.pth')
                    torch.save(
                        {'epoch': epoch, 'best_f1': best_f1, 'best_cov_f1': best_cov_f1, 'best_comofod': best_comofod,'best_CoMoFoD': best_CoMoFoD,'best_CASIA':best_CASIA,'best_three_f1':best_three_f1,'best_nine':best_nine,
                         'model': model.state_dict(), 'optimizer': optimizer.state_dict(),'scaler': scaler.state_dict()}, lastRoot)
                    print('当前epoch:%d, 当前最佳记录：best_f1:%f, best_cov_f1:%f, best_combine:%f, best_CoMoFoD:%f, best_CASIA:%f,best_three_f1:%f' % (epoch, best_f1, best_cov_f1, best_comofod, best_CoMoFoD,best_CASIA,best_three_f1))
                    print(time.strftime('当前时间：%H:%M:%S =====================================================================',time.localtime(time.time())))

                model.train()  # 测试完开启训练模式
        if epoch >= 20:
            scheduler.step(best_nine)
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch: {epoch+1}, best_nine: {best_nine:.4f}, Learning Rate: {current_lr:.6f},best:{scheduler.best:.4f},num_bad_epochs:{scheduler.num_bad_epochs}')
